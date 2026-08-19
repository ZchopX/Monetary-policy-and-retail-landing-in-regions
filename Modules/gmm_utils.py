"""Capability tests for the agreed ConsCred dynamic-panel GMM instrument design."""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import platform
from importlib.metadata import version
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEPENDENT = "d_Int_Rate_ConsCred"
PREDTERMINED = ("Cred_nagr_lag1", "Zakred_lag1", "Cap_to_assets_lag1", "CPI_reg_lag1")
DIRECT = (
    "ln_Fin_Dostup", "d_Inflation_Expectations", "d_Mon_Shock_pos", "d_Mon_Shock_neg",
    "d_Mon_Shock_pos_Cluster_new_cd_1", "d_Mon_Shock_neg_Cluster_new_cd_1",
    "d_Mon_Shock_pos_Cluster_new_cd_3", "d_Mon_Shock_neg_Cluster_new_cd_3",
    "d_Mon_Shock_pos_Cluster_new_cd_4", "d_Mon_Shock_neg_Cluster_new_cd_4",
)
PROFILE_LAGS = {"A": (2, 3), "B": (2, 4), "C": (2, 3)}


def profile_contract(profile: str, estimator: str) -> dict[str, Any]:
    """Return the fixed, collapsed instrument contract for a test attempt."""
    if profile not in PROFILE_LAGS or estimator not in {"difference", "system"}:
        raise ValueError(f"Unknown profile/estimator: {profile}/{estimator}")
    controls_role = "endogenous" if profile == "C" else "predetermined"
    gmm_variables = [f"L1_{DEPENDENT}", *PREDTERMINED]
    return {
        "profile": profile,
        "estimator": estimator,
        "lag_window": list(PROFILE_LAGS[profile]),
        "collapse": True,
        "difference_gmm_levels": gmm_variables,
        "levels_gmm_differences": gmm_variables if estimator == "system" else [],
        "levels_gmm_lag": 1 if estimator == "system" else None,
        "gmm_roles": {f"L1_{DEPENDENT}": "endogenous", **{name: controls_role for name in PREDTERMINED}},
        "direct_regressors": list(DIRECT),
    }


def load_conscred_preflight_data(workbook: Path, start: str, end: str) -> pd.DataFrame:
    """Load the fixed h=0 panel and validate its required structure."""
    data = pd.read_excel(workbook).drop(columns=["Unnamed: 0"], errors="ignore")
    data["Date"] = pd.to_datetime(data["Date"])
    required = {"Region", "Date", DEPENDENT, "d_Mon_Shock_pos", "d_Mon_Shock_neg", "ln_Fin_Dostup", "d_Inflation_Expectations",
                "Cred_nagr", "Zakred", "Cap_to_assets", "CPI_reg", "Cluster_new_cd_1", "Cluster_new_cd_3", "Cluster_new_cd_4"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    data = data.sort_values(["Region", "Date"]).copy()
    if data.duplicated(["Region", "Date"]).any():
        raise ValueError("Duplicate Region-Date keys in the preflight input")
    reconstructed = data["d_Mon_Shock_pos"] + data["d_Mon_Shock_neg"]
    if "d_Mon_Shock" in data and not data["d_Mon_Shock"].round(10).equals(reconstructed.round(10)):
        raise ValueError("Positive and negative shocks do not reconstruct source d_Mon_Shock")
    data["d_Mon_Shock"] = reconstructed
    for column in ("Cred_nagr", "Zakred", "Cap_to_assets", "CPI_reg"):
        data[f"{column}_lag1"] = data.groupby("Region")[column].shift(1)
    for cluster in ("Cluster_new_cd_1", "Cluster_new_cd_3", "Cluster_new_cd_4"):
        for shock in ("d_Mon_Shock_pos", "d_Mon_Shock_neg"):
            data[f"{shock}_{cluster}"] = data[shock] * data[cluster]
    data = data.loc[data["Date"].between(pd.Timestamp(start), pd.Timestamp(end))].copy()
    _validate_preflight_panel(data)
    return data


def _validate_preflight_panel(data: pd.DataFrame) -> None:
    """Reject changed panel structure before a third-party estimator runs."""
    dates = pd.DatetimeIndex(sorted(data["Date"].unique()))
    if not dates.equals(pd.date_range(dates.min(), dates.max(), freq="MS")):
        raise ValueError("Preflight dates are not consecutive monthly observations")
    counts = data.groupby("Region")["Date"].nunique()
    if len(counts) != 75 or counts.nunique() != 1:
        raise ValueError("Preflight requires a balanced 75-region panel")
    clusters = data[["Cluster_new_cd_1", "Cluster_new_cd_3", "Cluster_new_cd_4"]].sum(axis=1)
    if not clusters.isin((0, 1)).all():
        raise ValueError("Cluster encoding is not mutually exclusive with cluster 2 as reference")


def _expected_systemgmmkit_names(contract: dict[str, Any]) -> set[str]:
    lags = range(contract["lag_window"][0], contract["lag_window"][1] + 1)
    expected = {f"D:{variable}:L{lag}" for variable in contract["difference_gmm_levels"] for lag in lags}
    direct_prefix = "IV" if contract["estimator"] == "system" else "D:iv"
    expected.update(f"{direct_prefix}:{variable}" for variable in DIRECT)
    if contract["estimator"] == "system":
        expected.update(f"L:diff:{variable}:L1" for variable in contract["levels_gmm_differences"])
        expected.add("L:constant")
    return expected


def _systemgmmkit_instrument_check(contract: dict[str, Any], names: Any) -> dict[str, Any]:
    """Compare native backend names to the approved collapsed-instrument contract."""
    if names is None:
        return {"status": "unavailable", "missing": [], "unexpected": []}
    try:
        parsed = ast.literal_eval(names) if isinstance(names, str) and names.startswith("[") else names
    except (SyntaxError, ValueError):
        return {"status": "unparseable", "missing": [], "unexpected": []}
    actual = set(parsed if isinstance(parsed, (list, tuple)) else [str(parsed)])
    expected = _expected_systemgmmkit_names(contract)
    return {"status": "match" if actual == expected else "mismatch", "missing": sorted(expected - actual), "unexpected": sorted(actual - expected)}


def build_pydynpd_command(profile: str, stage: str, estimator: str = "system") -> str:
    """Build the direct pydynpd command using its required ``L1.y`` notation."""
    if stage not in {"one_step", "two_step"} or estimator not in {"difference", "system"}:
        raise ValueError(f"Unknown pydynpd stage/estimator: {stage}/{estimator}")
    lags = PROFILE_LAGS[profile]
    gmm_variables = " ".join([DEPENDENT, *PREDTERMINED])
    regressors = " ".join([f"L1.{DEPENDENT}", *PREDTERMINED, *DIRECT])
    options = ["collapse"]
    if estimator == "difference":
        options.append("nolevel")
    if stage == "one_step":
        options.append("onestep")
    return f"{DEPENDENT} {regressors} | gmm({gmm_variables}, {lags[0]}:{lags[1]}) iv({' '.join(DIRECT)}) | {' '.join(options)}"


def _systemgmmkit_attempt(data: pd.DataFrame, profile: str, estimator: str, windmeijer: bool) -> dict[str, Any]:
    """Test systemgmmkit native mode; its public wrapper has no one-step selector."""
    import systemgmmkit as sgk

    contract = profile_contract(profile, estimator)
    roles = list(PREDTERMINED) if profile != "C" else []
    endogenous = list(PREDTERMINED) if profile == "C" else []
    function = sgk.system_gmm if estimator == "system" else sgk.difference_gmm
    stage = "two_step_windmeijer" if windmeijer else "two_step_uncorrected"
    kwargs = {"data": data, "entity": "Region", "time": "Date", "dependent": DEPENDENT, "lagged_dependent": 1,
              "lagged_dependent_role": "endogenous", "regressors": [*PREDTERMINED, *DIRECT], "endogenous": endogenous,
              "predetermined": roles, "exogenous": list(DIRECT), "gmm_lags": tuple(contract["lag_window"]), "collapse": True,
              "backend": "native", "windmeijer": windmeijer, "return_workflow": True}
    try:
        workflow = function(**kwargs)
        result = getattr(workflow, "result", workflow)
        names = getattr(result, "instrument_names", None)
        return {"package": "systemgmmkit", "backend": "native", "estimator": estimator, "profile": profile, "stage": stage,
                "status": "completed", "contract": contract, "nobs": getattr(result, "nobs", None), "groups": getattr(result, "n_groups", None),
                "instrument_count": getattr(result, "n_instruments", None), "instrument_names": names,
                "instrument_check": _systemgmmkit_instrument_check(contract, names), "ar1_p": getattr(result, "ar1_p", None),
                "ar2_p": getattr(result, "ar2_p", None), "hansen_p": getattr(result, "hansen_p", None), "sargan_p": getattr(result, "sargan_p", None),
                "difference_in_hansen_p": getattr(result, "difference_in_hansen_p", None), "covariance_type": getattr(result, "covariance_type", None),
                "one_step_available": False}
    except BaseException as error:
        return {"package": "systemgmmkit", "backend": "native", "estimator": estimator, "profile": profile, "stage": stage,
                "status": "failed", "contract": contract, "error_type": type(error).__name__, "error": str(error)}


def _pydynpd_attempt(data: pd.DataFrame, profile: str, estimator: str, stage: str) -> dict[str, Any]:
    """Run direct pydynpd after its upstream NumPy compatibility bridge is applied."""
    from systemgmmkit.pydynpd_backend import _apply_numpy_compatibility_shims
    from pydynpd import regression

    command = build_pydynpd_command(profile, stage, estimator)
    try:
        _apply_numpy_compatibility_shims()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            fitted = regression.abond(command, data, ["Region", "Date"])
        model = fitted.models[0]
        info = model.z_information
        expected_difference = len(PREDTERMINED) + 1
        expected_difference *= len(range(PROFILE_LAGS[profile][0], PROFILE_LAGS[profile][1] + 1))
        expected_levels = len(PREDTERMINED) + 1 if estimator == "system" else 0
        expected_total = expected_difference + expected_levels + len(DIRECT) + (1 if estimator == "system" else 0)
        ar = {item.lag: item.P_value for item in model.AR_list}
        return {"package": "pydynpd", "package_version": version("pydynpd"), "backend": "direct", "estimator": estimator, "profile": profile, "stage": stage,
                "status": "completed", "command": command, "compatibility": "systemgmmkit._apply_numpy_compatibility_shims",
                "nobs": int(model.N * (model.T - 2)), "groups": int(model.N), "instrument_count": int(info.num_instr),
                "difference_gmm_instruments": int(info.num_Dgmm_instr), "levels_gmm_instruments": int(info.num_Lgmm_instr),
                "levels_height": int(info.level_height), "instrument_check": {"status": "match" if (info.num_Dgmm_instr, info.num_Lgmm_instr, info.num_instr) == (expected_difference, expected_levels, expected_total) else "mismatch", "expected": {"difference": expected_difference, "levels": expected_levels, "total": expected_total}},
                "ar1_p": ar.get(1), "ar2_p": ar.get(2), "hansen_p": model.hansen.p_value, "difference_in_hansen_p": None,
                "covariance_available": bool(getattr(model.step_results[-1], "vcov", None) is not None), "windmeijer": stage == "two_step"}
    except BaseException as error:
        return {"package": "pydynpd", "backend": "direct", "estimator": estimator, "profile": profile, "stage": stage,
                "status": "failed", "command": command, "error_type": type(error).__name__, "error": str(error)}


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# ConsCred GMM Package Capability Test", "", f"Decision: **{report['decision']}**", "", "## Scope", "",
             "The test uses the 2024-01 to 2025-12 h=0 monthly panel. It validates package behavior and realised instruments, not economic model acceptance.", "",
             "## Attempts", "", "| Package | Backend | Estimator | Profile | Stage | Status | Instruments | AR(2) | DiH |", "|---|---|---|---|---|---|---:|---|---|"]
    for attempt in report["attempts"]:
        lines.append(f"| {attempt['package']} | {attempt['backend']} | {attempt['estimator']} | {attempt['profile']} | {attempt['stage']} | {attempt['status']} | {attempt.get('instrument_count', 'n/a')} | {attempt.get('ar2_p', 'n/a')} | {attempt.get('difference_in_hansen_p', 'n/a')} |")
    lines.extend(["", "## Instrument Checks", ""])
    for attempt in report["attempts"]:
        check = attempt.get("instrument_check")
        if check:
            lines.append(f"- {attempt['package']} {attempt['estimator']} {attempt['profile']} {attempt['stage']}: `{check['status']}`; {json.dumps(check, ensure_ascii=False)}")
    lines.extend(["", "## Decision Reason", "", report["decision_reason"], ""])
    return "\n".join(lines)


def run_backend_preflight(start: str = "2024-01", end: str = "2025-12", workbook: Path = Path("Operations/conscred_reg_analys.xlsx"),
                          results_dir: Path = Path("Results")) -> tuple[dict[str, Any], Path, Path]:
    """Test native systemgmmkit and direct pydynpd with separate evidence."""
    data = load_conscred_preflight_data(workbook, start, end)
    attempts: list[dict[str, Any]] = []
    for profile in ("A", "B", "C"):
        for estimator in ("difference", "system"):
            for windmeijer in (False, True):
                attempts.append(_systemgmmkit_attempt(data, profile, estimator, windmeijer))
        if profile != "C":
            for estimator in ("difference", "system"):
                for stage in ("one_step", "two_step"):
                    attempts.append(_pydynpd_attempt(data, profile, estimator, stage))
        else:
            attempts.append({"package": "pydynpd", "backend": "direct", "estimator": "difference/system", "profile": profile,
                             "stage": "not_run", "status": "skipped", "reason": "Direct pydynpd command grammar cannot preserve profile C's separate control-role classification."})
    required_system = [item for item in attempts if item["package"] == "systemgmmkit" and item["estimator"] == "system"]
    system_ready = all(item["status"] == "completed" and item.get("instrument_check", {}).get("status") == "match" and item.get("ar1_p") is not None and item.get("ar2_p") is not None and item.get("difference_in_hansen_p") is not None for item in required_system)
    report = {"decision": "pass" if system_ready else "needs_backend_diagnostics", "decision_reason": "Native System GMM lacks one or more mandatory AR or Difference-in-Hansen diagnostics." if not system_ready else "A backend met the current instrument and diagnostic contract.",
              "python": platform.python_version(), "packages": {"systemgmmkit": version("systemgmmkit"), "pydynpd": version("pydynpd")},
              "source_window": {"start": start, "end": end, "rows": len(data), "regions": int(data['Region'].nunique())}, "attempts": attempts}
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"ConsCred_GMM_package_capability_{tag}"
    json_path, markdown_path = results_dir / f"{stem}.json", results_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return report, markdown_path, json_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-preflight", action="store_true")
    parser.add_argument("--start", default="2024-01")
    parser.add_argument("--end", default="2025-12")
    args = parser.parse_args()
    if not args.backend_preflight:
        parser.error("--backend-preflight is required")
    report, path, _ = run_backend_preflight(args.start, args.end)
    print(f"{report['decision']}: {path}")
    return 0 if report["decision"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
