"""Capability tests for the agreed ConsCred dynamic-panel GMM instrument design."""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import inspect
import io
import json
import platform
import re
from itertools import combinations
from importlib.metadata import version
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from scipy.stats import norm


DEPENDENT = "d_Int_Rate_ConsCred"
PREDTERMINED = ("Cred_nagr_lag1", "Zakred_lag1", "Cap_to_assets_lag1", "CPI_reg_lag1")
DIRECT = (
    "ln_Fin_Dostup", "d_Inflation_Expectations", "d_Mon_Shock_pos", "d_Mon_Shock_neg",
    "d_Mon_Shock_pos_Cluster_new_cd_1", "d_Mon_Shock_neg_Cluster_new_cd_1",
    "d_Mon_Shock_pos_Cluster_new_cd_3", "d_Mon_Shock_neg_Cluster_new_cd_3",
    "d_Mon_Shock_pos_Cluster_new_cd_4", "d_Mon_Shock_neg_Cluster_new_cd_4",
)
PROFILE_LAGS = {"A": (2, 3), "B": (2, 4), "C": (2, 3)}
LOCKED_PREPARED_PANEL = Path("Results/GMM/ConsCred_GMM_20260820_120000_000001/prepared_panel.csv")
LOCKED_PREPARED_PANEL_SHA256 = "f754e752b82fc83e880136d3f49ceedea5c77a1071acbef0249da8f3b95ceb45"
HISTORICAL_FULL_PANEL = Path("Results/GMM/ConsCred_GMM_20260819_164017_026078/full_analysis_panel.csv")
HISTORICAL_FULL_PANEL_SHA256 = "3c8a6c00ff0eb13b26173d12de60b95aa869f8c3542c1b4855be34a548b00f10"
HISTORICAL_REGIME_DIRECT = (*DIRECT, "Covid_dum", "Sank_dum")
QUARTERLY_DEPENDENT = "d_Int_Rate_ConsCred_Q"
QUARTERLY_CONTROLS = ("Cred_nagr_Q_lag1", "Zakred_Q_lag1", "Cap_to_assets_Q_lag1")
QUARTERLY_DIRECT = ("ln_Fin_Dostup_Q",)


def _pydynpd_gmm_windows(profile: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """Map the approved raw-time profile to pydynpd's equation-relative lags."""
    dynamic = PROFILE_LAGS[profile]
    return dynamic, dynamic if profile == "C" else (dynamic[0] - 1, dynamic[1] - 1)


def profile_contract(profile: str, estimator: str) -> dict[str, Any]:
    """Return the fixed, collapsed instrument contract for a test attempt."""
    if profile not in PROFILE_LAGS or estimator not in {"difference", "system"}:
        raise ValueError(f"Unknown profile/estimator: {profile}/{estimator}")
    controls_role = "endogenous" if profile == "C" else "predetermined"
    dynamic = f"L1_{DEPENDENT}"
    gmm_variables = [dynamic, *PREDTERMINED]
    dynamic_window, control_window = _pydynpd_gmm_windows(profile)
    return {
        "profile": profile,
        "estimator": estimator,
        "lag_window": list(PROFILE_LAGS[profile]),
        "gmm_lag_windows": {dynamic: list(dynamic_window), **{name: list(control_window) for name in PREDTERMINED}},
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


def load_locked_preflight_data(prepared_panel: Path, start: str, end: str) -> tuple[pd.DataFrame, str]:
    """Load the immutable Phase-2b input instead of rebuilding it from the workbook."""
    panel_sha256 = hashlib.sha256(prepared_panel.read_bytes()).hexdigest()
    metadata_path = prepared_panel.parent / "phase2b_input.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if panel_sha256 != metadata["copied_prepared_panel_sha256"]:
        raise ValueError("Locked prepared panel checksum differs from phase2b_input.json")
    if prepared_panel == LOCKED_PREPARED_PANEL and panel_sha256 != LOCKED_PREPARED_PANEL_SHA256:
        raise ValueError("Locked prepared panel checksum differs from the approved Phase-2b checksum")
    data = pd.read_csv(prepared_panel)
    data["Date"] = pd.to_datetime(data["Date"])
    required = {"Region", "Date", DEPENDENT, *PREDTERMINED, *DIRECT}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Locked prepared panel is missing: {', '.join(missing)}")
    data = data.loc[data["Date"].between(pd.Timestamp(start), pd.Timestamp(end))].copy()
    _validate_preflight_panel(data, require_clusters=False)
    return data, panel_sha256


def load_historical_regime_data(full_panel: Path = HISTORICAL_FULL_PANEL) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the checksummed historical h=0 panel with varying national regime dummies."""
    observed_sha256 = hashlib.sha256(full_panel.read_bytes()).hexdigest()
    manifest_path = full_panel.parent / "specification_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha256 = manifest.get("artifacts", {}).get(full_panel.name)
    if manifest_sha256 != observed_sha256:
        raise ValueError("Historical full-panel checksum differs from its Phase-1 manifest")
    if full_panel == HISTORICAL_FULL_PANEL and observed_sha256 != HISTORICAL_FULL_PANEL_SHA256:
        raise ValueError("Historical full-panel checksum differs from the approved Phase-1 checksum")
    data = pd.read_csv(full_panel)
    data["Date"] = pd.to_datetime(data["Date"])
    required = {"Region", "Date", DEPENDENT, f"{DEPENDENT}_lag1", *PREDTERMINED, *DIRECT, "Covid_dum", "Sank_dum"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Historical full panel is missing: {', '.join(missing)}")
    data = data.loc[data["Date"].between(pd.Timestamp("2019-04-01"), pd.Timestamp("2025-12-01"))].copy()
    if data.duplicated(["Region", "Date"]).any():
        raise ValueError("Historical panel has duplicate Region-Date keys")
    _validate_preflight_panel(data, require_clusters=False)
    for name in ("Covid_dum", "Sank_dum"):
        if not data[name].isin((0, 1)).all():
            raise ValueError(f"Historical regime dummy is not binary: {name}")
        by_date = data.groupby("Date")[name].nunique()
        if by_date.gt(1).any():
            raise ValueError(f"Historical regime dummy is not nationally common within date: {name}")
        if data[name].nunique() != 2:
            raise ValueError(f"Historical regime dummy lacks historical variation: {name}")
    return data, {"parent_full_panel": str(full_panel), "parent_manifest": str(manifest_path), "manifest_sha256": manifest_sha256,
                  "expected_sha256": HISTORICAL_FULL_PANEL_SHA256 if full_panel == HISTORICAL_FULL_PANEL else None,
                  "observed_sha256": observed_sha256, "rows": len(data), "regions": int(data.Region.nunique()),
                  "dates": int(data.Date.nunique()), "start": str(data.Date.min().date()), "end": str(data.Date.max().date()),
                  "regime_dummy_validation": {name: {"binary": True, "nationally_common_within_date": True, "historically_varying": True}
                                              for name in ("Covid_dum", "Sank_dum")}}


def _regime_transition_audit(data: pd.DataFrame) -> dict[str, Any]:
    """Describe level and first-difference transition months for national step dummies."""
    dates = pd.DatetimeIndex(sorted(data["Date"].unique()))
    audit: dict[str, Any] = {}
    for name in ("Covid_dum", "Sank_dum"):
        series = data.groupby("Date", sort=True)[name].first().reindex(dates).astype(int)
        changes = series.diff().fillna(0)
        audit[name] = {"level_one_dates": [str(date.date()) for date in dates[series.eq(1)]],
                       "level_coverage": {"zero_months": int(series.eq(0).sum()), "one_months": int(series.eq(1).sum())},
                       "level_transition_dates": [str(date.date()) for date in dates[changes.ne(0)]],
                       "differenced_nonzero_dates": [str(date.date()) for date in dates[changes.ne(0)]],
                       "differenced_values": {str(date.date()): int(changes.loc[date]) for date in dates[changes.ne(0)]},
                       "observed_exit_transition": bool((changes < 0).any())}
    return audit


def _validate_preflight_panel(data: pd.DataFrame, require_clusters: bool = True) -> None:
    """Reject changed panel structure before a third-party estimator runs."""
    dates = pd.DatetimeIndex(sorted(data["Date"].unique()))
    if not dates.equals(pd.date_range(dates.min(), dates.max(), freq="MS")):
        raise ValueError("Preflight dates are not consecutive monthly observations")
    counts = data.groupby("Region")["Date"].nunique()
    if len(counts) != 75 or counts.nunique() != 1:
        raise ValueError("Preflight requires a balanced 75-region panel")
    if require_clusters:
        clusters = data[["Cluster_new_cd_1", "Cluster_new_cd_3", "Cluster_new_cd_4"]].sum(axis=1)
        if not clusters.isin((0, 1)).all():
            raise ValueError("Cluster encoding is not mutually exclusive with cluster 2 as reference")


def _expected_systemgmmkit_names(contract: dict[str, Any]) -> set[str]:
    expected = {
        f"D:{variable}:L{lag}"
        for variable, bounds in contract["gmm_lag_windows"].items()
        for lag in range(bounds[0], bounds[1] + 1)
    }
    direct_prefix = "IV" if contract["estimator"] == "system" else "D:iv"
    expected.update(f"{direct_prefix}:{variable}" for variable in DIRECT)
    if contract["estimator"] == "system":
        expected.add(f"L:diff:L1_{DEPENDENT}:L1")
        expected.update(f"L:diff:{variable}:L0" for variable in PREDTERMINED)
        expected.add("L:constant")
    return expected


def _systemgmmkit_instrument_check(contract: dict[str, Any], names: Any, instrument_count: Any = None) -> dict[str, Any]:
    """Compare native backend names to the approved collapsed-instrument contract."""
    if names is None:
        return {"status": "unavailable", "names_status": "unavailable", "count_status": "unavailable", "expected_count": 26 if contract["estimator"] == "system" else 20, "missing": [], "unexpected": []}
    try:
        parsed = ast.literal_eval(names) if isinstance(names, str) and names.startswith("[") else names
    except (SyntaxError, ValueError):
        return {"status": "unparseable", "names_status": "unparseable", "count_status": "unavailable", "expected_count": 26 if contract["estimator"] == "system" else 20, "missing": [], "unexpected": []}
    actual = set(parsed if isinstance(parsed, (list, tuple)) else [str(parsed)])
    expected = _expected_systemgmmkit_names(contract)
    expected_count = 26 if contract["estimator"] == "system" else 20
    names_status = "match" if actual == expected else "mismatch"
    count_status = "match" if instrument_count == expected_count else "mismatch"
    status = "match" if names_status == count_status == "match" else "mismatch"
    result = {"status": status, "names_status": names_status, "count_status": count_status, "expected_count": expected_count,
              "missing": sorted(expected - actual), "unexpected": sorted(actual - expected)}
    if contract["estimator"] == "system" and status == "match":
        result.update({"status": "different_system_convention", "reason": "Matches the shared-IV plus level-intercept layout, not pgmm's duplicated-IV no-intercept matrix."})
    return result


def build_pydynpd_command(profile: str, stage: str, estimator: str = "system",
                          retained_controls: tuple[str, ...] = PREDTERMINED,
                          control_window: tuple[int, int] | None = None,
                          direct_terms: tuple[str, ...] = DIRECT,
                          dependent: str = DEPENDENT,
                          control_order: tuple[str, ...] = PREDTERMINED,
                          control_windows: dict[str, tuple[int, int]] | None = None) -> str:
    """Build the direct pydynpd command using its required ``L1.y`` notation."""
    if stage not in {"one_step", "two_step"} or estimator not in {"difference", "system"}:
        raise ValueError(f"Unknown pydynpd stage/estimator: {stage}/{estimator}")
    if any(name not in control_order for name in retained_controls) or tuple(name for name in control_order if name in retained_controls) != retained_controls:
        raise ValueError("retained_controls must be an ordered, duplicate-free subset of the control order")
    dynamic_lags, control_lags = _pydynpd_gmm_windows(profile)
    if control_window is not None:
        if (not isinstance(control_window, tuple) or len(control_window) != 2
                or any(isinstance(lag, bool) or not isinstance(lag, int) for lag in control_window)
                or control_window[0] < 1 or control_window[1] != control_window[0] + 1):
            raise ValueError("control_window must be two consecutive positive integer lags")
        control_lags = control_window
    if control_windows is not None:
        if set(control_windows) != set(retained_controls):
            raise ValueError("control_windows must define one window for every retained control")
        for window in control_windows.values():
            if (not isinstance(window, tuple) or len(window) != 2 or window[0] < 1
                    or window[1] != window[0] + 1):
                raise ValueError("control windows must be two consecutive positive integer lags")
    regressors = " ".join([f"L1.{dependent}", *retained_controls, *direct_terms])
    options = ["collapse"]
    if estimator == "difference":
        options.append("nolevel")
    if stage == "one_step":
        options.append("onestep")
    if control_windows is not None:
        gmm_controls = "".join(f" gmm({name}, {window[0]}:{window[1]})" for name, window in control_windows.items())
    else:
        gmm_controls = f" gmm({' '.join(retained_controls)}, {control_lags[0]}:{control_lags[1]})" if retained_controls else ""
    return f"{dependent} {regressors} | gmm({dependent}, {dynamic_lags[0]}:{dynamic_lags[1]}){gmm_controls} iv({' '.join(direct_terms)}) | {' '.join(options)}"


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
              "predetermined": roles, "exogenous": list(DIRECT), "gmm_lags": tuple(contract["lag_window"]),
              "gmm_lags_by_role": {"endogenous": tuple(contract["gmm_lag_windows"][f"L1_{DEPENDENT}"]), "predetermined": (1, 2)}, "collapse": True,
              "backend": "native", "windmeijer": windmeijer, "return_workflow": True}
    try:
        workflow = function(**kwargs)
        result = getattr(workflow, "result", workflow)
        names = getattr(result, "instrument_names", None)
        return {"package": "systemgmmkit", "backend": "native", "estimator": estimator, "profile": profile, "stage": stage,
                "status": "completed", "contract": contract, "nobs": getattr(result, "nobs", None), "groups": getattr(result, "n_groups", None),
                "instrument_count": getattr(result, "n_instruments", None), "instrument_names": names,
                "instrument_check": _systemgmmkit_instrument_check(contract, names, getattr(result, "n_instruments", None)), "ar1_p": getattr(result, "ar1_p", None),
                "ar2_p": getattr(result, "ar2_p", None), "hansen_p": getattr(result, "hansen_p", None), "sargan_p": getattr(result, "sargan_p", None),
                "hansen_robustness": "not_verified_by_probe", "difference_in_hansen_p": getattr(result, "difference_in_hansen_p", None), "covariance_type": getattr(result, "covariance_type", None),
                "one_step_available": False}
    except BaseException as error:
        return {"package": "systemgmmkit", "backend": "native", "estimator": estimator, "profile": profile, "stage": stage,
                "status": "failed", "contract": contract, "error_type": type(error).__name__, "error": str(error)}


def _pydynpd_attempt(data: pd.DataFrame, profile: str, estimator: str, stage: str,
                     retained_controls: tuple[str, ...] = PREDTERMINED,
                     control_window: tuple[int, int] | None = None,
                     direct_terms: tuple[str, ...] = DIRECT,
                     dependent: str = DEPENDENT,
                     control_order: tuple[str, ...] = PREDTERMINED,
                     control_windows: dict[str, tuple[int, int]] | None = None) -> dict[str, Any]:
    """Run direct pydynpd after its upstream NumPy compatibility bridge is applied."""
    command = build_pydynpd_command(profile, stage, estimator, retained_controls, control_window, direct_terms, dependent, control_order, control_windows)
    local_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = io.StringIO()
    try:
        from systemgmmkit.pydynpd_backend import _apply_numpy_compatibility_shims
        from pydynpd import regression
        from pydynpd import specification_tests

        _apply_numpy_compatibility_shims()
        with contextlib.redirect_stdout(output):
            fitted = regression.abond(command, data, ["Region", "Date"])
        model = fitted.models[0]
        info = model.z_information
        dynamic_window, resolved_control_window = _pydynpd_gmm_windows(profile)
        if control_window is not None:
            resolved_control_window = control_window
        expected_difference = dynamic_window[1] - dynamic_window[0] + 1 + sum(
            window[1] - window[0] + 1 for window in (control_windows or {name: resolved_control_window for name in retained_controls}).values())
        expected_levels = len(retained_controls) + 1 if estimator == "system" else 0
        expected_total = expected_difference + len(direct_terms) if estimator == "difference" else expected_difference + expected_levels + len(direct_terms) + 1
        ar = {item.lag: item.P_value for item in model.AR_list}
        realised = (info.num_Dgmm_instr, info.num_Lgmm_instr, info.num_instr)
        expected = (expected_difference, expected_levels, expected_total)
        instrument_status = "match" if estimator == "difference" and realised == expected else "different_system_convention" if estimator == "system" and realised == expected else "mismatch"
        z_height = int(model.z_list.shape[0] / model.N)
        if z_height != int(info.num_instr) or z_height * model.N != model.z_list.shape[0]:
            raise ValueError("pydynpd instrument matrix dimensions do not match the reported instrument count")
        z_matrix = np.concatenate([model.z_list[index * z_height:(index + 1) * z_height, :] for index in range(model.N)], axis=1)
        singular_values = np.linalg.svd(z_matrix, compute_uv=False)
        tolerance = float(np.finfo(float).eps * max(z_matrix.shape) * singular_values[0]) if singular_values.size else 0.0
        instrument_rank = int((singular_values > tolerance).sum())
        rank_status = "valid" if instrument_rank == int(info.num_instr) else "invalid"
        hansen_statistic = getattr(model.hansen, "test_value", None)
        hansen_df = getattr(model.hansen, "df", None)
        hansen_status = "available" if hansen_statistic is not None and hansen_df is not None and model.hansen.p_value is not None else "unavailable"
        package_nobs = int(match.group(1)) if (match := re.search(r"Number of obs\s*=\s*(\d+)", output.getvalue())) else int(model.N * (model.T - 2))
        usable_observations = {"package_nobs": package_nobs, "model_T": int(model.T),
                               "min_obs_per_group": int(match.group(1)) if (match := re.search(r"Min obs per group:\s*(\d+)", output.getvalue())) else None,
                               "max_obs_per_group": int(match.group(1)) if (match := re.search(r"Max obs per group:\s*(\d+)", output.getvalue())) else None,
                               "avg_obs_per_group": float(match.group(1)) if (match := re.search(r"Avg obs per group:\s*([\d.]+)", output.getvalue())) else None}
        return {"package": "pydynpd", "package_version": version("pydynpd"), "backend": "direct", "estimator": estimator, "profile": profile, "stage": stage,
                "status": "completed", "command": command, "raw_console": output.getvalue(), "compatibility": "systemgmmkit._apply_numpy_compatibility_shims",
                "retained_controls": list(retained_controls), "omitted_controls": [name for name in control_order if name not in retained_controls], "direct_terms": list(direct_terms),
                "nobs": package_nobs, "groups": int(model.N), "instrument_count": int(info.num_instr),
                "control_window": list(resolved_control_window), "local_source_sha256": local_source_sha256, "usable_observations": usable_observations,
                "difference_gmm_instruments": int(info.num_Dgmm_instr), "levels_gmm_instruments": int(info.num_Lgmm_instr),
                "levels_height": int(info.level_height), "instrument_check": {"status": instrument_status, "expected": {"difference": expected_difference, "levels": expected_levels, "total": expected_total}, "reason": "Shared direct-IV columns plus a level intercept differ from pgmm's 35-column, no-intercept System matrix." if estimator == "system" else None},
                "expected_nominal_hansen_df": expected_total - (1 + len(retained_controls) + len(direct_terms)) if estimator == "difference" else None,
                "instrument_rank": instrument_rank, "rank_status": rank_status, "hansen_statistic": hansen_statistic, "hansen_df": hansen_df, "hansen_status": hansen_status,
                "instrument_conditioning": {"label": "descriptive_not_formal_weak_instrument_test", "singular_values": singular_values.tolist(), "tolerance": tolerance,
                                             "condition_number": "infinity" if not singular_values.size or singular_values[-1] <= tolerance else float(singular_values[0] / singular_values[-1])},
                "ar1_p": ar.get(1), "ar2_p": ar.get(2), "ar2_status": "passed" if ar.get(2) is not None and ar[2] >= 0.05 else "failed" if ar.get(2) is not None else "unavailable", "hansen_p": model.hansen.p_value, "hansen_robustness": "two_step_empirical_moment_covariance" if stage == "two_step" else "not_verified_one_step",
                "hansen_robustness_evidence": {"function": "pydynpd.specification_tests.hansen_overid", "test_source_sha256": hashlib.sha256(inspect.getsource(specification_tests.hansen_overid).encode()).hexdigest(), "weighting_function": "pydynpd.regression.abond.GMM", "weighting_source_sha256": hashlib.sha256(inspect.getsource(regression.abond.GMM).encode()).hexdigest()} if stage == "two_step" else None,
                "difference_in_hansen_p": None,
                "covariance_available": bool(getattr(model.step_results[-1], "vcov", None) is not None), "windmeijer": stage == "two_step"}
    except BaseException as error:
        return {"package": "pydynpd", "backend": "direct", "estimator": estimator, "profile": profile, "stage": stage,
                "status": "failed", "command": command, "retained_controls": list(retained_controls), "omitted_controls": [name for name in control_order if name not in retained_controls], "direct_terms": list(direct_terms), "control_window": list(control_window) if control_window is not None else None, "local_source_sha256": local_source_sha256,
                "raw_console": output.getvalue(), "error_type": type(error).__name__, "error": str(error)}


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# ConsCred GMM Package Capability Test", "", f"Decision: **{report['decision']}**", "", "## Scope", "",
             "The test reads the checksum-recorded Phase-2b h=0 panel. It validates package behavior and realised instruments, not economic model acceptance.", "",
             "## Attempts", "", "| Package | Estimator | Stage | Status | Instruments | Hansen p | Hansen robustness | AR(2) | Contract |", "|---|---|---|---|---:|---|---|---|---|"]
    for attempt in report["attempts"]:
        lines.append(f"| {attempt['package']} | {attempt['estimator']} | {attempt['stage']} | {attempt['status']} | {attempt.get('instrument_count', 'n/a')} | {attempt.get('hansen_p', 'n/a')} | {attempt.get('hansen_robustness', 'n/a')} | {attempt.get('ar2_p', 'n/a')} | {attempt.get('instrument_check', {}).get('status', 'n/a')} |")
    lines.extend(["", "## Instrument Checks", ""])
    for attempt in report["attempts"]:
        check = attempt.get("instrument_check")
        if check:
            lines.append(f"- {attempt['package']} {attempt['estimator']} {attempt['profile']} {attempt['stage']}: `{check['status']}`; {json.dumps(check, ensure_ascii=False)}")
    lines.extend(["", "## Decision Reason", "", report["decision_reason"], ""])
    return "\n".join(lines)


def run_backend_preflight(start: str = "2024-01", end: str = "2025-12", prepared_panel: Path = LOCKED_PREPARED_PANEL,
                          results_dir: Path = Path("Results/GMM/preflight")) -> tuple[dict[str, Any], Path, Path]:
    """Probe candidate backends against the immutable locked Profile-A input."""
    data, panel_sha256 = load_locked_preflight_data(prepared_panel, start, end)
    attempts: list[dict[str, Any]] = []
    for profile in ("A",):
        for estimator in ("difference", "system"):
            for windmeijer in (False, True):
                attempts.append(_systemgmmkit_attempt(data, profile, estimator, windmeijer))
        for estimator in ("difference", "system"):
            for stage in ("one_step", "two_step"):
                attempts.append(_pydynpd_attempt(data, profile, estimator, stage))
    robust_difference = next((item for item in attempts if item["package"] == "pydynpd" and item["estimator"] == "difference" and item["stage"] == "two_step" and item["status"] == "completed" and item.get("instrument_check", {}).get("status") == "match"), None)
    if robust_difference is None:
        decision, reason = "robust_overid_unresolved", "No candidate produced a verified robust overall test on the locked Difference-GMM instrument contract."
    elif robust_difference["hansen_p"] < 0.05:
        decision, reason = "robust_overid_rejects", "pydynpd's verified two-step robust Hansen J rejects the locked Difference-GMM instrument set; heteroskedasticity alone does not explain the R Sargan rejection."
    else:
        decision, reason = "robust_overid_does_not_reject", "pydynpd's verified two-step robust Hansen J does not reject the locked Difference-GMM instrument set."
    report = {"decision": decision, "decision_reason": reason, "python": platform.python_version(), "packages": {"systemgmmkit": version("systemgmmkit"), "pydynpd": version("pydynpd")},
              "input": {"prepared_panel": str(prepared_panel), "sha256": panel_sha256}, "source_window": {"start": start, "end": end, "rows": len(data), "regions": int(data['Region'].nunique())}, "attempts": attempts}
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"ConsCred_GMM_package_capability_{tag}"
    json_path, markdown_path = results_dir / f"{stem}.json", results_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return report, markdown_path, json_path


def run_robust_hansen_diagnosis(start: str = "2024-01", end: str = "2025-12", prepared_panel: Path = LOCKED_PREPARED_PANEL,
                                results_dir: Path = Path("Results/GMM/preflight")) -> tuple[dict[str, Any], Path, Path]:
    """Run the approved Difference-GMM A/B/C diagnostic matrix without selecting a model."""
    data, panel_sha256 = load_locked_preflight_data(prepared_panel, start, end)
    attempts = [_pydynpd_attempt(data, profile, "difference", "two_step") for profile in ("A", "B", "C")]
    for attempt in attempts:
        check = attempt.get("instrument_check", {})
        attempt["diagnostic_gate"] = (
            "passes_available_diagnostics" if attempt["status"] == "completed" and check.get("status") == "match"
            and attempt.get("hansen_p", 0) >= 0.05 and attempt.get("ar2_p", 0) >= 0.05 and attempt.get("instrument_count", 75) < 75
            else "rejected_or_unresolved"
        )
    passing = [item["profile"] for item in attempts if item["diagnostic_gate"] == "passes_available_diagnostics"]
    rejected = [item["profile"] for item in attempts if item["diagnostic_gate"] != "passes_available_diagnostics"]
    conclusion = (
        "The A/B/C matrix changes only approved GMM timing/classification restrictions; direct ordinary-IV terms are unchanged and cannot be identified as the source from this matrix. "
        + (f"Profiles {', '.join(passing)} pass the available robust-J/AR(2)/instrument-count gates, while {', '.join(rejected) or 'none'} do not. " if passing else f"All approved profiles ({', '.join(rejected)}) remain rejected or unresolved. ")
        + "This is diagnostic evidence only; it does not accept a production model or resolve the separate System-GMM parity question."
    )
    report = {"decision": "diagnostic_complete_no_model_accepted", "conclusion": conclusion,
              "input": {"prepared_panel": str(prepared_panel), "sha256": panel_sha256},
              "source_window": {"start": start, "end": end, "rows": len(data), "regions": int(data["Region"].nunique())},
              "scope": {"estimator": "difference", "profiles": ["A", "B", "C"], "stage": "two_step", "direct_terms": "unchanged ordinary IV", "system_gmm": "not evaluated for pgmm parity"},
              "attempts": attempts}
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / f"ConsCred_GMM_robust_hansen_diagnosis_{tag}"
    run_dir.mkdir(parents=True, exist_ok=False)
    for attempt in attempts:
        (run_dir / f"{attempt['profile']}_raw_console.log").write_text(attempt.get("raw_console", attempt.get("error", "")), encoding="utf-8")
    json_path, markdown_path = run_dir / "diagnostics.json", run_dir / "report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = ["# ConsCred Difference-GMM Robust Hansen Diagnosis", "", "## Scope", "",
             "Approved Profile-A/B/C Difference-GMM two-step attempts only. The locked Phase-2b panel checksum was verified before fitting. This is diagnostic evidence, not model acceptance.", "",
             "## Attempts", "", "| Profile | Control role | GMM lags (dynamic / controls) | Instruments | Robust Hansen p | AR(2) p | Outcome |", "|---|---|---|---:|---:|---:|---|"]
    for item in attempts:
        dynamic_window, control_window = _pydynpd_gmm_windows(item["profile"])
        lines.append(f"| {item['profile']} | {profile_contract(item['profile'], 'difference')['gmm_roles'][PREDTERMINED[0]]} | {dynamic_window[0]}:{dynamic_window[1]} / {control_window[0]}:{control_window[1]} | {item.get('instrument_count', 'n/a')} | {item.get('hansen_p', 'n/a')} | {item.get('ar2_p', 'n/a')} | {item['diagnostic_gate']} |")
    lines.extend(["", "## Conclusion", "", conclusion, "", "## Evidence", "", "Each attempt's exact command, realised instrument counts, source hashes, and raw console output are retained in `diagnostics.json` and the adjacent per-profile log.", ""])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return report, markdown_path, json_path


def _control_subsets() -> tuple[tuple[str, ...], ...]:
    """Return the predeclared full-to-core control matrix in canonical order."""
    return tuple(subset for size in range(len(PREDTERMINED), -1, -1) for subset in combinations(PREDTERMINED, size))


def _control_subset_gate(attempt: dict[str, Any]) -> str:
    """Classify evidence without turning a non-rejection into model acceptance."""
    if attempt["status"] != "completed" or attempt.get("instrument_check", {}).get("status") != "match" or attempt.get("rank_status") != "valid" or attempt.get("hansen_status") != "available":
        return "unresolved"
    return "robust_J_rejects" if attempt["hansen_p"] < 0.05 else "robust_J_nonrejection_not_accepted"


def run_control_subset_diagnosis(start: str = "2024-01", end: str = "2025-12", prepared_panel: Path = LOCKED_PREPARED_PANEL,
                                 results_dir: Path = Path("Results/GMM/preflight")) -> tuple[dict[str, Any], Path, Path]:
    """Run the authorized control-subset diagnostic without selecting a model."""
    data, panel_sha256 = load_locked_preflight_data(prepared_panel, start, end)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / f"ConsCred_GMM_control_subset_diagnosis_{tag}"
    run_dir.mkdir(parents=True, exist_ok=False)
    attempts: list[dict[str, Any]] = []
    for retained_controls in _control_subsets():
        attempt = _pydynpd_attempt(data, "A", "difference", "two_step", retained_controls)
        attempt_id = "retained_" + ("__".join(retained_controls) if retained_controls else "none")
        attempt["attempt_id"] = attempt_id
        attempt["diagnostic_gate"] = _control_subset_gate(attempt)
        warnings: list[str] = []
        if attempt.get("hansen_df") is not None and attempt["hansen_df"] < 5:
            warnings.append("plan_specific_low_power_caution_fewer_than_five_realised_restrictions")
        if attempt.get("hansen_p") is not None and attempt["hansen_p"] > 0.95:
            warnings.append("active_design_hansen_above_0_95_warning")
        attempt["warnings"] = warnings
        (run_dir / f"{attempt_id}_raw_console.log").write_text(attempt.get("raw_console", attempt.get("error", "")), encoding="utf-8")
        attempts.append(attempt)
    report = {
        "decision": "diagnostic_complete_no_model_accepted",
        "user_authorization": "2026-08-20 bounded test-only control-subset diagnostic; no production, Phase 3, or model selection.",
        "parent_evidence": "Results/GMM/preflight/ConsCred_GMM_robust_hansen_diagnosis_20260820_171106/diagnostics.json",
        "input": {"prepared_panel": str(prepared_panel), "sha256": panel_sha256},
        "source_window": {"start": start, "end": end, "rows": len(data), "regions": int(data["Region"].nunique())},
        "scope": {"estimator": "difference", "profile": "A", "stage": "two_step", "direct_terms": list(DIRECT), "system_gmm": "not evaluated"},
        "attempts": attempts,
    }
    json_path, markdown_path = run_dir / "diagnostics.json", run_dir / "report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = ["# ConsCred Difference-GMM Control-Subset Diagnosis", "", "Decision: **diagnostic complete; no model accepted.**", "",
             "All direct shock, interaction, financial-access, and inflation-expectation terms are unchanged. This matrix localizes restrictions only; it does not choose a reduced model.", "",
             "| Retained controls | Omitted controls | Instruments | Rank | Hansen J (df) | Hansen p | AR(2) p | Outcome | Warnings |", "|---|---|---:|---|---|---:|---:|---|---|"]
    for item in attempts:
        lines.append(f"| {', '.join(item.get('retained_controls', [])) or 'none'} | {', '.join(item.get('omitted_controls', [])) or 'none'} | {item.get('instrument_count', 'n/a')} | {item.get('instrument_rank', 'n/a')} ({item.get('rank_status', 'n/a')}) | {item.get('hansen_statistic', 'n/a')} ({item.get('hansen_df', 'n/a')}) | {item.get('hansen_p', 'n/a')} | {item.get('ar2_p', 'n/a')} | {item['diagnostic_gate']} | {', '.join(item['warnings']) or 'none'} |")
    lines.extend(["", "## Interpretation boundary", "", "`robust_J_nonrejection_not_accepted` is not economic-model acceptance. Fewer than five realised restrictions is a plan-specific low-power caution; Hansen p-values above .95 retain the active-design warning. Each exact command, source-hash evidence, and raw console log is retained beside this report.", ""])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return report, markdown_path, json_path


def _delayed_control_lag_gate(attempt: dict[str, Any]) -> str:
    """Classify a delayed-control attempt without accepting an economic model."""
    if (attempt.get("status") != "completed" or attempt.get("instrument_check", {}).get("status") != "match"
            or attempt.get("rank_status") != "valid" or attempt.get("hansen_status") != "available"
            or attempt.get("ar2_status") != "passed" or attempt.get("sample_comparability_status") != "match"):
        return "unresolved"
    return "robust_J_rejects" if attempt["hansen_p"] < 0.05 else "robust_J_nonrejection_not_accepted"


def run_delayed_control_lag_diagnosis(start: str = "2024-01", end: str = "2025-12", prepared_panel: Path = LOCKED_PREPARED_PANEL,
                                     results_dir: Path = Path("Results/GMM/preflight")) -> tuple[dict[str, Any], Path, Path]:
    """Run the authorized three-window control-timing diagnosis only."""
    data, panel_sha256 = load_locked_preflight_data(prepared_panel, start, end)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / f"ConsCred_GMM_delayed_control_lag_diagnosis_{tag}"
    run_dir.mkdir(parents=True, exist_ok=False)
    windows = ((1, 2), (2, 3), (3, 4))
    attempts: list[dict[str, Any]] = []
    for window in windows:
        attempt = _pydynpd_attempt(data, "A", "difference", "two_step", control_window=window)
        attempt["attempt_id"] = f"controls_{window[0]}_{window[1]}"
        attempt["raw_control_timing"] = [window[0] + 1, window[1] + 1]
        (run_dir / f"{attempt['attempt_id']}_raw_console.log").write_text(attempt.get("raw_console", attempt.get("error", "")), encoding="utf-8")
        attempts.append(attempt)
    baseline = attempts[0]
    sample_keys = ("nobs", "groups", "usable_observations")
    baseline_sample = tuple(json.dumps(baseline.get(key), sort_keys=True, default=str) for key in sample_keys)
    for attempt in attempts:
        sample = tuple(json.dumps(attempt.get(key), sort_keys=True, default=str) for key in sample_keys)
        attempt["sample_comparability_status"] = "match" if sample == baseline_sample else "unresolved"
        attempt["diagnostic_gate"] = _delayed_control_lag_gate(attempt)
    report = {
        "decision": "diagnostic_complete_no_model_accepted",
        "user_authorization": "2026-08-24 three-window delayed-control-lag diagnostic only; no production estimation or model selection.",
        "diagnostic_boundary": "Difference-GMM pydynpd evidence only; System GMM parity and backend selection are not evaluated.",
        "parent_evidence": "Results/GMM/preflight/ConsCred_GMM_robust_hansen_diagnosis_20260820_171106/diagnostics.json",
        "input": {"prepared_panel": str(prepared_panel), "sha256": panel_sha256},
        "source_window": {"start": start, "end": end, "rows": len(data), "regions": int(data["Region"].nunique())},
        "scope": {"estimator": "difference", "profile": "A", "stage": "two_step", "dynamic_window": [2, 3], "direct_terms": list(DIRECT), "system_gmm": "not evaluated"},
        "attempts": attempts,
    }
    json_path, markdown_path = run_dir / "diagnostics.json", run_dir / "report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = ["# ConsCred Difference-GMM Delayed-Control-Lag Diagnosis", "", "Decision: **diagnostic complete; no model accepted.**", "",
             "The dynamic window is fixed at `2:3`; all four controls and all direct terms are unchanged. Only the control instruments are delayed. A non-rejection is diagnostic only, never model acceptance.", "",
             "| pydynpd control window | Raw control timing | Instruments | Rank | Hansen J (df) | Hansen p | AR(2) p/status | Sample comparability | Outcome |", "|---|---|---:|---|---|---:|---|---|---|"]
    for item in attempts:
        window = item.get("control_window", [])
        raw_window = item.get("raw_control_timing", [])
        lines.append(f"| {':'.join(map(str, window)) or 'n/a'} | {'t-' + ':t-'.join(map(str, raw_window)) if raw_window else 'n/a'} | {item.get('instrument_count', 'n/a')} | {item.get('instrument_rank', 'n/a')} ({item.get('rank_status', 'n/a')}) | {item.get('hansen_statistic', 'n/a')} ({item.get('hansen_df', 'n/a')}) | {item.get('hansen_p', 'n/a')} | {item.get('ar2_p', 'n/a')} ({item.get('ar2_status', 'n/a')}) | {item.get('sample_comparability_status', 'n/a')} | {item.get('diagnostic_gate', 'n/a')} |")
    lines.extend(["", "Each completed equal-width design should have 20 instruments and Hansen df 5. Raw console logs, exact commands, input checksum, upstream source hashes, and the executed local-module hash are retained in this immutable bundle.", ""])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return report, markdown_path, json_path


def _sample_comparability(attempt: dict[str, Any], baseline: dict[str, Any]) -> str:
    """Compare realised estimator samples without inferring comparability from commands."""
    keys = ("nobs", "groups", "usable_observations")
    left = tuple(json.dumps(attempt.get(key), sort_keys=True, default=str) for key in keys)
    right = tuple(json.dumps(baseline.get(key), sort_keys=True, default=str) for key in keys)
    return "match" if left == right else "unresolved"


def _instrument_validity_gate(attempt: dict[str, Any], expected_count: int, expected_df: int) -> str:
    """Classify a diagnostic attempt without accepting or selecting an economic model."""
    if (attempt.get("status") != "completed" or attempt.get("instrument_count") != expected_count
            or attempt.get("instrument_rank") != expected_count or attempt.get("hansen_df") != expected_df
            or attempt.get("hansen_status") != "available" or attempt.get("ar2_status") != "passed"
            or attempt.get("sample_comparability_status") != "match"):
        return "unresolved"
    return "robust_J_rejects" if attempt["hansen_p"] < 0.05 else "robust_J_nonrejection_not_accepted"


def _benchmark_cd(data: pd.DataFrame, time_effects: bool, direct_terms: tuple[str, ...] = DIRECT) -> dict[str, Any]:
    """Fit the declared FE benchmark and calculate a Pesaran-style residual CD statistic."""
    direct_terms = list(direct_terms)
    excluded = ["d_Inflation_Expectations", "d_Mon_Shock_pos", "d_Mon_Shock_neg"] if time_effects else []
    regressors = [f"{DEPENDENT}_lag1", *PREDTERMINED, *(term for term in direct_terms if term not in excluded)]
    model_data = data[["Region", "Date", DEPENDENT, *regressors]].dropna().copy()
    entity = pd.get_dummies(model_data["Region"], drop_first=True, dtype=float)
    pieces = [pd.Series(1.0, index=model_data.index, name="constant"), model_data[regressors].astype(float), entity]
    if time_effects:
        pieces.append(pd.get_dummies(model_data["Date"], drop_first=True, dtype=float))
    design = pd.concat(pieces, axis=1)
    selected: list[str] = []
    current = np.empty((len(design), 0))
    dropped: list[str] = []
    for index, name in enumerate(design.columns.astype(str)):
        candidate = np.column_stack((current, design.iloc[:, [index]].to_numpy()))
        if np.linalg.matrix_rank(candidate) == candidate.shape[1]:
            current = candidate
            selected.append(name)
        else:
            dropped.append(name)
    beta, _, _, _ = np.linalg.lstsq(current, model_data[DEPENDENT].to_numpy(dtype=float), rcond=None)
    model_data["residual"] = model_data[DEPENDENT].to_numpy(dtype=float) - current @ beta
    residuals = model_data.pivot(index="Date", columns="Region", values="residual").dropna(axis=0, how="any")
    correlations = residuals.corr().to_numpy()
    pairwise = correlations[np.triu_indices_from(correlations, k=1)]
    n, t = residuals.shape[1], residuals.shape[0]
    statistic = float(np.sqrt(2 * t / (n * (n - 1))) * pairwise.sum())
    return {"attempt_id": "cd_benchmark_time_effects" if time_effects else "cd_benchmark_no_time_effects",
            "residual_source": "entity-and-time-FE OLS benchmark" if time_effects else "entity-FE OLS benchmark",
            "time_effects": time_effects, "excluded_time_only_terms": excluded, "additional_dropped_collinear_terms": dropped,
            "regressors": regressors, "residual_coverage": {"dates": int(t), "regions": int(n), "rows": int(len(model_data))},
            "pesaran_cd_statistic": statistic, "pesaran_cd_two_sided_p": float(2 * norm.sf(abs(statistic)))}


def _direct_terms_audit() -> list[dict[str, str]]:
    """Return the required provenance-first audit; unknown timing is intentionally unresolved."""
    return [{"term": term, "source": "locked Phase-2b prepared panel / variable provenance record",
             "scope": "national" if "Cluster" not in term and term != "ln_Fin_Dostup" else "regional interaction" if "Cluster" in term else "regional",
             "publication_or_effective_timing": "unresolved", "contemporaneous_feedback_risk": "unresolved",
             "treatment": "ordinary direct IV", "status": "unresolved"} for term in DIRECT]


def run_instrument_validity_diagnosis(start: str = "2024-01", end: str = "2025-12", prepared_panel: Path = LOCKED_PREPARED_PANEL,
                                      results_dir: Path = Path("Results/GMM/preflight")) -> tuple[dict[str, Any], Path, Path]:
    """Run the finite, diagnostic-only monthly instrument-validity matrix."""
    data, panel_sha256 = load_locked_preflight_data(prepared_panel, start, end)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / f"ConsCred_GMM_instrument_validity_diagnosis_{tag}"
    run_dir.mkdir(parents=True, exist_ok=False)
    baseline = _pydynpd_attempt(data, "A", "difference", "two_step")
    baseline["attempt_id"] = "baseline_reproduction"
    baseline["sample_comparability_status"] = "match"
    baseline["diagnostic_gate"] = _instrument_validity_gate(baseline, 20, 5)
    attempts = [baseline]
    for term in DIRECT:
        attempt = _pydynpd_attempt(data, "A", "difference", "two_step", direct_terms=tuple(item for item in DIRECT if item != term))
        attempt.update({"attempt_id": f"direct_{term}_omitted_auxiliary", "classification": "auxiliary_not_primary_equation",
                        "sample_comparability_status": _sample_comparability(attempt, baseline)})
        attempt["diagnostic_gate"] = _instrument_validity_gate(attempt, 19, 5)
        attempts.append(attempt)
    for attempt_id, direct_terms in (
        ("direct_without_shock_interactions_auxiliary", DIRECT[:4]),
        ("direct_without_full_shock_block_auxiliary", DIRECT[:2]),
    ):
        attempt = _pydynpd_attempt(data, "A", "difference", "two_step", direct_terms=direct_terms)
        expected = 14 if len(direct_terms) == 4 else 12
        attempt.update({"attempt_id": attempt_id, "classification": "auxiliary_not_primary_equation",
                        "sample_comparability_status": _sample_comparability(attempt, baseline)})
        attempt["diagnostic_gate"] = _instrument_validity_gate(attempt, expected, 5)
        attempts.append(attempt)
    for mask in range(16):
        bits = f"{mask:04b}"
        windows = {name: (2, 3) if bit == "1" else (1, 2) for name, bit in zip(PREDTERMINED, bits, strict=True)}
        attempt = _pydynpd_attempt(data, "A", "difference", "two_step", control_windows=windows)
        attempt.update({"attempt_id": f"controls_mask_{bits}", "classification": "preflight_only_governance_exception_2026-08-26",
                        "control_timing": {name: {"pydynpd_window": list(window), "raw_time": [window[0] + 1, window[1] + 1]} for name, window in windows.items()},
                        "sample_comparability_status": _sample_comparability(attempt, baseline)})
        attempt["diagnostic_gate"] = _instrument_validity_gate(attempt, 20, 5)
        attempts.append(attempt)
    try:
        from pydynpd import command
        source = inspect.getsource(command.command.process_GMM)
        dynamic = {"attempt_id": "dynamic_two_lag_backend_unavailable", "status": "unavailable", "classification": "auxiliary_not_primary_equation",
                   "reason": "pydynpd 0.2.2 rejects a second GMM declaration for the dependent variable, so L1 and L2 cannot receive separate 2:3 and 3:4 collapsed blocks.",
                   "pydynpd_command_source_sha256": hashlib.sha256(source.encode()).hexdigest(), "required_instruments": 22, "required_rank": 22,
                   "required_nominal_hansen_df": 6, "sample_comparability_status": "unresolved"}
    except BaseException as error:
        dynamic = {"attempt_id": "dynamic_two_lag_backend_unavailable", "status": "unavailable", "reason": str(error), "sample_comparability_status": "unresolved"}
    cd_attempts = [_benchmark_cd(data, False), _benchmark_cd(data, True)]
    for attempt in attempts:
        (run_dir / f"{attempt['attempt_id']}_raw_console.log").write_text(attempt.get("raw_console", attempt.get("error", attempt.get("reason", ""))), encoding="utf-8")
    audit = _direct_terms_audit()
    (run_dir / "direct_terms_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {"decision": "diagnostic_complete_no_model_accepted", "user_authorization": "2026-08-26 finite preflight-only direct/control validity matrix; no production, selection, System GMM, or further grid.",
              "input": {"prepared_panel": str(prepared_panel), "sha256": panel_sha256}, "source_window": {"start": start, "end": end, "rows": len(data), "regions": int(data.Region.nunique())},
              "direct_terms_audit": audit, "attempts": attempts, "dynamic_attempt": dynamic, "cd_benchmarks": cd_attempts,
              "interpretation_limit": "An overall Hansen J cannot identify an individual direct term or prove strict exogeneity; non-rejection never accepts a model."}
    json_path, markdown_path = run_dir / "diagnostics.json", run_dir / "report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = ["# ConsCred Instrument-Validity Diagnosis", "", "Decision: **diagnostic_complete_no_model_accepted**.", "",
             "All runs are finite pydynpd Difference-GMM preflight evidence, not production estimation or a model-selection exercise.", "",
             "## Monthly Attempts", "", "| Attempt | Instruments/rank | Hansen p | AR(2) p | Sample | Outcome |", "|---|---:|---:|---:|---|---|"]
    for item in attempts:
        lines.append(f"| {item['attempt_id']} | {item.get('instrument_count', 'n/a')}/{item.get('instrument_rank', 'n/a')} | {item.get('hansen_p', 'n/a')} | {item.get('ar2_p', 'n/a')} | {item.get('sample_comparability_status', 'n/a')} | {item.get('diagnostic_gate', 'n/a')} |")
    lines.extend(["", "## Direct-Term Audit", "", "Timing/provenance gaps are retained as unresolved; an overall Hansen J does not test one direct term in isolation.", "",
                  "| Term | Scope | Timing | Feedback risk | Status |", "|---|---|---|---|---|"])
    for item in audit:
        lines.append(f"| {item['term']} | {item['scope']} | {item['publication_or_effective_timing']} | {item['contemporaneous_feedback_risk']} | {item['status']} |")
    lines.extend(["", "## Cross-Section Dependence Benchmarks", "", "Residual-source evidence only; these are not GMM Hansen or AR diagnostics.", "",
                  "| Benchmark | CD statistic | Two-sided p | Coverage |", "|---|---:|---:|---|"])
    for item in cd_attempts:
        lines.append(f"| {item['attempt_id']} | {item['pesaran_cd_statistic']} | {item['pesaran_cd_two_sided_p']} | {item['residual_coverage']} |")
    lines.extend(["", "## Dynamic Auxiliary", "", dynamic["reason"], "", "Every command, raw console, source hash, sample/rank evidence, conditioning evidence, and audit is retained in this immutable bundle.", ""])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return report, markdown_path, json_path


def run_historical_regime_diagnosis(full_panel: Path = HISTORICAL_FULL_PANEL,
                                    results_dir: Path = Path("Results/GMM/preflight")) -> tuple[dict[str, Any], Path, Path]:
    """Run the fixed historical Difference-GMM regime-dummy diagnostic, never model selection."""
    tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = results_dir / f"ConsCred_GMM_historical_regime_diagnosis_{tag}"
    run_dir.mkdir(parents=True, exist_ok=False)
    attempts: list[dict[str, Any]] = []
    input_evidence: dict[str, Any] = {"parent_full_panel": str(full_panel), "expected_sha256": HISTORICAL_FULL_PANEL_SHA256}
    try:
        data, input_evidence = load_historical_regime_data(full_panel)
        transition_audit = _regime_transition_audit(data)
        snapshot = run_dir / "prepared_panel.csv"
        data[["Region", "Date", DEPENDENT, f"{DEPENDENT}_lag1", *PREDTERMINED, *HISTORICAL_REGIME_DIRECT]].to_csv(snapshot, index=False)
        input_evidence["prepared_panel_snapshot_sha256"] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        registry = (
            ("historical_baseline", DIRECT, 20),
            ("historical_covid_sank_main_effects_auxiliary", HISTORICAL_REGIME_DIRECT, 22),
        )
        try:
            installed_version = version("pydynpd")
        except BaseException as error:
            installed_version = None
            version_error = f"{type(error).__name__}: {error}"
        else:
            version_error = None if installed_version == "0.2.2" else f"required pydynpd 0.2.2; found {installed_version}"
        for attempt_id, direct_terms, expected_count in registry:
            if version_error:
                attempt = {"attempt_id": attempt_id, "status": "unresolved", "package": "pydynpd", "package_version": installed_version,
                           "direct_terms": list(direct_terms), "required_instrument_count": expected_count, "required_rank": expected_count,
                           "required_hansen_df": 5, "error": version_error}
            else:
                attempt = _pydynpd_attempt(data, "A", "difference", "two_step", direct_terms=direct_terms)
                attempt.update({"attempt_id": attempt_id, "required_instrument_count": expected_count, "required_rank": expected_count,
                                "required_hansen_df": 5, "package_version_required": "0.2.2"})
            attempt["sample_comparability_status"] = "match" if not attempts else _sample_comparability(attempt, attempts[0])
            attempt["diagnostic_gate"] = _instrument_validity_gate(attempt, expected_count, 5)
            if attempt.get("groups") != 75 or attempt.get("nobs") != 5925:
                attempt["diagnostic_gate"] = "unresolved"
            (run_dir / f"{attempt_id}_raw_console.log").write_text(attempt.get("raw_console", attempt.get("error", "")), encoding="utf-8")
            attempts.append(attempt)
        cd_benchmarks = [_benchmark_cd(data, False, DIRECT), _benchmark_cd(data, False, HISTORICAL_REGIME_DIRECT)]
        decision = "diagnostic_complete_no_model_accepted"
        validation_error = None
    except BaseException as error:
        transition_audit = {}
        cd_benchmarks = []
        decision = "historical_regime_feasibility_unresolved"
        validation_error = f"{type(error).__name__}: {error}"
        try:
            input_evidence["observed_sha256"] = hashlib.sha256(full_panel.read_bytes()).hexdigest()
        except OSError:
            pass
    report = {"decision": decision,
              "scope": "historical 2019-04 through 2025-12 Difference-GMM two-step auxiliary; no production, selection, System GMM, time effects, or shock-by-regime interactions.",
              "input": input_evidence, "transition_audit": transition_audit, "attempt_registry": ["historical_baseline", "historical_covid_sank_main_effects_auxiliary"],
              "attempts": attempts, "cd_benchmarks": cd_benchmarks, "validation_error": validation_error,
              "backend_limitation": "pydynpd warns that Difference-GMM is not ideal when T >= N; this historical result is restricted diagnostic evidence.",
              "interpretation_limit": "In Difference-GMM step dummies enter through onset/exit changes, not as full month fixed effects. A lower CD statistic or Hansen non-rejection never accepts a model."}
    json_path, markdown_path = run_dir / "diagnostics.json", run_dir / "report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = ["# ConsCred Historical COVID/Sanctions-Regime Diagnosis", "", f"Decision: **{decision}**.", "",
             "This is a bounded historical Difference-GMM diagnostic. Neither attempt is a production model or a model-acceptance decision.", "",
             "## Input Validation", "", f"Parent checksum: `{input_evidence.get('observed_sha256', 'unavailable')}`. Both regime dummies passed binary, nationally-common-within-date, and historical-variation checks before fitting.", "",
             "## Regime-Dummy Interpretation", "", "Difference-GMM transforms step dummies into their transition months. They test discrete regime changes, not full time effects that absorb every national monthly shock.", "",
             "| Dummy | Level-one months | Transition months | Observed exit transition |", "|---|---:|---|---|"]
    for name, item in transition_audit.items():
        lines.append(f"| {name} | {item['level_coverage']['one_months']} | {', '.join(item['differenced_nonzero_dates']) or 'none'} | {item['observed_exit_transition']} |")
    if attempts:
        lines.extend(["", "## Fixed Attempts", "", "| Attempt | Instruments/rank | Hansen J (df) | Hansen p | AR(2) | Groups/observations | Outcome |", "|---|---:|---|---:|---|---|---|"])
        for item in attempts:
            lines.append(f"| {item['attempt_id']} | {item.get('instrument_count', 'n/a')}/{item.get('instrument_rank', 'n/a')} | {item.get('hansen_statistic', 'n/a')} ({item.get('hansen_df', 'n/a')}) | {item.get('hansen_p', 'n/a')} | {item.get('ar2_p', 'n/a')} ({item.get('ar2_status', 'n/a')}) | {item.get('groups', 'n/a')}/{item.get('nobs', 'n/a')} | {item.get('diagnostic_gate', 'n/a')} |")
    if cd_benchmarks:
        lines.extend(["", "## Residual CD Benchmarks", "", "| Equation | CD statistic | Two-sided p | Coverage |", "|---|---:|---:|---|"])
        for item in cd_benchmarks:
            lines.append(f"| {'+ Covid_dum + Sank_dum' if 'Covid_dum' in item['regressors'] else 'baseline'} | {item['pesaran_cd_statistic']} | {item['pesaran_cd_two_sided_p']} | {item['residual_coverage']} |")
    if validation_error:
        lines.extend(["", f"Feasibility failure retained: {validation_error}"])
    lines.extend(["", "## Limits", "", report["backend_limitation"], "", report["interpretation_limit"], ""])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return report, markdown_path, json_path


def prepare_quarterly_instrument_validity_data(workbook: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read and transform the fixed quarterly no-shock auxiliary panel without using Date for lags."""
    source_sha256 = hashlib.sha256(workbook.read_bytes()).hexdigest()
    data = pd.read_excel(workbook, sheet_name="data_quarter")
    required = {"Region", "Date", "Date_quarter", "Cred_nagr_Q", "Zakred_Q", "Def_Zadolg_Fl_Q", "New_Loans_Fl_Q",
                "Int_Rate_ConsCred_Q", "Fin_Dostup_Q", "Cap_to_assets_Q"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Quarterly workbook is missing: {', '.join(missing)}")
    parsed = data["Date_quarter"].astype(str).str.fullmatch(r"[1-4]кв\d{2}")
    if not parsed.all():
        raise ValueError("Date_quarter must use the NквYY convention")
    data["Quarter"] = data["Date_quarter"].str.extract(r"([1-4])кв(\d{2})").apply(lambda row: f"20{row.iloc[1]}Q{row.iloc[0]}", axis=1)
    periods = pd.PeriodIndex(data["Quarter"], freq="Q")
    data["Quarter"] = periods.astype(str)
    if data.duplicated(["Region", "Quarter"]).any() or data.groupby("Quarter")["Date"].nunique().gt(1).any():
        raise ValueError("Quarterly Region-Quarter keys or source Date mapping are not one-to-one")
    expected = pd.period_range(periods.min(), periods.max(), freq="Q").astype(str)
    if not pd.Index(sorted(data["Quarter"].unique())).equals(expected):
        raise ValueError("Quarter IDs are not consecutive")
    if (data["Fin_Dostup_Q"] <= 0).any():
        raise ValueError("Fin_Dostup_Q must be strictly positive")
    data = data.sort_values(["Region", "Quarter"]).copy()
    data[QUARTERLY_DEPENDENT] = data.groupby("Region")["Int_Rate_ConsCred_Q"].diff()
    for source, target in zip(("Cred_nagr_Q", "Zakred_Q", "Cap_to_assets_Q"), QUARTERLY_CONTROLS, strict=True):
        data[target] = data.groupby("Region")[source].shift(1)
    data[QUARTERLY_DIRECT[0]] = np.log(data["Fin_Dostup_Q"])
    panel = data[["Region", "Quarter", "Date", QUARTERLY_DEPENDENT, *QUARTERLY_CONTROLS, *QUARTERLY_DIRECT]].dropna().copy()
    counts = panel.groupby("Region")["Quarter"].nunique()
    usable_periods = pd.PeriodIndex(panel["Quarter"], freq="Q")
    if (len(counts) != 85 or counts.nunique() != 1 or not pd.Index(sorted(panel["Quarter"].unique())).equals(pd.period_range(usable_periods.min(), usable_periods.max(), freq="Q").astype(str))
            or panel.duplicated(["Region", "Quarter"]).any() or panel.isna().any().any()):
        raise ValueError("Quarterly transformed panel is not balanced, consecutive, unique, and complete")
    metadata = {"source": {"workbook": str(workbook), "sha256": source_sha256, "sheet": "data_quarter"},
                "coverage": {"raw_rows": len(data), "prepared_rows": len(panel), "regions": int(panel.Region.nunique()),
                             "quarters": int(panel.Quarter.nunique()), "start": panel.Quarter.min(), "end": panel.Quarter.max()},
                "date_timestamp_semantics": "unresolved", "lag_key": "Quarter", "transformations": {"dependent": "diff(Int_Rate_ConsCred_Q)",
                "controls": {target: f"lag1({source})" for source, target in zip(("Cred_nagr_Q", "Zakred_Q", "Cap_to_assets_Q"), QUARTERLY_CONTROLS, strict=True)},
                "direct": "log(Fin_Dostup_Q)", "available_but_excluded": ["Def_Zadolg_Fl_Q", "New_Loans_Fl_Q"]}}
    return panel.drop(columns="Date").rename(columns={"Quarter": "Date"}), metadata


def run_quarterly_instrument_validity_diagnosis(workbook: Path, results_dir: Path = Path("Results/GMM/preflight")) -> tuple[dict[str, Any], Path, Path]:
    """Run the two predeclared quarterly no-shock auxiliary Difference-GMM attempts."""
    tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = results_dir / f"ConsCred_GMM_quarterly_instrument_validity_diagnosis_{tag}"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        panel, metadata = prepare_quarterly_instrument_validity_data(workbook)
        prepared_path = run_dir / "prepared_panel.csv"
        panel.to_csv(prepared_path, index=False)
        metadata["prepared_panel_sha256"] = hashlib.sha256(prepared_path.read_bytes()).hexdigest()
        baseline = _pydynpd_attempt(panel, "A", "difference", "two_step", QUARTERLY_CONTROLS, direct_terms=QUARTERLY_DIRECT,
                                    dependent=QUARTERLY_DEPENDENT, control_order=QUARTERLY_CONTROLS)
        baseline.update({"attempt_id": "quarterly_no_shock_baseline", "classification": "no_shock_auxiliary_not_monthly_profile_A_or_monetary_shock_model",
                         "sample_comparability_status": "match"})
        baseline["diagnostic_gate"] = _instrument_validity_gate(baseline, 9, 4)
        delayed = _pydynpd_attempt(panel, "A", "difference", "two_step", QUARTERLY_CONTROLS, control_window=(2, 3), direct_terms=QUARTERLY_DIRECT,
                                   dependent=QUARTERLY_DEPENDENT, control_order=QUARTERLY_CONTROLS)
        delayed.update({"attempt_id": "quarterly_no_shock_controls_2_3_auxiliary", "classification": "no_shock_auxiliary_not_monthly_profile_A_or_monetary_shock_model",
                        "sample_comparability_status": _sample_comparability(delayed, baseline)})
        delayed["diagnostic_gate"] = _instrument_validity_gate(delayed, 9, 4)
        attempts = [baseline, delayed]
        for attempt in attempts:
            (run_dir / f"{attempt['attempt_id']}_raw_console.log").write_text(attempt.get("raw_console", attempt.get("error", "")), encoding="utf-8")
        report = {"decision": "diagnostic_complete_no_model_accepted", "scope": "quarterly_no_shock_auxiliary; not a monthly Profile-A or monetary-shock model",
                  "metadata": metadata, "attempts": attempts,
                  "interpretation_limit": "This branch jointly changes frequency, sample, transformations, controls, and the shock block; it cannot attribute any result to monthly shocks."}
    except BaseException as error:
        report = {"decision": "quarterly_feasibility_unresolved", "scope": "quarterly_no_shock_auxiliary", "error_type": type(error).__name__, "error": str(error)}
    json_path, markdown_path = run_dir / "diagnostics.json", run_dir / "report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = ["# ConsCred Quarterly No-Shock Instrument-Validity Auxiliary", "", f"Decision: **{report['decision']}**.", "",
             "This is auxiliary evidence only. It is not a monthly Profile-A model, a monetary-shock model, or a model-acceptance decision.", ""]
    if "attempts" in report:
        lines.extend(["| Attempt | Instruments/rank | Hansen p | AR(2) p | Sample | Outcome |", "|---|---:|---:|---:|---|---|"])
        for item in report["attempts"]:
            lines.append(f"| {item['attempt_id']} | {item.get('instrument_count', 'n/a')}/{item.get('instrument_rank', 'n/a')} | {item.get('hansen_p', 'n/a')} | {item.get('ar2_p', 'n/a')} | {item.get('sample_comparability_status', 'n/a')} | {item.get('diagnostic_gate', 'n/a')} |")
    else:
        lines.append(f"Feasibility failure retained: {report['error_type']}: {report['error']}")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report, markdown_path, json_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-preflight", action="store_true")
    parser.add_argument("--robust-hansen-diagnosis", action="store_true")
    parser.add_argument("--control-subset-diagnosis", action="store_true")
    parser.add_argument("--delayed-control-lag-diagnosis", action="store_true")
    parser.add_argument("--instrument-validity-diagnosis", action="store_true")
    parser.add_argument("--historical-regime-diagnosis", action="store_true")
    parser.add_argument("--quarterly-instrument-validity-diagnosis", action="store_true")
    parser.add_argument("--quarterly-workbook", type=Path, default=Path("Квартальные данные.xlsx"))
    parser.add_argument("--start", default="2024-01")
    parser.add_argument("--end", default="2025-12")
    parser.add_argument("--prepared-panel", type=Path, default=LOCKED_PREPARED_PANEL)
    args = parser.parse_args()
    if sum((args.backend_preflight, args.robust_hansen_diagnosis, args.control_subset_diagnosis, args.delayed_control_lag_diagnosis,
            args.instrument_validity_diagnosis, args.historical_regime_diagnosis, args.quarterly_instrument_validity_diagnosis)) != 1:
        parser.error("choose exactly one diagnostic mode")
    if args.control_subset_diagnosis:
        report, path, _ = run_control_subset_diagnosis(args.start, args.end, args.prepared_panel)
        print(f"{report['decision']}: {path}")
        return 0
    if args.robust_hansen_diagnosis:
        report, path, _ = run_robust_hansen_diagnosis(args.start, args.end, args.prepared_panel)
        print(f"{report['decision']}: {path}")
        return 0
    if args.delayed_control_lag_diagnosis:
        report, path, _ = run_delayed_control_lag_diagnosis(args.start, args.end, args.prepared_panel)
        print(f"{report['decision']}: {path}")
        return 0
    if args.instrument_validity_diagnosis:
        report, path, _ = run_instrument_validity_diagnosis(args.start, args.end, args.prepared_panel)
        print(f"{report['decision']}: {path}")
        return 0
    if args.historical_regime_diagnosis:
        report, path, _ = run_historical_regime_diagnosis()
        print(f"{report['decision']}: {path}")
        return 0
    if args.quarterly_instrument_validity_diagnosis:
        report, path, _ = run_quarterly_instrument_validity_diagnosis(args.quarterly_workbook)
        print(f"{report['decision']}: {path}")
        return 0
    report, path, _ = run_backend_preflight(args.start, args.end, args.prepared_panel)
    print(f"{report['decision']}: {path}")
    return 0 if report["decision"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
