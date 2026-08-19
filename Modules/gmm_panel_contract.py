"""Build the reproducible ConsCred Phase 1 GMM input contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


WORKBOOK = Path("Operations/conscred_reg_analys.xlsx")
CLUSTER_MAP = Path("region_cluster_final.pkl")
RESULTS_ROOT = Path("Results/GMM")
PANEL_START = pd.Timestamp("2019-03-01")
VALIDATION_START = pd.Timestamp("2024-01-01")
VALIDATION_END = pd.Timestamp("2025-12-01")
KEYS = ("Region", "Date")
EXPECTED_REGION_COUNT = 75
OUTCOME = "d_Int_Rate_ConsCred"
CONTROLS = ("Cred_nagr", "Zakred", "Cap_to_assets", "CPI_reg")
CLUSTERS = ("Cluster_new_cd_1", "Cluster_new_cd_3", "Cluster_new_cd_4")
SHOCKS = ("d_Mon_Shock_pos", "d_Mon_Shock_neg")
REQUIRED = (OUTCOME, *CONTROLS, "ln_Fin_Dostup", "d_Inflation_Expectations", *SHOCKS, *CLUSTERS)
LIMITATIONS = [
    "The source/formula for Cred_nagr, Zakred, Cap_to_assets, CPI_reg, and ln_Fin_Dostup is not yet documented.",
    "The survey statistic behind Inflation_Expectations is not yet documented.",
    "The workbook shocks are the approved input; their underlying Taylor-run inputs and coefficients are not reconstructed here.",
]


class ContractError(ValueError):
    """A source contract violation that blocks Phase 2."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _lag_name(column: str, horizon: int) -> str:
    return column if horizon == 0 else f"{column}_lag{horizon}"


def _model_columns(horizon: int) -> list[str]:
    shocks = [_lag_name(shock, horizon) for shock in SHOCKS]
    interactions = [f"{shock}_{cluster}" for shock in shocks for cluster in CLUSTERS]
    return [OUTCOME, f"{OUTCOME}_lag1", *[f"{column}_lag1" for column in CONTROLS], "ln_Fin_Dostup", "d_Inflation_Expectations", *shocks, *interactions]


def _audit_raw(data: pd.DataFrame, mapping: pd.DataFrame) -> dict[str, Any]:
    missing = sorted(set((*KEYS, *REQUIRED)).difference(data.columns))
    if missing:
        raise ContractError(f"Missing required source columns: {', '.join(missing)}")
    if data["Date"].isna().any():
        raise ContractError("Invalid Date values in source panel")
    if data.duplicated(list(KEYS)).any():
        raise ContractError("Duplicate Region-Date keys in source panel")
    if mapping.duplicated("Region").any() or mapping["Region"].isna().any():
        raise ContractError("Cluster mapping must have one non-null row per Region")
    if set(data.Region) != set(mapping.Region) or data.Region.nunique() != EXPECTED_REGION_COUNT or mapping.Region.nunique() != EXPECTED_REGION_COUNT:
        raise ContractError(f"Source panel and cluster mapping must contain the same {EXPECTED_REGION_COUNT} regions")
    if data.Date.min() != PANEL_START:
        raise ContractError(f"Source panel must start at {PANEL_START.date()}")
    expected_dates = pd.date_range(PANEL_START, data.Date.max(), freq="MS")
    counts: dict[str, int] = {}
    for region, dates in data.groupby("Region", sort=False)["Date"]:
        if not pd.DatetimeIndex(dates).equals(expected_dates):
            raise ContractError(f"Incomplete or non-consecutive monthly coverage for Region {region}")
        counts[str(region)] = len(dates)
    return {
        "status": "passed", "rows": len(data), "regions": int(data.Region.nunique()),
        "date_range": {"start": str(data.Date.min().date()), "end": str(data.Date.max().date())},
        "per_region_counts": counts,
        "required_column_missing_counts": {column: int(data[column].isna().sum()) for column in REQUIRED},
    }


def _audit_clusters(data: pd.DataFrame, mapping: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing = sorted(set(("Region", *CLUSTERS)).difference(mapping.columns))
    if missing:
        raise ContractError(f"Cluster mapping missing columns: {', '.join(missing)}")
    joined = data.merge(mapping[["Region", *CLUSTERS]], on="Region", how="left", validate="many_to_one", suffixes=("", "_map"))
    if joined[[f"{column}_map" for column in CLUSTERS]].isna().any().any():
        raise ContractError("Cluster mapping is missing a source region")
    for column in CLUSTERS:
        if not joined[column].isin((0, 1)).all() or not joined[f"{column}_map"].isin((0, 1)).all():
            raise ContractError("Cluster indicators must be binary")
        if not joined[column].equals(joined[f"{column}_map"]):
            raise ContractError(f"Workbook cluster differs from authoritative mapping: {column}")
    membership = joined[list(CLUSTERS)].sum(axis=1)
    if not membership.isin((0, 1)).all():
        raise ContractError("Cluster indicators must be mutually exclusive; cluster 2 is the implicit reference")
    return joined.drop(columns=[f"{column}_map" for column in CLUSTERS]), {
        "status": "passed", "membership_counts": {column: int(joined[column].sum()) for column in CLUSTERS},
        "reference_cluster_count": int((membership == 0).sum()),
    }


def _transform(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    panel = data.copy()
    reconstructed = panel[SHOCKS[0]] + panel[SHOCKS[1]]
    if (panel[SHOCKS[0]] < 0).any() or (panel[SHOCKS[1]] > 0).any():
        raise ContractError("Positive/negative shock columns violate their sign contract")
    signed_source = "d_Mon_Shock" in panel
    residual = panel["d_Mon_Shock"] - reconstructed if signed_source else reconstructed * 0
    if signed_source and not residual.abs().le(1e-10).all():
        raise ContractError("Positive and negative shocks do not reconstruct d_Mon_Shock within 1e-10")
    panel["d_Mon_Shock"] = reconstructed
    grouped = panel.groupby("Region", sort=False)
    panel[f"{OUTCOME}_lag1"] = grouped[OUTCOME].shift(1)
    for column in CONTROLS:
        panel[f"{column}_lag1"] = grouped[column].shift(1)
    for horizon in range(1, 7):
        for shock in SHOCKS:
            panel[_lag_name(shock, horizon)] = grouped[shock].shift(horizon)
    for horizon in range(7):
        for shock in SHOCKS:
            name = _lag_name(shock, horizon)
            for cluster in CLUSTERS:
                panel[f"{name}_{cluster}"] = panel[name] * panel[cluster]
    boundary = {str(horizon): int(panel[_lag_name(SHOCKS[0], horizon)].isna().sum()) for horizon in range(7)}
    return panel, {"status": "passed", "tolerance": 1e-10, "reconstruction": {"signed_source_present": signed_source, "max_abs_residual": float(residual.abs().max()), "nonzero_residuals": int(residual.ne(0).sum())}, "shock_lag_boundary_counts": boundary}


def _eligibility(panel: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    results: dict[str, Any] = {}
    audit: dict[str, Any] = {"status": "passed", "horizons": results}
    for horizon in range(7):
        columns = _model_columns(horizon)
        missing = panel[columns].isna()
        mask = ~missing.any(axis=1)
        rejected = panel.loc[~mask]
        by_column = {column: int(missing.loc[~mask, column].sum()) for column in columns if missing[column].any()}
        expected_shocks = [_lag_name(shock, horizon) for shock in SHOCKS]
        expected_missing = pd.DataFrame(False, index=panel.index, columns=columns)
        first_month = panel.Date.eq(PANEL_START)
        for column in (f"{OUTCOME}_lag1", *[f"{column}_lag1" for column in CONTROLS], "d_Inflation_Expectations"):
            expected_missing.loc[first_month, column] = True
        if horizon:
            lag_boundary = panel.Date.lt(PANEL_START + pd.DateOffset(months=horizon))
            for shock in expected_shocks:
                expected_missing.loc[lag_boundary, shock] = True
                for cluster in CLUSTERS:
                    expected_missing.loc[lag_boundary, f"{shock}_{cluster}"] = True
        unexpected_missing = missing & ~expected_missing
        unexpected = {column: int(unexpected_missing[column].sum()) for column in columns if unexpected_missing[column].any()}
        validation_mask = mask & panel.Date.between(VALIDATION_START, VALIDATION_END)
        validation_rows = int(validation_mask.sum())
        validation_expected = int(panel.loc[panel.Date.between(VALIDATION_START, VALIDATION_END), "Region"].nunique()) * 24
        balance = "unbalanced" if unexpected else "balanced"
        results[str(horizon)] = {
            "model_columns": columns, "input_rows": len(panel), "eligible_rows": int(mask.sum()), "rejected_rows": int((~mask).sum()),
            "eligible_date_range": {"start": str(panel.loc[mask, "Date"].min().date()), "end": str(panel.loc[mask, "Date"].max().date())},
            "missing_column_counts": by_column, "unexpected_missing_counts": unexpected, "per_date_rejected": {str(key.date()): int(value) for key, value in rejected.groupby("Date").size().items()},
            "per_region_rejected": {str(key): int(value) for key, value in rejected.groupby("Region").size().items()},
            "balance_status": balance, "validation_rows": validation_rows, "validation_expected_rows": validation_expected,
            "validation_clean": validation_rows == validation_expected,
        }
    return results, audit


def _report(status: str, run_id: str, message: str, eligibility: dict[str, Any] | None = None) -> str:
    lines = ["# ConsCred GMM Phase 1", "", f"Status: **{status}**", "", f"Run ID: `{run_id}`", "", message, ""]
    if eligibility:
        lines += ["## Eligibility", "", "| Horizon | Eligible rows | Validation rows | Balance |", "|---:|---:|---:|---|"]
        lines += [f"| {h} | {item['eligible_rows']} | {item['validation_rows']} | {item['balance_status']} |" for h, item in eligibility.items()]
    return "\n".join(lines)


def run_contract(workbook: Path = WORKBOOK, cluster_map: Path = CLUSTER_MAP, output_root: Path = RESULTS_ROOT, run_id: str | None = None) -> tuple[dict[str, Any], Path]:
    """Create one immutable Phase 1 evidence directory and return its manifest."""
    run_id = run_id or f"ConsCred_GMM_{datetime.now():%Y%m%d_%H%M%S_%f}"
    run_dir = output_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    manifest: dict[str, Any] = {"phase": 1, "run_id": run_id, "status": "failed", "sources": {}, "limitations": LIMITATIONS, "artifacts": {}}
    eligibility: dict[str, Any] | None = None
    try:
        if not workbook.is_file() or not cluster_map.is_file():
            raise ContractError("Workbook and cluster-map paths must exist")
        manifest["sources"] = {"workbook": {"path": str(workbook), "sha256": _sha256(workbook)}, "cluster_map": {"path": str(cluster_map), "sha256": _sha256(cluster_map)}}
        data = pd.read_excel(workbook).drop(columns=["Unnamed: 0"], errors="ignore")
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        data = data.sort_values(list(KEYS)).reset_index(drop=True)
        mapping = pd.read_pickle(cluster_map)
        raw = _audit_raw(data, mapping)
        _json(run_dir / "raw_panel_audit.json", raw)
        panel, clusters = _audit_clusters(data, mapping)
        _json(run_dir / "cluster_audit.json", clusters)
        panel, shocks = _transform(panel)
        _json(run_dir / "shock_audit.json", shocks)
        eligibility, loss = _eligibility(panel)
        _json(run_dir / "row_loss_audit.json", loss)
        manifest.update({"raw_panel": raw, "cluster_audit": clusters, "shock_audit": shocks, "horizons": eligibility,
                         "full_panel_window": {"start": str(PANEL_START.date()), "end": str(panel.Date.max().date())},
                         "current_baseline_end": "2025-12-01", "transformations": {"regional_lags": [f"{OUTCOME}_lag1", *[f"{column}_lag1" for column in CONTROLS]], "shock_horizons": list(range(7))}})
        panel.to_csv(run_dir / "full_analysis_panel.csv", index=False)
        manifest["artifacts"]["full_analysis_panel.csv"] = _sha256(run_dir / "full_analysis_panel.csv")
        manifest["schema"] = {"full_analysis_panel.csv": list(panel.columns), "prepared_panel.csv": [*KEYS, *_model_columns(0)]}
        manifest["variable_roles"] = {"outcome": OUTCOME, "dynamic_regressor": f"{OUTCOME}_lag1", "predetermined_controls": [f"{column}_lag1" for column in CONTROLS], "ordinary_direct_h0": _model_columns(0)[6:], "fixed_cluster_reference": "Cluster_new_cd_2"}
        h0 = eligibility["0"]
        if not h0["validation_clean"]:
            manifest["status"] = "unresolved"
            message = "The h=0 validation slice is not clean; Phase 2 is not authorized."
        else:
            validation = panel.loc[panel.Date.between(VALIDATION_START, VALIDATION_END) & panel[_model_columns(0)].notna().all(axis=1), [*KEYS, *_model_columns(0)]]
            validation.to_csv(run_dir / "prepared_panel.csv", index=False)
            manifest["artifacts"]["prepared_panel.csv"] = _sha256(run_dir / "prepared_panel.csv")
            manifest["status"] = "passed"
            message = "Phase 1 passed; the clean h=0 validation panel may be handed to Phase 2."
        manifest["artifacts"].update({name: _sha256(run_dir / name) for name in ("raw_panel_audit.json", "cluster_audit.json", "shock_audit.json", "row_loss_audit.json")})
    except (ContractError, KeyError, ValueError) as error:
        manifest["error"] = str(error)
        message = f"Phase 1 failed: {error}. Correct the source contract and create a new run."
    _json(run_dir / "specification_manifest.json", manifest)
    # A manifest cannot checksum itself without a circular rewrite.
    (run_dir / "report.md").write_text(_report(manifest["status"], run_id, message, eligibility), encoding="utf-8")
    return manifest, run_dir


def verify_manifest(run_dir: Path) -> bool:
    """Return whether every non-self artifact checksum in a manifest still matches."""
    manifest = json.loads((run_dir / "specification_manifest.json").read_text(encoding="utf-8"))
    return all(_sha256(run_dir / name) == digest for name, digest in manifest["artifacts"].items() if name != "specification_manifest.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, default=WORKBOOK)
    parser.add_argument("--cluster-map", type=Path, default=CLUSTER_MAP)
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.output_root.resolve() != RESULTS_ROOT.resolve():
        parser.error("CLI output root must be Results/GMM")
    manifest, run_dir = run_contract(args.workbook, args.cluster_map, args.output_root, args.run_id)
    print(run_dir)
    return 0 if manifest["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
