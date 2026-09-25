"""Level-data Difference-GMM search for the consumer-credit rate.

The estimator, rather than Python, takes the sole first difference.  Results
are an auditable search ledger; a non-rejected Hansen test is not model
selection or economic acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


WORKBOOK = Path("База данных_рег и фед показатели.xlsx")
SHOCK = Path("mon_shock_dataset.pkl")
CLEAN_WORKBOOK = Path("Operations/conscred_reg_analys.xlsx")
FEDERAL_WORKBOOK = Path("Operations/fed_analys.xlsx")
RESULTS_ROOT = Path("Results/GMM")
LEGACY_PANEL = Path("Results/GMM/ConsCred_GMM_20260820_120000_000001/prepared_panel.csv")
LEGACY_DIAGNOSTICS = Path("Results/GMM/preflight/ConsCred_GMM_control_subset_diagnosis_20260820_175804/diagnostics.json")
RSCRIPT = Path(r"C:\Program Files\R\R-4.4.2\bin\x64\Rscript.exe")
R_LIBRARY = Path(".agent/tmp/r-library").resolve()
DEPENDENT = "Int_Rate_ConsCred"
DIRECT = ("Mon_Shock", "Covid_dum", "Sank_dum")
CLEAN_CLUSTER_COLUMNS = ("Cluster_new_cd_1", "Cluster_new_cd_3", "Cluster_new_cd_4")
CLEAN_DIRECT = ("Mon_Shock", *(f"Mon_Shock_{name}" for name in CLEAN_CLUSTER_COLUMNS))
CLEAN_SPECS = {
    "core": ("Zakred", "CPI_reg", "Cap_to_assets", "Cred_nagr"),
    "access": ("Zakred", "CPI_reg", "Cap_to_assets", "ln_Fin_Dostup"),
    "distress": ("Zakred", "Def_Zadolg_ConsCred", "Cap_to_assets", "ln_Fin_Dostup"),
    "without_zakred": ("CPI_reg", "Cap_to_assets", "Cred_nagr"),
    "without_cpi": ("Zakred", "Cap_to_assets", "Cred_nagr"),
    "without_capital": ("Zakred", "CPI_reg", "Cred_nagr"),
    "without_credit_burden": ("Zakred", "CPI_reg", "Cap_to_assets"),
    "default_for_credit_burden": ("Zakred", "CPI_reg", "Cap_to_assets", "Def_Zadolg_ConsCred"),
}
FEDERAL_COLUMNS = ("Oil_p", "REER", "Inflation_Expectations")
FEDERAL_SENSITIVITY_SPECS = {
    "core": ((), {}),
    "core_plus_ln_fin_dostup": (("ln_Fin_Dostup",), {"ln_Fin_Dostup": "endogenous"}),
    "core_plus_oil_p": (("Oil_p",), {"Oil_p": "direct"}),
    "core_plus_reer": (("REER",), {"REER": "direct"}),
    "core_plus_inflation_expectations": (("Inflation_Expectations",), {"Inflation_Expectations": "direct"}),
}
LAGS = (2, 3)
EXCLUDED = {"Region", "Date", DEPENDENT, "RUONIA", "ROISFIX", "MIACR", "Key_Rate", "Inflation_Expectations", "Nominal_Percent_Rate", "Ent_conf_ind_mining", "Ent_conf_ind_manufactoring", "d_Mon_Shock", "d_Mon_Shock_pos", "d_Mon_Shock_neg"}
MAP = {"MaP_Fact", "MaP_Announcement", "Mortgage_MaP_Fact", "Mortgage_MaP_Announcement", "Conscred_MaP_Fact", "Conscred_MaP_Announcement", "MaP_Tight_Fact", "MaP_Ease_Fact", "MaP_Tight_Announcement", "MaP_Ease_Announcement", "Mortgage_MaP_Tight_Fact", "Mortgage_MaP_Ease_Fact", "Mortgage_MaP_Tight_Announcement", "Mortgage_MaP_Ease_Announcement", "Conscred_MaP_Tight_Fact", "Conscred_MaP_Ease_Fact", "Conscred_MaP_Tight_Announcement", "Conscred_MaP_Ease_Announcement"}
REGIONAL = {"D_top5_rozn", "Zakred", "Zadolg_Fl", "Zadolg_Mort", "Zadolg_ConsCred", "Def_Zadolg_Fl", "Def_Zadolg_Mort", "Def_Zadolg_ConsCred", "New_Loans_Fl", "New_Loans_Mort", "New_Loans_ConsCred", "Int_Rate_FL", "Int_Rate_Mort", "Int_Rate_Progr_DOMRF", "Fin_Dostup", "CPI_reg", "Cap_to_assets", "Credit_load_Mort", "Cred_nagr"}
NATIONAL = {"Oil_p", "CPI_fed", "REER"}
LEGACY_CONTROLS = ("Cred_nagr_lag1", "Zakred_lag1", "Cap_to_assets_lag1", "CPI_reg_lag1")
APPROVED_PYTHON: Path | None = None


def sha256(value: Path | pd.DataFrame) -> str:
    """Hash an immutable source or exact input table."""
    raw = value.read_bytes() if isinstance(value, Path) else value.to_csv(index=False, date_format="%Y-%m-%d").encode()
    return hashlib.sha256(raw).hexdigest()


def stable_id(controls: tuple[str, ...]) -> str:
    """Return a filesystem-safe deterministic specification id."""
    return "level__" + ("__".join(controls) if controls else "baseline")


def build_pydynpd_command(controls: tuple[str, ...], roles: dict[str, str], direct: tuple[str, ...] = DIRECT) -> str:
    """Build the only permitted level-data, collapsed Difference-GMM command."""
    if len(controls) != len(set(controls)) or any(roles.get(x) not in {"endogenous", "direct"} for x in controls):
        raise ValueError("Controls must be unique and assigned endogenous or direct roles")
    endogenous = tuple(x for x in controls if roles[x] == "endogenous")
    direct = (*direct, *(x for x in controls if roles[x] == "direct"))
    # pydynpd requires ordinary-IV terms in both the equation segment and its
    # explicit iv() declaration; omitting the first occurrence silently drops
    # them from the regression.
    return f"{DEPENDENT} L1.{DEPENDENT} {' '.join((*endogenous, *direct))} | gmm({DEPENDENT}{(' ' + ' '.join(endogenous)) if endogenous else ''}, 2:3) iv({' '.join(direct)}) | collapse nolevel"


def build_r_formula(controls: tuple[str, ...], roles: dict[str, str], direct: tuple[str, ...] = DIRECT) -> str:
    """Build the matching pgmm level-data formula."""
    endogenous = tuple(x for x in controls if roles[x] == "endogenous")
    direct = (*direct, *(x for x in controls if roles[x] == "direct"))
    equation = (f"lag({DEPENDENT}, 1)", *endogenous, *direct)
    gmm = (f"lag({DEPENDENT}, 2:3)", *(f"lag({x}, 2:3)" for x in endogenous))
    return f"{DEPENDENT} ~ {' + '.join(equation)} | {' + '.join(gmm)} | {' + '.join(direct)}"


def load_panel(workbook: Path = WORKBOOK, shock: Path = SHOCK) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read the level sources and fail fast on an ambiguous monthly merge."""
    regional = pd.read_excel(workbook, sheet_name="data", skiprows=1).drop(columns=lambda x: str(x).startswith("Unnamed:"), errors="ignore")
    federal = pd.read_excel(workbook, sheet_name="data_RF", skiprows=1).drop(columns=lambda x: str(x).startswith("Unnamed:"), errors="ignore")
    for frame in (regional, federal):
        frame["Date"] = pd.to_datetime(frame["Date"], errors="raise").dt.to_period("M").dt.to_timestamp()
    if regional.duplicated(["Region", "Date"]).any() or federal.duplicated("Date").any():
        raise ValueError("Duplicate regional or federal merge keys")
    expected = pd.date_range(regional.Date.min(), regional.Date.max(), freq="MS")
    if any(not pd.DatetimeIndex(dates).equals(expected) for _, dates in regional.sort_values("Date").groupby("Region")["Date"]):
        raise ValueError("Regional calendar is incomplete or unbalanced")
    regional = regional.rename(columns={"CPI": "CPI_reg"})
    federal = federal.rename(columns={"CPI": "CPI_fed"}).drop(columns="Region", errors="ignore")
    if set(regional.Date).difference(federal.Date):
        raise ValueError("Federal source is missing regional calendar dates")
    data = regional.merge(federal, on="Date", how="left", validate="many_to_one")
    shocks = pd.read_pickle(shock).copy(); shocks["Date"] = pd.to_datetime(shocks["Date"], errors="raise").dt.to_period("M").dt.to_timestamp()
    if shocks.duplicated("Date").any() or set(("Date", "Mon_Shock")).difference(shocks):
        raise ValueError("Shock source requires one Date and one Mon_Shock per month")
    # The level shock begins in 2019-03; the unmatched regional lead month is
    # outside the agreed common monthly panel, not an imputable missing value.
    data = data.loc[data["Date"] >= shocks["Date"].min()].merge(shocks[["Date", "Mon_Shock"]], on="Date", how="left", validate="many_to_one")
    if data[list(DIRECT)[1:]].isna().any().any() or data["Mon_Shock"].isna().any() or data.filter(regex=r"CPI_[xy]$").columns.any():
        raise ValueError("Source merge has missing shocks or ambiguous CPI columns")
    data = data.sort_values(["Region", "Date"]).reset_index(drop=True)
    audit = {"workbook_sha256": sha256(workbook), "shock_sha256": sha256(shock), "rows": len(data), "regions": int(data.Region.nunique()), "dates": int(data.Date.nunique()), "window": [str(data.Date.min().date()), str(data.Date.max().date())], "balanced": True, "missingness": data.isna().sum().to_dict(), "transformations": "level variables only; Difference-GMM applies the sole first difference"}
    return data, audit


def load_clean_panel(workbook: Path = CLEAN_WORKBOOK, shock: Path = SHOCK) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the cleaned regional panel and its level Taylor residual."""
    data = pd.read_excel(workbook).drop(columns=lambda name: str(name).startswith("Unnamed:"), errors="ignore")
    required = {"Region", "Date", DEPENDENT, *CLEAN_CLUSTER_COLUMNS, *{name for controls in CLEAN_SPECS.values() for name in controls}}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Clean panel is missing required columns: {sorted(missing)}")
    data["Date"] = pd.to_datetime(data["Date"], errors="raise").dt.to_period("M").dt.to_timestamp()
    if data.duplicated(["Region", "Date"]).any():
        raise ValueError("Duplicate clean regional keys")
    expected = pd.date_range(data.Date.min(), data.Date.max(), freq="MS")
    if any(not pd.DatetimeIndex(dates).equals(expected) for _, dates in data.sort_values("Date").groupby("Region")["Date"]):
        raise ValueError("Clean regional calendar is incomplete or unbalanced")
    clusters = data[list(CLEAN_CLUSTER_COLUMNS)]
    if not clusters.isin((0, 1)).all().all() or not clusters.sum(axis=1).isin((0, 1)).all():
        raise ValueError("Cluster columns must be mutually exclusive zero-one dummies")
    shocks = pd.read_pickle(shock).copy()
    if not {"Date", "Mon_Shock"}.issubset(shocks):
        raise ValueError("Shock source requires Date and Mon_Shock")
    shocks["Date"] = pd.to_datetime(shocks["Date"], errors="raise").dt.to_period("M").dt.to_timestamp()
    if shocks.duplicated("Date").any():
        raise ValueError("Shock source has duplicate monthly keys")
    data = data.merge(shocks[["Date", "Mon_Shock"]], on="Date", how="left", validate="many_to_one")
    if data["Mon_Shock"].isna().any():
        raise ValueError("Clean panel has missing merged Mon_Shock values")
    for cluster in CLEAN_CLUSTER_COLUMNS:
        data[f"Mon_Shock_{cluster}"] = data["Mon_Shock"] * data[cluster]
    data = data.sort_values(["Region", "Date"]).reset_index(drop=True)
    audit = {
        "workbook_sha256": sha256(workbook),
        "shock_sha256": sha256(shock),
        "merged_panel_sha256": sha256(data),
        "rows": len(data),
        "regions": int(data.Region.nunique()),
        "dates": int(data.Date.nunique()),
        "window": [str(data.Date.min().date()), str(data.Date.max().date())],
        "balanced": True,
        "available_split_shock_columns": sorted(name for name in data if name.startswith("d_Mon_Shock")),
        "transformations": "level Int_Rate_ConsCred and level Mon_Shock only; Difference-GMM applies the sole first difference",
    }
    return data, audit


def national_series_audit(data: pd.DataFrame, columns: tuple[str, ...] = FEDERAL_COLUMNS) -> dict[str, dict[str, Any]]:
    """Validate nationally common terms and summarise their usable time variation."""
    audit: dict[str, dict[str, Any]] = {}
    for name in columns:
        if name not in data or data[name].isna().any():
            raise ValueError(f"Federal source has missing {name}")
        series = data.groupby("Date", sort=True)[name]
        if (series.nunique(dropna=False) != 1).any():
            raise ValueError(f"Federal source is not nationally constant for {name}")
        changes = series.first().diff().dropna()
        audit[name] = {"within_date_constant": True, "monthly_change_observations": int(len(changes)), "distinct_monthly_changes": int(changes.nunique())}
    return audit


def load_clean_federal_panel(workbook: Path = CLEAN_WORKBOOK, federal_workbook: Path = FEDERAL_WORKBOOK,
                             shock: Path = SHOCK) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the clean panel with federal terms validated against their source."""
    data, audit = load_clean_panel(workbook, shock)
    federal = pd.read_excel(federal_workbook, usecols=lambda name: name in {"Date", *FEDERAL_COLUMNS})
    missing = {"Date", *FEDERAL_COLUMNS}.difference(federal.columns)
    if missing:
        raise ValueError(f"Federal workbook is missing required columns: {sorted(missing)}")
    federal["Date"] = pd.to_datetime(federal["Date"], errors="raise").dt.to_period("M").dt.to_timestamp()
    if federal.duplicated("Date").any():
        raise ValueError("Federal source has duplicate monthly keys")
    federal = federal.loc[federal["Date"].isin(data["Date"].unique())].sort_values("Date").reset_index(drop=True)
    if set(data["Date"].unique()).difference(federal["Date"]):
        raise ValueError("Federal source is missing clean-panel calendar dates")
    if federal[list(FEDERAL_COLUMNS)].isna().any().any():
        raise ValueError("Federal source has missing candidate values")
    for name in ("REER", "Inflation_Expectations"):
        if name in data and not data[["Date", name]].drop_duplicates().sort_values("Date")[name].reset_index(drop=True).equals(federal[name]):
            raise ValueError(f"Clean and federal sources disagree for {name}")
    data = data.drop(columns=[name for name in FEDERAL_COLUMNS if name in data], errors="ignore").merge(federal, on="Date", how="left", validate="many_to_one")
    federal_audit = national_series_audit(data)
    audit.update({"federal_workbook_sha256": sha256(federal_workbook), "federal_source_rows": len(federal), "federal_source_equality": {name: True for name in ("REER", "Inflation_Expectations")}, "national_time_series": federal_audit, "merged_panel_sha256": sha256(data)})
    return data.sort_values(["Region", "Date"]).reset_index(drop=True), audit


def candidate_inventory(data: pd.DataFrame, threshold: float = .95) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Inventory all sources once, retaining only usable agreed-role candidates."""
    base = data[["Region", "Date", DEPENDENT, *DIRECT]].dropna()
    rows: list[dict[str, Any]] = []
    eligible: list[dict[str, str]] = []
    for name in data.columns:
        role = "excluded"; reason = "excluded_by_scope"; scope = "regional" if name in REGIONAL else "national_common"
        if name in MAP: role, reason = "endogenous", "pending"
        elif name in REGIONAL: role, reason = "endogenous", "pending"
        elif name in NATIONAL: role, reason = "direct", "pending"
        elif name in DIRECT: role, reason = "required_direct_not_searched", "required"
        coverage = float(data.loc[base.index, name].notna().mean()) if name in data else 0.0
        varying = bool(data.loc[base.index, name].nunique(dropna=True) > 1) if name in data else False
        series = data.groupby("Date")[name].first() if name in MAP else None
        support = int(((series.shift(2).ne(0) | series.shift(3).ne(0)) & series.shift(3).notna()).sum()) if series is not None else None
        if reason == "pending" and (coverage < threshold or not varying): reason = "ineligible_coverage_or_variation"
        if reason == "pending" and name in MAP and (support or 0) < 3: reason = "national_event_instrument_weak"
        row = {"source": name, "search_column": name if reason == "pending" else "", "role": role, "representation": "level", "variation_scope": scope, "coverage_post_lag": coverage, "nonzero_transition_count": int(data.groupby("Date")[name].first().diff().ne(0).sum()) if name in MAP else None, "lag_2_3_support": support, "status": "eligible" if reason == "pending" else reason}
        rows.append(row)
        if reason == "pending": eligible.append({"name": name, "role": role, "source": name})
    return pd.DataFrame(rows), eligible


def specification(data: pd.DataFrame, controls: tuple[str, ...], roles: dict[str, str], direct: tuple[str, ...] = DIRECT,
                  attempt_id: str | None = None, comparison_family: str | None = None) -> dict[str, Any]:
    """Create one exact, level-data estimator input record without transforming it."""
    columns = list(dict.fromkeys(("Region", "Date", DEPENDENT, *direct, *controls)))
    sample = data[columns].dropna().sort_values(["Region", "Date"]).reset_index(drop=True)
    groups = int(sample.Region.nunique())
    per_group = sample.groupby("Region").size() if groups else pd.Series(dtype=int)
    endogenous = tuple(x for x in controls if roles[x] == "endogenous")
    all_direct = (*direct, *(x for x in controls if roles[x] == "direct"))
    labels = [f"gmm:{DEPENDENT}:L{lag}" for lag in LAGS] + [f"gmm:{name}:L{lag}" for name in endogenous for lag in LAGS] + [f"iv:{name}" for name in all_direct]
    return {"attempt_id": attempt_id or stable_id(controls), "controls": list(controls), "roles": {x: roles[x] for x in controls}, "direct_terms": list(direct), "comparison_family": comparison_family, "columns": columns, "input_sha256": sha256(sample), "key_sha256": sha256(sample[["Region", "Date"]]), "rows": len(sample), "groups": groups, "min_periods": int(per_group.min()) if groups else 0, "different_sample": bool(len(sample) != len(data)), "r_formula": build_r_formula(controls, roles, direct), "pydynpd_command": build_pydynpd_command(controls, roles, direct), "instrument_labels": labels, "normal_iv_count": len(all_direct), "input": sample}


R_BACKEND = r'''args <- commandArgs(trailingOnly=TRUE)
.libPaths(c(normalizePath(args[4]), .libPaths())); library(jsonlite); library(plm)
input <- read.csv(args[1], check.names=FALSE); spec <- fromJSON(args[3]); f <- as.formula(spec$r_formula)
h <- function(x) if(inherits(x,"htest")) list(statistic=unname(x$statistic),df=unname(x$parameter),p_value=x$p.value,method=x$method) else list(error=conditionMessage(x))
out <- tryCatch({ fit <- pgmm(f,input,index=c("region_id","month_id"),effect="individual",model="twosteps",transformation="d",fsm="G",collapse=TRUE); W <- do.call(rbind,fit$W); if(!is.null(spec$instrument_labels) && ncol(W)!=length(spec$instrument_labels)) stop("R instrument labels do not match canonical specification"); colnames(W) <- if(is.null(spec$instrument_labels)) paste0("Z",seq_len(ncol(W))) else unlist(spec$instrument_labels); write.csv(W,args[5],row.names=FALSE); d <- svd(W,nu=0,nv=0)$d; rr <- tryCatch(do.call(cbind,fit$residuals),error=function(e) NULL); cf <- if(is.null(rr)||ncol(rr)<2) NULL else {q<-svd(scale(rr,scale=FALSE),nu=0,nv=0)$d; q[1]^2/sum(q^2)}; v<-tryCatch(vcovHC(fit),error=function(e) NULL); cv<-if(is.null(v)) NULL else list(names=colnames(v),values=unname(v)); list(status="completed",formula=spec$r_formula,nobs=nobs(fit),groups=length(fit$W),instrument_count=ncol(W),instrument_rank=qr(W)$rank,singular_values=d,instrument_labels=colnames(W),key_sha256=spec$key_sha256,input_sha256=spec$input_sha256,residual_common_factor_share=cf,coefficients=as.list(coef(fit)),robust_covariance=cv,ar1=h(tryCatch(mtest(fit,order=1L,vcov=vcovHC(fit)),error=function(e)e)),ar2=h(tryCatch(mtest(fit,order=2L,vcov=vcovHC(fit)),error=function(e)e)),sargan=h(tryCatch(sargan(fit,weights="twosteps"),error=function(e)e)),sargan_label="classical_heteroskedasticity_sensitive") },error=function(e) list(status="failed",error=conditionMessage(e),formula=spec$r_formula))
write_json(out,args[2],auto_unbox=TRUE,null="null",na="null")'''


PY_BACKEND = r'''import contextlib, io, json, sys, hashlib, inspect
import numpy as np, pandas as pd
from systemgmmkit.pydynpd_backend import _apply_numpy_compatibility_shims
from pydynpd import regression, specification_tests
data=pd.read_csv(sys.argv[1]); spec=json.load(open(sys.argv[2],encoding="utf8")); out=io.StringIO()
try:
 _apply_numpy_compatibility_shims()
 with contextlib.redirect_stdout(out): fitted=regression.abond(spec["pydynpd_command"],data,["Region","Date"])
 model=fitted.models[0]; info=model.z_information; height=int(model.z_list.shape[0]/model.N); z=np.concatenate([model.z_list[i*height:(i+1)*height,:] for i in range(model.N)],axis=1).T; np.savetxt(sys.argv[4],z,delimiter=","); s=np.linalg.svd(z,compute_uv=False); tol=float(np.finfo(float).eps*max(z.shape)*s[0]) if len(s) else 0.; ar={x.lag:x.P_value for x in model.AR_list}; h=model.hansen
 if z.shape[1]!=len(spec["instrument_labels"]): raise ValueError("pydynpd instrument labels do not match canonical specification")
 result={"status":"completed","raw_console":out.getvalue(),"groups":int(model.N),"instrument_count":int(info.num_instr),"instrument_rank":int((s>tol).sum()),"singular_values":s.tolist(),"instrument_labels":spec["instrument_labels"],"key_sha256":spec["key_sha256"],"input_sha256":spec["input_sha256"],"ar1_p":ar.get(1),"ar2_p":ar.get(2),"hansen_statistic":getattr(h,"test_value",None),"hansen_df":getattr(h,"df",None),"hansen_p":getattr(h,"p_value",None),"hansen_robustness":"two_step_empirical_moment_covariance","hansen_source_sha256":hashlib.sha256(inspect.getsource(specification_tests.hansen_overid).encode()).hexdigest(),"weighting_source_sha256":hashlib.sha256(inspect.getsource(regression.abond.GMM).encode()).hexdigest()}
except BaseException as e: result={"status":"failed","raw_console":out.getvalue(),"error_type":type(e).__name__,"error":str(e)}
json.dump(result,open(sys.argv[3],"w",encoding="utf8"),ensure_ascii=False)'''


def _run_r(spec: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    input_path = run_dir / f"{spec['attempt_id']}_r_input.csv"; result_path = run_dir / f"{spec['attempt_id']}_r.json"; spec_path = run_dir / f"{spec['attempt_id']}_spec.json"; matrix_path = run_dir / f"{spec['attempt_id']}_r_Z.csv"
    data = spec["input"].copy(); data["region_id"] = pd.factorize(data.Region)[0] + 1; data["month_id"] = pd.factorize(data.Date)[0] + 1; data.to_csv(input_path, index=False); spec_path.write_text(json.dumps({k: v for k, v in spec.items() if k != "input"}, default=str), encoding="utf8")
    call = subprocess.run([str(RSCRIPT), "--vanilla", str(run_dir / "r_backend.R"), str(input_path), str(result_path), str(spec_path), str(R_LIBRARY), str(matrix_path)], capture_output=True, text=True, check=False)
    result = json.loads(result_path.read_text(encoding="utf8")) if result_path.exists() else {"status": "failed", "error": "R backend produced no JSON"}
    result["raw_console"] = call.stdout + call.stderr; result["matrix_path"] = str(matrix_path) if matrix_path.exists() else None
    return result


def robust_preflight(run_dir: Path) -> dict[str, Any]:
    """Verify the immutable approved pydynpd environment before broad search."""
    probe = "from importlib.metadata import version; import pydynpd,systemgmmkit; print(version('pydynpd'))"
    global APPROVED_PYTHON
    interpreters = sorted(Path(".uv-cache/archive-v0").glob("*/Scripts/python.exe"))
    approved = next((path for path in interpreters if subprocess.run([str(path), "-c", probe], capture_output=True, text=True, check=False).stdout.strip() == "0.2.2"), None)
    if approved is None: return {"status": "blocked", "classification": "robust_backend_blocked", "error": "No cached Python interpreter exposes approved pydynpd 0.2.2"}
    APPROVED_PYTHON = approved
    call = subprocess.run([str(approved), "-c", probe], capture_output=True, text=True, check=False)
    probe_result = run_dir / "historical_probe.json"
    historical = "import json,pandas as pd; from Modules.gmm_utils import _pydynpd_attempt; x=_pydynpd_attempt(pd.read_csv(r'''%s'''), 'A', 'difference', 'two_step'); json.dump(x,open(r'''%s''','w'),default=str)" % (LEGACY_PANEL.resolve(), probe_result.resolve())
    historical_call = subprocess.run([str(approved), "-c", historical], capture_output=True, text=True, check=False)
    attempt = json.loads(probe_result.read_text(encoding="utf8")) if probe_result.exists() else {}
    expected = next((x for x in json.loads(LEGACY_DIAGNOSTICS.read_text(encoding="utf8")).get("attempts", []) if x.get("attempt_id") == "retained_Cred_nagr_lag1__Zakred_lag1__Cap_to_assets_lag1__CPI_reg_lag1"), {})
    matches = attempt.get("status") == "completed" and attempt.get("instrument_rank") == expected.get("instrument_rank") and np.isclose(attempt.get("hansen_p", np.nan), expected.get("hansen_p", np.nan)) and np.isclose(attempt.get("ar2_p", np.nan), expected.get("ar2_p", np.nan)) and bool(attempt.get("hansen_robustness_evidence"))
    result = {"status": "passed" if call.returncode == 0 and "0.2.2" in call.stdout and historical_call.returncode == 0 and matches else "blocked", "python": str(approved), "stdout": call.stdout, "stderr": call.stderr, "historical_probe": {"status": attempt.get("status"), "instrument_rank": attempt.get("instrument_rank"), "hansen_p": attempt.get("hansen_p"), "ar2_p": attempt.get("ar2_p"), "source_hash_evidence": attempt.get("hansen_robustness_evidence")}}
    if result["status"] == "blocked": result["classification"] = "robust_backend_blocked"
    (run_dir / "robust_preflight.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    return result


def _run_robust(spec: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    input_path = run_dir / f"{spec['attempt_id']}_py_input.csv"; spec_path = run_dir / f"{spec['attempt_id']}_py_spec.json"; result_path = run_dir / f"{spec['attempt_id']}_py.json"; matrix_path = run_dir / f"{spec['attempt_id']}_py_Z.csv"
    spec["input"].to_csv(input_path, index=False); spec_path.write_text(json.dumps({k: v for k, v in spec.items() if k != "input"}, default=str), encoding="utf8")
    approved = APPROVED_PYTHON
    if approved is None: return {"status": "failed", "error": "approved robust environment unavailable"}
    call = subprocess.run([str(approved), str(run_dir / "pydynpd_backend.py"), str(input_path), str(spec_path), str(result_path), str(matrix_path)], capture_output=True, text=True, check=False)
    result = json.loads(result_path.read_text(encoding="utf8")) if result_path.exists() else {"status": "failed", "error": "pydynpd backend produced no JSON"}
    result["raw_console"] = result.get("raw_console", "") + call.stdout + call.stderr; result["matrix_path"] = str(matrix_path) if matrix_path.exists() else None
    return result


def parity(r: dict[str, Any], py: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Require observable global matrix and structural equality before validation."""
    checks = {"terms": r.get("formula") == spec["r_formula"], "roles": bool(spec["roles"] is not None), "groups": r.get("groups") == py.get("groups") == spec["groups"], "instrument_count": r.get("instrument_count") == py.get("instrument_count") == len(spec["instrument_labels"]), "rank": r.get("instrument_rank") == py.get("instrument_rank"), "key_sha256": r.get("key_sha256") == py.get("key_sha256") == spec["key_sha256"] and r.get("input_sha256") == py.get("input_sha256") == spec["input_sha256"], "lag_window": list(LAGS) == [2, 3], "normal_iv_count": spec["normal_iv_count"] == len(spec["instrument_labels"]) - 2 - 2 * sum(role == "endogenous" for role in spec["roles"].values()), "labels": r.get("instrument_labels") == py.get("instrument_labels") == spec["instrument_labels"]}
    python_column_order_for_r: list[int] | None = None
    if r.get("matrix_path") and py.get("matrix_path"):
        try:
            rz = pd.read_csv(r["matrix_path"]).to_numpy(float); pz = np.loadtxt(py["matrix_path"], delimiter=",")
            checks["matrix_shape"] = rz.shape == pz.shape
            order: list[int] = []
            if checks["matrix_shape"]:
                for column in range(rz.shape[1]):
                    matches = [other for other in range(pz.shape[1]) if np.allclose(rz[:, column], pz[:, other], rtol=1e-8, atol=1e-10)]
                    if len(matches) != 1:
                        order = []
                        break
                    order.append(matches[0])
            checks["matrix_values"] = bool(order and len(set(order)) == len(order) and np.allclose(rz, pz[:, order], rtol=1e-8, atol=1e-10))
            if checks["matrix_values"]:
                python_column_order_for_r = order
        except (OSError, ValueError): checks["matrix_values"] = False
    else: checks["matrix_values"] = False
    return {"status": "match" if all(checks.values()) else "mismatch", "checks": checks, "python_column_order_for_r": python_column_order_for_r}


def classify(r: dict[str, Any], py: dict[str, Any], check: dict[str, Any]) -> list[str]:
    """Return independent diagnostic labels, never an acceptance decision."""
    labels: list[str] = []
    if r.get("status") != "completed" or py.get("status") != "completed": return ["fit_failed"]
    if r.get("instrument_count", 0) >= r.get("groups", 0) or r.get("instrument_rank") != r.get("instrument_count"): labels.append("instrument_cap_invalid")
    if (py.get("hansen_df") or 0) < 5: labels.append("low_power_warning")
    labels.append("r_screen_nonrejecting" if (r.get("sargan", {}).get("p_value") or -1) >= .05 else "r_sargan_rejects")
    if check["status"] != "match": labels.append("robust_hansen_unresolved_instrument_mismatch")
    elif py.get("hansen_p") is None or py.get("ar2_p") is None: labels.append("fit_failed")
    elif py["hansen_p"] >= .05 and py["ar2_p"] >= .05: labels.append("robust_hansen_nonrejecting")
    else: labels.append("robust_hansen_rejects")
    return labels


def paired_comparison_panel(data: pd.DataFrame, left: tuple[str, ...], right: tuple[str, ...]) -> pd.DataFrame:
    """Return the exact common complete-case panel for a comparison with core."""
    columns = list(dict.fromkeys(("Region", "Date", DEPENDENT, *CLEAN_DIRECT, *left, *right)))
    return data[columns].dropna().sort_values(["Region", "Date"]).reset_index(drop=True)


def cluster_shock_totals(r: dict[str, Any]) -> dict[str, Any]:
    """Calculate reference and cluster-specific level-shock responses from R output."""
    coefficients = r.get("coefficients") or {}
    covariance = r.get("robust_covariance") or {}
    names = covariance.get("names") or []
    values = covariance.get("values")
    if not isinstance(coefficients, dict) or values is None or "Mon_Shock" not in coefficients:
        return {"status": "failed", "error": "R coefficients or robust covariance missing"}
    try:
        matrix = np.asarray(values, dtype=float)
        position = {name: index for index, name in enumerate(names)}
        if matrix.shape != (len(names), len(names)) or "Mon_Shock" not in position:
            raise ValueError("covariance names do not match matrix")
        base = "Mon_Shock"
        out: dict[str, Any] = {"status": "completed"}
        for label, term in (("reference", base), *[(cluster.rsplit("_", 1)[-1], cluster) for cluster in CLEAN_DIRECT[1:]]):
            if term == base:
                estimate, variance = float(coefficients[base]), float(matrix[position[base], position[base]])
            else:
                if term not in coefficients or term not in position:
                    raise ValueError(f"missing interaction coefficient: {term}")
                estimate = float(coefficients[base]) + float(coefficients[term])
                variance = float(matrix[position[base], position[base]] + matrix[position[term], position[term]] + 2 * matrix[position[base], position[term]])
            out[label] = {"estimate": estimate, "robust_se": float(np.sqrt(max(variance, 0.0)))}
        return out
    except (TypeError, ValueError, KeyError):
        return {"status": "failed", "error": "R robust covariance is not usable for cluster totals"}


def diagnosis_gate(r: dict[str, Any], py: dict[str, Any], check: dict[str, Any], totals: dict[str, Any], groups: int) -> dict[str, Any]:
    """Apply the fixed, non-ranking promotion gates for one realised sample."""
    failures: list[str] = []
    if r.get("status") != "completed" or py.get("status") != "completed":
        return {"outcome": "backend_failure", "failures": ["R or Python backend did not complete"], "warnings": []}
    if check["status"] != "match":
        return {"outcome": "parity_failure", "failures": ["R/Python realised matrices differ"], "warnings": []}
    if r.get("instrument_count", groups) >= groups or r.get("instrument_rank") != r.get("instrument_count"):
        failures.append("instrument count/rank gate")
    if (r.get("sargan", {}).get("p_value") or -1) < .05:
        failures.append("R Sargan p < 0.05")
    if py.get("hansen_p") is None or py["hansen_p"] < .05:
        failures.append("Python Hansen p < 0.05")
    if py.get("ar2_p") is None or py["ar2_p"] < .05:
        failures.append("Python AR(2) p < 0.05")
    if totals.get("status") != "completed":
        failures.append("cluster shock total unavailable")
    warnings = [name for name, value in (("R Sargan p > 0.95", r.get("sargan", {}).get("p_value")), ("Python Hansen p > 0.95", py.get("hansen_p"))) if value is not None and value > .95]
    return {"outcome": "shortlist" if not failures else "diagnostic_failure", "failures": failures, "warnings": warnings}


def _clean_attempt(data: pd.DataFrame, specification_id: str, controls: tuple[str, ...], run_dir: Path,
                   sample_kind: str, comparison_family: str | None = None,
                   roles: dict[str, str] | None = None) -> dict[str, Any]:
    roles = roles or {name: "endogenous" for name in controls}
    attempt_id = specification_id if sample_kind == "ordinary" else f"{specification_id}__{comparison_family}"
    spec = specification(data, controls, roles, CLEAN_DIRECT, attempt_id, comparison_family)
    if spec["groups"] == 0 or spec["min_periods"] < 5:
        return {**{key: value for key, value in spec.items() if key != "input"}, "specification_id": specification_id, "sample_kind": sample_kind, "status": "invalid", "gate": {"outcome": "backend_failure", "failures": ["insufficient realised panel"], "warnings": []}}
    r, py = _run_r(spec, run_dir), _run_robust(spec, run_dir)
    check = parity(r, py, spec)
    totals = cluster_shock_totals(r)
    gate = diagnosis_gate(r, py, check, totals, spec["groups"])
    return {**{key: value for key, value in spec.items() if key != "input"}, "specification_id": specification_id, "sample_kind": sample_kind, "status": "completed" if gate["outcome"] == "shortlist" else "failed", "r": r, "pydynpd": py, "parity": check, "cluster_totals": totals, "gate": gate}


def _clean_ledger(attempts: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten the decision fields while retaining raw diagnostics in JSON."""
    rows: list[dict[str, Any]] = []
    for item in attempts:
        r, py, totals = item.get("r", {}), item.get("pydynpd", {}), item.get("cluster_totals", {})
        row = {"specification_id": item["specification_id"], "sample_kind": item["sample_kind"], "comparison_family": item.get("comparison_family"), "outcome": item["gate"]["outcome"], "failure_reason": "; ".join(item["gate"]["failures"]), "warning": "; ".join(item["gate"]["warnings"]), "controls": ", ".join(item["controls"]), "rows": item["rows"], "groups": item["groups"], "input_sha256": item["input_sha256"], "key_sha256": item["key_sha256"], "instrument_count": r.get("instrument_count"), "instrument_rank": r.get("instrument_rank"), "r_sargan_p": r.get("sargan", {}).get("p_value"), "python_hansen_p": py.get("hansen_p"), "python_ar1_p": py.get("ar1_p"), "python_ar2_p": py.get("ar2_p"), "parity": item.get("parity", {}).get("status")}
        for label in ("reference", "1", "3", "4"):
            row[f"shock_{label}"] = totals.get(label, {}).get("estimate")
            row[f"shock_{label}_robust_se"] = totals.get(label, {}).get("robust_se")
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_effects(attempts: list[dict[str, Any]]) -> pd.DataFrame:
    """State the gate change for each core comparison on its common sample."""
    rows: list[dict[str, Any]] = []
    for family in sorted({item.get("comparison_family") for item in attempts if item.get("comparison_family")}):
        pair = [item for item in attempts if item.get("comparison_family") == family]
        core = next(item for item in pair if item["specification_id"] == "core")
        alternative = next(item for item in pair if item["specification_id"] != "core")
        added_failures = sorted(set(alternative["gate"]["failures"]).difference(core["gate"]["failures"]))
        rows.append({"comparison_family": family, "alternative": alternative["specification_id"], "common_keys_equal": core["key_sha256"] == alternative["key_sha256"], "core_outcome": core["gate"]["outcome"], "alternative_outcome": alternative["gate"]["outcome"], "diagnostic_change": "; ".join(added_failures) or "no additional failed gate", "core_controls": ", ".join(core["controls"]), "alternative_controls": ", ".join(alternative["controls"])})
    return pd.DataFrame(rows)


def _sensitivity_effects(attempts: list[dict[str, Any]]) -> pd.DataFrame:
    """Report paired gate and shock-response changes without selecting a model."""
    effects = _paired_effects(attempts)
    for row in effects.itertuples():
        pair = [item for item in attempts if item.get("comparison_family") == row.comparison_family]
        core = next(item for item in pair if item["specification_id"] == "core")
        alternative = next(item for item in pair if item["specification_id"] != "core")
        for label in ("reference", "1", "3", "4"):
            effects.loc[row.Index, f"shock_{label}_change_from_core"] = alternative.get("cluster_totals", {}).get(label, {}).get("estimate", np.nan) - core.get("cluster_totals", {}).get(label, {}).get("estimate", np.nan)
    return effects


def _write_clean_bundle(run_dir: Path, manifest: dict[str, Any], panel: pd.DataFrame, audit: dict[str, Any], attempts: list[dict[str, Any]]) -> None:
    """Write the immutable clean-control diagnostic bundle."""
    for item in attempts:
        for backend in ("r", "pydynpd"):
            if backend in item:
                (run_dir / f"{item['attempt_id']}_{backend}_console.log").write_text(item[backend].get("raw_console", ""), encoding="utf8")
    ledger = _clean_ledger(attempts)
    paired_effects = _paired_effects(attempts)
    panel.to_csv(run_dir / "prepared_clean_level_panel.csv", index=False)
    (run_dir / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf8")
    (run_dir / "diagnostics.json").write_text(json.dumps({"manifest": manifest, "attempts": attempts}, ensure_ascii=False, indent=2, default=str), encoding="utf8")
    ledger.to_csv(run_dir / "diagnostic_ledger.csv", index=False)
    text = ["# ConsCred level-GMM control diagnosis", "", f"Outcome: **{manifest['status']}**", "", f"Selected control block: **{manifest.get('selected_control_block') or 'unresolved'}**", "", "| Specification | Sample | Outcome | Failed gate |", "|---|---|---|---|"]
    text += [f"| {row.specification_id} | {row.sample_kind} | {row.outcome} | {row.failure_reason or '—'} |" for row in ledger.itertuples()]
    text += ["", "## Paired diagnostic effects", "", "| Comparison | Alternative | Common keys | Diagnostic change |", "|---|---|---|---|"]
    text += [f"| {row.comparison_family} | {row.alternative} | {row.common_keys_equal} | {row.diagnostic_change} |" for row in paired_effects.itertuples()]
    (run_dir / "report.md").write_text("\n".join(text) + "\n", encoding="utf8")
    with pd.ExcelWriter(run_dir / "summary.xlsx") as writer:
        pd.DataFrame([manifest]).to_excel(writer, sheet_name="Summary", index=False)
        ledger.to_excel(writer, sheet_name="Diagnostic ledger", index=False)
        ledger.loc[ledger.sample_kind == "paired"].to_excel(writer, sheet_name="Paired comparisons", index=False)
        paired_effects.to_excel(writer, sheet_name="Paired effects", index=False)
        pd.DataFrame(audit.items(), columns=["field", "value"]).to_excel(writer, sheet_name="Data audit", index=False)
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf8")


def _write_federal_sensitivity_bundle(run_dir: Path, manifest: dict[str, Any], panel: pd.DataFrame,
                                      audit: dict[str, Any], attempts: list[dict[str, Any]]) -> None:
    """Write the immutable federal-sensitivity evidence bundle."""
    for item in attempts:
        for backend in ("r", "pydynpd"):
            if backend in item:
                (run_dir / f"{item['attempt_id']}_{backend}_console.log").write_text(item[backend].get("raw_console", ""), encoding="utf8")
    ledger = _clean_ledger(attempts)
    ledger["r_formula"] = [item.get("r_formula") for item in attempts]
    ledger["pydynpd_command"] = [item.get("pydynpd_command") for item in attempts]
    ledger["roles"] = [json.dumps(item.get("roles"), ensure_ascii=False) for item in attempts]
    ledger["direct_terms"] = [json.dumps(item.get("direct_terms"), ensure_ascii=False) for item in attempts]
    ledger["lag_window"] = [json.dumps(list(LAGS))] * len(ledger)
    ledger["instrument_labels"] = [json.dumps(item.get("instrument_labels"), ensure_ascii=False) for item in attempts]
    for field in ("workbook_sha256", "federal_workbook_sha256", "shock_sha256", "merged_panel_sha256"):
        ledger[field] = audit.get(field)
    for field in ("sensitivity_variable", "sensitivity_role", "sensitivity_scope", "monthly_change_observations", "distinct_monthly_changes"):
        ledger[field] = [item.get(field) for item in attempts]
    ledger["reported_outcome"] = ledger["outcome"].replace({"shortlist": "diagnostically_viable_sensitivity"})
    paired_effects = _sensitivity_effects(attempts)
    panel.to_csv(run_dir / "prepared_clean_federal_panel.csv", index=False)
    (run_dir / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf8")
    (run_dir / "diagnostics.json").write_text(json.dumps({"manifest": manifest, "attempts": attempts}, ensure_ascii=False, indent=2, default=str), encoding="utf8")
    ledger.to_csv(run_dir / "diagnostic_ledger.csv", index=False)
    paired_effects.to_csv(run_dir / "paired_sensitivity_effects.csv", index=False)
    text = ["# ConsCred Core federal sensitivity", "", f"Outcome: **{manifest['status']}**", "", "| Specification | Sample | Added variable | Outcome | Failed gate |", "|---|---|---|---|---|"]
    text += [f"| {row.specification_id} | {row.sample_kind} | {row.sensitivity_variable or '—'} | {row.reported_outcome} | {row.failure_reason or '—'} |" for row in ledger.itertuples()]
    text += ["", "## Paired Core changes", "", "| Comparison | Addition | Common keys | Reference shock change | Cluster 1 change | Cluster 3 change | Cluster 4 change |", "|---|---|---|---:|---:|---:|---:|"]
    text += [f"| {row.comparison_family} | {row.alternative} | {row.common_keys_equal} | {row.shock_reference_change_from_core:.6g} | {row.shock_1_change_from_core:.6g} | {row.shock_3_change_from_core:.6g} | {row.shock_4_change_from_core:.6g} |" for row in paired_effects.itertuples()]
    (run_dir / "report.md").write_text("\n".join(text) + "\n", encoding="utf8")
    with pd.ExcelWriter(run_dir / "summary.xlsx") as writer:
        pd.DataFrame([manifest]).to_excel(writer, sheet_name="Summary", index=False)
        ledger.to_excel(writer, sheet_name="Diagnostic ledger", index=False)
        paired_effects.to_excel(writer, sheet_name="Paired effects", index=False)
        pd.DataFrame(audit.items(), columns=["field", "value"]).to_excel(writer, sheet_name="Data audit", index=False)
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf8")


def run_clean_control_diagnosis(workbook: Path = CLEAN_WORKBOOK, shock: Path = SHOCK,
                                 results_root: Path = RESULTS_ROOT) -> tuple[str, Path]:
    """Run the fixed eight-specification cleaned level-rate diagnosis."""
    run_dir = results_root / f"level_cluster_control_diagnosis_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "r_backend.R").write_text(R_BACKEND, encoding="utf8")
    (run_dir / "pydynpd_backend.py").write_text(PY_BACKEND, encoding="utf8")
    panel, audit = load_clean_panel(workbook, shock)
    manifest: dict[str, Any] = {"status": "running", "created": datetime.now().isoformat(), "sources": {"workbook": str(workbook), "shock": str(shock)}, "level_contract": {"dependent": DEPENDENT, "direct_terms": list(CLEAN_DIRECT), "python_pre_differencing": False, "lag_window": list(LAGS)}, "fixed_specifications": {name: list(controls) for name, controls in CLEAN_SPECS.items()}}
    preflight = robust_preflight(run_dir)
    if preflight["status"] != "passed":
        manifest.update(status="backend_failure", selected_control_block=None, preflight=preflight)
        _write_clean_bundle(run_dir, manifest, panel, audit, [])
        return manifest["status"], run_dir
    attempts = [_clean_attempt(panel, name, controls, run_dir, "ordinary") for name, controls in CLEAN_SPECS.items()]
    for name, controls in CLEAN_SPECS.items():
        if name == "core":
            continue
        family = f"core_vs_{name}"
        common = paired_comparison_panel(panel, CLEAN_SPECS["core"], controls)
        attempts.extend((_clean_attempt(common, "core", CLEAN_SPECS["core"], run_dir, "paired", family), _clean_attempt(common, name, controls, run_dir, "paired", family)))
    eligible: list[str] = []
    for name, controls in CLEAN_SPECS.items():
        related = [item for item in attempts if item["specification_id"] == name]
        if related and all(item["gate"]["outcome"] == "shortlist" for item in related):
            eligible.append(name)
    scores = {name: (len(CLEAN_SPECS[name]), name != "core") for name in eligible}
    best = min(scores.values()) if scores else None
    finalists = [name for name, score in scores.items() if score == best]
    selected = finalists[0] if len(finalists) == 1 else None
    manifest.update(status="selected" if selected else "unresolved", selected_control_block=selected, eligible_control_blocks=eligible, selection_tie=finalists if len(finalists) > 1 else [], preflight=preflight)
    _write_clean_bundle(run_dir, manifest, panel, audit, attempts)
    return manifest["status"], run_dir


def run_clean_federal_sensitivity(workbook: Path = CLEAN_WORKBOOK, federal_workbook: Path = FEDERAL_WORKBOOK,
                                  shock: Path = SHOCK, results_root: Path = RESULTS_ROOT) -> tuple[str, Path]:
    """Run the fixed Core-plus-one federal sensitivity schedule."""
    run_dir = results_root / f"level_cluster_federal_sensitivity_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "r_backend.R").write_text(R_BACKEND, encoding="utf8")
    (run_dir / "pydynpd_backend.py").write_text(PY_BACKEND, encoding="utf8")
    panel, audit = load_clean_federal_panel(workbook, federal_workbook, shock)
    core = CLEAN_SPECS["core"]
    manifest: dict[str, Any] = {"status": "running", "created": datetime.now().isoformat(), "sources": {"workbook": str(workbook), "federal_workbook": str(federal_workbook), "shock": str(shock)}, "level_contract": {"dependent": DEPENDENT, "direct_terms": list(CLEAN_DIRECT), "python_pre_differencing": False, "lag_window": list(LAGS)}, "fixed_specifications": {name: list((*core, *addition)) for name, (addition, _) in FEDERAL_SENSITIVITY_SPECS.items()}}
    preflight = robust_preflight(run_dir)
    if preflight["status"] != "passed":
        manifest.update(status="backend_failure", preflight=preflight)
        _write_federal_sensitivity_bundle(run_dir, manifest, panel, audit, [])
        return manifest["status"], run_dir
    attempts = [_clean_attempt(panel, "core", core, run_dir, "ordinary")]
    for name, (addition, extra_roles) in FEDERAL_SENSITIVITY_SPECS.items():
        if name == "core":
            continue
        controls = (*core, *addition)
        roles = {control: "endogenous" for control in core} | extra_roles
        family = f"core_vs_{name}"
        common = paired_comparison_panel(panel, core, controls)
        paired_core = _clean_attempt(common, "core", core, run_dir, "paired", family)
        expanded = _clean_attempt(common, name, controls, run_dir, "paired", family, roles)
        if addition:
            variable = addition[0]
            expanded.update(sensitivity_variable=variable, sensitivity_role=roles[variable], sensitivity_scope="regional" if variable == "ln_Fin_Dostup" else "common_time_series_sensitivity", **audit["national_time_series"].get(variable, {}))
        attempts.extend((paired_core, expanded))
    manifest.update(status="completed", preflight=preflight)
    _write_federal_sensitivity_bundle(run_dir, manifest, panel, audit, attempts)
    return manifest["status"], run_dir


def initial_schedule(candidates: list[dict[str, str]]) -> list[tuple[str, ...]]:
    """Enumerate every required initial tuple, without a hidden size limit."""
    names = tuple(x["name"] for x in candidates); endogenous = tuple(x["name"] for x in candidates if x["role"] == "endogenous")
    return [(), *combinations(names, 1), *combinations(names, 2), *combinations(names, 3), *combinations(endogenous, 4)]


def descendants(attempts: Iterable[dict[str, Any]], candidates: list[dict[str, str]]) -> list[tuple[str, ...]]:
    """Breadth-first deterministic expansions from fully valid robust parents."""
    names = tuple(x["name"] for x in candidates); done = {tuple(a["controls"]) for a in attempts}; out: list[tuple[str, ...]] = []
    for item in attempts:
        controls = tuple(item["controls"])
        terminal = {"fit_failed", "instrument_cap_invalid", "robust_hansen_unresolved_instrument_mismatch"}
        if "robust_hansen_nonrejecting" in item.get("classification", []) and not terminal.intersection(item.get("classification", [])) and tuple(controls) in done:
            for name in names:
                candidate = (*controls, name)
                if name not in controls and (not controls or names.index(name) > names.index(controls[-1])) and candidate not in done and candidate not in out: out.append(candidate)
    return out


def _write_bundle(run_dir: Path, manifest: dict[str, Any], panel: pd.DataFrame, audit: dict[str, Any], inventory: pd.DataFrame, attempts: list[dict[str, Any]], legacy: list[dict[str, Any]], reproduction: list[dict[str, Any]]) -> None:
    """Persist all states, including a blocked robust backend, as immutable evidence."""
    for item in attempts:
        for backend in ("r", "pydynpd"):
            if backend in item: (run_dir / f"{item['attempt_id']}_{backend}_console.log").write_text(item[backend].get("raw_console", ""), encoding="utf8")
    clean = [{k: v for k, v in item.items() if k not in {"input", "r", "pydynpd", "parity"}} for item in attempts]
    panel.to_csv(run_dir / "prepared_level_panel.csv", index=False); inventory.to_csv(run_dir / "candidate_inventory.csv", index=False); pd.DataFrame(clean).to_csv(run_dir / "attempts.csv", index=False)
    (run_dir / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf8")
    detailed = [{k: v for k, v in item.items() if k != "input"} for item in attempts]
    (run_dir / "diagnostics.json").write_text(json.dumps({"manifest": manifest, "attempts": detailed, "legacy_immutable_reference": legacy}, ensure_ascii=False, indent=2, default=str), encoding="utf8")
    text = ["# ConsCred level-data robust Difference-GMM search", "", f"Status: **{manifest['status']}**", "", "R Sargan is classical and heteroskedasticity-sensitive. Robust Hansen labels require pydynpd and exact matrix parity.", "", "## Search multiplicity", "", f"Initial universe: {manifest.get('initial_universe', 0)} specifications. A searched non-rejection is not confirmatory.", "", "| ID | Controls | Labels |", "|---|---|---|"]
    text += [f"| {x['attempt_id']} | {', '.join(x['controls'])} | {', '.join(x.get('classification', []))} |" for x in clean]
    (run_dir / "report.md").write_text("\n".join(text) + "\n", encoding="utf8")
    with pd.ExcelWriter(run_dir / "summary.xlsx") as writer:
        pd.DataFrame({"metric": ["status", "attempts", "initial_universe"], "value": [manifest["status"], len(clean), manifest.get("initial_universe", 0)]}).to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame(clean).to_excel(writer, sheet_name="All attempts", index=False); inventory.to_excel(writer, sheet_name="Candidate inventory", index=False)
        pd.DataFrame([{**x, "pydynpd": x.get("pydynpd", {})} for x in attempts]).to_excel(writer, sheet_name="Robust Hansen", index=False)
        pd.DataFrame(legacy).to_excel(writer, sheet_name="Legacy reference", index=False); pd.DataFrame(reproduction).to_excel(writer, sheet_name="R reproduction", index=False); pd.DataFrame(audit.items(), columns=["field", "value"]).to_excel(writer, sheet_name="Data audit", index=False)
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf8")


def legacy_reference() -> list[dict[str, Any]]:
    """Load historical Python output only; it is never reclassified as current evidence."""
    report = json.loads(LEGACY_DIAGNOSTICS.read_text(encoding="utf8"))
    return [{"label": "legacy_immutable_reference", "attempt_id": x.get("attempt_id"), "status": x.get("status"), "hansen_p": x.get("hansen_p"), "source": str(LEGACY_DIAGNOSTICS)} for x in report.get("attempts", [])]


def legacy_r_reproduction(run_dir: Path) -> list[dict[str, Any]]:
    """Run the separate pre-differenced 75-region R reference matrix."""
    from Modules.gmm_utils import DIRECT as legacy_direct

    expected = json.loads(LEGACY_DIAGNOSTICS.read_text(encoding="utf8"))
    panel = pd.read_csv(LEGACY_PANEL); panel["Date"] = pd.to_datetime(panel["Date"])
    source_sha = sha256(LEGACY_PANEL)
    expected_sha = expected.get("input", {}).get("sha256")
    if source_sha != expected_sha:
        return [{"label": "legacy_r_reproduction_pre_differenced_75_region", "status": "blocked", "error": "legacy panel SHA-256 mismatch", "source_sha256": source_sha, "expected_sha256": expected_sha}]
    results: list[dict[str, Any]] = []
    for size in range(len(LEGACY_CONTROLS) + 1):
        for controls in combinations(LEGACY_CONTROLS, size):
            direct = tuple(legacy_direct); equation = ("lag(d_Int_Rate_ConsCred, 1)", *controls, *direct)
            gmm = ("lag(d_Int_Rate_ConsCred, 2:3)", *(f"lag({x}, 1:2)" for x in controls))
            spec = {"attempt_id": "legacy_r__" + ("__".join(controls) or "none"), "input": panel, "r_formula": f"d_Int_Rate_ConsCred ~ {' + '.join(equation)} | {' + '.join(gmm)} | {' + '.join(direct)}"}
            fitted = _run_r(spec, run_dir)
            (run_dir / f"{spec['attempt_id']}_r_console.log").write_text(fitted.get("raw_console", ""), encoding="utf8")
            results.append({"label": "legacy_r_reproduction_pre_differenced_75_region", "controls": list(controls), "source_sha256": source_sha, "window": "2024-01 through 2025-12", "status": fitted.get("status"), "formula": fitted.get("formula"), "instrument_rank": fitted.get("instrument_rank"), "ar2": fitted.get("ar2"), "sargan": fitted.get("sargan"), "note": "R reproduction only; no robust-Hansen equality claim."})
    return results


def postfit_diagnostics(item: dict[str, Any], panel: pd.DataFrame, roles: dict[str, str], run_dir: Path) -> dict[str, Any]:
    """Record predeclared stability and dependence evidence for a robust pass."""
    share = item["r"].get("residual_common_factor_share")
    out: dict[str, Any] = {"common_factor_share": share, "common_shock_dependence_unresolved": bool(share is not None and share >= .5), "stability_only_not_confirmatory": True}
    early = panel.loc[panel.Date <= pd.Timestamp("2024-12-01")]
    spec = specification(early, tuple(item["controls"]), roles)
    r, py = _run_r(spec, run_dir), _run_robust(spec, run_dir)
    out["ending_2024_12"] = {"r_status": r.get("status"), "hansen_p": py.get("hansen_p"), "ar2_p": py.get("ar2_p"), "parity": parity(r, py, spec).get("status")}
    # National-common terms are collinear with month effects; this is only an
    # identification sensitivity, never a replacement specification.
    regional_controls = tuple(x for x in item["controls"] if roles[x] == "endogenous" and x in REGIONAL)
    sensitivity = specification(panel, regional_controls, roles)
    out["month_effects_sensitivity"] = {"status": "not_estimated", "controls": list(regional_controls), "label": "identification_sensitivity_not_replacement", "reason": "national-common regressors excluded as collinear with month effects"}
    return out


def run_search(workbook: Path = WORKBOOK, shock: Path = SHOCK, results_root: Path = RESULTS_ROOT, resume: Path | None = None) -> tuple[str, Path]:
    """Run the resumable initial level search, stopping safely if robust validation fails."""
    run_dir = resume or results_root / f"level_robust_expanded_search_{datetime.now():%Y%m%d_%H%M%S}"
    if resume is None: run_dir.mkdir(parents=True, exist_ok=False)
    elif not (run_dir / "diagnostics.json").is_file(): raise ValueError("Resume bundle must contain diagnostics.json")
    (run_dir / "r_backend.R").write_text(R_BACKEND, encoding="utf8"); (run_dir / "pydynpd_backend.py").write_text(PY_BACKEND, encoding="utf8")
    panel, audit = load_panel(workbook, shock); inventory, candidates = candidate_inventory(panel); legacy = legacy_reference(); reproduction = legacy_r_reproduction(run_dir); manifest: dict[str, Any] = {"status": "running", "created": datetime.now().isoformat(), "sources": {"workbook": str(workbook), "shock": str(shock)}, "initial_universe": len(initial_schedule(candidates)), "level_contract": {"dependent": DEPENDENT, "direct": list(DIRECT), "python_pre_differencing": False}}
    preflight = robust_preflight(run_dir)
    if preflight["status"] != "passed":
        manifest["status"] = "robust_backend_blocked"; _write_bundle(run_dir, manifest, panel, audit, inventory, [], legacy, reproduction); return manifest["status"], run_dir
    roles = {x["name"]: x["role"] for x in candidates}; prior = json.loads((run_dir / "diagnostics.json").read_text(encoding="utf8")).get("attempts", []) if resume else []; attempts: list[dict[str, Any]] = prior
    done = {tuple(x["controls"]) for x in attempts}; pending = [x for x in initial_schedule(candidates) if x not in done]
    while pending:
        controls = pending.pop(0); spec = specification(panel, controls, roles)
        if spec["groups"] == 0 or spec["min_periods"] < 5:
            attempts.append({**{k: v for k, v in spec.items() if k != "input"}, "status": "invalid", "classification": ["fit_failed", "terminal_no_usable_panel"]}); continue
        r, py = _run_r(spec, run_dir), _run_robust(spec, run_dir); check = parity(r, py, spec); labels = classify(r, py, check)
        item = {**{k: v for k, v in spec.items() if k != "input"}, "status": "completed" if r.get("status") == py.get("status") == "completed" else "failed", "classification": labels, "r": r, "pydynpd": py, "parity": check}
        if "robust_hansen_nonrejecting" in labels: item["postfit"] = postfit_diagnostics(item, panel, roles, run_dir)
        attempts.append(item)
        pending.extend(x for x in descendants(attempts, candidates) if x not in pending)
    manifest["status"] = "partial" if any("fit_failed" in x.get("classification", []) for x in attempts) else "completed"; _write_bundle(run_dir, manifest, panel, audit, inventory, attempts, legacy, reproduction); return manifest["status"], run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, default=WORKBOOK)
    parser.add_argument("--shock", type=Path, default=SHOCK)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--clean-control-diagnosis", action="store_true")
    parser.add_argument("--clean-workbook", type=Path, default=CLEAN_WORKBOOK)
    parser.add_argument("--federal-workbook", type=Path, default=FEDERAL_WORKBOOK)
    parser.add_argument("--clean-federal-sensitivity", action="store_true")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    args = parser.parse_args(argv)
    if args.clean_federal_sensitivity:
        status, bundle = run_clean_federal_sensitivity(args.clean_workbook, args.federal_workbook, args.shock, args.results_root)
        print(f"status={status} bundle={bundle}")
        return 0 if status == "completed" else 2
    if args.clean_control_diagnosis:
        status, bundle = run_clean_control_diagnosis(args.clean_workbook, args.shock, args.results_root)
        print(f"status={status} bundle={bundle}")
        return 0 if status in {"selected", "unresolved"} else 2
    status, bundle = run_search(args.workbook, args.shock, args.results_root, args.resume)
    print(f"status={status} bundle={bundle}")
    return 0 if status == "completed" else 2


if __name__ == "__main__": raise SystemExit(main())
