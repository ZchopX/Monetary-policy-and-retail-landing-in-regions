import pandas as pd
import pytest
import numpy as np
import sys
import types
from pathlib import Path
import shutil

from Modules import gmm_utils
from Modules.gmm_utils import DIRECT, PREDTERMINED, _benchmark_cd, _control_subset_gate, _control_subsets, _delayed_control_lag_gate, _instrument_validity_gate, _pydynpd_attempt, _systemgmmkit_instrument_check, _validate_preflight_panel, build_pydynpd_command, prepare_quarterly_instrument_validity_data, profile_contract
from Modules import gmm_panel_contract
from Modules.gmm_panel_contract import CLUSTERS, REQUIRED, run_contract, verify_manifest


def test_profile_contract_keeps_one_level_difference_lag() -> None:
    contract = profile_contract("A", "system")
    assert contract["lag_window"] == [2, 3]
    assert contract["gmm_lag_windows"]["L1_d_Int_Rate_ConsCred"] == [2, 3]
    assert contract["gmm_lag_windows"]["Cred_nagr_lag1"] == [1, 2]
    assert contract["levels_gmm_lag"] == 1
    assert profile_contract("C", "difference")["gmm_roles"]["Cred_nagr_lag1"] == "endogenous"


def test_pydynpd_command_uses_its_lagged_dependent_notation() -> None:
    command = build_pydynpd_command("A", "one_step")
    assert "L1.d_Int_Rate_ConsCred" in command
    assert "gmm(d_Int_Rate_ConsCred, 2:3)" in command
    assert "gmm(Cred_nagr_lag1 Zakred_lag1 Cap_to_assets_lag1 CPI_reg_lag1, 1:2)" in command
    assert command.endswith("collapse onestep")
    assert build_pydynpd_command("A", "two_step", "difference").endswith("collapse nolevel")
    assert "gmm(Cred_nagr_lag1 Zakred_lag1 Cap_to_assets_lag1 CPI_reg_lag1, 1:3)" in build_pydynpd_command("B", "two_step", "difference")
    assert "gmm(Cred_nagr_lag1 Zakred_lag1 Cap_to_assets_lag1 CPI_reg_lag1, 2:3)" in build_pydynpd_command("C", "two_step", "difference")


def test_pydynpd_command_supports_each_canonical_control_subset() -> None:
    for retained_count, expected_instruments in zip(range(4, -1, -1), (20, 18, 16, 14, 12), strict=True):
        retained = PREDTERMINED[:retained_count]
        command = build_pydynpd_command("A", "two_step", "difference", retained)
        assert command.endswith("collapse nolevel")
        assert "gmm(, " not in command
        assert expected_instruments == 12 + 2 * retained_count
    assert "gmm(Cred_nagr_lag1, 1:2)" in build_pydynpd_command("A", "two_step", "difference", PREDTERMINED[:1])
    assert "gmm(Cred_nagr_lag1" not in build_pydynpd_command("A", "two_step", "difference", ())
    with pytest.raises(ValueError, match="ordered, duplicate-free"):
        build_pydynpd_command("A", "two_step", "difference", (PREDTERMINED[1], PREDTERMINED[0]))
    with pytest.raises(ValueError, match="ordered, duplicate-free"):
        build_pydynpd_command("A", "two_step", "difference", (PREDTERMINED[0], PREDTERMINED[0]))


def test_pydynpd_command_supports_only_valid_delayed_control_windows() -> None:
    for window in ((1, 2), (2, 3), (3, 4)):
        command = build_pydynpd_command("A", "two_step", "difference", control_window=window)
        assert f"gmm(Cred_nagr_lag1 Zakred_lag1 Cap_to_assets_lag1 CPI_reg_lag1, {window[0]}:{window[1]})" in command
        assert "gmm(d_Int_Rate_ConsCred, 2:3)" in command
        assert 20 == 2 + 2 * len(PREDTERMINED) + 10
        assert 5 == 20 - (1 + len(PREDTERMINED) + 10)
    for window in ((0, 1), (2, 2), (1, 3), (3, 2), (1.0, 2), [1, 2]):
        with pytest.raises(ValueError, match="two consecutive positive integer"):
            build_pydynpd_command("A", "two_step", "difference", control_window=window)  # type: ignore[arg-type]


def test_instrument_validity_commands_keep_direct_and_control_variants_separate() -> None:
    omitted = build_pydynpd_command("A", "two_step", "difference", direct_terms=DIRECT[:-1])
    assert DIRECT[-1] not in omitted
    windows = dict(zip(PREDTERMINED, ((1, 2), (2, 3), (1, 2), (2, 3)), strict=True))
    command = build_pydynpd_command("A", "two_step", "difference", control_windows=windows)
    assert "gmm(Cred_nagr_lag1, 1:2)" in command
    assert "gmm(Zakred_lag1, 2:3)" in command
    assert command.count("gmm(") == 5


def test_instrument_validity_gate_never_accepts_a_model() -> None:
    attempt = {"status": "completed", "instrument_count": 20, "instrument_rank": 20, "hansen_df": 5,
               "hansen_status": "available", "ar2_status": "passed", "sample_comparability_status": "match", "hansen_p": 0.2}
    assert _instrument_validity_gate(attempt, 20, 5) == "robust_J_nonrejection_not_accepted"
    attempt["instrument_rank"] = 19
    assert _instrument_validity_gate(attempt, 20, 5) == "unresolved"


def test_benchmark_cd_reports_residual_source() -> None:
    rows = []
    for region in ("a", "b", "c"):
        for month in range(8):
            row = {"Region": region, "Date": pd.Timestamp("2024-01-01") + pd.offsets.MonthBegin(month),
                   "d_Int_Rate_ConsCred": float(month), "d_Int_Rate_ConsCred_lag1": float(month - 1)}
            row.update({name: float(month + offset) for offset, name in enumerate(PREDTERMINED)})
            row.update({name: float(month * (offset + 1) + (region == "c")) for offset, name in enumerate(DIRECT)})
            rows.append(row)
    result = _benchmark_cd(pd.DataFrame(rows), False)
    assert result["residual_source"] == "entity-FE OLS benchmark"
    assert result["residual_coverage"] == {"dates": 8, "regions": 3, "rows": 24}
    assert _benchmark_cd(pd.DataFrame(rows), True)["excluded_time_only_terms"] == ["d_Inflation_Expectations", "d_Mon_Shock_pos", "d_Mon_Shock_neg"]
    augmented = _benchmark_cd(pd.DataFrame(rows), False, DIRECT)
    assert augmented["regressors"][-1] == DIRECT[-1]


def test_historical_regime_transition_audit_keeps_onsets_and_exits() -> None:
    rows = []
    for region in ("a", "b"):
        for date, covid, sank in zip(pd.date_range("2020-01-01", periods=4, freq="MS"), (0, 1, 1, 0), (0, 0, 1, 1), strict=True):
            rows.append({"Region": region, "Date": date, "Covid_dum": covid, "Sank_dum": sank})
    audit = gmm_utils._regime_transition_audit(pd.DataFrame(rows))
    assert audit["Covid_dum"]["differenced_values"] == {"2020-02-01": 1, "2020-04-01": -1}
    assert audit["Covid_dum"]["observed_exit_transition"] is True
    assert audit["Sank_dum"]["differenced_nonzero_dates"] == ["2020-03-01"]
    assert audit["Sank_dum"]["observed_exit_transition"] is False


def test_historical_regime_runner_has_exactly_two_fixed_attempts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = pd.DataFrame({"Region": ["R"] * 2, "Date": pd.date_range("2020-01-01", periods=2, freq="MS"),
                         "Covid_dum": [0, 1], "Sank_dum": [0, 1],
                         **{name: [1.0, 2.0] for name in (gmm_utils.DEPENDENT, f"{gmm_utils.DEPENDENT}_lag1", *PREDTERMINED, *DIRECT)}})
    monkeypatch.setattr(gmm_utils, "load_historical_regime_data", lambda _: (data, {"observed_sha256": "locked"}))
    monkeypatch.setattr(gmm_utils, "version", lambda _: "0.2.2")
    monkeypatch.setattr(gmm_utils, "_benchmark_cd", lambda data, time_effects, direct_terms: {"regressors": list(direct_terms), "pesaran_cd_statistic": 0.0, "pesaran_cd_two_sided_p": 1.0, "residual_coverage": {}})

    def attempt(data: pd.DataFrame, profile: str, estimator: str, stage: str, direct_terms: tuple[str, ...] = DIRECT, **_: object) -> dict[str, object]:
        count = len(direct_terms) + 10
        return {"status": "completed", "raw_console": "ok", "direct_terms": list(direct_terms), "instrument_count": count,
                "instrument_rank": count, "hansen_df": 5, "hansen_status": "available", "hansen_p": 0.2,
                "ar2_status": "passed", "ar2_p": 0.2, "groups": 75, "nobs": 5925, "usable_observations": {"package_nobs": 5925}}

    monkeypatch.setattr(gmm_utils, "_pydynpd_attempt", attempt)
    report, markdown_path, json_path = gmm_utils.run_historical_regime_diagnosis(Path("parent.csv"), tmp_path)
    assert report["decision"] == "diagnostic_complete_no_model_accepted"
    assert [item["attempt_id"] for item in report["attempts"]] == ["historical_baseline", "historical_covid_sank_main_effects_auxiliary"]
    assert [item["instrument_count"] for item in report["attempts"]] == [20, 22]
    assert all(not ("Cluster" in term and ("Covid_dum" in term or "Sank_dum" in term)) for term in report["attempts"][1]["direct_terms"])
    assert markdown_path.exists() and json_path.exists()


def test_quarterly_preparer_uses_quarter_ids() -> None:
    panel, metadata = prepare_quarterly_instrument_validity_data(Path("Квартальные данные.xlsx"))
    assert panel.shape == (2295, 7)
    assert panel.Date.min() == "2019Q2"
    assert panel.Date.max() == "2025Q4"
    assert metadata["date_timestamp_semantics"] == "unresolved"


def test_delayed_control_lag_gate_requires_ar2_and_comparable_sample() -> None:
    attempt = {"status": "completed", "instrument_check": {"status": "match"}, "rank_status": "valid", "hansen_status": "available", "hansen_p": 0.2,
               "ar2_status": "passed", "sample_comparability_status": "match"}
    assert _delayed_control_lag_gate(attempt) == "robust_J_nonrejection_not_accepted"
    attempt["ar2_status"] = "unavailable"
    assert _delayed_control_lag_gate(attempt) == "unresolved"
    attempt["ar2_status"] = "passed"
    attempt["sample_comparability_status"] = "unresolved"
    assert _delayed_control_lag_gate(attempt) == "unresolved"


def test_delayed_control_lag_runner_persists_timing_and_sample_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gmm_utils, "load_locked_preflight_data", lambda *args: (pd.DataFrame({"Region": ["R"]}), "locked"))

    def attempt(data: pd.DataFrame, profile: str, estimator: str, stage: str, retained_controls: tuple[str, ...] = PREDTERMINED,
                control_window: tuple[int, int] | None = None) -> dict[str, object]:
        nobs = 1575 if control_window == (3, 4) else 1650
        return {"status": "completed", "command": "fixed", "raw_console": "console", "control_window": list(control_window or ()),
                "nobs": nobs, "groups": 75, "usable_observations": {"package_nobs": nobs}, "instrument_count": 20,
                "instrument_rank": 20, "rank_status": "valid", "instrument_check": {"status": "match"}, "hansen_status": "available",
                "hansen_p": 0.2, "hansen_df": 5, "ar2_p": 0.3, "ar2_status": "passed", "local_source_sha256": "a" * 64,
                "hansen_robustness_evidence": {"test_source_sha256": "b" * 64, "weighting_source_sha256": "c" * 64}}

    monkeypatch.setattr(gmm_utils, "_pydynpd_attempt", attempt)
    report, markdown_path, json_path = gmm_utils.run_delayed_control_lag_diagnosis(results_dir=tmp_path)
    assert [item["attempt_id"] for item in report["attempts"]] == ["controls_1_2", "controls_2_3", "controls_3_4"]
    assert [item["raw_control_timing"] for item in report["attempts"]] == [[2, 3], [3, 4], [4, 5]]
    assert report["attempts"][-1]["sample_comparability_status"] == "unresolved"
    assert report["attempts"][-1]["diagnostic_gate"] == "unresolved"
    assert json_path.exists() and markdown_path.exists()
    assert len(list(tmp_path.rglob("*_raw_console.log"))) == 3


def test_control_subsets_are_the_complete_deterministic_matrix() -> None:
    subsets = _control_subsets()
    assert len(subsets) == len(set(subsets)) == 16
    assert subsets[0] == PREDTERMINED
    assert subsets[-1] == ()


def test_pydynpd_attempt_preserves_partial_output_and_rank_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = types.ModuleType("systemgmmkit.pydynpd_backend")
    backend._apply_numpy_compatibility_shims = lambda: None
    systemgmmkit = types.ModuleType("systemgmmkit")
    systemgmmkit.__path__ = []
    systemgmmkit.pydynpd_backend = backend
    regression = types.ModuleType("pydynpd.regression")
    specification_tests = types.ModuleType("pydynpd.specification_tests")

    def hansen_overid() -> None:
        return None

    def weighting() -> None:
        return None

    specification_tests.hansen_overid = hansen_overid

    def abond(command: str, data: pd.DataFrame, index: list[str]) -> object:
        print("partial output")
        raise RuntimeError("fit failed")

    abond.GMM = weighting
    regression.abond = abond
    pydynpd = types.ModuleType("pydynpd")
    pydynpd.__path__ = []
    pydynpd.regression = regression
    pydynpd.specification_tests = specification_tests
    for name, module in {"systemgmmkit": systemgmmkit, "systemgmmkit.pydynpd_backend": backend, "pydynpd": pydynpd,
                         "pydynpd.regression": regression, "pydynpd.specification_tests": specification_tests}.items():
        monkeypatch.setitem(sys.modules, name, module)
    failed = _pydynpd_attempt(pd.DataFrame(), "A", "difference", "two_step")
    assert failed["status"] == "failed"
    assert "partial output" in failed["raw_console"]

    model = types.SimpleNamespace(
        z_information=types.SimpleNamespace(num_Dgmm_instr=10, num_Lgmm_instr=0, num_instr=20, level_height=0),
        AR_list=[], hansen=types.SimpleNamespace(test_value=17.7, df=5, p_value=0.003), N=75, T=24,
        z_list=np.tile(np.pad(np.eye(20), ((0, 0), (0, 2))), (75, 1)), step_results=[types.SimpleNamespace(vcov=np.eye(1))],
    )
    regression.abond = lambda command, data, index: types.SimpleNamespace(models=[model])
    regression.abond.GMM = weighting
    monkeypatch.setattr(gmm_utils, "version", lambda _: "0.2.2")
    completed = _pydynpd_attempt(pd.DataFrame(), "A", "difference", "two_step")
    assert completed["hansen_statistic"] == 17.7
    assert completed["hansen_df"] == 5
    assert completed["instrument_rank"] == 20
    assert completed["rank_status"] == "valid"
    assert completed["hansen_status"] == "available"
    assert completed["ar2_status"] == "unavailable"
    assert len(completed["local_source_sha256"]) == 64
    model.z_list = np.ones((20 * 75, 22))
    rank_invalid = _pydynpd_attempt(pd.DataFrame(), "A", "difference", "two_step")
    assert rank_invalid["rank_status"] == "invalid"
    assert _control_subset_gate(rank_invalid) == "unresolved"


def test_profile_contract_translates_approved_control_timing() -> None:
    assert profile_contract("B", "difference")["gmm_lag_windows"]["Cred_nagr_lag1"] == [1, 3]
    assert profile_contract("C", "difference")["gmm_lag_windows"]["Cred_nagr_lag1"] == [2, 3]


def test_systemgmmkit_names_reject_extra_level_lag() -> None:
    contract = profile_contract("A", "system")
    names = ["L:diff:L1_d_Int_Rate_ConsCred:L2"]
    result = _systemgmmkit_instrument_check(contract, names)
    assert result["status"] == "mismatch"
    assert "L:diff:L1_d_Int_Rate_ConsCred:L2" in result["unexpected"]


def test_systemgmmkit_system_layout_is_not_pgmm_matrix_parity() -> None:
    contract = profile_contract("A", "system")
    names = [
        "D:L1_d_Int_Rate_ConsCred:L2", "D:L1_d_Int_Rate_ConsCred:L3",
        "D:Cred_nagr_lag1:L1", "D:Cred_nagr_lag1:L2", "D:Zakred_lag1:L1", "D:Zakred_lag1:L2",
        "D:Cap_to_assets_lag1:L1", "D:Cap_to_assets_lag1:L2", "D:CPI_reg_lag1:L1", "D:CPI_reg_lag1:L2",
        *[f"IV:{name}" for name in contract["direct_regressors"]], "L:diff:L1_d_Int_Rate_ConsCred:L1",
        "L:diff:Cred_nagr_lag1:L0", "L:diff:Zakred_lag1:L0", "L:diff:Cap_to_assets_lag1:L0", "L:diff:CPI_reg_lag1:L0", "L:constant",
    ]
    result = _systemgmmkit_instrument_check(contract, names, 26)
    assert result["status"] == "different_system_convention"
    assert result["expected_count"] == 26


def test_unbalanced_panel_is_rejected() -> None:
    data = pd.DataFrame({"Region": ["a", "a", "b"], "Date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-01"]),
                         "Cluster_new_cd_1": [0, 0, 0], "Cluster_new_cd_3": [0, 0, 0], "Cluster_new_cd_4": [0, 0, 0]})
    with pytest.raises(ValueError, match="balanced 75-region"):
        _validate_preflight_panel(data)


@pytest.fixture(scope="session")
def panel_template(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("panel-template")
    dates = pd.date_range("2019-03-01", "2025-12-01", freq="MS")
    rows = []
    mapping = []
    for number in range(2):
        region = f"R{number:02d}"
        cluster = [int(number % 4 == value) for value in (1, 2, 3)]
        mapping.append({"Region": region, **dict(zip(CLUSTERS, cluster, strict=True))})
        for index, date in enumerate(dates):
            shock = float((index % 5) - 2)
            rows.append({"Region": region, "Date": date, "d_Int_Rate_ConsCred": float(index), "Cred_nagr": number + index,
                         "Zakred": number + index + 1, "Cap_to_assets": number + index + 2, "CPI_reg": number + index + 3,
                         "ln_Fin_Dostup": float(number), "d_Inflation_Expectations": None if index == 0 else float(index),
                         "d_Mon_Shock": shock, "d_Mon_Shock_pos": max(shock, 0), "d_Mon_Shock_neg": min(shock, 0), **dict(zip(CLUSTERS, cluster, strict=True))})
    workbook, cluster_map = root / "panel.xlsx", root / "clusters.pkl"
    pd.DataFrame(rows).to_excel(workbook, index=False)
    pd.DataFrame(mapping).to_pickle(cluster_map)
    return workbook, cluster_map


@pytest.fixture
def panel_sources(panel_template: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    monkeypatch.setattr(gmm_panel_contract, "EXPECTED_REGION_COUNT", 2)
    workbook, cluster_map = (tmp_path / "panel.xlsx", tmp_path / "clusters.pkl")
    shutil.copy(panel_template[0], workbook)
    shutil.copy(panel_template[1], cluster_map)
    return workbook, cluster_map, tmp_path / "results"


def test_panel_manifest_horizons_and_checksums(panel_sources: tuple[Path, Path, Path]) -> None:
    manifest, run_dir = run_contract(*panel_sources, run_id="happy")
    assert manifest["status"] == "passed"
    assert [manifest["horizons"][str(h)]["eligible_rows"] for h in range(7)] == [162, 162, 160, 158, 156, 154, 152]
    assert all(manifest["horizons"][str(h)]["validation_rows"] == 48 for h in range(7))
    assert all(manifest["horizons"][str(h)]["balance_status"] == "balanced" for h in range(7))
    assert manifest["schema"]["prepared_panel.csv"][:2] == ["Region", "Date"]
    assert manifest["variable_roles"]["dynamic_regressor"] == "d_Int_Rate_ConsCred_lag1"
    assert verify_manifest(run_dir)
    assert (run_dir / "prepared_panel.csv").exists()


def test_panel_retains_unbalanced_value_gap(panel_sources: tuple[Path, Path, Path]) -> None:
    workbook, cluster_map, root = panel_sources
    data = pd.read_excel(workbook)
    data.loc[(data.Region == "R00") & (data.Date == pd.Timestamp("2023-01-01")), "ln_Fin_Dostup"] = None
    data.to_excel(workbook, index=False)
    manifest, _ = run_contract(workbook, cluster_map, root, "gap")
    assert manifest["status"] == "passed"
    assert manifest["horizons"]["0"]["balance_status"] == "unbalanced"
    assert manifest["horizons"]["0"]["eligible_rows"] == 161


@pytest.mark.parametrize("column, changed_date", [("d_Inflation_Expectations", "2023-01-01"), ("Cred_nagr", "2023-01-01")])
def test_panel_marks_later_expected_column_gaps_unbalanced(panel_sources: tuple[Path, Path, Path], column: str, changed_date: str) -> None:
    workbook, cluster_map, root = panel_sources
    data = pd.read_excel(workbook)
    data.loc[(data.Region == "R00") & (data.Date == pd.Timestamp(changed_date)), column] = None
    data.to_excel(workbook, index=False)
    manifest, _ = run_contract(workbook, cluster_map, root, f"late-{column}")
    assert manifest["horizons"]["0"]["balance_status"] == "unbalanced"


@pytest.mark.parametrize("change, message", [("duplicate", "Duplicate"), ("missing", "Missing required"), ("shock", "reconstruct"), ("calendar", "Incomplete or non-consecutive")])
def test_panel_failures_retain_manifest(panel_sources: tuple[Path, Path, Path], change: str, message: str) -> None:
    workbook, cluster_map, root = panel_sources
    data = pd.read_excel(workbook)
    if change == "duplicate":
        data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    elif change == "missing":
        data = data.drop(columns=[REQUIRED[0]])
    elif change == "calendar":
        data = data.drop(data.index[(data.Region == "R00") & (data.Date == pd.Timestamp("2025-12-01"))])
    else:
        data.loc[0, "d_Mon_Shock"] = 99
    data.to_excel(workbook, index=False)
    manifest, run_dir = run_contract(workbook, cluster_map, root, change)
    assert manifest["status"] == "failed"
    assert message in manifest["error"]
    assert (run_dir / "specification_manifest.json").exists()
    assert not (run_dir / "prepared_panel.csv").exists()


@pytest.mark.parametrize("change, message", [("mismatch", "differs"), ("encoding", "binary")])
def test_panel_cluster_contract_failures(panel_sources: tuple[Path, Path, Path], change: str, message: str) -> None:
    workbook, cluster_map, root = panel_sources
    mapping = pd.read_pickle(cluster_map)
    if change == "mismatch":
        mapping.loc[0, CLUSTERS[0]] = 1 - mapping.loc[0, CLUSTERS[0]]
    else:
        mapping.loc[0, CLUSTERS[0]] = 2
    mapping.to_pickle(cluster_map)
    manifest, run_dir = run_contract(workbook, cluster_map, root, change)
    assert manifest["status"] == "failed"
    assert message in manifest["error"]
    assert (run_dir / "raw_panel_audit.json").exists()


def test_panel_unclean_validation_is_unresolved(panel_sources: tuple[Path, Path, Path]) -> None:
    workbook, cluster_map, root = panel_sources
    data = pd.read_excel(workbook)
    data.loc[(data.Region == "R00") & (data.Date == pd.Timestamp("2025-01-01")), "ln_Fin_Dostup"] = None
    data.to_excel(workbook, index=False)
    manifest, run_dir = run_contract(workbook, cluster_map, root, "unclean")
    assert manifest["status"] == "unresolved"
    assert not (run_dir / "prepared_panel.csv").exists()


def test_manifest_refuses_overwrite(panel_sources: tuple[Path, Path, Path]) -> None:
    run_contract(*panel_sources, run_id="existing")
    with pytest.raises(FileExistsError):
        run_contract(*panel_sources, run_id="existing")
