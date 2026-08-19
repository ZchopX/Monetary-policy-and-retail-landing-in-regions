import pandas as pd
import pytest
from pathlib import Path
import shutil

from Modules.gmm_utils import _systemgmmkit_instrument_check, _validate_preflight_panel, build_pydynpd_command, profile_contract
from Modules import gmm_panel_contract
from Modules.gmm_panel_contract import CLUSTERS, REQUIRED, run_contract, verify_manifest


def test_profile_contract_keeps_one_level_difference_lag() -> None:
    contract = profile_contract("A", "system")
    assert contract["lag_window"] == [2, 3]
    assert contract["levels_gmm_lag"] == 1
    assert profile_contract("C", "difference")["gmm_roles"]["Cred_nagr_lag1"] == "endogenous"


def test_pydynpd_command_uses_its_lagged_dependent_notation() -> None:
    command = build_pydynpd_command("A", "one_step")
    assert "L1.d_Int_Rate_ConsCred" in command
    assert "gmm(d_Int_Rate_ConsCred Cred_nagr_lag1 Zakred_lag1 Cap_to_assets_lag1 CPI_reg_lag1, 2:3)" in command
    assert command.endswith("collapse onestep")
    assert build_pydynpd_command("A", "two_step", "difference").endswith("collapse nolevel")


def test_systemgmmkit_names_reject_extra_level_lag() -> None:
    contract = profile_contract("A", "system")
    names = ["L:diff:L1_d_Int_Rate_ConsCred:L2"]
    result = _systemgmmkit_instrument_check(contract, names)
    assert result["status"] == "mismatch"
    assert "L:diff:L1_d_Int_Rate_ConsCred:L2" in result["unexpected"]


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
