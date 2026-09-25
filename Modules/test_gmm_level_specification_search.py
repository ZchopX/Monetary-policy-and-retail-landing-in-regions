"""Focused contract checks for the level-data expanded GMM search."""

from pathlib import Path

import pandas as pd
import pytest

from Modules import gmm_level_specification_search as search


def _sources(tmp_path: Path, duplicate: bool = False) -> tuple[Path, Path]:
    dates = pd.date_range("2020-01-01", periods=7, freq="MS")
    regional = pd.DataFrame([(region, date, i + 10, i + 1) for region in ("A", "B") for i, date in enumerate(dates)], columns=["Region", "Date", "Int_Rate_ConsCred", "Zakred"])
    if duplicate: regional = pd.concat((regional, regional.iloc[[0]]), ignore_index=True)
    federal = pd.DataFrame({"Region": "RU", "Date": dates, "CPI": range(7), "Oil_p": range(7), "REER": range(7), "Covid_dum": [0, 0, 1, 1, 1, 0, 0], "Sank_dum": [0, 0, 0, 1, 1, 1, 1], "MaP_Fact": [0, 1, 0, 1, 0, 1, 0]})
    book, shock = tmp_path / "input.xlsx", tmp_path / "shock.pkl"
    with pd.ExcelWriter(book) as writer:
        regional.to_excel(writer, sheet_name="data", startrow=1, index=False); federal.to_excel(writer, sheet_name="data_RF", startrow=1, index=False)
    pd.DataFrame({"Date": dates[1:], "Mon_Shock": range(6)}).to_pickle(shock)
    return book, shock


def test_level_loader_has_no_python_difference_or_shock_split(tmp_path: Path) -> None:
    data, audit = search.load_panel(*_sources(tmp_path))
    assert {"Int_Rate_ConsCred", "Mon_Shock"}.issubset(data.columns)
    assert not any(name.startswith("d_") or "_pos" in name or "_neg" in name for name in data)
    assert "sole first difference" in audit["transformations"]


def test_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        search.load_panel(*_sources(tmp_path, duplicate=True))


def test_commands_keep_direct_terms_in_equation_and_iv() -> None:
    roles = {"Zakred": "endogenous", "Oil_p": "direct"}
    py = search.build_pydynpd_command(("Zakred", "Oil_p"), roles)
    r = search.build_r_formula(("Zakred", "Oil_p"), roles)
    assert "L1.Int_Rate_ConsCred Zakred Mon_Shock Covid_dum Sank_dum Oil_p" in py
    assert "gmm(Int_Rate_ConsCred Zakred, 2:3)" in py and "iv(Mon_Shock Covid_dum Sank_dum Oil_p)" in py
    assert "lag(Zakred, 2:3)" in r and "lag(Oil_p, 2:3)" not in r
    assert r.count("Oil_p") == 2  # equation plus normal-IV block, never duplicated in either block


def test_inventory_roles_and_exclusions(tmp_path: Path) -> None:
    data, _ = search.load_panel(*_sources(tmp_path)); data["MIACR"] = 1.; data["Key_Rate"] = 1.
    inventory, candidates = search.candidate_inventory(data)
    rows = inventory.set_index("source")
    assert rows.loc["Zakred", "role"] == "endogenous"
    assert rows.loc["Oil_p", "role"] == "direct"
    assert rows.loc["MIACR", "status"] == "excluded_by_scope"
    assert rows.loc["Mon_Shock", "status"] == "required"
    assert {x["name"] for x in candidates} >= {"Zakred", "Oil_p"}


def test_schedule_contains_every_initial_tuple_and_all_endogenous_fourth() -> None:
    candidates = [{"name": x, "role": "endogenous", "source": x} for x in "abcd"] + [{"name": "x", "role": "direct", "source": "x"}]
    schedule = search.initial_schedule(candidates)
    assert len(schedule) == 1 + 5 + 10 + 10 + 1
    assert ("a", "b", "c", "d") in schedule
    assert not any("x" in item and len(item) == 4 for item in schedule)


def test_parity_mismatch_forbids_robust_nonrejection() -> None:
    spec = {"r_formula": "f", "groups": 2, "key_sha256": "x", "input_sha256": "x", "roles": {}, "instrument_labels": ["a", "b", "c", "d"], "normal_iv_count": 2}
    r = {"status": "completed", "formula": "f", "groups": 2, "instrument_count": 4, "instrument_rank": 4, "key_sha256": "x", "input_sha256": "x", "sargan": {"p_value": .2}}
    py = {"status": "completed", "groups": 2, "instrument_count": 4, "instrument_rank": 4, "key_sha256": "x", "input_sha256": "x", "hansen_df": 5, "hansen_p": .2, "ar2_p": .2}
    check = search.parity(r, py, spec)
    assert check["status"] == "mismatch"
    assert "robust_hansen_nonrejecting" not in search.classify(r, py, check)
    assert "robust_hansen_unresolved_instrument_mismatch" in search.classify(r, py, check)


def test_descendants_keep_low_power_nonrejecting_parent() -> None:
    candidates = [{"name": "a", "role": "endogenous", "source": "a"}, {"name": "b", "role": "direct", "source": "b"}]
    children = search.descendants([{"controls": ["a"], "classification": ["low_power_warning", "robust_hansen_nonrejecting"]}], candidates)
    assert children == [("a", "b")]


def test_descendants_reject_invalid_instrument_parent() -> None:
    candidates = [{"name": "a", "role": "endogenous", "source": "a"}, {"name": "b", "role": "direct", "source": "b"}]
    children = search.descendants([{"controls": ["a"], "classification": ["instrument_cap_invalid", "robust_hansen_nonrejecting"]}], candidates)
    assert children == []


def test_blocked_bundle_keeps_required_sheets(tmp_path: Path) -> None:
    search._write_bundle(tmp_path, {"status": "robust_backend_blocked", "initial_universe": 1}, pd.DataFrame({"Region": ["A"]}), {}, pd.DataFrame(), [], [{"label": "legacy_immutable_reference"}], [{"status": "completed"}])
    assert set(pd.ExcelFile(tmp_path / "summary.xlsx").sheet_names) == {"Summary", "All attempts", "Candidate inventory", "Robust Hansen", "Legacy reference", "R reproduction", "Data audit"}
    assert "searched non-rejection is not confirmatory" in (tmp_path / "report.md").read_text(encoding="utf8")


def _clean_sources(tmp_path: Path, duplicate: bool = False, incomplete: bool = False,
                   missing_shock: bool = False) -> tuple[Path, Path]:
    dates = pd.date_range("2020-01-01", periods=7, freq="MS")
    controls = {name: i for i, name in enumerate({item for block in search.CLEAN_SPECS.values() for item in block}, start=1)}
    rows = []
    for region, cluster in (("A", (1, 0, 0)), ("B", (0, 0, 0)), ("C", (0, 1, 0)), ("D", (0, 0, 1))):
        for month, date in enumerate(dates):
            rows.append({"Region": region, "Date": date, "Int_Rate_ConsCred": month + 10, "Cluster_new_cd_1": cluster[0], "Cluster_new_cd_3": cluster[1], "Cluster_new_cd_4": cluster[2], **{name: month + value for name, value in controls.items()}})
    panel = pd.DataFrame(rows)
    if duplicate:
        panel = pd.concat((panel, panel.iloc[[0]]), ignore_index=True)
    if incomplete:
        panel = panel.drop(panel.index[0])
    book, shock = tmp_path / "clean.xlsx", tmp_path / "shock.pkl"
    panel.to_excel(book, index=False)
    pd.DataFrame({"Date": dates[:-1] if missing_shock else dates, "Mon_Shock": range(len(dates) - int(missing_shock))}).to_pickle(shock)
    return book, shock


def _federal_sources(tmp_path: Path, duplicate: bool = False, missing: bool = False,
                     mismatch: bool = False, missing_value: bool = False) -> tuple[Path, Path, Path]:
    book, shock = _clean_sources(tmp_path)
    panel = pd.read_excel(book)
    dates = pd.date_range("2020-01-01", periods=7, freq="MS")
    panel["REER"] = panel["Date"].map(dict(zip(dates, range(10, 17))))
    panel["Inflation_Expectations"] = panel["Date"].map(dict(zip(dates, range(20, 27))))
    if mismatch:
        panel.loc[0, "REER"] += 1
    panel.to_excel(book, index=False)
    federal = pd.DataFrame({"Date": dates, "Oil_p": range(30, 37), "REER": range(10, 17), "Inflation_Expectations": range(20, 27)})
    if duplicate:
        federal = pd.concat((federal, federal.iloc[[0]]), ignore_index=True)
    if missing:
        federal = federal.iloc[1:]
    if missing_value:
        federal.loc[0, "Oil_p"] = None
    federal_book = tmp_path / "federal.xlsx"
    federal.to_excel(federal_book, index=False)
    return book, federal_book, shock


def test_clean_loader_rejects_duplicate_and_incomplete_monthly_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        search.load_clean_panel(*_clean_sources(tmp_path, duplicate=True))
    with pytest.raises(ValueError, match="incomplete"):
        search.load_clean_panel(*_clean_sources(tmp_path, incomplete=True))


def test_clean_loader_rejects_missing_merged_shock(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing merged Mon_Shock"):
        search.load_clean_panel(*_clean_sources(tmp_path, missing_shock=True))


def test_clean_direct_block_uses_level_shock_and_cluster_interactions(tmp_path: Path) -> None:
    data, _ = search.load_clean_panel(*_clean_sources(tmp_path))
    controls = search.CLEAN_SPECS["core"]
    spec = search.specification(data, controls, {name: "endogenous" for name in controls}, search.CLEAN_DIRECT)
    assert tuple(spec["direct_terms"]) == search.CLEAN_DIRECT
    assert "L1.Int_Rate_ConsCred" in spec["pydynpd_command"]
    assert not any("d_Mon_Shock" in name or name in {"Covid_dum", "Sank_dum"} for name in spec["columns"])
    assert not any("MaP" in name or "Int_Rate_Mort" in name for name in spec["columns"])


def test_clean_schedule_is_fixed_and_excludes_unavailable_mortgage_default() -> None:
    assert list(search.CLEAN_SPECS) == ["core", "access", "distress", "without_zakred", "without_cpi", "without_capital", "without_credit_burden", "default_for_credit_burden"]
    assert all("Def_Zadolg_Mort" not in controls for controls in search.CLEAN_SPECS.values())


def test_paired_comparison_uses_identical_keys_for_both_sides(tmp_path: Path) -> None:
    data, _ = search.load_clean_panel(*_clean_sources(tmp_path))
    data.loc[data.index[0], "ln_Fin_Dostup"] = None
    common = search.paired_comparison_panel(data, search.CLEAN_SPECS["core"], search.CLEAN_SPECS["access"])
    core = search.specification(common, search.CLEAN_SPECS["core"], {name: "endogenous" for name in search.CLEAN_SPECS["core"]}, search.CLEAN_DIRECT)
    access = search.specification(common, search.CLEAN_SPECS["access"], {name: "endogenous" for name in search.CLEAN_SPECS["access"]}, search.CLEAN_DIRECT)
    assert core["key_sha256"] == access["key_sha256"]
    assert "d_Int_Rate_ConsCred" not in core["input"]


def test_paired_effects_report_only_new_failed_gates() -> None:
    attempts = [
        {"comparison_family": "core_vs_access", "specification_id": "core", "key_sha256": "same", "controls": ["a"], "gate": {"outcome": "shortlist", "failures": []}},
        {"comparison_family": "core_vs_access", "specification_id": "access", "key_sha256": "same", "controls": ["b"], "gate": {"outcome": "diagnostic_failure", "failures": ["Python Hansen p < 0.05"]}},
    ]
    effect = search._paired_effects(attempts).iloc[0]
    assert effect.common_keys_equal and effect.diagnostic_change == "Python Hansen p < 0.05"


def test_clean_federal_loader_validates_sources_and_monthly_keys(tmp_path: Path) -> None:
    data, audit = search.load_clean_federal_panel(*_federal_sources(tmp_path))
    assert set(search.FEDERAL_COLUMNS).issubset(data)
    assert audit["federal_source_equality"] == {"REER": True, "Inflation_Expectations": True}
    assert audit["national_time_series"]["Oil_p"]["monthly_change_observations"] == 6
    with pytest.raises(ValueError, match="duplicate"):
        search.load_clean_federal_panel(*_federal_sources(tmp_path, duplicate=True))
    with pytest.raises(ValueError, match="missing clean-panel"):
        search.load_clean_federal_panel(*_federal_sources(tmp_path, missing=True))
    with pytest.raises(ValueError, match="missing candidate"):
        search.load_clean_federal_panel(*_federal_sources(tmp_path, missing_value=True))
    with pytest.raises(ValueError, match="disagree"):
        search.load_clean_federal_panel(*_federal_sources(tmp_path, mismatch=True))


def test_federal_schedule_roles_pairs_and_national_constancy(tmp_path: Path) -> None:
    data, _ = search.load_clean_federal_panel(*_federal_sources(tmp_path))
    assert list(search.FEDERAL_SENSITIVITY_SPECS) == ["core", "core_plus_ln_fin_dostup", "core_plus_oil_p", "core_plus_reer", "core_plus_inflation_expectations"]
    core = search.CLEAN_SPECS["core"]
    for name, (addition, extra_roles) in search.FEDERAL_SENSITIVITY_SPECS.items():
        assert len(addition) <= 1
        if name == "core":
            continue
        controls = (*core, *addition)
        roles = {control: "endogenous" for control in core} | extra_roles
        spec = search.specification(data, controls, roles, search.CLEAN_DIRECT)
        common = search.paired_comparison_panel(data, core, controls)
        paired_core = search.specification(common, core, {control: "endogenous" for control in core}, search.CLEAN_DIRECT)
        paired_expanded = search.specification(common, controls, roles, search.CLEAN_DIRECT)
        assert paired_core["key_sha256"] == paired_expanded["key_sha256"]
        assert "d_Int_Rate_ConsCred" not in spec["input"] and "Date" not in spec["r_formula"]
    oil_spec = search.specification(data, (*core, "Oil_p"), {**{control: "endogenous" for control in core}, "Oil_p": "direct"}, search.CLEAN_DIRECT)
    assert "lag(Oil_p, 2:3)" not in oil_spec["r_formula"] and oil_spec["r_formula"].count("Oil_p") == 2
    fin_spec = search.specification(data, (*core, "ln_Fin_Dostup"), {control: "endogenous" for control in (*core, "ln_Fin_Dostup")}, search.CLEAN_DIRECT)
    assert "lag(ln_Fin_Dostup, 2:3)" in fin_spec["r_formula"]
    data.loc[data["Region"] == "A", "Oil_p"] += 1
    with pytest.raises(ValueError, match="not nationally constant"):
        search.national_series_audit(data)


def test_sensitivity_effects_report_shock_changes() -> None:
    totals = lambda value: {"status": "completed", **{label: {"estimate": value} for label in ("reference", "1", "3", "4")}}
    attempts = [
        {"comparison_family": "core_vs_oil", "specification_id": "core", "key_sha256": "same", "controls": ["a"], "gate": {"outcome": "shortlist", "failures": []}, "cluster_totals": totals(1)},
        {"comparison_family": "core_vs_oil", "specification_id": "core_plus_oil_p", "key_sha256": "same", "controls": ["a", "Oil_p"], "gate": {"outcome": "shortlist", "failures": []}, "cluster_totals": totals(1.5)},
    ]
    effect = search._sensitivity_effects(attempts).iloc[0]
    assert effect.shock_reference_change_from_core == .5 and effect.shock_4_change_from_core == .5
