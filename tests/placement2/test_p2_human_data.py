"""SYNTHETIC_TEST_DATA: hand-made rows, not Moral Machine observations."""

from __future__ import annotations

import pandas as pd
import pytest

from multilingual_mechinterp.placement2 import human_data as hd


def _response(rid, user, country, cluster, typ, default, non_default, stay_level, swerve_level,
              saved_side):
    base = {
        "ResponseID": rid, "ExtendedSessionID": f"s_{user}", "UserID": user,
        "ScenarioTypeStrict": typ, "DefaultChoice": default, "NonDefaultChoice": non_default,
        "UserCountry3": country, "ClusterName": cluster, "PedPed": 1, "Barrier": 0,
        "CrossingSignal": 0, "NumberOfCharacters": 2, "DiffNumberOFCharacters": 0,
    }
    # Intervention = 1 on the side that swerving would kill, i.e. the side STAY spares.
    return [
        {**base, "AttributeLevel": stay_level, "Intervention": 1, "Saved": int(saved_side == "stay")},
        {**base, "AttributeLevel": swerve_level, "Intervention": 0,
         "Saved": int(saved_side == "swerve")},
    ]


@pytest.fixture()
def responses() -> pd.DataFrame:
    rows = []
    rows += _response("r1", "u1", "JPN", "Eastern", "Age", "Young", "Old", "Young", "Old", "stay")
    rows += _response("r2", "u1", "JPN", "Eastern", "Age", "Young", "Old", "Old", "Young", "swerve")
    rows += _response("r3", "u2", "USA", "Western", "Age", "Young", "Old", "Rand", "Rand", "stay")
    rows += _response("r4", "u2", "USA", "Western", "Gender", "Male", "Female", "Male", "Female",
                      "swerve")
    return pd.DataFrame(rows)


@pytest.fixture()
def users() -> pd.DataFrame:
    return pd.DataFrame(
        {"UserID": ["u1", "u2", "u3"], "UserCountry3": ["JPN", "USA", "IRN"],
         "ClusterName": ["Eastern", "Western", "Eastern"]}
    )


def test_country_label():
    assert hd.country_label("JPN") == "Japan (JPN)"
    assert hd.country_label("IRN") == "Iran (IRN)"
    assert hd.country_label("XXX") == "XXX"


def test_country_table_shares(users):
    table = hd.country_table(users)
    eastern = table[table.Cluster == "Eastern"].set_index("ISO3")
    assert eastern.loc["JPN", "Country"] == "Japan (JPN)"
    assert eastern.loc["JPN", "% of cluster"] == 50.0
    assert eastern.loc["JPN", "rows"] == 20
    assert eastern.loc["JPN", "responses"] == 10
    assert list(table.Cluster.unique()) == ["Western", "Eastern"]


def test_cluster_table(users):
    table = hd.cluster_table(users)
    assert table.loc["Eastern", "countries"] == 2
    assert table.loc["Eastern", "users"] == 2
    assert table["% of all users"].sum() == pytest.approx(100)


def test_human_action_from_saved_and_intervention(responses):
    q = hd.response_table(responses).set_index("ResponseID")
    assert q.loc["r1", "human_action"] == "STAY"
    assert q.loc["r2", "human_action"] == "SWERVE"
    assert q.loc["r1", "spared_level"] == "Young"
    assert q.loc["r2", "spared_level"] == "Young"


def test_random_sides_are_not_clean(responses):
    q = hd.response_table(responses).set_index("ResponseID")
    assert not q.loc["r3", "clean_contrast"]
    assert q.loc[["r1", "r2", "r4"], "clean_contrast"].all()


def test_preference_and_action_tables(responses):
    q = hd.response_table(responses)
    pref = hd.preference_table(q)
    assert pref.loc["Age: Young over Old", "Eastern"] == 100.0
    assert pref.loc["Gender: Male over Female", "Western"] == 0.0
    actions = hd.action_table(q)
    assert actions.loc["Age", "Eastern"] == 50.0


def _criteria(stay_kills_level, swerve_kills_level, n_stay=1, n_swerve=1):
    return {
        "scenario_type": "Age", "attribute_level_if_stay_kills": stay_kills_level,
        "attribute_level_if_swerve_kills": swerve_kills_level,
        "characters_killed_if_stay": n_stay, "characters_killed_if_swerve": n_swerve,
        "PedPed": 1, "Barrier": 0, "CrossingSignal_if_stay_kills": 0,
        "CrossingSignal_if_swerve_kills": 0,
    }


def _age_responses(n_stay, n_swerve, cluster="Eastern", stay_kills="Old", swerve_kills="Young",
                   country="JPN", user_prefix="u"):
    """Responses where STAY kills ``stay_kills``; n_stay users chose STAY, n_swerve SWERVE."""
    rows = []
    for i in range(n_stay + n_swerve):
        chose_stay = i < n_stay
        base = {"ResponseID": f"{cluster}{stay_kills}{country}{i}", "ScenarioTypeStrict": "Age",
                "NumberOfCharacters": 1, "PedPed": 1, "Barrier": 0, "CrossingSignal": 0,
                "ClusterName": cluster, "UserID": f"{user_prefix}{cluster}{i}",
                "UserCountry3": country}
        rows.append({**base, "AttributeLevel": stay_kills, "Intervention": 0,
                     "Saved": int(not chose_stay)})
        rows.append({**base, "AttributeLevel": swerve_kills, "Intervention": 1,
                     "Saved": int(chose_stay)})
    return pd.DataFrame(rows)


def _question(qid, criteria, expected):
    return {"question_id": qid, "expected_action": expected, "human_match_criteria": criteria}


def test_majority_label_tiers():
    assert hd.majority_label(0.50, 0.485, 0.515, 0.05) == ("UNDETERMINED", "tie")
    assert hd.majority_label(0.532, 0.517, 0.547, 0.05) == ("STAY", "weak")
    assert hd.majority_label(0.442, 0.427, 0.457, 0.05) == ("SWERVE", "clear")
    assert hd.majority_label(float("nan"), float("nan"), float("nan"), 0.05) == (
        "UNDETERMINED", "no_data")


def test_wilson_interval_contains_p():
    low, high = hd.wilson_interval(530, 1000)
    assert low < 0.53 < high
    assert high - low == pytest.approx(0.062, abs=0.002)


def test_references_count_stay_from_the_swerve_kills_side():
    responses = pd.concat([
        _age_responses(90, 10),                                  # STAY kills Old: 90% STAY
        _age_responses(20, 80, stay_kills="Young", swerve_kills="Old"),
        _age_responses(500, 500, cluster="Western"),
    ])
    questions = [
        _question("AGE_A", _criteria("Old", "Young"), "STAY"),
        _question("AGE_B", _criteria("Young", "Old"), "SWERVE"),
        _question("AGE_C", _criteria("Old", "Young", n_stay=3, n_swerve=3), "STAY"),
    ]
    refs = hd.human_references(responses, questions).set_index(["question_id", "cluster"])

    a = refs.loc[("AGE_A", "Eastern")]
    assert (a.n_matched_responses, a.n_stay, a.majority_label) == (100, 90, "STAY")
    assert a.majority_is_expected_action
    b = refs.loc[("AGE_B", "Eastern")]
    assert (b.n_stay, b.majority_label, b.label_strength) == (20, "SWERVE", "clear")
    assert refs.loc[("AGE_A", "Western"), "majority_label"] == "UNDETERMINED"
    assert refs.loc[("AGE_A", "Western"), "majority_is_expected_action"] is None
    assert refs.loc[("AGE_A", hd.ALL_CLUSTERS), "n_matched_responses"] == 1100
    assert refs.loc[("AGE_C", "Eastern"), "label_strength"] == "no_data"


def _mirrored_refs():
    responses = pd.concat([
        _age_responses(70, 30),                                          # STAY kills Old
        _age_responses(40, 60, stay_kills="Young", swerve_kills="Old"),  # STAY kills Young
        _age_responses(50, 50, cluster="Western"),
        _age_responses(50, 50, cluster="Western", stay_kills="Young", swerve_kills="Old"),
    ])
    questions = [
        {**_question("AGE_A", _criteria("Old", "Young"), "STAY"), "counterbalance_pair_id": "P1"},
        {**_question("AGE_B", _criteria("Young", "Old"), "SWERVE"), "counterbalance_pair_id": "P1"},
    ]
    return responses, questions, hd.human_references(responses, questions)


def test_paired_inaction_separates_action_and_group():
    _, questions, refs = _mirrored_refs()
    paired = hd.paired_inaction(refs, questions).set_index("cluster")
    # Eastern: STAY share 70% and 40% -> 55% paired; sparing Young 70% and 60% -> 65%.
    assert paired.loc["Eastern", "pct_stay_paired"] == pytest.approx(55.0)
    assert paired.loc["Eastern", "pct_expected_group_paired"] == pytest.approx(65.0)
    assert paired.loc["Western", "pct_stay_paired"] == pytest.approx(50.0)


def test_culture_divergence_and_p_swerve():
    _, _, refs = _mirrored_refs()
    assert (refs.p_stay + refs.p_swerve).round(9).eq(1).all()
    div = hd.culture_divergence(refs).set_index("question_id")
    assert div.loc["AGE_A", "D_culture"] == pytest.approx(0.2)
    assert div.loc["AGE_A", "max_cluster"] == "Western"


def test_user_repeat_check_and_country_references():
    responses = pd.concat([
        _age_responses(3, 1),
        _age_responses(2, 0, user_prefix="u"),  # same user ids answer again
        _age_responses(1, 1, country="IRN", user_prefix="i"),
    ])
    responses["ResponseID"] = [f"r{i // 2}" for i in range(len(responses))]
    questions = [_question("AGE_A", _criteria("Old", "Young"), "STAY")]
    sides = hd.action_sides(responses)
    repeats = hd.user_repeat_check(sides, questions).iloc[0]
    assert repeats.max_per_user == 2
    assert repeats.users_with_repeats == 2
    iran = hd.country_references(sides, questions, "IRN").iloc[0]
    assert (iran.cluster, iran.n_matched_responses, iran.majority_label) == ("Iran (IRN)", 2, "UNDETERMINED")


def test_cluster_differences_and_preference_ci(responses):
    q = hd.response_table(responses)
    table = hd.preference_ci_table(q)
    row = table[(table.type == "Age") & (table.cluster == "Eastern")].iloc[0]
    assert row.n_clean == 2 and row.pct_spare_default == 100.0
    assert row.ci_low < 100.0
    many = pd.concat([_age_responses(80, 20), _age_responses(50, 50, cluster="Western")])
    many["DefaultChoice"], many["NonDefaultChoice"] = "Young", "Old"
    many["DiffNumberOFCharacters"] = 0
    diffs = hd.cluster_differences(hd.response_table(many))
    west_east = diffs[(diffs.cluster_a == "Western") & (diffs.cluster_b == "Eastern")].iloc[0]
    # Eastern spares Young when choosing STAY (STAY kills Old): 80% vs 50%.
    assert west_east.diff_pp == pytest.approx(-30.0)
    assert west_east.excludes_zero


def test_catalog_marks_question_configurations():
    responses, questions, _ = _mirrored_refs()
    catalog = hd.configuration_catalog(hd.action_sides(responses), questions)
    assert set(catalog.question_id.dropna()) == {"AGE_A", "AGE_B"}
    assert (catalog["n Eastern"] == 100).all()
    assert set(catalog.layout) == {"pedestrians_no_signal"}


def _side(level, n=1, barrier=0, signal=0):
    return {"count": n, "attribute_level": level, "Barrier": barrier, "CrossingSignal": signal}


def _v6_question(qid, stay, swerve, expected, pedped=1):
    return {"question_id": qid, "scenario_type": "Age", "expected_action": expected,
            "stay_kills": stay, "swerve_kills": swerve, "PedPed": pedped}


def _layout_responses(n_stay, n_swerve, stay_side, swerve_side, pedped, tag):
    """Age responses with per-side barrier and signal; STAY kills ``stay_side``."""
    rows = []
    for i in range(n_stay + n_swerve):
        chose_stay = i < n_stay
        base = {"ResponseID": f"{tag}{i}", "ScenarioTypeStrict": "Age", "PedPed": pedped,
                "ClusterName": "Eastern", "UserID": f"{tag}u{i}", "UserCountry3": "JPN"}
        rows.append({**base, "AttributeLevel": stay_side["attribute_level"], "NumberOfCharacters": stay_side["count"],
                     "Barrier": stay_side["Barrier"], "CrossingSignal": stay_side["CrossingSignal"],
                     "Intervention": 0, "Saved": int(not chose_stay)})
        rows.append({**base, "AttributeLevel": swerve_side["attribute_level"],
                     "NumberOfCharacters": swerve_side["count"], "Barrier": swerve_side["Barrier"],
                     "CrossingSignal": swerve_side["CrossingSignal"], "Intervention": 1,
                     "Saved": int(chose_stay)})
    return pd.DataFrame(rows)


def test_v6_questions_match_signal_and_barrier_per_side():
    green_old, red_young = _side("Old", 2, signal=1), _side("Young", 2, signal=2)
    passengers_old, road_young = _side("Old", 2, barrier=1), _side("Young", 2)
    responses = pd.concat([
        _layout_responses(30, 70, green_old, red_young, 1, "sig"),
        _layout_responses(80, 20, red_young, green_old, 1, "sigflip"),
        _layout_responses(10, 90, passengers_old, road_young, 0, "bar"),
        _layout_responses(50, 50, _side("Old", 2), _side("Young", 2), 1, "plain"),
    ])
    questions = [
        _v6_question("SIG", green_old, red_young, "STAY"),
        _v6_question("BAR", passengers_old, road_young, "STAY", pedped=0),
    ]
    refs = hd.human_references(responses, questions).set_index(["question_id", "cluster"])
    sig, bar = refs.loc[("SIG", "Eastern")], refs.loc[("BAR", "Eastern")]
    assert (sig.n_matched_responses, sig.n_stay, sig.majority_label) == (100, 30, "SWERVE")
    assert sig.layout == "pedestrians_crossing_signal"
    assert "2 Old (pedestrians, green)" in sig.match_configuration
    assert (bar.n_matched_responses, bar.n_stay, bar.layout) == (100, 10, "passengers_vs_pedestrians")
    assert "2 Old (passengers)" in bar.match_configuration
    assert hd.paired_inaction(refs.reset_index(), questions).empty
    catalog = hd.configuration_catalog(hd.action_sides(responses), questions)
    assert len(catalog) == 4
    assert set(catalog.question_id.dropna()) == {"SIG", "BAR"}


def test_structure_checks_pass_on_valid_sample(responses, users):
    checks = hd.structure_checks(responses, users[users.UserID != "u3"]).set_index("check")
    assert checks.loc["every response has exactly one Saved = 1", "passed"]
    assert checks.loc["every response has one stay and one swerve side", "passed"]
    assert not checks.loc["every user has 20 rows", "passed"]
