"""Load and summarise the balanced Moral Machine sample in ``data_raw_output/``.

Each CSV row is one side of one dilemma; the two sides share a ``ResponseID``.
One response (a user's answer to one dilemma) is therefore two rows.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

RESPONSES_FILE = "balanced_cluster_responses_300k.csv"
USERS_FILE = "balanced_cluster_users_300k.csv"

CLUSTER_ORDER = ("Western", "Eastern", "Southern")
TYPE_ORDER = ("Utilitarian", "Gender", "Fitness", "Age", "Species")
ID_COLUMNS = ("ResponseID", "ExtendedSessionID", "UserID")

COUNTRY_NAMES = {
    "AFG": "Afghanistan", "AGO": "Angola", "ALB": "Albania", "AND": "Andorra",
    "ARE": "United Arab Emirates", "ARG": "Argentina", "ARM": "Armenia", "AUS": "Australia",
    "AUT": "Austria", "AZE": "Azerbaijan", "BEL": "Belgium", "BGD": "Bangladesh",
    "BGR": "Bulgaria", "BHR": "Bahrain", "BHS": "Bahamas", "BIH": "Bosnia and Herzegovina",
    "BLR": "Belarus", "BOL": "Bolivia", "BRA": "Brazil", "BRB": "Barbados", "BRN": "Brunei",
    "CAN": "Canada", "CHE": "Switzerland", "CHL": "Chile", "CHN": "China", "COL": "Colombia",
    "CRI": "Costa Rica", "CYP": "Cyprus", "CZE": "Czechia", "DEU": "Germany", "DNK": "Denmark",
    "DOM": "Dominican Republic", "DZA": "Algeria", "ECU": "Ecuador", "EGY": "Egypt",
    "ESP": "Spain", "EST": "Estonia", "FIN": "Finland", "FRA": "France", "GBR": "United Kingdom",
    "GEO": "Georgia", "GGY": "Guernsey", "GRC": "Greece", "GTM": "Guatemala", "GUM": "Guam",
    "HKG": "Hong Kong", "HND": "Honduras", "HRV": "Croatia", "HUN": "Hungary", "IDN": "Indonesia",
    "IMN": "Isle of Man", "IND": "India", "IRL": "Ireland", "IRN": "Iran", "IRQ": "Iraq",
    "ISL": "Iceland", "ISR": "Israel", "ITA": "Italy", "JAM": "Jamaica", "JEY": "Jersey",
    "JOR": "Jordan", "JPN": "Japan", "KAZ": "Kazakhstan", "KEN": "Kenya", "KGZ": "Kyrgyzstan",
    "KHM": "Cambodia", "KOR": "South Korea", "KWT": "Kuwait", "LBN": "Lebanon",
    "LKA": "Sri Lanka", "LTU": "Lithuania", "LUX": "Luxembourg", "LVA": "Latvia", "MAC": "Macao",
    "MAR": "Morocco", "MCO": "Monaco", "MDA": "Moldova", "MDG": "Madagascar", "MDV": "Maldives",
    "MEX": "Mexico", "MKD": "North Macedonia", "MLT": "Malta", "MMR": "Myanmar",
    "MNE": "Montenegro", "MNG": "Mongolia", "MTQ": "Martinique", "MUS": "Mauritius",
    "MYS": "Malaysia", "NCL": "New Caledonia", "NGA": "Nigeria", "NIC": "Nicaragua",
    "NLD": "Netherlands", "NOR": "Norway", "NPL": "Nepal", "NZL": "New Zealand", "OMN": "Oman",
    "PAK": "Pakistan", "PAN": "Panama", "PER": "Peru", "PHL": "Philippines", "POL": "Poland",
    "PRI": "Puerto Rico", "PRT": "Portugal", "PRY": "Paraguay", "PSE": "Palestine",
    "PYF": "French Polynesia", "QAT": "Qatar", "REU": "Reunion", "ROU": "Romania",
    "RUS": "Russia", "SAU": "Saudi Arabia", "SGP": "Singapore", "SLV": "El Salvador",
    "SRB": "Serbia", "SVK": "Slovakia", "SVN": "Slovenia", "SWE": "Sweden", "SYR": "Syria",
    "THA": "Thailand", "TTO": "Trinidad and Tobago", "TUN": "Tunisia", "TUR": "Turkey",
    "TWN": "Taiwan", "UKR": "Ukraine", "URY": "Uruguay", "USA": "United States",
    "UZB": "Uzbekistan", "VEN": "Venezuela", "VNM": "Vietnam", "ZAF": "South Africa",
}


def country_label(code: str) -> str:
    """``"JPN"`` -> ``"Japan (JPN)"``; unknown codes are returned unchanged."""
    name = COUNTRY_NAMES.get(code)
    return f"{name} ({code})" if name else code


def load_responses(data_dir: str | Path) -> pd.DataFrame:
    # IDs must stay text: as floats the 16-digit IDs lose precision.
    return pd.read_csv(
        Path(data_dir) / RESPONSES_FILE,
        dtype={col: str for col in (*ID_COLUMNS, "UserCountry3")},
    )


def load_users(data_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / USERS_FILE, dtype=str)


def structure_checks(responses: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """One row per check with the observed value and whether it passed."""
    per_question = responses.groupby("ResponseID").agg(
        rows=("Saved", "size"),
        saved=("Saved", "sum"),
        swerve_sides=("Intervention", "sum"),
        types=("ScenarioTypeStrict", "nunique"),
    )
    rows_per_user = responses.groupby("UserID").size()
    checks = [
        ("users in users file", len(users), True),
        ("users in responses file", responses.UserID.nunique(), True),
        ("same user IDs in both files", None, set(users.UserID) == set(responses.UserID)),
        ("rows", len(responses), True),
        ("responses (ResponseIDs)", len(per_question), len(per_question) * 2 == len(responses)),
        ("every user has 20 rows", int(rows_per_user.min()), bool((rows_per_user == 20).all())),
        ("every response has 2 rows", None, bool((per_question.rows == 2).all())),
        ("every response has exactly one Saved = 1", None, bool((per_question.saved == 1).all())),
        ("every response has one stay and one swerve side", None,
         bool((per_question.swerve_sides == 1).all())),
        ("both rows of a response have the same type", None, bool((per_question.types == 1).all())),
        ("only the 5 kept types", None, set(responses.ScenarioTypeStrict) <= set(TYPE_ORDER)),
    ]
    return pd.DataFrame(checks, columns=["check", "value", "passed"])


def cluster_table(users: pd.DataFrame) -> pd.DataFrame:
    table = users.groupby("ClusterName").agg(
        countries=("UserCountry3", "nunique"), users=("UserID", "size")
    )
    table = table.reindex([c for c in CLUSTER_ORDER if c in table.index])
    table["responses"] = table.users * 10
    table["rows"] = table.users * 20
    table["% of all users"] = (100 * table.users / table.users.sum()).round(2)
    return table


def country_table(users: pd.DataFrame) -> pd.DataFrame:
    """Users, responses and rows per country, with its share of the cluster."""
    table = (
        users.groupby(["ClusterName", "UserCountry3"]).size().rename("users").reset_index()
    )
    table["Country"] = table.UserCountry3.map(country_label)
    table["responses"] = table.users * 10
    table["rows"] = table.users * 20
    cluster_total = table.groupby("ClusterName").users.transform("sum")
    table["% of cluster"] = (100 * table.users / cluster_total).round(2)
    table["% of all users"] = (100 * table.users / table.users.sum()).round(3)
    table["ClusterName"] = pd.Categorical(table.ClusterName, CLUSTER_ORDER, ordered=True)
    table = table.sort_values(["ClusterName", "users"], ascending=[True, False])
    table = table.rename(columns={"ClusterName": "Cluster", "UserCountry3": "ISO3"})
    columns = ["Cluster", "Country", "ISO3", "users", "responses", "rows",
               "% of cluster", "% of all users"]
    return table[columns].reset_index(drop=True)


def response_table(responses: pd.DataFrame) -> pd.DataFrame:
    """One row per response with the human action and the spared level.

    ``Intervention`` says which action kills a side, so sparing the side with
    ``Intervention == 1`` means the user chose STAY, and vice versa.
    """
    saved = responses[responses.Saved == 1]
    other = responses[responses.Saved == 0][["ResponseID", "AttributeLevel"]]
    table = saved.merge(other, on="ResponseID", suffixes=("", "_other"))
    table = table.rename(
        columns={
            "AttributeLevel": "spared_level",
            "AttributeLevel_other": "killed_level",
            "ScenarioTypeStrict": "type",
            "ClusterName": "cluster",
            "UserCountry3": "country",
        }
    )
    table["human_action"] = table.Intervention.map({1: "STAY", 0: "SWERVE"})
    table["clean_contrast"] = [
        {spared, killed} == {default, non_default} and spared != killed
        for spared, killed, default, non_default in zip(
            table.spared_level, table.killed_level, table.DefaultChoice, table.NonDefaultChoice
        )
    ]
    table["spared_default"] = table.spared_level == table.DefaultChoice
    columns = ["ResponseID", "UserID", "country", "cluster", "type", "DefaultChoice",
               "NonDefaultChoice", "spared_level", "killed_level", "human_action",
               "clean_contrast", "spared_default", "PedPed", "Barrier", "CrossingSignal",
               "NumberOfCharacters", "DiffNumberOFCharacters"]
    return table[columns]


def type_table(questions: pd.DataFrame) -> pd.DataFrame:
    """Responses per type, and how many compare the two levels cleanly."""
    table = questions.groupby("type").agg(
        responses=("ResponseID", "size"),
        clean=("clean_contrast", "sum"),
        default=("DefaultChoice", "first"),
        non_default=("NonDefaultChoice", "first"),
    )
    table["both sides random"] = table.responses - table.clean
    table["% clean"] = (100 * table.clean / table.responses).round(2)
    return table.reindex([t for t in TYPE_ORDER if t in table.index])


def preference_table(questions: pd.DataFrame) -> pd.DataFrame:
    """% of clean responses sparing the default level (e.g. Young over Old), per cluster."""
    clean = questions[questions.clean_contrast]
    table = clean.pivot_table(
        index="type", columns="cluster", values="spared_default", aggfunc="mean"
    )
    table = 100 * table.reindex(index=[t for t in TYPE_ORDER if t in table.index],
                                columns=[c for c in CLUSTER_ORDER if c in table.columns])
    pairs = clean.groupby("type")[["DefaultChoice", "NonDefaultChoice"]].first()
    table.index = [f"{t}: {pairs.loc[t, 'DefaultChoice']} over {pairs.loc[t, 'NonDefaultChoice']}"
                   for t in table.index]
    table["All clusters"] = 100 * clean.groupby("type").spared_default.mean().reindex(
        [t for t in TYPE_ORDER if t in pairs.index]
    ).to_numpy()
    return table.round(1)


def action_table(questions: pd.DataFrame) -> pd.DataFrame:
    """% of responses where the human chose STAY, per type and cluster."""
    stay = questions.assign(stay=questions.human_action == "STAY")
    table = 100 * stay.pivot_table(index="type", columns="cluster", values="stay", aggfunc="mean")
    table = table.reindex(index=[t for t in TYPE_ORDER if t in table.index],
                          columns=[c for c in CLUSTER_ORDER if c in table.columns])
    table.loc["All types"] = 100 * stay.groupby("cluster").stay.mean().reindex(table.columns)
    table["All clusters"] = list(100 * stay.groupby("type").stay.mean().reindex(
        table.index[:-1]
    )) + [100 * stay.stay.mean()]
    return table.round(1)


ALL_CLUSTERS = "All clusters"


def load_questions(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def action_sides(responses: pd.DataFrame) -> pd.DataFrame:
    """One row per response: columns ending ``_sk`` describe the side STAY kills
    (``Intervention == 0``), ``_wk`` the side SWERVE kills (``Intervention == 1``)."""
    columns = ["ResponseID", "ScenarioTypeStrict", "AttributeLevel", "NumberOfCharacters",
               "PedPed", "Barrier", "CrossingSignal", "Saved", "ClusterName", "UserID",
               "UserCountry3"]
    sides = responses[columns]
    stay_kills = sides[responses.Intervention == 0].set_index("ResponseID")
    swerve_kills = sides[responses.Intervention == 1].set_index("ResponseID")
    table = stay_kills.join(swerve_kills, lsuffix="_sk", rsuffix="_wk", how="inner")
    table["chose_stay"] = table.Saved_wk == 1
    return table


SIGNAL_NAMES = {0: "no signal", 1: "green", 2: "red"}


def match_criteria(question: dict) -> dict:
    """The configuration a question is matched on, per side.

    Reads the per-side ``stay_kills``/``swerve_kills`` fields (question files v3 onward,
    which carry ``Barrier`` and ``CrossingSignal`` per side) or the older single
    ``human_match_criteria`` dict.
    """
    stay, swerve = question.get("stay_kills", {}), question.get("swerve_kills", {})
    if "Barrier" in stay and "Barrier" in swerve:
        return {
            "scenario_type": question["scenario_type"],
            "level_sk": stay["attribute_level"], "level_wk": swerve["attribute_level"],
            "n_sk": stay["count"], "n_wk": swerve["count"],
            "barrier_sk": stay["Barrier"], "barrier_wk": swerve["Barrier"],
            "signal_sk": stay["CrossingSignal"], "signal_wk": swerve["CrossingSignal"],
            "pedped": question["PedPed"],
        }
    c = question["human_match_criteria"]
    return {
        "scenario_type": c["scenario_type"],
        "level_sk": c["attribute_level_if_stay_kills"], "level_wk": c["attribute_level_if_swerve_kills"],
        "n_sk": c["characters_killed_if_stay"], "n_wk": c["characters_killed_if_swerve"],
        "barrier_sk": c["Barrier"], "barrier_wk": c["Barrier"],
        "signal_sk": c["CrossingSignal_if_stay_kills"], "signal_wk": c["CrossingSignal_if_swerve_kills"],
        "pedped": c["PedPed"],
    }


def match_mask(sides: pd.DataFrame, criteria: dict) -> pd.Series:
    """Responses with the configuration ``criteria`` (from :func:`match_criteria`)."""
    return (
        (sides.ScenarioTypeStrict_sk == criteria["scenario_type"])
        & (sides.AttributeLevel_sk == criteria["level_sk"])
        & (sides.AttributeLevel_wk == criteria["level_wk"])
        & (sides.NumberOfCharacters_sk == criteria["n_sk"])
        & (sides.NumberOfCharacters_wk == criteria["n_wk"])
        & (sides.PedPed_sk == criteria["pedped"])
        & (sides.Barrier_sk == criteria["barrier_sk"])
        & (sides.Barrier_wk == criteria["barrier_wk"])
        & (sides.CrossingSignal_sk == criteria["signal_sk"])
        & (sides.CrossingSignal_wk == criteria["signal_wk"])
    )


def layout_name(barrier_sk: int, barrier_wk: int, signal_sk: int, signal_wk: int) -> str:
    if barrier_sk or barrier_wk:
        return "passengers_vs_pedestrians"
    if signal_sk or signal_wk:
        return "pedestrians_crossing_signal"
    return "pedestrians_no_signal"


def _side_label(level: str, n: int, barrier: int, signal: int) -> str:
    where = "passengers" if barrier else f"pedestrians, {SIGNAL_NAMES.get(signal, signal)}"
    return f"{n} {level} ({where})"


def configuration_label(criteria: dict) -> str:
    c = criteria
    return (
        f"{c['scenario_type']}: STAY kills {_side_label(c['level_sk'], c['n_sk'], c['barrier_sk'], c['signal_sk'])}, "
        f"SWERVE kills {_side_label(c['level_wk'], c['n_wk'], c['barrier_wk'], c['signal_wk'])}"
    )


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def majority_label(p_stay: float, ci_low: float, ci_high: float, near_tie: float) -> tuple[str, str]:
    """(label, strength): UNDETERMINED/tie when the interval includes 50%;
    otherwise the majority, ``weak`` when within ``near_tie`` of 50%."""
    if math.isnan(p_stay):
        return "UNDETERMINED", "no_data"
    if ci_low <= 0.5 <= ci_high:
        return "UNDETERMINED", "tie"
    label = "STAY" if p_stay > 0.5 else "SWERVE"
    return label, "clear" if abs(p_stay - 0.5) >= near_tie else "weak"


def _reference_row(question: dict, group_name: str, group: pd.DataFrame, near_tie: float) -> dict:
    criteria = match_criteria(question)
    n, k = len(group), int(group.chose_stay.sum())
    p = k / n if n else math.nan
    low, high = wilson_interval(k, n)
    label, strength = majority_label(p, low, high, near_tie)
    return {
        "question_id": question["question_id"],
        "scenario_type": criteria["scenario_type"],
        "layout": layout_name(criteria["barrier_sk"], criteria["barrier_wk"],
                              criteria["signal_sk"], criteria["signal_wk"]),
        "expected_action": question["expected_action"],
        "cluster": group_name,
        "match_configuration": configuration_label(criteria),
        "n_matched_responses": n,
        "n_stay": k,
        "n_swerve": n - k,
        "p_stay": p,
        "p_swerve": 1 - p,
        "p_stay_ci_low": low,
        "p_stay_ci_high": high,
        "majority_label": label,
        "label_strength": strength,
        "near_tie_flag": strength != "clear",
        "majority_is_expected_action": (
            None if label == "UNDETERMINED" else label == question["expected_action"]
        ),
        "aggregation": "pooled_respondents_balanced_cluster_sample",
    }


def human_references(
    responses: pd.DataFrame, questions: list[dict], *, near_tie: float = 0.05,
    sides: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-question, per-cluster human reference from matched responses.

    ``All clusters`` pools respondents of the balanced sample (equal users per cluster).
    """
    sides = action_sides(responses) if sides is None else sides
    clusters = [c for c in CLUSTER_ORDER if c in set(sides.ClusterName_sk)]
    records = []
    for question in questions:
        matched = sides[match_mask(sides, match_criteria(question))]
        groups = [(c, matched[matched.ClusterName_sk == c]) for c in clusters]
        groups.append((ALL_CLUSTERS, matched))
        records.extend(_reference_row(question, name, group, near_tie) for name, group in groups)
    return pd.DataFrame(records)


def country_references(
    sides: pd.DataFrame, questions: list[dict], country: str, *, near_tie: float = 0.05
) -> pd.DataFrame:
    """Same as :func:`human_references` for one country (usually too sparse to label)."""
    in_country = sides[sides.UserCountry3_sk == country]
    rows = [
        _reference_row(q, country_label(country), in_country[match_mask(in_country, match_criteria(q))],
                       near_tie)
        for q in questions
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Phase 1 completion: uncertainty, cluster differences, paired inaction, quality
# ---------------------------------------------------------------------------


def preference_ci_table(questions_table: pd.DataFrame) -> pd.DataFrame:
    """Share of clean responses sparing the default level, with 95% Wilson intervals.

    ``questions_table`` is the output of :func:`response_table`.
    """
    clean = questions_table[questions_table.clean_contrast]
    rows = [_share_row(kind, cluster, group) for (kind, cluster), group in clean.groupby(["type", "cluster"])]
    for kind, group in clean.groupby("type"):
        rows.append(_share_row(kind, ALL_CLUSTERS, group))
    table = pd.DataFrame(rows)
    table["type"] = pd.Categorical(table.type, TYPE_ORDER, ordered=True)
    table["cluster"] = pd.Categorical(table.cluster, [*CLUSTER_ORDER, ALL_CLUSTERS], ordered=True)
    return table.sort_values(["type", "cluster"]).reset_index(drop=True)


def _share_row(kind: str, cluster: str, group: pd.DataFrame) -> dict:
    n, k = len(group), int(group.spared_default.sum())
    low, high = wilson_interval(k, n)
    return {
        "type": kind,
        "cluster": cluster,
        "default_level": group.DefaultChoice.iloc[0] if n else None,
        "other_level": group.NonDefaultChoice.iloc[0] if n else None,
        "n_clean": n,
        "pct_spare_default": 100 * k / n if n else math.nan,
        "ci_low": 100 * low,
        "ci_high": 100 * high,
    }


def _diff_ci(k1: int, n1: int, k2: int, n2: int, z: float = 1.96) -> tuple[float, float, float]:
    p1, p2 = k1 / n1, k2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p1 - p2
    return 100 * diff, 100 * (diff - z * se), 100 * (diff + z * se)


def cluster_differences(questions_table: pd.DataFrame) -> pd.DataFrame:
    """Pairwise cluster differences (percentage points) in sparing the default level."""
    clean = questions_table[questions_table.clean_contrast]
    counts = clean.groupby(["type", "cluster"]).spared_default.agg(["sum", "size"])
    rows = []
    for kind in [t for t in TYPE_ORDER if t in counts.index.get_level_values(0)]:
        for i, a in enumerate(CLUSTER_ORDER):
            for b in CLUSTER_ORDER[i + 1:]:
                if (kind, a) not in counts.index or (kind, b) not in counts.index:
                    continue
                ka, na = counts.loc[(kind, a)]
                kb, nb = counts.loc[(kind, b)]
                diff, low, high = _diff_ci(int(ka), int(na), int(kb), int(nb))
                rows.append({
                    "type": kind, "cluster_a": a, "cluster_b": b,
                    "diff_pp": diff, "ci_low_pp": low, "ci_high_pp": high,
                    "excludes_zero": low > 0 or high < 0,
                })
    return pd.DataFrame(rows)


PAIRED_COLUMNS = ["pair_id", "scenario_type", "cluster", "questions", "n_matched", "pct_stay_paired",
                  "stay_ci_low", "stay_ci_high", "pct_expected_group_paired", "expected_ci_low",
                  "expected_ci_high"]


def paired_inaction(references: pd.DataFrame, questions: list[dict]) -> pd.DataFrame:
    """Mirrored pairs separate the preference for STAY from the group preference.

    In a mirrored pair the two groups swap paths, so the mean STAY share over the pair is
    the preference for not intervening (50 = none) with the group held fixed, and the mean
    share choosing ``expected_action`` is the group preference with the action held fixed.
    Questions without a ``counterbalance_pair_id`` are skipped; with none the table is empty.
    """
    pair_of = {q["question_id"]: q.get("counterbalance_pair_id") for q in questions}
    if not any(pair_of.values()):
        return pd.DataFrame(columns=PAIRED_COLUMNS)
    table = references.assign(pair_id=references.question_id.map(pair_of))
    table["p_expected"] = [
        p if action == "STAY" else 1 - p
        for p, action in zip(table.p_stay, table.expected_action)
    ]
    table["var"] = table.p_stay * (1 - table.p_stay) / table.n_matched_responses
    rows = []
    for (pair_id, cluster), group in table.groupby(["pair_id", "cluster"], sort=False):
        if len(group) != 2:
            continue
        se = math.sqrt(group["var"].sum()) / 2
        stay = group.p_stay.mean()
        expected = group.p_expected.mean()
        rows.append({
            "pair_id": pair_id,
            "scenario_type": group.scenario_type.iloc[0],
            "cluster": cluster,
            "questions": " / ".join(group.question_id),
            "n_matched": int(group.n_matched_responses.sum()),
            "pct_stay_paired": 100 * stay,
            "stay_ci_low": 100 * (stay - 1.96 * se),
            "stay_ci_high": 100 * (stay + 1.96 * se),
            "pct_expected_group_paired": 100 * expected,
            "expected_ci_low": 100 * (expected - 1.96 * se),
            "expected_ci_high": 100 * (expected + 1.96 * se),
        })
    return pd.DataFrame(rows, columns=PAIRED_COLUMNS)


def structural_table(sides: pd.DataFrame) -> pd.DataFrame:
    """Descriptive STAY share by barrier and crossing-signal layout, per cluster.

    Layouts mix dilemma types and group sizes, so differences are not causal effects.
    """
    def layout(row_barrier_sk, row_barrier_wk, signal_sk, signal_wk, pedped):
        if row_barrier_sk or row_barrier_wk:
            where = "STAY kills passengers" if row_barrier_sk else "SWERVE kills passengers"
            return f"barrier: {where}"
        if signal_sk == 0 and signal_wk == 0:
            return "pedestrians, no signal"
        return f"pedestrians, signal STAY-side={signal_sk} SWERVE-side={signal_wk}"

    keys = sides[["Barrier_sk", "Barrier_wk", "CrossingSignal_sk", "CrossingSignal_wk", "PedPed_sk"]]
    labels = [layout(*row) for row in keys.itertuples(index=False)]
    table = sides.assign(layout=labels)
    pivot = table.pivot_table(index="layout", columns="ClusterName_sk", values="chose_stay",
                              aggfunc=["mean", "size"])
    share = 100 * pivot["mean"].reindex(columns=list(CLUSTER_ORDER))
    counts = pivot["size"].reindex(columns=list(CLUSTER_ORDER))
    share.columns = [f"% STAY {c}" for c in share.columns]
    counts.columns = [f"n {c}" for c in counts.columns]
    return pd.concat([share.round(1), counts], axis=1).sort_values(f"n {CLUSTER_ORDER[0]}", ascending=False)


def data_quality(responses: pd.DataFrame) -> pd.DataFrame:
    """Missing values, duplicates and head-count consistency, one row per check."""
    sides = responses.groupby("ResponseID").NumberOfCharacters.agg(["min", "max"])
    diff = responses.groupby("ResponseID").DiffNumberOFCharacters.first()
    rows = [("missing cells", int(responses.isna().sum().sum()))]
    rows += [(f"missing in {col}", int(n)) for col, n in responses.isna().sum().items() if n]
    rows += [
        ("duplicated rows", int(responses.duplicated().sum())),
        ("duplicated (ResponseID, Intervention)",
         int(responses.duplicated(["ResponseID", "Intervention"]).sum())),
        ("responses where |N_a - N_b| != DiffNumberOFCharacters",
         int(((sides["max"] - sides["min"]) != diff).sum())),
        ("characters per side, min", int(responses.NumberOfCharacters.min())),
        ("characters per side, max", int(responses.NumberOfCharacters.max())),
        ("countries", int(responses.UserCountry3.nunique())),
    ]
    return pd.DataFrame(rows, columns=["check", "value"])


# ---------------------------------------------------------------------------
# Phase 2 completion: cultural divergence, catalog, repeats, country sensitivity
# ---------------------------------------------------------------------------


def culture_divergence(references: pd.DataFrame) -> pd.DataFrame:
    """D_culture = max - min of p_human(SWERVE) over the three clusters, per question."""
    per = references[references.cluster.isin(CLUSTER_ORDER)]
    wide = per.pivot(index="question_id", columns="cluster", values="p_swerve")
    wide = wide[[c for c in CLUSTER_ORDER if c in wide.columns]]
    labels = per.pivot(index="question_id", columns="cluster", values="majority_label")
    out = wide.add_prefix("p_swerve_")
    out["D_culture"] = wide.max(axis=1) - wide.min(axis=1)
    out["max_cluster"] = wide.idxmax(axis=1)
    out["min_cluster"] = wide.idxmin(axis=1)
    out["labels_differ"] = labels.nunique(axis=1) > 1
    return out.sort_values("D_culture", ascending=False).reset_index()


def configuration_catalog(sides: pd.DataFrame, questions: list[dict] | None = None) -> pd.DataFrame:
    """Every configuration in the sample: type, level and head count per side, and the
    barrier, crossing signal and PedPed fields the questions are matched on."""
    keys = ["ScenarioTypeStrict_sk", "AttributeLevel_sk", "NumberOfCharacters_sk", "Barrier_sk",
            "CrossingSignal_sk", "AttributeLevel_wk", "NumberOfCharacters_wk", "Barrier_wk",
            "CrossingSignal_wk", "PedPed_sk"]
    counts = sides.pivot_table(index=keys, columns="ClusterName_sk", values="chose_stay",
                               aggfunc=["size", "mean"])
    sizes = counts["size"].reindex(columns=list(CLUSTER_ORDER))
    shares = counts["mean"].reindex(columns=list(CLUSTER_ORDER))
    table = pd.DataFrame(index=counts.index)
    for cluster in CLUSTER_ORDER:
        table[f"n {cluster}"] = sizes[cluster]
        table[f"% STAY {cluster}"] = (100 * shares[cluster]).round(1)
    table = table.reset_index().rename(columns={
        "ScenarioTypeStrict_sk": "type", "AttributeLevel_sk": "stay_kills_level",
        "NumberOfCharacters_sk": "stay_kills_n", "Barrier_sk": "stay_kills_barrier",
        "CrossingSignal_sk": "stay_kills_signal", "AttributeLevel_wk": "swerve_kills_level",
        "NumberOfCharacters_wk": "swerve_kills_n", "Barrier_wk": "swerve_kills_barrier",
        "CrossingSignal_wk": "swerve_kills_signal", "PedPed_sk": "PedPed",
    })
    table.insert(1, "layout", [
        layout_name(*row) for row in table[["stay_kills_barrier", "swerve_kills_barrier",
                                            "stay_kills_signal", "swerve_kills_signal"]].itertuples(index=False)
    ])
    table["min n per cluster"] = table[[f"n {c}" for c in CLUSTER_ORDER]].min(axis=1)
    if questions is not None:
        names = ["type", "stay_kills_level", "stay_kills_n", "stay_kills_barrier", "stay_kills_signal",
                 "swerve_kills_level", "swerve_kills_n", "swerve_kills_barrier", "swerve_kills_signal", "PedPed"]
        used = {}
        for q in questions:
            c = match_criteria(q)
            key = (c["scenario_type"], c["level_sk"], c["n_sk"], c["barrier_sk"], c["signal_sk"],
                   c["level_wk"], c["n_wk"], c["barrier_wk"], c["signal_wk"], c["pedped"])
            used[key] = q["question_id"]
        table["question_id"] = [used.get(tuple(row)) for row in table[names].itertuples(index=False)]
    return table.sort_values(["type", "layout", "min n per cluster"],
                             ascending=[True, True, False]).reset_index(drop=True)


def user_repeat_check(sides: pd.DataFrame, questions: list[dict]) -> pd.DataFrame:
    """How often one user answered the same matched configuration more than once."""
    rows = []
    for q in questions:
        matched = sides[match_mask(sides, match_criteria(q))]
        per_user = matched.groupby("UserID_sk").size()
        rows.append({
            "question_id": q["question_id"],
            "responses": len(matched),
            "users": int(per_user.size),
            "max_per_user": int(per_user.max()) if len(per_user) else 0,
            "users_with_repeats": int((per_user > 1).sum()),
            "responses_from_repeaters": int(per_user[per_user > 1].sum()),
        })
    return pd.DataFrame(rows)


def country_type_preferences(questions_table: pd.DataFrame, country: str) -> pd.DataFrame:
    """Type-level preference for one country next to its cluster, with Wilson intervals."""
    clean = questions_table[questions_table.clean_contrast]
    in_country = clean[clean.country == country]
    if in_country.empty:
        return pd.DataFrame()
    cluster = in_country.cluster.iloc[0]
    rows = []
    for kind in TYPE_ORDER:
        for name, group in ((country_label(country), in_country), (cluster, clean[clean.cluster == cluster])):
            group = group[group.type == kind]
            if len(group):
                rows.append(_share_row(kind, name, group))
    return pd.DataFrame(rows)
