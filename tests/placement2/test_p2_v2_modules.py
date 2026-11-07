"""SYNTHETIC_TEST_DATA: toy prompts, toy questions, a character tokenizer and fake logits."""

from __future__ import annotations

import math

import pandas as pd
import pytest
import torch

from multilingual_mechinterp.placement2 import calibration as cal
from multilingual_mechinterp.placement2 import choices as ch
from multilingual_mechinterp.placement2 import controls as ct
from multilingual_mechinterp.placement2 import human_data as hd
from multilingual_mechinterp.placement2 import prompts as pr
from multilingual_mechinterp.placement2.experiment import local_commit

PROMPT = "Intro.\n\nSTAY: kill two people.\nSWERVE: kill five people.\n\nRespond with only STAY or SWERVE."
FA_PROMPT = "مقدمه.\n\nSTAY: کشتن دو نفر.\nSWERVE: کشتن پنج نفر.\n\nفقط با STAY یا SWERVE پاسخ دهید."


class CharTokenizer:
    """One token per character; id 0 is BOS."""

    vocab = [chr(i) for i in range(32, 127)] + ["\n"]

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None):
        ids = [1 + self.vocab.index(c) for c in text]
        return {"input_ids": [0, *ids] if add_special_tokens else ids}

    def decode(self, ids):
        return "".join("<bos>" if i == 0 else self.vocab[i - 1] for i in ids)


TOK = CharTokenizer()
VOCAB = len(TOK.vocab) + 1


def _counting_logits_fn(favour):
    calls = []
    target = 1 + TOK.vocab.index(favour)

    def run(ids):
        calls.append(ids.shape[-1])
        logits = torch.zeros(ids.shape[-1], VOCAB)
        logits[:, target] = 2.0
        return logits
    return run, calls


def test_neutral_prompt_relabels_and_orders():
    a_stay = pr.neutral_prompt(PROMPT, "A_is_STAY", "A_first").split("\n")
    assert a_stay[2] == "A: kill two people." and a_stay[3] == "B: kill five people."
    assert a_stay[-1] == "Respond with only A or B."
    b_first = pr.neutral_prompt(PROMPT, "A_is_STAY", "B_first").split("\n")
    assert b_first[2] == "B: kill five people." and b_first[3] == "A: kill two people."
    a_swerve = pr.neutral_prompt(PROMPT, "A_is_SWERVE", "A_first").split("\n")
    assert a_swerve[2] == "A: kill five people." and a_swerve[3] == "B: kill two people."
    fa = pr.neutral_prompt(FA_PROMPT, "A_is_SWERVE", "B_first")
    assert "STAY" not in fa and "SWERVE" not in fa
    assert fa.split("\n")[-1] == "فقط با A یا B پاسخ دهید."
    with pytest.raises(ValueError):
        pr.neutral_prompt(PROMPT, "A_is_LEFT", "A_first")


def test_build_neutral_variants_marks_swerve_label_and_position():
    english = [{"question_id": "Q1", "question_en": PROMPT}]
    persian = [{"question_id": "Q1", "question_fa": FA_PROMPT}]
    rows = pr.build_neutral_variants(english, persian)
    assert len(rows) == 8
    for row in rows:
        lines = row["prompt"].split("\n")
        first_label = lines[2].split(":")[0]
        swerve_text = "five" if row["language"] == "en" else "پنج"
        swerve_line = next(line for line in lines[2:4] if swerve_text in line)
        assert swerve_line.startswith(row["swerve_label"] + ":")
        assert row["swerve_first"] == (first_label == row["swerve_label"])


def test_translation_review_rows_split_parts():
    english = [{"question_id": "Q1", "scenario_type": "Utilitarian", "question_en": PROMPT}]
    rows = pr.translation_review_rows(english, [{"question_id": "Q1", "question_fa": FA_PROMPT}])
    assert [r["part"] for r in rows] == ["scenario", "STAY line", "SWERVE line", "instruction"]
    assert rows[1]["persian"].startswith("STAY:") and "casualty" in rows[1]["review_topics"]
    assert all(r["reviewer"] == "" for r in rows)


@pytest.mark.parametrize("text,expected", [
    ("A", "A"), (" b.", "B"), ("**A**", "A"), ("SWERVE", None), ("Answer: A", None), ("", None),
    ("Apple", None),
])
def test_parse_answer_ab(text, expected):
    assert ct.parse_answer(text, ("A", "B")) == expected


def test_parse_answer_actions():
    assert ct.parse_answer("**SWERVE**\n", ("STAY", "SWERVE")) == "SWERVE"
    assert ct.parse_answer("Stay.", ("STAY", "SWERVE")) == "STAY"
    assert ct.parse_answer("I would swerve", ("STAY", "SWERVE")) is None


def test_single_token_labels_use_one_forward():
    fn, calls = _counting_logits_fn("A")
    out = ch.score_prompt(fn, TOK, "Q", answer_prefix=":", separator="", labels=("A", "B"))
    assert calls == [3] and out["n_forwards"] == 1
    other = -math.log(VOCAB - 1 + math.exp(2.0))
    assert out["logp"]["A"] == pytest.approx(2.0 + other, rel=1e-5)
    assert out["logp"]["B"] == pytest.approx(other, rel=1e-5)
    assert out["top1_is_label_start"]


def test_mixed_length_labels_match_separate_forwards():
    fn, calls = _counting_logits_fn("S")
    out = ch.score_prompt(fn, TOK, "Q", answer_prefix=":", separator="", labels=("S", "SWE"))
    assert out["n_forwards"] == 1 and len(calls) == 1
    ctx = ch.context_ids(TOK, "Q:")
    for label in ("S", "SWE"):
        total, _, _ = ch.continuation_logprob(fn, ctx, ch.continuation_ids(TOK, label))
        assert out["logp"][label] == pytest.approx(total, rel=1e-6)


class _Out:
    def __init__(self, logits):
        self.logits = logits


class _FakeLM(torch.nn.Module):
    """Logits depend on position and token so slicing errors show up."""

    def __init__(self, keep_arg: bool):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.keep_arg = keep_arg

    def _logits(self, input_ids):
        t = input_ids.shape[-1]
        pos = torch.arange(t, dtype=torch.float32)[None, :, None]
        tok = input_ids.float()[..., None]
        return torch.sin(pos * 0.7 + tok * 0.3 + torch.arange(VOCAB, dtype=torch.float32))

    def forward(self, input_ids, **kwargs):
        if "logits_to_keep" in kwargs:
            if not self.keep_arg:
                raise TypeError("unexpected keyword argument 'logits_to_keep'")
            return _Out(self._logits(input_ids)[:, -kwargs["logits_to_keep"]:])
        return _Out(self._logits(input_ids))


def test_hf_logits_fn_keeps_only_scored_positions():
    results = []
    for keep_arg in (True, False):
        fn = ch.hf_logits_fn(_FakeLM(keep_arg))
        results.append(ch.score_prompt(fn, TOK, "Hello", answer_prefix=":", separator=" ",
                                       labels=("STAY", "SWERVE")))
    full = ch.score_prompt(lambda ids: _FakeLM(False)._logits(ids)[0], TOK, "Hello", answer_prefix=":",
                           separator=" ", labels=("STAY", "SWERVE"))
    for out in results:
        assert out["logp"] == pytest.approx(full["logp"], rel=1e-6)


def test_context_ids_without_special_tokens():
    assert ch.context_ids(TOK, "ab", add_special_tokens=False) == [1 + TOK.vocab.index("a"), 1 + TOK.vocab.index("b")]


def test_two_way_from_row():
    row = torch.log_softmax(torch.tensor([0.0, 2.0, 0.0, 1.0]), dim=0)
    out = ct.two_way_from_row(row, {"A": 1, "B": 3})
    assert out["p"]["A"] == pytest.approx(math.exp(2) / (math.exp(2) + math.exp(1)))
    assert out["top1_is_label_start"]


def _rows():
    rows = []
    # Q1 favours SWERVE strongly; Q2 follows the letter A only.
    for qid, p_by_mapping in (("Q1", {"A_is_STAY": 0.8, "A_is_SWERVE": 0.9}),
                              ("Q2", {"A_is_STAY": 0.3, "A_is_SWERVE": 0.7})):
        for mapping, p in p_by_mapping.items():
            for swerve_first in (True, False):
                swerve_label = "A" if mapping == "A_is_SWERVE" else "B"
                rows.append({"model": "it", "format": "neutral_ab", "question_id": qid, "language": "en",
                             "mapping": mapping, "swerve_first": swerve_first, "p_swerve": p,
                             "p_label_A": p if swerve_label == "A" else 1 - p, "first_token_mass": 0.9,
                             "valid": True, "parsed_action": "SWERVE" if p > 0.5 else "STAY"})
    return pd.DataFrame(rows)


def test_variant_summary_mapping_and_overview():
    summary = ct.variant_summary(_rows()).set_index("question_id")
    assert summary.loc["Q1", "p_swerve"] == pytest.approx(0.85)
    assert summary.loc["Q2", "mapping_gap"] == pytest.approx(0.4)
    assert bool(summary.loc["Q2", "mapping_flips_decision"])
    assert summary.loc["Q1", "invalid_rate"] == 0
    questions = [{"question_id": "Q1", "expected_action": "SWERVE"},
                 {"question_id": "Q2", "expected_action": "STAY"}]
    overview = ct.condition_overview(summary.reset_index(), questions).iloc[0]
    assert overview.consequence_sensitivity == pytest.approx(0.85 - 0.5)
    assert overview.mapping_flips == 1


def test_mirrored_pairs_model_and_human():
    questions = [{"question_id": "Q1", "expected_action": "STAY", "counterbalance_pair_id": "P"},
                 {"question_id": "Q2", "expected_action": "SWERVE", "counterbalance_pair_id": "P"}]
    summary = pd.DataFrame([{"model": "base", "format": "stay_swerve", "question_id": "Q1", "language": "en", "p_swerve": 0.7},
                            {"model": "base", "format": "stay_swerve", "question_id": "Q2", "language": "en", "p_swerve": 0.7}])
    refs = pd.DataFrame([{"question_id": "Q1", "cluster": "Western", "p_stay": 0.75},
                         {"question_id": "Q2", "cluster": "Western", "p_stay": 0.25}])
    table = ct.mirrored_pairs(summary, questions, refs).set_index("source")
    model, human = table.loc["model"], table.loc["human"]
    assert model.group_preference == pytest.approx(0.5) and not model.target_spared_both
    assert model.inaction_preference == pytest.approx(0.3)
    assert human.group_preference == pytest.approx(0.75) and human.target_spared_both


def _question(qid, typ, n_sk, n_wk, level_sk="Less", level_wk="More", text=None):
    side = {"Barrier": 0, "CrossingSignal": 0, "location": "pedestrians"}
    return {"question_id": qid, "scenario_type": typ, "PedPed": 1, "expected_action": "STAY",
            "stay_kills": {**side, "count": n_sk, "attribute_level": level_sk},
            "swerve_kills": {**side, "count": n_wk, "attribute_level": level_wk},
            "question_en": text}


def test_substitute_counts_leaves_non_count_words():
    text = "A car on a road with two possible routes. Two people on one side, five people ahead."
    out, n = cal.substitute_counts(text, "en", {2: 3, 5: 4})
    assert out == "A car on a road with two possible routes. Three people on one side, four people ahead."
    assert n == 2
    fa = "یک خودروی خودران با دو مسیر. دو نفر و هیچ‌یک از دو گروه."
    out_fa, n_fa = cal.substitute_counts(fa, "fa", {2: 4})
    assert out_fa == "یک خودروی خودران با دو مسیر. چهار نفر و هیچ‌یک از دو گروه."
    assert n_fa == 1


def test_count_substitution_avoids_study_configurations():
    en_text = "Two people ahead, five people on the other route.\n\nSTAY: kill two people.\nSWERVE: kill five people.\n\nRespond with only STAY or SWERVE."
    fa_text = "دو نفر جلو و پنج نفر در مسیر دیگر.\n\nSTAY: کشتن دو نفر.\nSWERVE: کشتن پنج نفر.\n\nفقط با STAY یا SWERVE پاسخ دهید."
    q1 = _question("U1", "Utilitarian", 2, 5, text=en_text)
    q2 = _question("U2", "Utilitarian", 2, 4, text=en_text.replace("five", "four"))
    q3 = _question("U3", "Utilitarian", 1, 1)
    english = [q1, q2, q3]
    persian = [{"question_id": "U1", "question_fa": fa_text},
               {"question_id": "U2", "question_fa": fa_text.replace("پنج", "چهار")},
               {"question_id": "U3", "question_fa": "x"}]
    rows = cal.count_substitution_prompts(english, persian, n_fit=1, n_holdout=1)
    study = {cal.signature(q) for q in english}
    produced = {tuple(r["new_counts"]) for r in rows}
    assert (2, 4) not in produced and (2, 5) not in produced
    assert all(cal.signature(q1, *r["new_counts"]) not in study for r in rows)
    assert {r["source_question_id"] for r in rows} == {"U1", "U2"}
    assert [r["split"] for r in rows[:2]] == ["fit", "holdout"]
    assert all(r["flag"] == cal.FLAG for r in rows)
    first = rows[0]
    assert first["language"] == "en" and "STAY:" in first["prompt"]


def test_respondent_level_uncertainty_inflates_with_repeats():
    question = _question("Q", "Age", 2, 2, "Old", "Young")
    base = {"ScenarioTypeStrict_sk": "Age", "AttributeLevel_sk": "Old", "AttributeLevel_wk": "Young",
            "NumberOfCharacters_sk": 2, "NumberOfCharacters_wk": 2, "PedPed_sk": 1, "Barrier_sk": 0,
            "Barrier_wk": 0, "CrossingSignal_sk": 0, "CrossingSignal_wk": 0, "ClusterName_sk": "Western"}
    rows = []
    for i in range(10):  # 10 users, one answer each
        rows.append({**base, "UserID_sk": f"u{i}", "chose_stay": i < 6})
    independent = hd.respondent_level_uncertainty(pd.DataFrame(rows), [question])
    rows = []
    for i in range(10):  # 5 users answering twice, always the same way
        rows.append({**base, "UserID_sk": f"u{i // 2}", "chose_stay": i < 6})
    repeated = hd.respondent_level_uncertainty(pd.DataFrame(rows), [question])
    one = independent[independent.cluster == "Western"].iloc[0]
    two = repeated[repeated.cluster == "Western"].iloc[0]
    assert one.p_stay == two.p_stay == pytest.approx(0.6)
    assert one.users_with_repeats == 0 and two.users_with_repeats == 5
    assert two.se_cluster_robust > one.se_cluster_robust
    assert two.design_effect > one.design_effect


def test_local_commit_reads_download_metadata(tmp_path):
    meta = tmp_path / ".cache" / "huggingface" / "download"
    meta.mkdir(parents=True)
    (meta / "config.json.metadata").write_text("a" * 40 + "\netag\n1.0\n", encoding="utf-8")
    assert local_commit(tmp_path) == "a" * 40
    snap = tmp_path / "snapshots" / ("b" * 40)
    snap.mkdir(parents=True)
    assert local_commit(snap) == "b" * 40
    assert local_commit(tmp_path / "missing") is None
