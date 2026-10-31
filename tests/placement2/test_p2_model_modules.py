"""SYNTHETIC_TEST_DATA: toy prompts, a character tokenizer and fake logits; no model or human data."""

from __future__ import annotations

import math

import pandas as pd
import pytest
import torch

from multilingual_mechinterp.placement2 import behavior as bh
from multilingual_mechinterp.placement2 import choices as ch
from multilingual_mechinterp.placement2 import jlens_analysis as jl
from multilingual_mechinterp.placement2 import prompts as pr

PROMPT = "Intro.\n\nSTAY: kill A.\nSWERVE: kill B.\n\nRespond with only STAY or SWERVE."


class CharTokenizer:
    """One token per character; id 0 is BOS."""

    vocab = [chr(i) for i in range(32, 127)] + ["\n"]

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None,
                 return_offsets_mapping=False):
        ids = [1 + self.vocab.index(c) for c in text]
        offsets = [(i, i + 1) for i in range(len(text))]
        if add_special_tokens:
            ids, offsets = [0, *ids], [(0, 0), *offsets]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out

    def decode(self, ids):
        return "".join("<bos>" if i == 0 else self.vocab[i - 1] for i in ids)


TOK = CharTokenizer()
VOCAB = len(TOK.vocab) + 1


def test_present_swaps_option_lines_only():
    swapped = pr.present(PROMPT, "SWERVE_first")
    lines = swapped.split("\n")
    assert lines[2].startswith("SWERVE:") and lines[3].startswith("STAY:")
    assert pr.present(swapped, "STAY_first") == PROMPT
    assert pr.present(PROMPT, "STAY_first") == PROMPT
    with pytest.raises(ValueError):
        pr.present("no options here", "STAY_first")


def test_validate_persian_blocks_bad_rows():
    english = [{"question_id": "Q1", "prompt_en": PROMPT}, {"question_id": "Q2", "prompt_en": PROMPT}]
    good = "مقدمه.\n\nSTAY: کشتن الف.\nSWERVE: کشتن ب.\n\nفقط پاسخ دهید."
    rows = [
        {"question_id": "Q1", "language": "fa", "prompt_fa": good},
        {"question_id": "Q3", "language": "fa", "prompt_fa": "TODO"},
    ]
    problems = pr.validate_persian(english, rows)
    assert any("missing Persian prompts for ['Q2']" in p for p in problems)
    assert any("Q3" in p and "TODO" in p for p in problems)
    assert not any(p.startswith("Q1") for p in problems)
    template = pr.persian_template_rows(english)
    assert pr.validate_persian(english, template)  # empty template never validates


def test_build_variants_skips_missing_language():
    english = [{"question_id": "Q1", "prompt_en": PROMPT}]
    rows = pr.build_variants(english, None)
    assert {(r["language"], r["order"]) for r in rows} == {("en", "STAY_first"), ("en", "SWERVE_first")}


def test_question_fields_and_persian_metadata_parity():
    side = {"count": 1, "attribute_level": "Old", "Barrier": 0, "CrossingSignal": 0}
    english = [{"question_id": "Q1", "question_en": PROMPT, "expected_action": "STAY", "stay_kills": side}]
    fa_text = "مقدمه.\n\nSTAY: کشتن الف.\nSWERVE: کشتن ب.\n\nفقط پاسخ دهید."
    persian = [{"question_id": "Q1", "language": "fa", "question_fa": fa_text, "expected_action": "STAY",
                "stay_kills": side}]
    assert pr.validate_persian(english, persian) == []
    rows = pr.build_variants(english, persian)
    assert {r["language"] for r in rows} == {"en", "fa"}
    assert pr.prompt_text(english[0], "en") == PROMPT
    persian[0]["expected_action"] = "SWERVE"
    assert pr.validate_persian(english, persian) == ["Q1: expected_action differs from the English file"]


def test_question_families_reads_per_side_levels():
    question = {"scenario_type": "Age", "PedPed": 0,
                "stay_kills": {"count": 2, "attribute_level": "Old", "Barrier": 1, "CrossingSignal": 0},
                "swerve_kills": {"count": 2, "attribute_level": "Young", "Barrier": 0, "CrossingSignal": 0}}
    fams = jl.question_families(question, {"Old": "OLD", "Young": "YOUNG"})
    assert fams == {"stay_kills": "OLD", "swerve_kills": "YOUNG"}


def _logits_fn_preferring(char):
    target = 1 + TOK.vocab.index(char)

    def run(ids):
        logits = torch.zeros(ids.shape[-1], VOCAB)
        logits[:, target] = 3.0
        return logits
    return run


def test_full_sequence_scoring_sums_continuation_tokens():
    result = ch.score_prompt(_logits_fn_preferring("S"), TOK, PROMPT, answer_prefix="\nA:", separator="")
    # Uniform except 'S'; both labels start with 'S', STAY has 4 tokens and SWERVE 6.
    other = -math.log(VOCAB - 1 + math.exp(3.0))
    s_tok = 3.0 + other
    assert result["logp"]["STAY"] == pytest.approx(s_tok + 3 * other, rel=1e-5)
    assert result["logp"]["SWERVE"] == pytest.approx(s_tok + 5 * other, rel=1e-5)
    assert result["decision"] == "STAY"
    assert result["p_stay"] + result["p_swerve"] == pytest.approx(1.0)
    assert result["n_continuation_tokens"] == {"STAY": 4, "SWERVE": 6}
    assert result["top_next_tokens"][0]["token"] == "S"


def test_two_way_margin_and_entropy():
    out = ch.two_way({"STAY": math.log(0.25), "SWERVE": math.log(0.75)})
    assert out["p_swerve"] == pytest.approx(0.75)
    assert out["decision"] == "SWERVE"
    assert out["entropy_bits"] == pytest.approx(0.8113, abs=1e-4)


def _scores():
    rows = []
    values = {  # (question, language) -> p_swerve for STAY_first, SWERVE_first
        ("Q1", "en"): (0.2, 0.3), ("Q1", "fa"): (0.7, 0.8),   # D1
        ("Q2", "en"): (0.8, 0.9), ("Q2", "fa"): (0.3, 0.2),   # D2
        ("Q3", "en"): (0.6, 0.7), ("Q3", "fa"): (0.9, 0.9),   # B1
        ("Q4", "en"): (0.6, 0.6), ("Q4", "fa"): (0.61, 0.61), # B2
    }
    for (qid, lang), (a, b) in values.items():
        for order, p in (("STAY_first", a), ("SWERVE_first", b)):
            rows.append({"question_id": qid, "language": lang, "order": order, "p_stay": 1 - p,
                         "p_swerve": p, "margin_stay_minus_swerve": math.log((1 - p) / p),
                         "entropy_bits": bh.entropy_bits(p), "decision": "SWERVE" if p > 0.5 else "STAY"})
    return pd.DataFrame(rows)


def test_order_summary_and_groups():
    summary = bh.order_summary(_scores())
    q1 = summary[(summary.question_id == "Q1") & (summary.language == "en")].iloc[0]
    assert q1.p_swerve == pytest.approx(0.25)
    assert q1.position_bias == pytest.approx(0.1)
    comparison = bh.language_comparison(summary).set_index("question_id")
    assert comparison.group.to_dict() == {"Q1": "D1", "Q2": "D2", "Q3": "B1", "Q4": "B2"}
    assert comparison.loc["Q1", "delta_p_lang"] == pytest.approx(-0.5)
    assert bh.js_divergence(0.5, 0.5) == pytest.approx(0.0)
    picked = bh.select_cases(comparison.reset_index(), per_group=1)
    assert list(picked.group) == ["D1", "D2", "B1", "B2"]


def test_alignment_excludes_undetermined():
    summary = bh.order_summary(_scores())
    refs = pd.DataFrame([
        {"question_id": "Q1", "cluster": "Western", "p_swerve": 0.2, "majority_label": "STAY",
         "label_strength": "clear", "n_matched_responses": 100},
        {"question_id": "Q2", "cluster": "Western", "p_swerve": 0.5, "majority_label": "UNDETERMINED",
         "label_strength": "tie", "n_matched_responses": 100},
    ])
    table = bh.alignment_table(summary, refs)
    en_q1 = table[(table.question_id == "Q1") & (table.language == "en")].iloc[0]
    assert en_q1.agrees_with_majority
    assert en_q1.abs_error == pytest.approx(0.05)
    assert table[table.question_id == "Q2"].agrees_with_majority.isna().all()
    legacy = bh.alignment_table(summary, refs.assign(p_stay=1 - refs.p_swerve).drop(columns="p_swerve"))
    assert legacy.abs_error.tolist() == pytest.approx(table.abs_error.tolist())
    summary_rows = bh.alignment_summary(table, n_boot=50).set_index("language")
    assert summary_rows.loc["en", "n_labelled"] == 1
    assert summary_rows.loc["fa", "agreement_labelled"] == 0.0


def test_format_filter_and_family_scores():
    assert jl.is_format_token(" STAY") and jl.is_format_token(":") and jl.is_format_token("<bos>")
    assert not jl.is_format_token(" children")
    tops = [{"token": " SWERVE"}, {"token": " dog"}, {"token": "."}, {"token": " cat"}]
    assert [t["token"] for t in jl.content_tokens(tops, k=5)] == [" dog", " cat"]

    families = {"single": {"en": ["a", "b"]}, "multi": {"en": ["ab"]}}
    ids, dropped = jl.family_token_ids(TOK, families, "en")
    assert dropped == {"single": [], "multi": ["ab"]}
    logits = torch.zeros(VOCAB)
    logits[ids["single"]] = 5.0
    scores = jl.family_scores(logits, ids)
    assert scores["multi"] is None
    assert scores["single"] > math.log(0.5)
    assert jl.family_ranks(logits, ids)["single"] == 1


def test_option_line_positions():
    pos = jl.option_line_positions(TOK, PROMPT)
    stay_end = PROMPT.index("kill A.") + len("kill A.") - 1
    assert pos["STAY"] == stay_end + 1  # +1 for BOS
    assert pos["last"] == len(PROMPT)
    assert PROMPT[pos["SWERVE"] - 1] == "."


def test_backprojected_direction_follows_jacobian():
    weight = torch.eye(4)
    jac = torch.tensor([[0.0, 1.0, 0, 0], [1.0, 0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])
    direction = jl.backprojected_direction(jac, weight, [0])
    # J maps source dim 1 to final dim 0, so raising token 0 needs source dim 1.
    assert torch.allclose(direction, torch.tensor([0.0, 1.0, 0, 0]))
    residual = torch.tensor([3.0, 4.0, 0, 0])
    edited = jl.edited_residual(residual, direction, 0.1)
    assert (edited - residual).norm() == pytest.approx(0.5)
    assert jl.random_direction(4, 0).norm() == pytest.approx(1.0)
