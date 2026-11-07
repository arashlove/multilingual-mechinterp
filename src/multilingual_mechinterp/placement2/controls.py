"""Behavioural controls for Placement 2 (v2): instruction-tuned model, chat template,
neutral A/B labels, generated answers and mirrored pairs.

Every scored variant is stored as one row with ``p_swerve`` (the two-way probability of
the option that means SWERVE, whatever its label), ``swerve_first`` and, for A/B prompts,
``mapping``. Summaries average over all variants of a question, so position and label
preferences cancel; the size of each is reported separately.
"""

from __future__ import annotations

import re

import pandas as pd
import torch

from .behavior import entropy_bits

CONDITION_COLUMNS = ("model", "format")


def chat_text(tokenizer, prompt: str) -> str:
    """The prompt as one user turn with the generation prompt appended (Gemma chat format).
    The returned text already contains BOS; tokenise it with ``add_special_tokens=False``."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


_LEADING = re.compile(r"^[\s*_`\"'«»“”‘’()\[\]#>.:-]+")


def parse_answer(text: str, labels) -> str | None:
    """The label the generated text starts with, or None.

    Leading whitespace, markdown and quotes are ignored and the match is case-insensitive;
    the label must be a whole word ("A", "A.", "**SWERVE**" count; "Answer: A" does not).
    """
    rest = _LEADING.sub("", text or "")
    match = re.match(r"[A-Za-z]+", rest)
    if not match:
        return None
    word = match.group(0).upper()
    for label in labels:
        if word == label.upper():
            return label
    return None


def label_token_ids(tokenizer, labels, separator: str = "") -> dict[str, int] | None:
    """Token id of each label when every label is one token, else None."""
    ids = {}
    for label in labels:
        toks = tokenizer(separator + label, add_special_tokens=False)["input_ids"]
        if len(toks) != 1:
            return None
        ids[label] = int(toks[0])
    return ids


def two_way_from_row(logprobs: torch.Tensor, ids: dict[str, int]) -> dict:
    """Two-way probabilities of single-token labels from one next-token log-prob row."""
    labels = list(ids)
    values = torch.stack([logprobs[ids[label]] for label in labels]).float()
    probs = torch.softmax(values, dim=0)
    return {
        "p": {label: float(p) for label, p in zip(labels, probs)},
        "first_token_mass": float(logprobs.exp()[list(ids.values())].sum()),
        "top1_is_label_start": int(torch.argmax(logprobs)) in set(ids.values()),
    }


def generate_answer(hf_model, tokenizer, context_text: str, *, max_new_tokens: int = 6,
                    max_length: int = 1024) -> tuple[str, torch.Tensor, int]:
    """Greedy continuation of chat-template text: (decoded text, first-step log-probs, context length)."""
    enc = tokenizer(context_text, add_special_tokens=False, truncation=True, max_length=max_length,
                    return_tensors="pt")
    device = next(hf_model.parameters()).device
    ids = enc["input_ids"].to(device)
    with torch.no_grad():
        out = hf_model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_logits=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    new = out.sequences[0, ids.shape[1]:]
    text = tokenizer.decode(new, skip_special_tokens=True)
    first = torch.log_softmax(out.logits[0][0].float(), dim=-1).cpu()
    return text, first, int(ids.shape[1])


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def variant_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Per condition x question x language: p(SWERVE) averaged over every variant, with
    the position effect, the label-mapping effect and answer validity.

    ``position_bias`` = p(SWERVE | SWERVE listed first) - p(SWERVE | listed second).
    ``mapping_gap`` (A/B only) = p(SWERVE | SWERVE is A) - p(SWERVE | SWERVE is B), the
    pull towards the letter A. ``*_flips`` are True when the two halves of a split give
    different decisions.
    """
    keys = [*CONDITION_COLUMNS, "question_id", "language"]
    out = []
    for key, group in rows.groupby(keys, sort=False):
        p = float(group.p_swerve.mean())
        first = group[group.swerve_first].p_swerve.mean()
        second = group[~group.swerve_first].p_swerve.mean()
        entry = dict(zip(keys, key))
        entry.update({
            "n_variants": len(group),
            "p_swerve": p,
            "p_stay": 1 - p,
            "decision": "SWERVE" if p > 0.5 else "STAY",
            "entropy_bits": entropy_bits(p),
            "position_bias": first - second,
            "order_flips_decision": (first > 0.5) != (second > 0.5),
            "median_first_token_mass": float(group.first_token_mass.median()),
        })
        if "mapping" in group and group.mapping.notna().any():
            a_is_swerve = group[group.mapping == "A_is_SWERVE"].p_swerve.mean()
            b_is_swerve = group[group.mapping == "A_is_STAY"].p_swerve.mean()
            entry["mapping_gap"] = a_is_swerve - b_is_swerve
            entry["mapping_flips_decision"] = (a_is_swerve > 0.5) != (b_is_swerve > 0.5)
            entry["mean_p_label_A"] = float(group.p_label_A.mean())
        if "valid" in group and group.valid.notna().any():
            entry["invalid_rate"] = float(1 - group.valid.astype(float).mean())
            entry["generated_decisions"] = " ".join(sorted(group.parsed_action.fillna("INVALID")))
        out.append(entry)
    return pd.DataFrame(out)


def condition_overview(summary: pd.DataFrame, questions: list[dict]) -> pd.DataFrame:
    """Per condition x language: decision share, spread, consequence sensitivity and biases.

    ``consequence_sensitivity`` is mean p(SWERVE) on questions designed so that SWERVE
    spares the expected group minus mean p(SWERVE) where STAY does. A model that reads
    the consequences gives a clearly positive value; a constant action preference gives 0.
    """
    expected = {q["question_id"]: q["expected_action"] for q in questions}
    table = summary.assign(expected_action=summary.question_id.map(expected))
    rows = []
    for key, group in table.groupby([*CONDITION_COLUMNS, "language"], sort=False):
        swerve_expected = group[group.expected_action == "SWERVE"].p_swerve.mean()
        stay_expected = group[group.expected_action == "STAY"].p_swerve.mean()
        entry = dict(zip([*CONDITION_COLUMNS, "language"], key))
        entry.update({
            "n_questions": len(group),
            "share_swerve_decisions": float((group.decision == "SWERVE").mean()),
            "mean_p_swerve": float(group.p_swerve.mean()),
            "sd_p_swerve": float(group.p_swerve.std(ddof=1)),
            "min_p_swerve": float(group.p_swerve.min()),
            "max_p_swerve": float(group.p_swerve.max()),
            "consequence_sensitivity": float(swerve_expected - stay_expected),
            "decision_matches_expected_action": float((group.decision == group.expected_action).mean()),
            "mean_position_bias": float(group.position_bias.mean()),
            "order_flips": int(group.order_flips_decision.sum()),
            "median_first_token_mass": float(group.median_first_token_mass.median()),
        })
        if "mapping_gap" in group and group.mapping_gap.notna().any():
            entry["mean_p_label_A"] = float(group.mean_p_label_A.mean())
            entry["mean_mapping_gap"] = float(group.mapping_gap.mean())
            entry["mapping_flips"] = int(group.mapping_flips_decision.sum())
        if "invalid_rate" in group and group.invalid_rate.notna().any():
            entry["invalid_rate"] = float(group.invalid_rate.mean())
        rows.append(entry)
    return pd.DataFrame(rows)


def target_sparing(p_swerve: float, expected_action: str) -> float:
    """Probability of sparing the group named in ``expected_preference``."""
    return p_swerve if expected_action == "SWERVE" else 1 - p_swerve


def mirrored_pairs(summary: pd.DataFrame, questions: list[dict], references: pd.DataFrame | None = None) -> pd.DataFrame:
    """Model and human behaviour on mirrored pairs (``counterbalance_pair_id``).

    In a mirrored pair the same two groups swap paths. ``p_target_*`` is the probability
    of sparing the expected group in each orientation; their mean is the group preference
    with the action held fixed, and the mean p(STAY) over the pair is the preference for
    not intervening with the groups held fixed. A model that follows the groups keeps
    ``p_target`` high in both orientations; a model with only an action preference has
    ``p_target`` high in one and low in the other.
    """
    by_id = {q["question_id"]: q for q in questions}
    pairs: dict[str, list[str]] = {}
    for q in questions:
        if q.get("counterbalance_pair_id"):
            pairs.setdefault(q["counterbalance_pair_id"], []).append(q["question_id"])
    rows = []

    def add(source: str, condition: dict, values: dict[str, float]):
        for pair_id, ids in pairs.items():
            if len(ids) != 2 or not all(i in values for i in ids):
                continue
            first, second = sorted(ids)
            t1 = target_sparing(values[first], by_id[first]["expected_action"])
            t2 = target_sparing(values[second], by_id[second]["expected_action"])
            rows.append({
                "pair_id": pair_id, "source": source, **condition,
                "questions": f"{first} / {second}",
                "p_swerve_first": values[first], "p_swerve_second": values[second],
                "p_target_first": t1, "p_target_second": t2,
                "group_preference": (t1 + t2) / 2,
                "inaction_preference": 1 - (values[first] + values[second]) / 2,
                "target_spared_both": t1 > 0.5 and t2 > 0.5,
            })

    for key, group in summary.groupby([*CONDITION_COLUMNS, "language"], sort=False):
        add("model", dict(zip([*CONDITION_COLUMNS, "language"], key)),
            dict(zip(group.question_id, group.p_swerve)))
    if references is not None:
        ref = references if "p_swerve" in references else references.assign(p_swerve=1 - references.p_stay)
        for cluster, group in ref.groupby("cluster", sort=False):
            add("human", {"model": "human", "format": cluster, "language": None},
                dict(zip(group.question_id, group.p_swerve)))
    return pd.DataFrame(rows)


def compare_runs(new: pd.DataFrame, old: pd.DataFrame, *, on=("question_id", "language", "order")) -> dict:
    """Max and mean |Δp(SWERVE)| between two score tables on their shared variants."""
    merged = new.merge(old, on=list(on), suffixes=("_new", "_old"))
    if merged.empty:
        return {"n": 0}
    diff = (merged.p_swerve_new - merged.p_swerve_old).abs()
    return {
        "n": len(merged),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "decision_changes": int(((merged.p_swerve_new > 0.5) != (merged.p_swerve_old > 0.5)).sum()),
    }
