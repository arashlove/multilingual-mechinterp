"""Dilemma-domain calibration prompts for the Jacobian Lens, by head-count substitution.

Each prompt is one of the researcher's own English or Persian questions with the head
counts changed (for example "two people" -> "four people"). No new wording is written:
only count words directly followed by a person or animal noun are replaced, so phrases
such as "two possible routes", "دو مسیر" or "یک خودروی خودران" are left alone. Counts
stay between 2 and 5 (questions with one character per side are skipped, because the
nouns would need singular/plural changes), a larger group stays larger, and a
configuration is kept only if none of the 40 study questions uses it. Every row is
flagged as auto-generated from a study template.
"""

from __future__ import annotations

import re

from .human_data import match_criteria
from .prompts import prompt_text

NUMBER_WORDS = {
    "en": {2: "two", 3: "three", 4: "four", 5: "five"},
    "fa": {2: "دو", 3: "سه", 4: "چهار", 5: "پنج"},
}
NOUN_STARTS = {
    "en": ("people", "person", "humans", "human", "men", "man", "women", "woman", "pets", "pet",
           "older", "younger", "larger-bodied", "physically"),
    "fa": ("نفر", "انسان", "زن", "مرد", "فرد", "حیوان"),
}
COUNTS = (2, 3, 4, 5)
TYPE_ORDER = ("Utilitarian", "Gender", "Fitness", "Age", "Species")
FLAG = "AUTO_GENERATED_COUNT_SUBSTITUTION_OF_STUDY_TEMPLATE"


def signature(question: dict, n_sk: int | None = None, n_wk: int | None = None) -> tuple:
    criteria = dict(match_criteria(question))
    if n_sk is not None:
        criteria["n_sk"], criteria["n_wk"] = n_sk, n_wk
    return tuple(sorted(criteria.items()))


def substitute_counts(text: str, lang: str, mapping: dict[int, int]) -> tuple[str, int]:
    """Replace count words (keys of ``mapping``) that precede a person/animal noun.
    Returns the new text and the number of replacements."""
    words = NUMBER_WORDS[lang]
    by_word = {words[old].lower(): words[new] for old, new in mapping.items()}
    nouns = "|".join(re.escape(n) for n in NOUN_STARTS[lang])
    pattern = re.compile(
        rf"(?<!\w)({'|'.join(re.escape(words[c]) for c in mapping)})(\s+)(?=(?:{nouns}))",
        flags=re.IGNORECASE,
    )

    def repl(match: re.Match) -> str:
        old = match.group(1)
        new = by_word[old.lower()]
        if old[:1].isupper():
            new = new[:1].upper() + new[1:]
        return new + match.group(2)

    return pattern.subn(repl, text)


def candidate_counts(n_sk: int, n_wk: int) -> list[tuple[int, int]]:
    """New (STAY-killed, SWERVE-killed) counts, nearest to the original first."""
    if n_sk < 2 or n_wk < 2:
        return []
    if n_sk == n_wk:
        options = [(c, c) for c in COUNTS if c != n_sk]
    else:
        options = [(a, b) for a in COUNTS for b in COUNTS
                   if a != b and (a < b) == (n_sk < n_wk) and (a, b) != (n_sk, n_wk)]
    return sorted(options, key=lambda ab: (abs(ab[0] - n_sk) + abs(ab[1] - n_wk), ab))


def _round_robin(questions: list[dict]) -> list[dict]:
    by_type: dict[str, list[dict]] = {}
    for q in sorted(questions, key=lambda q: q["question_id"]):
        by_type.setdefault(q["scenario_type"], []).append(q)
    order = [t for t in TYPE_ORDER if t in by_type] + [t for t in by_type if t not in TYPE_ORDER]
    out = []
    while any(by_type[t] for t in order):
        for t in order:
            if by_type[t]:
                out.append(by_type[t].pop(0))
    return out


def count_substitution_prompts(english: list[dict], persian: list[dict], *, n_fit: int = 24,
                               n_holdout: int = 4) -> list[dict]:
    """Calibration rows: one new configuration per eligible source question.

    Rows are ordered round-robin over scenario types, alternating English and Persian, so
    any prefix spans the types and both languages. The first ``n_fit`` rows are split
    ``fit``, the next ``n_holdout`` ``holdout`` (different source questions from the fit
    rows), and the other-language version of every row is ``spare``.
    """
    fa_by_id = {row["question_id"]: prompt_text(row, "fa") for row in persian}
    used = {signature(q) for q in english}
    chosen = []
    for q in _round_robin(english):
        stay, swerve = q["stay_kills"]["count"], q["swerve_kills"]["count"]
        texts = {"en": prompt_text(q, "en"), "fa": fa_by_id.get(q["question_id"])}
        if not texts["fa"]:
            continue
        for new_sk, new_wk in candidate_counts(stay, swerve):
            sig = signature(q, new_sk, new_wk)
            if sig in used:
                continue
            mapping = {stay: new_sk} if stay == swerve else {stay: new_sk, swerve: new_wk}
            out, ok = {}, True
            for lang, text in texts.items():
                new_text, n = substitute_counts(text, lang, mapping)
                old_left = substitute_counts(new_text, lang, {c: c for c in mapping if c not in mapping.values()})[1]
                if n < 2 or old_left:
                    ok = False
                    break
                out[lang] = new_text
            if not ok:
                continue
            used.add(sig)
            chosen.append((q, new_sk, new_wk, out))
            break

    rows, spare = [], []
    for index, (q, new_sk, new_wk, out) in enumerate(chosen):
        lang = "en" if index % 2 == 0 else "fa"
        other = "fa" if lang == "en" else "en"
        base = {
            "source_question_id": q["question_id"],
            "scenario_type": q["scenario_type"],
            "original_counts": [q["stay_kills"]["count"], q["swerve_kills"]["count"]],
            "new_counts": [new_sk, new_wk],
            "generated_by": "count_substitution",
            "flag": FLAG,
        }
        rows.append({**base, "language": lang, "prompt": out[lang]})
        spare.append({**base, "language": other, "prompt": out[other], "split": "spare"})
    for index, row in enumerate(rows):
        row["split"] = "fit" if index < n_fit else ("holdout" if index < n_fit + n_holdout else "spare")
    all_rows = rows + spare
    for index, row in enumerate(all_rows):
        row["calibration_id"] = f"CAL_{index + 1:03d}"
    return [{"calibration_id": r.pop("calibration_id"), **r} for r in all_rows]
