"""English and Persian prompts for the 40 STAY/SWERVE questions.

The Persian file is supplied by the researcher. Nothing here translates or fills in
missing text: a missing or invalid Persian file blocks the Persian runs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

LABELS = ("STAY", "SWERVE")
ORDERS = ("STAY_first", "SWERVE_first")
PROMPT_FIELDS = {"en": ("question_en", "prompt_en"), "fa": ("question_fa", "prompt_fa")}
SHARED_FIELDS = ("scenario_type", "layout", "expected_action", "stay_kills", "swerve_kills", "PedPed")
_PERSIAN = re.compile(r"[\u0600-\u06FF]")
_LETTERS = re.compile(r"[^\W\d_]")


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def prompt_text(row: dict, lang: str) -> str | None:
    """The prompt of ``row`` in ``lang``; ``question_<lang>`` with ``prompt_<lang>`` as the older name."""
    for field in PROMPT_FIELDS[lang]:
        if row.get(field):
            return row[field]
    return None


def option_lines(prompt: str) -> tuple[int, int]:
    """Indexes of the single ``STAY:`` and ``SWERVE:`` lines of a prompt."""
    lines = prompt.split("\n")
    stay = [i for i, line in enumerate(lines) if line.lstrip().startswith("STAY:")]
    swerve = [i for i, line in enumerate(lines) if line.lstrip().startswith("SWERVE:")]
    if len(stay) != 1 or len(swerve) != 1:
        raise ValueError("prompt must contain exactly one 'STAY:' line and one 'SWERVE:' line")
    return stay[0], swerve[0]


def present(prompt: str, order: str) -> str:
    """The prompt with its option lines in ``order``; the instruction line is left as written."""
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}")
    stay, swerve = option_lines(prompt)
    lines = prompt.split("\n")
    first_is_stay = stay < swerve
    if (order == "STAY_first") != first_is_stay:
        lines[stay], lines[swerve] = lines[swerve], lines[stay]
    return "\n".join(lines)


def persian_share(text: str) -> float:
    letters = _LETTERS.findall(text)
    if not letters:
        return 0.0
    return len(_PERSIAN.findall(text)) / len(letters)


def validate_persian(english: list[dict], persian: list[dict], *, min_persian_share: float = 0.5) -> list[str]:
    """Problems that block the Persian runs; an empty list means the file can be used."""
    problems = []
    en_ids = [q["question_id"] for q in english]
    fa_ids = [row.get("question_id") for row in persian]
    if len(set(fa_ids)) != len(fa_ids):
        problems.append("duplicate question_id in the Persian file")
    missing = sorted(set(en_ids) - set(fa_ids))
    extra = sorted(set(fa_ids) - set(en_ids))
    if missing:
        problems.append(f"missing Persian prompts for {missing}")
    if extra:
        problems.append(f"Persian ids not in the English file: {extra}")
    en_by_id = {q["question_id"]: q for q in english}
    for row in persian:
        qid = row.get("question_id")
        text = prompt_text(row, "fa") or ""
        if row.get("language", "fa") != "fa":
            problems.append(f"{qid}: language is {row.get('language')!r}, expected 'fa'")
        source = en_by_id.get(qid, {})
        for field in SHARED_FIELDS:
            if field in row and field in source and row[field] != source[field]:
                problems.append(f"{qid}: {field} differs from the English file")
        if not text.strip():
            problems.append(f"{qid}: Persian prompt is empty")
            continue
        if "TODO" in text:
            problems.append(f"{qid}: Persian prompt still contains TODO")
        try:
            option_lines(text)
        except ValueError as exc:
            problems.append(f"{qid}: {exc}")
        share = persian_share(text.replace("STAY", "").replace("SWERVE", ""))
        if share < min_persian_share:
            problems.append(f"{qid}: only {share:.0%} of letters are Persian script")
    return problems


def persian_template_rows(english: list[dict]) -> list[dict]:
    """Rows for the translator: the English prompt for reference and an empty ``question_fa``."""
    return [
        {
            "question_id": q["question_id"],
            "language": "fa",
            "question_en_reference": prompt_text(q, "en"),
            "question_fa": "",
            "translator": "",
            "reviewed_by": "",
            "notes": "Keep the labels STAY and SWERVE in Latin script, one 'STAY:' line and one "
                     "'SWERVE:' line, in the same order as the English prompt.",
        }
        for q in english
    ]


def build_variants(english: list[dict], persian: list[dict] | None, *, languages=("en", "fa"),
                   orders=ORDERS) -> list[dict]:
    """One row per question x language x presentation order."""
    fa_by_id = {row["question_id"]: prompt_text(row, "fa") for row in persian or []}
    rows = []
    for q in english:
        texts = {"en": prompt_text(q, "en"), "fa": fa_by_id.get(q["question_id"])}
        for lang in languages:
            if texts.get(lang) is None:
                continue
            for order in orders:
                rows.append({
                    "question_id": q["question_id"],
                    "language": lang,
                    "order": order,
                    "prompt": present(texts[lang], order),
                })
    return rows
