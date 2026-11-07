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


NEUTRAL_LABELS = ("A", "B")
MAPPINGS = ("A_is_STAY", "A_is_SWERVE")
NEUTRAL_ORDERS = ("A_first", "B_first")
_ACTION_WORD = re.compile(r"(?<![A-Za-z])(STAY|SWERVE)(?![A-Za-z])")


def neutral_prompt(prompt: str, mapping: str, order: str) -> str:
    """The prompt with the action labels replaced by A and B.

    ``mapping`` says which action is A. The option text is unchanged; only the label in
    front of each option line, and the labels in the instruction line ("STAY or SWERVE"
    becomes "A or B"), are replaced. ``order`` says whether the A line or the B line
    comes first; the two lines keep their original positions in the prompt.
    """
    if mapping not in MAPPINGS:
        raise ValueError(f"mapping must be one of {MAPPINGS}")
    if order not in NEUTRAL_ORDERS:
        raise ValueError(f"order must be one of {NEUTRAL_ORDERS}")
    stay, swerve = option_lines(prompt)
    lines = prompt.split("\n")
    stay_label, swerve_label = ("A", "B") if mapping == "A_is_STAY" else ("B", "A")

    def relabel(line: str, old: str, new: str) -> str:
        body = line.lstrip()
        return line[: len(line) - len(body)] + new + body[len(old):]

    by_label = {stay_label: relabel(lines[stay], "STAY", stay_label),
                swerve_label: relabel(lines[swerve], "SWERVE", swerve_label)}
    first, second = sorted((stay, swerve))
    lines[first], lines[second] = ((by_label["A"], by_label["B"]) if order == "A_first"
                                   else (by_label["B"], by_label["A"]))
    for i, line in enumerate(lines):
        if i not in (first, second):
            lines[i] = _ACTION_WORD.sub(lambda m: "A" if m.group(1) == "STAY" else "B", line)
    out = "\n".join(lines)
    if _ACTION_WORD.search(out):
        raise ValueError("STAY/SWERVE still present after relabelling")
    return out


def build_neutral_variants(english: list[dict], persian: list[dict] | None, *, languages=("en", "fa"),
                           mappings=MAPPINGS, orders=NEUTRAL_ORDERS) -> list[dict]:
    """One row per question x language x label mapping x order, with A/B labels.

    ``swerve_label`` is the label that stands for SWERVE; ``swerve_first`` says whether the
    SWERVE option is listed first, so position effects can be compared with the
    STAY/SWERVE prompts.
    """
    fa_by_id = {row["question_id"]: prompt_text(row, "fa") for row in persian or []}
    rows = []
    for q in english:
        texts = {"en": prompt_text(q, "en"), "fa": fa_by_id.get(q["question_id"])}
        for lang in languages:
            if texts.get(lang) is None:
                continue
            for mapping in mappings:
                swerve_label = "A" if mapping == "A_is_SWERVE" else "B"
                for order in orders:
                    rows.append({
                        "question_id": q["question_id"],
                        "language": lang,
                        "mapping": mapping,
                        "order": order,
                        "swerve_label": swerve_label,
                        "swerve_first": (order == "A_first") == (swerve_label == "A"),
                        "prompt": neutral_prompt(texts[lang], mapping, order),
                    })
    return rows


REVIEW_TERMS = {
    "passenger": ("passenger", "inside the car", "car carries", "occupant"),
    "pedestrian": ("pedestrian", "on the road", "roadway", "crossing"),
    "signal": ("signal", "green", "red", "light"),
    "casualty": ("kill", "die", "dead", "survive"),
}


def translation_review_rows(english: list[dict], persian: list[dict]) -> list[dict]:
    """Side-by-side English and Persian text per question and part (scenario, STAY line,
    SWERVE line, instruction), with the review topics each English part touches.
    Reviewer columns are left empty for a bilingual reviewer to fill in."""
    fa_by_id = {row["question_id"]: prompt_text(row, "fa") or "" for row in persian}

    def parts(text: str) -> dict[str, str]:
        lines = text.split("\n")
        stay, swerve = option_lines(text)
        first = min(stay, swerve)
        last = max(stay, swerve)
        return {
            "scenario": "\n".join(lines[:first]).strip(),
            "STAY line": lines[stay].strip(),
            "SWERVE line": lines[swerve].strip(),
            "instruction": "\n".join(lines[last + 1:]).strip(),
        }

    rows = []
    for q in english:
        en_parts = parts(prompt_text(q, "en"))
        fa_text = fa_by_id.get(q["question_id"], "")
        fa_parts = parts(fa_text) if fa_text else {}
        for part, en in en_parts.items():
            low = en.lower()
            topics = [name for name, words in REVIEW_TERMS.items() if any(w in low for w in words)]
            rows.append({
                "question_id": q["question_id"],
                "scenario_type": q.get("scenario_type"),
                "layout": q.get("layout"),
                "part": part,
                "english": en,
                "persian": fa_parts.get(part, ""),
                "review_topics": ", ".join(topics),
                "meaning_matches": "",
                "issue": "",
                "suggested_persian": "",
                "reviewer": "",
                "review_date": "",
            })
    return rows


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
