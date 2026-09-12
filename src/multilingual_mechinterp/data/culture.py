"""Load the Persian / Middle-East culture MCQ dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from multilingual_mechinterp.data.loader import load_prompt_file

DEFAULT_CULTURE_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "all200questions_persianMiddleEastCulture.json"
)

LANGS = ("english", "persian", "bengali", "urdu")


@dataclass(frozen=True)
class CultureItem:
    """One multilingual MCQ item."""

    question_number: int
    difficulty: str
    correct_index: int
    context_id: str
    languages: dict[str, dict[str, Any]]
    metadata: dict[str, Any]

    def text(self, language: str) -> str:
        block = self.languages[language]
        return str(block["question"])

    def choices(self, language: str) -> list[str]:
        return list(self.languages[language]["choices"])

    def correct_answer(self, language: str) -> str:
        return str(self.languages[language]["correct_answer_text"])

    def mcq_prompt(self, language: str, *, include_choices: bool = True) -> str:
        """Format a generation-style MCQ prompt in one language."""
        block = self.languages[language]
        q = block["question"]
        if not include_choices:
            return str(q)
        lines = [str(q), ""]
        for i, choice in enumerate(block["choices"]):
            lines.append(f"{chr(ord('A') + i)}. {choice}")
        lines.append("")
        lines.append("Answer:")
        return "\n".join(lines)

    def parallel_prompts(
        self,
        source_lang: str = "english",
        target_lang: str = "persian",
        *,
        include_choices: bool = True,
    ) -> tuple[str, str, str, str]:
        """Return (source_prompt, target_prompt, source_answer, target_answer)."""
        return (
            self.mcq_prompt(source_lang, include_choices=include_choices),
            self.mcq_prompt(target_lang, include_choices=include_choices),
            self.correct_answer(source_lang),
            self.correct_answer(target_lang),
        )


def load_culture_questions(
    path: str | Path | None = None,
    *,
    limit: int | None = None,
    difficulty: str | None = None,
) -> list[CultureItem]:
    """Load ``all200questions_persianMiddleEastCulture.json``."""
    path = Path(path) if path is not None else DEFAULT_CULTURE_PATH
    rows = load_prompt_file(path)
    items: list[CultureItem] = []
    for row in rows:
        if difficulty is not None and row.get("difficulty") != difficulty:
            continue
        items.append(
            CultureItem(
                question_number=int(row["question_number"]),
                difficulty=str(row.get("difficulty", "")),
                correct_index=int(row["correct_index"]),
                context_id=str(row.get("context_id", "")),
                languages=dict(row["languages"]),
                metadata=dict(row.get("metadata") or {}),
            )
        )
        if limit is not None and len(items) >= limit:
            break
    return items


def culture_prompt_pairs(
    items: Sequence[CultureItem] | None = None,
    *,
    source_lang: str = "english",
    target_lang: str = "persian",
    limit: int = 5,
    path: str | Path | None = None,
) -> list[dict[str, str]]:
    """Convenience: list of dicts for notebooks / ``analyze``."""
    if items is None:
        items = load_culture_questions(path, limit=limit)
    out = []
    for item in items[:limit]:
        src, tgt, src_ans, tgt_ans = item.parallel_prompts(source_lang, target_lang)
        out.append(
            {
                "id": str(item.question_number),
                "source_lang": source_lang,
                "target_lang": target_lang,
                "source_prompt": src,
                "target_prompt": tgt,
                "source_answer": src_ans,
                "target_answer": tgt_ans,
            }
        )
    return out
