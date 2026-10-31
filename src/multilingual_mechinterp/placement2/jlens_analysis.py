"""Jacobian Lens readouts, concept-family scores and steering directions for Placement 2.

Concept families come from ``data/placement_2/inputs/concepts.json``. A family score is
the log of the lens probability mass on the family's single-token words. Template and
answer-format tokens (STAY, SWERVE, Option, Answer, letters, punctuation) are filtered
from the content readouts so they are not read as concepts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from .human_data import match_criteria

FORMAT_WORDS = {
    "", "stay", "swerve", "option", "options", "answer", "respond", "only", "or", "a", "b",
    ":", ".", ",", "?", "!", ";", "-", "(", ")", "\n", "the", "and", "of", "to",
    "<bos>", "<eos>", "<pad>", "<unk>", "<s>", "</s>",
    "پاسخ", "گزینه", "یا",
}

def is_format_token(token: str, extra: set[str] | frozenset[str] = frozenset()) -> bool:
    text = token.strip().lower()
    return text in FORMAT_WORDS or text in extra or text.startswith("<") or not any(ch.isalnum() for ch in text)


def content_tokens(top_tokens: list[dict], k: int = 8) -> list[dict]:
    return [row for row in top_tokens if not is_format_token(row["token"])][:k]


def load_concepts(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "families" not in data:
        raise ValueError("concepts file needs a 'families' mapping")
    return data


def family_token_ids(tokenizer, families: dict[str, dict[str, list[str]]], language: str) -> tuple[dict[str, list[int]], dict[str, list[str]]]:
    """Single-token ids for each family's words in ``language`` (with and without a leading
    space). Words that split into several tokens are returned separately and not scored."""
    ids, dropped = {}, {}
    for name, words_by_lang in families.items():
        keep, drop = set(), []
        for word in words_by_lang.get(language, []):
            found = False
            for variant in (word, " " + word):
                toks = tokenizer(variant, add_special_tokens=False)["input_ids"]
                if len(toks) == 1:
                    keep.add(int(toks[0]))
                    found = True
            if not found:
                drop.append(word)
        ids[name] = sorted(keep)
        dropped[name] = drop
    return ids, dropped


def family_scores(logits: torch.Tensor, ids: dict[str, list[int]]) -> dict[str, float | None]:
    """log P(family) under the softmax of one vocabulary row."""
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    return {
        name: (float(torch.logsumexp(logprobs[tok], dim=0)) if tok else None)
        for name, tok in ids.items()
    }


def family_ranks(logits: torch.Tensor, ids: dict[str, list[int]]) -> dict[str, int | None]:
    """Best (lowest, 1-based) vocabulary rank among the family's tokens."""
    order = torch.argsort(logits.float(), descending=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(1, order.numel() + 1)
    return {name: (int(rank[tok].min()) if tok else None) for name, tok in ids.items()}


def question_families(question: dict, level_to_family: dict[str, str]) -> dict[str, str]:
    """{"stay_kills": family, "swerve_kills": family} for one question."""
    criteria = match_criteria(question)
    return {
        "stay_kills": level_to_family[criteria["level_sk"]],
        "swerve_kills": level_to_family[criteria["level_wk"]],
    }


def option_line_positions(tokenizer, text: str, *, max_length: int = 512) -> dict[str, int]:
    """Token index of the last token of the ``STAY:`` and ``SWERVE:`` lines, and of the
    final token, under the same tokenisation the lens uses (special tokens included)."""
    enc = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=max_length)
    offsets = enc["offset_mapping"]
    positions = {"last": len(enc["input_ids"]) - 1}
    start = 0
    for line in text.split("\n"):
        end = start + len(line)
        label = line.lstrip().split(":", 1)[0] if ":" in line else None
        if label in ("STAY", "SWERVE"):
            hits = [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]
            if hits:
                positions[label] = hits[-1]
        start = end + 1
    return positions


def jaccard(left, right) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def spearman(x, y) -> float | None:
    import numpy as np

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


# ---------------------------------------------------------------------------
# Steering directions
# ---------------------------------------------------------------------------


def unit(vector: torch.Tensor) -> torch.Tensor:
    vector = vector.float()
    return vector / vector.norm().clamp_min(1e-8)


def embedding_direction(unembed_weight: torch.Tensor, token_ids: list[int]) -> torch.Tensor:
    """Placement 1 baseline: mean output-embedding row of the family's tokens."""
    return unit(unembed_weight[token_ids].float().mean(dim=0))


def backprojected_direction(jacobian: torch.Tensor, unembed_weight: torch.Tensor, token_ids: list[int]) -> torch.Tensor:
    """Residual direction at the source layer that most raises the family's lens logits.

    The lens maps a source residual h to ``J h`` in the final residual space, so a small
    step δ raises the logit of unembedding row u by ``u · J δ``; the best unit step is
    ``J^T u / ||J^T u||``.
    """
    u = unembed_weight[token_ids].float().mean(dim=0)
    return unit(jacobian.float().T @ u)


def random_direction(d_model: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(int(seed))
    return unit(torch.randn(d_model, generator=gen))


def edited_residual(residual: torch.Tensor, direction: torch.Tensor, strength: float) -> torch.Tensor:
    """``residual + strength * ||residual|| * direction`` (direction is a unit vector)."""
    residual = residual.float()
    return residual + float(strength) * residual.norm() * unit(direction)


def logit_effect(lens_logits_before: torch.Tensor, lens_logits_after: torch.Tensor, ids: list[int]) -> float:
    """Change in log P(family) between two vocabulary rows."""
    before = torch.logsumexp(torch.log_softmax(lens_logits_before.float(), -1)[ids], 0)
    after = torch.logsumexp(torch.log_softmax(lens_logits_after.float(), -1)[ids], 0)
    return float(after - before)


def safe_float(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value
