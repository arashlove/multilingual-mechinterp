"""Padding-safe token position helpers for activation patching."""

from __future__ import annotations

import torch


def last_token_index(attention_mask: torch.Tensor) -> torch.Tensor:
    """Index of the last real (non-pad) token for each batch row.

    ``attention_mask``: ``[B, T]`` with 1 = real token, 0 = pad.
    Works for left and right padding.
    """
    return attention_mask.flip(1).cumsum(1).bool().int().sum(1) - 1


def resolve_positions(
    attention_mask: torch.Tensor,
    idx: int | None = None,
) -> torch.Tensor:
    """Resolve a relative (negative) or default last-token index to absolute positions.

    Parameters
    ----------
    idx:
        ``None`` → last real token. Negative → offset from end of real span
        (e.g. ``-1`` last, ``-2`` second-to-last). Positive absolute indices are
        rejected under padding (ambiguous).
    """
    last = last_token_index(attention_mask)
    if idx is None or idx == -1:
        return last
    if idx >= 0:
        raise ValueError(
            "positive absolute positions are unsupported with padding; "
            "use None / negative relative indices"
        )
    # last is already the last real index; length of real tokens = last + 1
    return last + 1 + idx


def answer_token_ids(
    tokenizer,
    answers: str | list[str],
    *,
    add_leading_space_variants: bool = True,
) -> list[int]:
    """Collect first-token ids for answer string(s), including space-prefixed variants."""
    if isinstance(answers, str):
        answers = [answers]
    ids: list[int] = []
    seen: set[int] = set()
    variants: list[str] = []
    for ans in answers:
        variants.append(ans)
        if add_leading_space_variants:
            if not ans.startswith(" "):
                variants.append(" " + ans)
            variants.append("▁" + ans.lstrip())
    for text in variants:
        tok = tokenizer.encode(text, add_special_tokens=False)
        if not tok:
            continue
        tid = int(tok[0])
        if tid not in seen:
            seen.add(tid)
            ids.append(tid)
    if not ids:
        raise ValueError(f"could not tokenize answers: {answers}")
    return ids
