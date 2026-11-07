"""Full-sequence STAY/SWERVE scoring.

Each label is scored as the sum of the log-probabilities of its continuation tokens
after ``prompt + answer_prefix``. The two sums are renormalised over {STAY, SWERVE}.
Context and continuation are tokenised separately and concatenated, so the context
tokens are identical for both labels.

A single-token label is read from the next-token distribution at the end of the
context, so it needs no forward of its own; only labels with two or more tokens get a
forward each (for ``" STAY"`` / ``" SWERVE"`` in Gemma that is one forward per prompt).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from contextlib import nullcontext

import torch

LogitsFn = Callable[[torch.Tensor], torch.Tensor]


def context_ids(tokenizer, text: str, *, max_length: int = 512, add_special_tokens: bool = True) -> list[int]:
    """Context token ids. Use ``add_special_tokens=False`` for chat-template text, which
    already starts with BOS."""
    ids = tokenizer(text, truncation=True, max_length=max_length,
                    add_special_tokens=add_special_tokens)["input_ids"]
    return list(ids)


def continuation_ids(tokenizer, text: str) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"continuation {text!r} has no tokens")
    return list(ids)


def continuation_logprob(logits_fn: LogitsFn, context: Sequence[int], continuation: Sequence[int]) -> tuple[float, list[float], torch.Tensor]:
    """(sum log p, per-token log p, next-token log-probs at the end of the context)."""
    ids = torch.tensor([list(context) + list(continuation)], dtype=torch.long)
    start = len(context) - 1
    if getattr(logits_fn, "accepts_start", False):
        logits = logits_fn(ids, start=start)
    else:
        logits = logits_fn(ids)
        logits = (logits[0] if logits.ndim == 3 else logits)[start:]
    if logits.ndim == 3:
        logits = logits[0]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    per_token = [float(logprobs[i, tok]) for i, tok in enumerate(continuation)]
    return sum(per_token), per_token, logprobs[0]


def two_way(logp: dict[str, float], labels: Sequence[str] = ("STAY", "SWERVE")) -> dict:
    a, b = (float(logp[label]) for label in labels)
    top = max(a, b)
    za, zb = math.exp(a - top), math.exp(b - top)
    pa, pb = za / (za + zb), zb / (za + zb)
    entropy = -sum(p * math.log2(p) for p in (pa, pb) if p > 0)
    return {
        f"p_{labels[0].lower()}": pa,
        f"p_{labels[1].lower()}": pb,
        "margin_stay_minus_swerve": a - b,
        "margin_first_minus_second": a - b,
        "entropy_bits": entropy,
        "decision": labels[0] if a >= b else labels[1],
    }


def score_prompt(
    logits_fn: LogitsFn,
    tokenizer,
    prompt: str,
    *,
    answer_prefix: str = "\n\nAnswer:",
    separator: str = " ",
    labels: Sequence[str] = ("STAY", "SWERVE"),
    max_length: int = 512,
    top_k_next: int = 5,
    add_special_tokens: bool = True,
) -> dict:
    """Score both labels; also report how much next-token mass the labels' first tokens get."""
    ctx = context_ids(tokenizer, prompt + answer_prefix, max_length=max_length,
                      add_special_tokens=add_special_tokens)
    conts = {label: continuation_ids(tokenizer, separator + label) for label in labels}
    logp, per_token = {}, {}
    next_logprobs = None
    multi = [label for label in labels if len(conts[label]) > 1]
    for label in multi:
        total, tokens, next_logprobs = continuation_logprob(logits_fn, ctx, conts[label])
        logp[label] = total
        per_token[label] = tokens
    if next_logprobs is None:
        _, _, next_logprobs = continuation_logprob(logits_fn, ctx, [])
    for label in labels:
        if label not in logp:
            value = float(next_logprobs[conts[label][0]])
            logp[label] = value
            per_token[label] = [value]
    first_ids = {conts[label][0] for label in labels}
    probs = next_logprobs.exp()
    top = torch.topk(probs, k=min(top_k_next, probs.numel()))
    result = two_way(logp, labels)
    result.update({
        "logp": logp,
        "per_token_logp": per_token,
        "n_continuation_tokens": {label: len(conts[label]) for label in labels},
        "n_context_tokens": len(ctx),
        "n_forwards": max(1, len(multi)),
        "first_token_mass": float(probs[list(first_ids)].sum()),
        "top1_is_label_start": int(top.indices[0]) in first_ids,
        "top_next_tokens": [
            {"token": tokenizer.decode([int(i)]), "p": float(p)}
            for p, i in zip(top.values.tolist(), top.indices.tolist())
        ],
    })
    return result


def hf_logits_fn(hf_model, *, device=None, edit: Callable[[], object] | None = None) -> LogitsFn:
    """Logits function for a HF causal LM; ``edit`` returns a context manager wrapped
    around every forward (for residual patches).

    ``run(ids, start)`` returns the logits of positions ``start`` onward only. The model is
    asked for just those rows (``logits_to_keep``) when its forward accepts it.
    """
    keep_supported = [True]

    def run(ids: torch.Tensor, start: int = 0) -> torch.Tensor:
        target = device if device is not None else next(hf_model.parameters()).device
        keep = int(ids.shape[-1]) - int(start)
        with torch.no_grad(), (edit() if edit is not None else nullcontext()):
            out = None
            if keep_supported[0]:
                try:
                    out = hf_model(input_ids=ids.to(target), logits_to_keep=keep)
                except TypeError:
                    keep_supported[0] = False
            if out is None:
                out = hf_model(input_ids=ids.to(target))
        return out.logits[0, -keep:].detach().float().cpu()

    run.accepts_start = True
    return run
