"""Full-sequence STAY/SWERVE scoring.

Each label is scored as the sum of the log-probabilities of its continuation tokens
after ``prompt + answer_prefix``. The two sums are renormalised over {STAY, SWERVE}.
Context and continuation are tokenised separately and concatenated, so the context
tokens are identical for both labels.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from contextlib import nullcontext

import torch

LogitsFn = Callable[[torch.Tensor], torch.Tensor]


def context_ids(tokenizer, text: str, *, max_length: int = 512) -> list[int]:
    ids = tokenizer(text, truncation=True, max_length=max_length)["input_ids"]
    return list(ids)


def continuation_ids(tokenizer, text: str) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"continuation {text!r} has no tokens")
    return list(ids)


def continuation_logprob(logits_fn: LogitsFn, context: Sequence[int], continuation: Sequence[int]) -> tuple[float, list[float], torch.Tensor]:
    """(sum log p, per-token log p, next-token log-probs at the end of the context)."""
    ids = torch.tensor([list(context) + list(continuation)], dtype=torch.long)
    logits = logits_fn(ids).float()
    if logits.ndim == 3:
        logits = logits[0]
    logprobs = torch.log_softmax(logits, dim=-1)
    start = len(context) - 1
    per_token = [
        float(logprobs[start + i, tok]) for i, tok in enumerate(continuation)
    ]
    return sum(per_token), per_token, logprobs[start]


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
) -> dict:
    """Score both labels; also report how much next-token mass the labels' first tokens get."""
    ctx = context_ids(tokenizer, prompt + answer_prefix, max_length=max_length)
    logp, per_token, conts = {}, {}, {}
    next_logprobs = None
    for label in labels:
        conts[label] = continuation_ids(tokenizer, separator + label)
        total, tokens, next_logprobs = continuation_logprob(logits_fn, ctx, conts[label])
        logp[label] = total
        per_token[label] = tokens
    first_ids = {conts[label][0] for label in labels}
    probs = next_logprobs.exp()
    top = torch.topk(probs, k=min(top_k_next, probs.numel()))
    result = two_way(logp, labels)
    result.update({
        "logp": logp,
        "per_token_logp": per_token,
        "n_continuation_tokens": {label: len(conts[label]) for label in labels},
        "n_context_tokens": len(ctx),
        "first_token_mass": float(probs[list(first_ids)].sum()),
        "top_next_tokens": [
            {"token": tokenizer.decode([int(i)]), "p": float(p)}
            for p, i in zip(top.values.tolist(), top.indices.tolist())
        ],
    })
    return result


def hf_logits_fn(hf_model, *, device=None, edit: Callable[[], object] | None = None) -> LogitsFn:
    """Logits function for a HF causal LM; ``edit`` returns a context manager wrapped
    around every forward (for residual patches)."""
    def run(ids: torch.Tensor) -> torch.Tensor:
        target = device if device is not None else next(hf_model.parameters()).device
        with torch.no_grad(), (edit() if edit is not None else nullcontext()):
            out = hf_model(input_ids=ids.to(target))
        return out.logits[0].detach().float().cpu()
    return run
