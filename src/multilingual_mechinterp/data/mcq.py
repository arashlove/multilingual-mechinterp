"""MCQ option-letter scoring for culture prompts and patching."""

from __future__ import annotations

from typing import Any, Sequence

import torch


def correct_letter(correct_index: int) -> str:
    """Map 0-based choice index → ``A``/``B``/``C``/``D``."""
    if correct_index < 0 or correct_index > 25:
        raise ValueError(f"correct_index out of range: {correct_index}")
    return chr(ord("A") + int(correct_index))


def option_letters(n: int = 4) -> list[str]:
    return [chr(ord("A") + i) for i in range(n)]


def first_token_id(tokenizer: Any, text: str) -> int:
    """First non-special token id for ``text`` (tries space-prefixed variant)."""
    for candidate in (text, f" {text}", f"\n{text}"):
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if ids:
            return int(ids[0])
    raise ValueError(f"could not tokenize {text!r}")


def option_token_ids(tokenizer: Any, letters: Sequence[str] | None = None) -> dict[str, int]:
    """Map option letter → first vocab id."""
    letters = list(letters) if letters is not None else option_letters()
    return {L: first_token_id(tokenizer, L) for L in letters}


def next_token_logits(model: Any, prompt: str, *, device: str | None = None) -> torch.Tensor:
    """Return next-token logits ``[V]`` for ``prompt`` (last real position)."""
    from multilingual_mechinterp.models.loader import LoadedModel

    if isinstance(model, LoadedModel):
        hf, tok, run_device = model.model, model.tokenizer, torch.device(device or model.device)
    else:
        hf, tok = model, model.tokenizer
        run_device = torch.device(device or next(hf.parameters()).device)

    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    enc = {k: v.to(run_device) for k, v in enc.items()}
    with torch.no_grad():
        out = hf(**enc)
        logits = out.logits if hasattr(out, "logits") else out
    # last non-pad position
    if "attention_mask" in enc:
        pos = int(enc["attention_mask"][0].sum().item()) - 1
    else:
        pos = logits.size(1) - 1
    return logits[0, pos].float().cpu()


def score_options_from_logits(
    logits: torch.Tensor,
    option_ids: dict[str, int],
    *,
    correct_letter: str,
) -> dict[str, Any]:
    """Softmax over A–D ids + correct-vs-wrong logit margin."""
    letters = list(option_ids)
    ids = [option_ids[L] for L in letters]
    opt_logits = torch.stack([logits[i] for i in ids])
    probs = opt_logits.softmax(0)
    correct = correct_letter.upper()
    if correct not in option_ids:
        raise ValueError(f"correct_letter {correct!r} not in {letters}")
    c_logit = float(logits[option_ids[correct]])
    wrong = [L for L in letters if L != correct]
    max_wrong = max(float(logits[option_ids[L]]) for L in wrong)
    margin = c_logit - max_wrong
    pred = letters[int(opt_logits.argmax())]
    return {
        "probs": {L: float(probs[i]) for i, L in enumerate(letters)},
        "logits": {L: float(opt_logits[i]) for i, L in enumerate(letters)},
        "correct_letter": correct,
        "pred_letter": pred,
        "p_correct": float(probs[letters.index(correct)]),
        "margin": margin,
        "correct": pred == correct,
    }


def score_mcq_prompt(
    model: Any,
    prompt: str,
    *,
    correct_letter: str,
    n_options: int = 4,
    device: str | None = None,
) -> dict[str, Any]:
    """Behavioural baseline for one MCQ prompt ending in ``Answer:``."""
    from multilingual_mechinterp.models.loader import LoadedModel

    tok = model.tokenizer if isinstance(model, LoadedModel) else model.tokenizer
    opt_ids = option_token_ids(tok, option_letters(n_options))
    logits = next_token_logits(model, prompt, device=device)
    out = score_options_from_logits(logits, opt_ids, correct_letter=correct_letter)
    out["option_ids"] = opt_ids
    return out


def margin_from_probs(
    probs: torch.Tensor,
    *,
    correct_id: int,
    option_ids: Sequence[int],
) -> torch.Tensor:
    """Logit-equivalent margin from vocab probs: log P_c − max log P_wrong.

    ``probs`` may be ``[V]`` or ``[..., V]``.
    """
    logp = probs.clamp_min(1e-12).log()
    c = logp[..., correct_id]
    wrong = [i for i in option_ids if i != correct_id]
    if not wrong:
        raise ValueError("need ≥1 wrong option id")
    w = torch.stack([logp[..., i] for i in wrong], dim=-1).max(dim=-1).values
    return c - w
