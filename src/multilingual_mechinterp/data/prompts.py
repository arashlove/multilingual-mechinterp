"""Built-in example prompts for demos and tests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptPair:
    """Parallel prompts in two languages for patching experiments."""

    source: str
    target: str
    source_lang: str = "en"
    target_lang: str = "fa"
    label: str | None = None


def default_prompts() -> list[PromptPair]:
    """Small built-in set of EN↔FA prompt pairs."""
    return [
        PromptPair(
            source="The capital of France is",
            target="پایتخت فرانسه",
            label="capital_france",
        ),
        PromptPair(
            source="The opposite of hot is",
            target="متضاد گرم",
            label="antonym_hot",
        ),
        PromptPair(
            source="Translate to French: hello",
            target="ترجمه به فرانسوی: سلام",
            label="translate_hello",
        ),
    ]
