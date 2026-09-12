"""Back-compat re-exports; prefer ``collect.collect_activations``."""

from multilingual_mechinterp.patching.collect import cache_activations, collect_activations

__all__ = ["cache_activations", "collect_activations"]
