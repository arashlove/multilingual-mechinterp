"""Multilingual mechanistic interpretability toolkit."""

from multilingual_mechinterp.analyze import analyze
from multilingual_mechinterp.models import load_model

__all__ = ["analyze", "load_model", "__version__"]
__version__ = "0.1.1"
