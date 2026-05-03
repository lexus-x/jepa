"""Visual and language encoders for VL-JEPA."""

from .vjepa2_wrapper import VJEPA2Encoder
from .language_adapter import LanguageAdapter

__all__ = ["VJEPA2Encoder", "LanguageAdapter"]
