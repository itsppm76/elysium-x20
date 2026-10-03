"""Elysium X 20 - an open-source emotion-state engine for AI agents and companion models."""
from .appraisal import Appraisal, Appraiser, LexiconAppraiser
from .engine import EmotionEngine, Personality
from .modulation import MemoryDecision, StyleParams
from .state import EMOTIONS, EmotionState

__all__ = ["EmotionEngine", "Personality", "EmotionState", "Appraisal", "Appraiser",
           "LexiconAppraiser", "MemoryDecision", "StyleParams", "EMOTIONS"]
__version__ = "0.1.0"
