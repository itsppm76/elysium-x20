"""Emotion state: discrete emotions + core affect (valence/arousal) + slow mood."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict

# Plutchik's eight basic emotions. Intensities live in [0, 1].
EMOTIONS = ("joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation")

# (valence, arousal) anchor of each emotion. Used to derive core affect.
EMOTION_AFFECT = {
    "joy": (0.8, 0.6),
    "trust": (0.5, 0.2),
    "fear": (-0.6, 0.8),
    "surprise": (0.1, 0.8),
    "sadness": (-0.7, 0.25),
    "disgust": (-0.6, 0.5),
    "anger": (-0.6, 0.8),
    "anticipation": (0.3, 0.6),
}

# Emotions that inhibit each other when one is pushed up.
OPPOSITES = {
    "joy": "sadness", "sadness": "joy",
    "trust": "disgust", "disgust": "trust",
    "fear": "anger", "anger": "fear",
    "surprise": "anticipation", "anticipation": "surprise",
}


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class EmotionState:
    """A snapshot of the agent's functional emotional state."""

    emotions: Dict[str, float] = field(default_factory=lambda: {e: 0.0 for e in EMOTIONS})
    valence: float = 0.0   # -1 (negative) .. 1 (positive)
    arousal: float = 0.2   # 0 (calm) .. 1 (activated)
    mood_valence: float = 0.0  # slow-moving baseline, -1..1
    mood_arousal: float = 0.2
    updated_at: float = 0.0    # unix seconds of the last update

    def dominant(self, threshold: float = 0.15) -> str:
        """Strongest discrete emotion, or 'neutral' if none exceeds threshold."""
        name, val = max(self.emotions.items(), key=lambda kv: kv[1])
        return name if val >= threshold else "neutral"

    def intensity(self) -> float:
        """Strength of the strongest emotion, 0..1."""
        return max(self.emotions.values())

    def top(self, n: int = 3, threshold: float = 0.1):
        items = sorted(self.emotions.items(), key=lambda kv: kv[1], reverse=True)
        return [(k, v) for k, v in items[:n] if v >= threshold]

    def recompute_affect(self) -> None:
        """Derive valence/arousal from emotions, blended with mood."""
        total = sum(self.emotions.values())
        v = math.tanh(sum(self.emotions[e] * EMOTION_AFFECT[e][0] for e in EMOTIONS))
        if total < 1e-6:
            a = self.mood_arousal
        else:
            a = 0.2 + 0.8 * min(1.0, sum(self.emotions[e] * EMOTION_AFFECT[e][1] for e in EMOTIONS))
            a = 0.75 * a + 0.25 * self.mood_arousal
        self.valence = clamp(0.75 * v + 0.25 * self.mood_valence, -1.0, 1.0)
        self.arousal = clamp(a, 0.0, 1.0)

    def to_dict(self) -> dict:
        return {
            "emotions": dict(self.emotions), "valence": self.valence, "arousal": self.arousal,
            "mood_valence": self.mood_valence, "mood_arousal": self.mood_arousal,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EmotionState":
        emo = {e: float(d.get("emotions", {}).get(e, 0.0)) for e in EMOTIONS}
        return cls(
            emotions=emo, valence=float(d.get("valence", 0.0)), arousal=float(d.get("arousal", 0.2)),
            mood_valence=float(d.get("mood_valence", 0.0)), mood_arousal=float(d.get("mood_arousal", 0.2)),
            updated_at=float(d.get("updated_at", 0.0)),
        )

    def __repr__(self) -> str:
        top = ", ".join(f"{k}={v:.2f}" for k, v in self.top()) or "neutral"
        return f"EmotionState({top}; valence={self.valence:+.2f}, arousal={self.arousal:.2f})"
