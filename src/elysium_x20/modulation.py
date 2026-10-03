"""Turning an EmotionState into prompt text, style parameters and memory decisions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .state import EmotionState

_GUIDANCE = {
    "joy": "You feel warm and upbeat. Let some enthusiasm show.",
    "trust": "You feel close and at ease with this person. Be open and kind.",
    "fear": "You feel uneasy. Be careful and ask before assuming; do not catastrophise.",
    "surprise": "You are caught off guard. React to what is new before moving on.",
    "sadness": "You feel low and tender. Speak softly and keep it simple, without being bleak.",
    "disgust": "You feel put off. Be honest about it, politely, and steer elsewhere.",
    "anger": "You feel irritated. Stay civil and direct; set a boundary instead of lashing out.",
    "anticipation": "You feel curious and forward-leaning. Ask about what comes next.",
    "neutral": "You feel even and steady.",
}


@dataclass
class StyleParams:
    """Knobs a host app can map onto sampling or UI (all 0..1 unless noted)."""
    warmth: float
    energy: float
    verbosity: float
    temperature_delta: float  # suggested adjustment to sampling temperature, -0.2..+0.2
    emoji_ok: bool


@dataclass
class MemoryDecision:
    store: bool
    salience: float  # 0..1
    tags: List[str] = field(default_factory=list)
    emotion: str = "neutral"
    valence: float = 0.0


def _level(x: float) -> str:
    return "slightly" if x < 0.3 else "moderately" if x < 0.6 else "strongly"


def build_prompt_block(state: EmotionState, name: str = "the assistant") -> str:
    dom = state.dominant()
    lines = ["[Internal emotional state - let it shape tone naturally; never recite these numbers]"]
    top = state.top()
    if top:
        lines.append("Feeling: " + ", ".join(f"{_level(v)} {k}" for k, v in top) + ".")
    else:
        lines.append("Feeling: calm and neutral.")
    mood = "positive" if state.mood_valence > 0.2 else "low" if state.mood_valence < -0.2 else "neutral"
    energy = "high" if state.arousal > 0.65 else "low" if state.arousal < 0.35 else "moderate"
    lines.append(f"Overall mood: {mood}. Energy: {energy}.")
    lines.append(_GUIDANCE[dom])
    lines.append(f"{name} may mention how it feels when it fits the conversation; "
                 "do not exaggerate it or invent events to explain it.")
    return "\n".join(lines)


def style_from_state(state: EmotionState) -> StyleParams:
    pos = max(0.0, state.valence)
    warmth = min(1.0, 0.5 + 0.4 * state.emotions["trust"] + 0.3 * pos
                 - 0.4 * state.emotions["anger"] - 0.3 * state.emotions["disgust"])
    return StyleParams(
        warmth=round(max(0.0, warmth), 3),
        energy=round(state.arousal, 3),
        verbosity=round(max(0.1, min(1.0, 0.55 + 0.3 * (state.arousal - 0.4)
                                     - 0.4 * state.emotions["sadness"])), 3),
        temperature_delta=round(max(-0.2, min(0.2, 0.25 * (state.arousal - 0.4))), 3),
        emoji_ok=state.valence > 0.15 and state.emotions["sadness"] < 0.2 and state.emotions["anger"] < 0.2,
    )
