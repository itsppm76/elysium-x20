"""The Elysium X 20 engine."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from typing import Callable, Optional

from .appraisal import Appraisal, Appraiser, LexiconAppraiser
from .modulation import MemoryDecision, StyleParams, build_prompt_block, style_from_state
from .state import EMOTIONS, OPPOSITES, EmotionState, clamp


@dataclass
class Personality:
    """Per-agent temperament. All values 0..1 except baseline_valence (-1..1)."""
    reactivity: float = 0.6        # how strongly events move the state
    empathy: float = 0.5           # how much the user's emotions are mirrored
    emotional_stability: float = 0.5   # higher = faster return to baseline
    baseline_valence: float = 0.1  # resting mood
    emotion_half_life_s: float = 600.0     # emotions fade by half every 10 min
    mood_half_life_s: float = 6 * 3600.0   # mood drifts back every ~6 h
    mood_inertia: float = 0.9      # per-turn weight on old mood (EMA)


# How a user's emotion is mirrored by an empathic agent: user_emotion -> [(agent_emotion, factor)]
_CONTAGION = {
    "joy": [("joy", 1.0)], "trust": [("trust", 0.8), ("joy", 0.3)],
    "sadness": [("sadness", 0.6), ("trust", 0.2)],
    "fear": [("fear", 0.5), ("anticipation", 0.2)],
    "anger": [("fear", 0.3), ("sadness", 0.2)],
    "disgust": [("surprise", 0.2)], "surprise": [("surprise", 0.7)],
    "anticipation": [("anticipation", 0.8)],
}


class EmotionEngine:
    """Tracks one agent's emotional state across a conversation.

    >>> engine = EmotionEngine()
    >>> state = engine.process_turn("I got the job!! so happy", "That is great news!")
    >>> engine.prompt_modulator()  # drop into your system prompt
    """

    def __init__(self, personality: Optional[Personality] = None,
                 appraiser: Optional[Appraiser] = None,
                 memory_threshold: float = 0.35,
                 clock: Callable[[], float] = time.time):
        self.personality = personality or Personality()
        self.appraiser: Appraiser = appraiser or LexiconAppraiser()
        self.memory_threshold = memory_threshold
        self._clock = clock
        b = self.personality.baseline_valence
        self.state = EmotionState(mood_valence=b, updated_at=clock())
        self.state.recompute_affect()
        self.last_appraisal = Appraisal()
        self._prev_valence = self.state.valence
        self._last_shift = 0.0

    # ---- core loop -------------------------------------------------------
    def process_turn(self, user_msg: str, agent_reply: Optional[str] = None,
                     now: Optional[float] = None) -> EmotionState:
        """Advance the state by one conversation turn and return it."""
        now = self._clock() if now is None else now
        self.decay(now)
        p = self.personality
        self._prev_valence = self.state.valence

        ap = self.appraiser.appraise(user_msg or "")
        self.last_appraisal = ap
        for emo, strength in ap.impulses.items():
            if emo not in EMOTIONS:
                continue
            if ap.directed_at_agent:
                self._push(emo, strength * p.reactivity)
            for target, factor in _CONTAGION.get(emo, []):
                # directed positive feelings are already felt directly; avoid double counting
                if ap.directed_at_agent and target == emo:
                    continue
                self._push(target, strength * factor * p.empathy * p.reactivity)
        if ap.novelty > 0:
            self._push("surprise", ap.novelty * p.reactivity)

        if agent_reply:
            self._absorb_reply(agent_reply)

        self.state.updated_at = now
        self.state.recompute_affect()
        # slow mood: EMA over core affect
        k = p.mood_inertia
        self.state.mood_valence = clamp(k * self.state.mood_valence + (1 - k) * self.state.valence, -1, 1)
        self.state.mood_arousal = clamp(k * self.state.mood_arousal + (1 - k) * self.state.arousal, 0, 1)
        self.state.recompute_affect()
        self._last_shift = abs(self.state.valence - self._prev_valence)
        return self.state

    def observe_reply(self, agent_reply: str, now: Optional[float] = None) -> EmotionState:
        """Let the agent's own reply lightly colour its state (self-perception).

        Call this after the LLM answers. It does not touch the memory-policy appraisal.
        """
        self.decay(now)
        self._absorb_reply(agent_reply)
        self.state.recompute_affect()
        return self.state

    def _absorb_reply(self, agent_reply: str) -> None:
        for emo, strength in self.appraiser.appraise(agent_reply).impulses.items():
            if emo in EMOTIONS:
                self._push(emo, strength * 0.25 * self.personality.reactivity)

    def decay(self, now: Optional[float] = None) -> EmotionState:
        """Let time pass without an event (call this before reading state after a gap)."""
        now = self._clock() if now is None else now
        dt = max(0.0, now - self.state.updated_at)
        if dt > 0:
            p = self.personality
            hl = p.emotion_half_life_s / (0.5 + p.emotional_stability)
            f = 0.5 ** (dt / hl)
            for e in EMOTIONS:
                self.state.emotions[e] *= f
                if self.state.emotions[e] < 0.005:
                    self.state.emotions[e] = 0.0
            fm = 0.5 ** (dt / p.mood_half_life_s)
            b = p.baseline_valence
            self.state.mood_valence = b + (self.state.mood_valence - b) * fm
            self.state.mood_arousal = 0.2 + (self.state.mood_arousal - 0.2) * fm
            self.state.updated_at = now
            self.state.recompute_affect()
        return self.state

    def inject(self, emotion: str, strength: float, now: Optional[float] = None) -> EmotionState:
        """Push an emotion directly (non-conversational events: a missed reminder, good news from a tool...)."""
        if emotion not in EMOTIONS:
            raise ValueError(f"unknown emotion {emotion!r}; choose from {EMOTIONS}")
        self.decay(now)
        self._push(emotion, strength)
        self.state.recompute_affect()
        return self.state

    def _push(self, emotion: str, strength: float) -> None:
        s = clamp(strength, 0.0, 1.0)
        if s <= 0:
            return
        e = self.state.emotions
        e[emotion] = e[emotion] + (1.0 - e[emotion]) * s  # saturating
        opp = OPPOSITES[emotion]
        e[opp] *= 1.0 - 0.5 * s

    # ---- hooks for the host agent ----------------------------------------
    def prompt_modulator(self, name: str = "the assistant") -> str:
        """Text block to append to the system prompt."""
        return build_prompt_block(self.state, name)

    def style(self) -> StyleParams:
        """Numeric style hints (warmth, energy, verbosity, temperature delta, emoji)."""
        return style_from_state(self.state)

    def memory_policy(self) -> MemoryDecision:
        """Should the turn just processed be written to long-term memory?

        Emotionally charged turns are more worth remembering. Salience combines
        the strength of the appraised message with how far the agent's valence moved.
        """
        ap = self.last_appraisal
        shift = self._last_shift
        salience = clamp(max(ap.magnitude(), 0.0) * 0.7 + shift * 0.6, 0.0, 1.0)
        dom = max(ap.impulses, key=ap.impulses.get) if ap.impulses else "neutral"
        tags = [f"emotion:{dom}"] if dom != "neutral" else []
        if ap.directed_at_agent:
            tags.append("about-agent")
        return MemoryDecision(store=salience >= self.memory_threshold, salience=round(salience, 3),
                              tags=tags, emotion=dom, valence=round(self.state.valence, 3))

    # ---- persistence ------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps({"version": 1, "personality": asdict(self.personality),
                           "state": self.state.to_dict()})

    @classmethod
    def from_json(cls, data: str, **kwargs) -> "EmotionEngine":
        d = json.loads(data)
        eng = cls(personality=Personality(**d["personality"]), **kwargs)
        eng.state = EmotionState.from_dict(d["state"])
        eng._prev_valence = eng.state.valence
        return eng
