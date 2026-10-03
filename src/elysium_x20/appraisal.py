"""Appraisal: turning a piece of text into emotional impulses.

The built-in ``LexiconAppraiser`` is a small, transparent heuristic. It is
deliberately simple and English-only. Anything with an ``appraise`` method that
returns an :class:`Appraisal` can replace it (a classifier, an LLM call, ...).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Protocol

from .state import EMOTIONS


@dataclass
class Appraisal:
    """Result of appraising one message.

    impulses: emotion -> strength in [0, 1] expressed by the *text*.
    directed_at_agent: the text is about / aimed at the agent itself
        (e.g. "you are useless", "thank you"), so the agent feels it directly
        instead of merely empathising.
    novelty: 0..1, how unexpected the message looks (used for surprise).
    """

    impulses: Dict[str, float] = field(default_factory=dict)
    directed_at_agent: bool = False
    novelty: float = 0.0

    def magnitude(self) -> float:
        return max(self.impulses.values(), default=0.0)


class Appraiser(Protocol):
    def appraise(self, text: str) -> Appraisal: ...


# word (stem) -> (emotion, weight)
_LEX = {
    # joy
    "happy": ("joy", .6), "glad": ("joy", .5), "great": ("joy", .5), "awesome": ("joy", .7),
    "amazing": ("joy", .7), "wonderful": ("joy", .7), "love": ("joy", .7), "loved": ("joy", .7),
    "excited": ("joy", .7), "yay": ("joy", .6), "won": ("joy", .6), "passed": ("joy", .5),
    "proud": ("joy", .6), "fun": ("joy", .5), "good": ("joy", .35), "nice": ("joy", .35),
    "lol": ("joy", .3), "haha": ("joy", .4), "perfect": ("joy", .6),
    # trust / affection
    "thanks": ("trust", .5), "thank": ("trust", .5), "appreciate": ("trust", .6),
    "trust": ("trust", .6), "friend": ("trust", .4), "safe": ("trust", .4), "helpful": ("trust", .5),
    # sadness
    "sad": ("sadness", .6), "cry": ("sadness", .6), "crying": ("sadness", .7), "lonely": ("sadness", .7),
    "miss": ("sadness", .5), "depressed": ("sadness", .8), "hopeless": ("sadness", .8),
    "lost": ("sadness", .5), "failed": ("sadness", .6), "hurt": ("sadness", .6), "grief": ("sadness", .8),
    "died": ("sadness", .8), "tired": ("sadness", .3), "alone": ("sadness", .6), "unhappy": ("sadness", .6), "awful": ("sadness", .5),
    # anger
    "angry": ("anger", .7), "furious": ("anger", .9), "hate": ("anger", .8), "hated": ("anger", .8),
    "stupid": ("anger", .6), "useless": ("anger", .6), "annoying": ("anger", .5), "annoyed": ("anger", .5),
    "idiot": ("anger", .7), "shut": ("anger", .4), "worst": ("anger", .5), "terrible": ("anger", .5),
    # fear
    "afraid": ("fear", .7), "scared": ("fear", .7), "worried": ("fear", .5), "anxious": ("fear", .6),
    "nervous": ("fear", .5), "panic": ("fear", .8), "terrified": ("fear", .9), "worry": ("fear", .5),
    # disgust
    "disgusting": ("disgust", .7), "gross": ("disgust", .6), "ew": ("disgust", .5),
    # surprise
    "wow": ("surprise", .6), "omg": ("surprise", .6), "unexpected": ("surprise", .5),
    "suddenly": ("surprise", .4), "shocked": ("surprise", .7), "whoa": ("surprise", .6),
    # anticipation
    "tomorrow": ("anticipation", .3), "soon": ("anticipation", .3), "waiting": ("anticipation", .4),
    "cant wait": ("anticipation", .7), "looking": ("anticipation", .2), "plan": ("anticipation", .3),
    "hope": ("anticipation", .4), "curious": ("anticipation", .4),
}
_NEGATORS = {"not", "no", "never", "dont", "don't", "isnt", "isn't", "cant", "can't", "wont", "won't",
             "didnt", "didn't", "aint", "ain't", "wasnt", "wasn't", "hardly"}
_BOOST = {"very": 1.4, "so": 1.4, "really": 1.4, "extremely": 1.6, "totally": 1.4, "super": 1.4,
          "absolutely": 1.5, "slightly": 0.6, "bit": 0.7, "kinda": 0.7, "somewhat": 0.7}
_POSITIVE = {"joy", "trust", "anticipation"}
_SELF = {"you", "your", "youre", "you're", "ur", "u"}
_TOKEN = re.compile(r"[a-z']+")


class LexiconAppraiser:
    """Word-level heuristic: lexicon + negation + intensifiers + punctuation."""

    def appraise(self, text: str) -> Appraisal:
        raw = text or ""
        low = raw.lower().replace("can't wait", "cant wait")
        toks = _TOKEN.findall(low)
        imp: Dict[str, float] = {e: 0.0 for e in EMOTIONS}
        directed = False
        for i, tok in enumerate(toks):
            hit = _LEX.get(tok)
            if tok == "cant" and i + 1 < len(toks) and toks[i + 1] == "wait":
                hit = _LEX["cant wait"]
            if not hit:
                continue
            emo, w = hit
            window = toks[max(0, i - 3):i]
            for prev in window:
                if prev in _BOOST:
                    w *= _BOOST[prev]
            if any(p in _NEGATORS for p in window) and tok != "cant":
                # negated positive -> mild sadness; negated negative -> mild relief (joy)
                emo, w = ("sadness", w * 0.5) if emo in _POSITIVE else ("joy", w * 0.35)
            imp[emo] = min(1.0, imp[emo] + w * (1.0 if imp[emo] == 0 else 0.6))
            near = toks[max(0, i - 4):i + 4]
            if any(p in _SELF for p in near) or tok in {"thanks", "thank"}:
                directed = True
        # punctuation / caps energy amplify whatever is there
        excl = min(raw.count("!"), 3)
        letters = [c for c in raw if c.isalpha()]
        shouting = len(letters) >= 6 and sum(c.isupper() for c in letters) / len(letters) > 0.7
        amp = 1.0 + 0.1 * excl + (0.2 if shouting else 0.0)
        imp = {k: min(1.0, v * amp) for k, v in imp.items() if v > 0}
        novelty = min(1.0, 0.08 * excl + (0.3 if "?!" in raw or "!?" in raw else 0.0)
                      + imp.get("surprise", 0.0))
        return Appraisal(impulses=imp, directed_at_agent=directed, novelty=novelty)
