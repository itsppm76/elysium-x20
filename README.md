---
license: mit
library_name: elysium-x20
language: [en]
tags: [emotion, affect, agents, companion, llm, appraisal, project-nhe]
---

# Elysium X 20

An open-source **emotion-state engine** for AI agents and companion models. Part of [Project NHE](https://projectnhe.tech).

Elysium X 20 sits between the user and your LLM. It keeps a persistent emotional state, nudges it with every conversation turn, lets it fade over time, and hands you three things to act on: a **system-prompt block**, **style parameters**, and a **memory decision**. No dependencies, pure Python.

## Philosophy

- **Functional, not claimed.** The engine models a *state that changes behaviour*. It makes no claim that an agent "really feels". Your product decides how to describe that to users.
- **State, not vibes in the prompt.** Most companions fake emotion by writing "be warm" in the system prompt. Here emotion is data: it accumulates, decays, has a slow mood, and persists across sessions.
- **Transparent and swappable.** The default appraiser is a small word-list heuristic you can read in one file. Replace it with a classifier or an LLM call without touching the rest.
- **Emotion shapes memory.** Charged moments are worth remembering. The engine tells you which ones.

## Install

```bash
pip install elysium-x20          # once published to PyPI
pip install git+https://github.com/itsppm76/elysium-x20   # or from the repo
pip install -e .                 # from a clone
```

## Quick start

```python
from elysium_x20 import EmotionEngine, Personality

engine = EmotionEngine(Personality(empathy=0.7))

engine.process_turn("I passed my exam!! so happy")     # feel the user's message
system = BASE_PROMPT + "\n\n" + engine.prompt_modulator("Nova")
reply = my_llm(system, user_msg)
engine.observe_reply(reply)                              # optional: reply colours state

decision = engine.memory_policy()                        # remember this turn?
if decision.store:
    save_memory(user_msg, tags=decision.tags, salience=decision.salience)

engine.style()            # warmth / energy / verbosity / temperature_delta / emoji_ok
engine.to_json()          # persist; EmotionEngine.from_json(...) to restore
```

See `examples/companion_loop.py` for a runnable loop.

## How it works

1. **Appraise.** The message becomes emotional *impulses* over Plutchik's eight emotions (joy, trust, fear, surprise, sadness, disgust, anger, anticipation). Negation, intensifiers, `!` and ALL CAPS are handled. Text aimed at the agent ("you are useless", "thank you") is flagged `directed_at_agent`.
2. **Update.** Directed feelings hit the agent fully. Otherwise the user's emotion is *mirrored* in proportion to `Personality.empathy` (user sadness gives the agent some sadness plus trust, user anger gives mild unease, not anger). Updates saturate at 1 and suppress the opposite emotion (joy vs sadness, fear vs anger, ...).
3. **Core affect.** Valence and arousal are derived from the emotions and blended with a slow **mood**.
4. **Decay.** Emotions halve every ~10 minutes (scaled by `emotional_stability`); mood drifts back to `baseline_valence` over hours. Time is injectable (`now=`), so behaviour is deterministic and testable.
5. **Modulate.** `prompt_modulator()`, `style()` and `memory_policy()` turn the state into things your agent can use.

## Plugging in a better appraiser

```python
from elysium_x20 import Appraisal, EmotionEngine

class MyAppraiser:
    def appraise(self, text: str) -> Appraisal:
        scores = my_classifier(text)             # {"joy": 0.7, ...}
        return Appraisal(impulses=scores, directed_at_agent=False)

engine = EmotionEngine(appraiser=MyAppraiser())
```

## Honest limitations

- The built-in appraiser is a **keyword heuristic**: English only, no sarcasm, no context beyond a 3-word window. It is a baseline, not a state-of-the-art emotion recognizer.
- The numbers (half-lives, mirroring factors, thresholds) are **hand-tuned defaults**, not fitted to human data. No benchmarks are claimed. Tune them for your product.
- It is not a safety system and not a mental-health tool. Do not use it to detect crises or to manipulate users' emotions.

## Roadmap

- **Elysium X 20 FR**: a trained appraisal model for this interface. Not released yet; this repo contains no trained weights.
- Multilingual lexicons, per-user relationship memory, an optional LLM-based appraiser.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT. See `LICENSE`.
