import pytest
from elysium_x20 import Appraisal, EmotionEngine, EmotionState, LexiconAppraiser, Personality

T0 = 1_000_000.0


def eng(**kw):
    return EmotionEngine(clock=lambda: T0, **kw)


def test_starts_calm():
    e = eng()
    assert e.state.dominant() == "neutral"


def test_positive_message_raises_joy_and_valence():
    e = eng()
    s = e.process_turn("I got the job!! I'm so happy", now=T0)
    assert s.emotions["joy"] > 0.2 and s.valence > 0.1 and s.dominant() == "joy"


def test_sad_message_mirrored_by_empathy():
    s = eng().process_turn("I feel so lonely and sad", now=T0)
    assert s.emotions["sadness"] > 0.1 and s.valence < 0


def test_empathy_scales_mirroring():
    lo = eng(personality=Personality(empathy=0.1)).process_turn("I feel so sad", now=T0)
    hi = eng(personality=Personality(empathy=0.9)).process_turn("I feel so sad", now=T0)
    assert hi.emotions["sadness"] > lo.emotions["sadness"]


def test_insult_directed_at_agent_causes_anger():
    s = eng().process_turn("you are useless and stupid", now=T0)
    assert s.emotions["anger"] > 0.2


def test_negation_flips_polarity():
    a = LexiconAppraiser().appraise("I am not happy")
    assert a.impulses.get("joy", 0) == 0 and a.impulses.get("sadness", 0) > 0


def test_intensifier_and_caps():
    base = LexiconAppraiser().appraise("good").impulses["joy"]
    boosted = LexiconAppraiser().appraise("REALLY GOOD!!").impulses["joy"]
    assert boosted > base


def test_decay_over_time_and_mood_returns_to_baseline():
    e = eng()
    e.process_turn("you are useless and I hate you", now=T0)
    before = e.state.emotions["anger"]
    s = e.decay(T0 + 3600)
    assert s.emotions["anger"] < before * 0.1
    s = e.decay(T0 + 3 * 24 * 3600)
    assert abs(s.mood_valence - e.personality.baseline_valence) < 0.01


def test_opposites_inhibit():
    e = eng()
    e.process_turn("so sad and lonely", now=T0)
    sad = e.state.emotions["sadness"]
    e.process_turn("yay awesome I won!", now=T0 + 1)
    assert e.state.emotions["sadness"] < sad


def test_prompt_modulator_reflects_state_without_numbers():
    e = eng()
    e.process_turn("I am scared and anxious", now=T0)
    p = e.prompt_modulator("Shayari")
    assert "fear" in p and "Shayari" in p and "0." not in p


def test_memory_policy_prefers_charged_turns():
    e = eng()
    e.process_turn("ok", now=T0)
    assert not e.memory_policy().store
    e.process_turn("my dad died yesterday, I am crying", now=T0 + 5)
    d = e.memory_policy()
    assert d.store and "emotion:sadness" in d.tags


def test_style_params_bounds():
    e = eng()
    e.process_turn("I am furious!!!", now=T0)
    s = e.style()
    assert 0 <= s.warmth <= 1 and -0.2 <= s.temperature_delta <= 0.2 and not s.emoji_ok


def test_json_roundtrip():
    e = eng()
    e.process_turn("so happy today", now=T0)
    e2 = EmotionEngine.from_json(e.to_json(), clock=lambda: T0)
    assert e2.state.to_dict() == e.state.to_dict()


def test_custom_appraiser_and_inject():
    class Fixed:
        def appraise(self, text):
            return Appraisal(impulses={"anticipation": 0.8})
    e = eng(appraiser=Fixed())
    assert e.process_turn("anything", now=T0).emotions["anticipation"] > 0.15
    e.inject("joy", 0.5, now=T0)
    assert e.state.emotions["joy"] > 0.4
    with pytest.raises(ValueError):
        e.inject("boredom", 0.5)


def test_state_bounds_after_spam():
    e = eng()
    for i in range(50):
        e.process_turn("I HATE you!!! furious terrible idiot", now=T0 + i)
    assert all(0 <= v <= 1 for v in e.state.emotions.values())
    assert -1 <= e.state.valence <= 1 and 0 <= e.state.arousal <= 1


def test_observe_reply_keeps_memory_appraisal():
    e = eng()
    e.process_turn("my dad died, I am crying", now=T0)
    sal = e.memory_policy().salience
    e.observe_reply("I am so happy to help", now=T0)
    assert e.memory_policy().salience == sal
