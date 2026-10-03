"""Minimal companion loop. Replace `fake_llm` with a real model call."""
from elysium_x20 import EmotionEngine, Personality

BASE_PROMPT = "You are Nova, a friendly companion. Keep replies short."
engine = EmotionEngine(Personality(empathy=0.7))
memory = []


def fake_llm(system_prompt: str, user: str) -> str:
    return "(a real LLM would answer here, shaped by the system prompt)"


for user_msg in ["hey! got great news, I passed my exam!!",
                 "but my best friend failed and I feel awful",
                 "you are useless, you never help"]:
    engine.process_turn(user_msg)                      # 1) feel the user's message
    system = BASE_PROMPT + "\n\n" + engine.prompt_modulator("Nova")   # 2) shape the prompt
    reply = fake_llm(system, user_msg)
    engine.observe_reply(reply)                  # 3) (optional) let the reply colour the state
    decision = engine.memory_policy()                 # 4) decide what to remember
    print(f"user: {user_msg}\n  state: {engine.state}\n  style: {engine.style()}")
    if decision.store:
        memory.append({"text": user_msg, "tags": decision.tags, "salience": decision.salience})
print("memories:", memory)
