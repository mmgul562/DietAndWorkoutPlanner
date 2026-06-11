import json
from training_agent import TrainingAssistant
from diet_agent import DietAssistant
from vector_store import VectorStore
from openai import OpenAI
from config import AI_MODEL, CLIENT, FOOD_DB_PATH


class UnifiedAssistant:
    def __init__(self, vector_store: VectorStore):
        self.training = TrainingAssistant(vector_store)
        self.diet = DietAssistant(vector_store, FOOD_DB_PATH)
        self.conversation_history = []  # single list of {"role": ..., "content": ...}

    def _classify_intent(self, user_msg: str) -> str:
        """Return 'training' or 'diet' based on the user's message."""
        # Use a short prompt with the last few messages for context
        context = self.conversation_history[-4:]  # last 2 exchanges
        messages = [
            {
                "role": "system",
                "content": (
                    "Classify the user's request as either 'training' or 'diet'.\n"
                    "- 'training' if it's about exercises, workout plans, injury prevention in sports, or specific muscles.\n"
                    "- 'diet' if it's about food, meals, nutrition, calories, or diet.\n"
                     "If the user asks how to do something, assume it's about training if not sure.\n"
                    "Answer with only one word: training or diet."
                )
            }
        ]
        if context:
            messages.append({"role": "system", "content": f"Recent conversation: {json.dumps(context)}"})
        messages.append({"role": "user", "content": user_msg})
        resp = CLIENT.chat.completions.create(
            model=AI_MODEL,
            messages=messages,
            max_tokens=10,
            temperature=0.0
        )
        intent = resp.choices[0].message.content.strip().lower()
        return intent if intent in ["training", "diet"] else "diet"  # fallback

    def clear_history(self):
        self.conversation_history = []

    def handle_message(self, user_msg: str) -> str:
        # 1. Classify intent
        intent = self._classify_intent(user_msg)

        # 2. Get response from the appropriate assistant (pass current history)
        if intent == "training":
            answer = self.training.handle_message(user_msg, self.conversation_history)
        else:
            answer = self.diet.generate_response(user_msg, self.conversation_history)

        # 3. Update unified history
        self.conversation_history.append({"role": "user", "content": user_msg})
        self.conversation_history.append({"role": "assistant", "content": answer})

        # 4. Trim history if too long (keep last 20 messages = 10 exchanges)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        return answer