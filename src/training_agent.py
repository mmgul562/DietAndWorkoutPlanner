import json
from openai import OpenAI
from config import AI_MODEL, CLIENT
from vector_store import VectorStore
from guardrails import is_medical_query, SAFETY_MESSAGE


SYSTEM_PROMPT = (
    "You are a knowledgeable and concise training assistant. "
    "You help users design safe, effective workout plans considering their injuries. "
    "You have access to a large database of exercises with instructions and safety notes. "
    "Rules:\n"
    "- Answer directly and briefly. No lengthy introductions.\n"
    "- If the user provides their fitness level, give specific recommendations: sets, reps, weight, equipment variations.\n"
    "- If the user has an injury, choose exercises that avoid contraindicated movements.\n"
    "- Never give medical advice. If asked for a diagnosis or treatment, politely decline.\n"
    "- Refer to previously suggested exercises when the user asks follow-up questions (e.g., about technique or modifications).\n"
    "- Use the context of the whole conversation."
)

class TrainingAssistant:
    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store

    def _get_full_context(self, history: list):
        """Return conversation as list of messages, capped to last N exchanges."""
        # Keep last 10 messages (5 exchanges) to save tokens
        return history[-10:]

    def extract_intent(self, user_msg: str, history: list) -> dict:
        # Use recent context to disambiguate references (like "that exercise")
        context = self._get_full_context(history)
        messages = [
            {
                "role": "system",
                "content": (
                    "Extract the target muscle group and mentioned injury from the user message. "
                    "If the user refers to a previously mentioned exercise or muscle, use context. "
                    "Return JSON: {\"target_muscle\": \"...\", \"injury\": \"...\"} "
                    "(null if none)."
                )
            }
        ]
        # Add a summary of last assistant response for context (optional)
        if context:
            messages.append({"role": "system", "content": f"Recent conversation: {json.dumps(context[-4:])}"})
        messages.append({"role": "user", "content": user_msg})
        resp = CLIENT.chat.completions.create(
            model=AI_MODEL,
            messages=messages,
            max_tokens=100,
            temperature=0.0
        )
        try:
            return json.loads(resp.choices[0].message.content.strip())
        except:
            return {"target_muscle": "unknown", "injury": None}

    def get_safe_exercises(self, target: str, injury: str = None, top_k=5):
        # Semantic search in vector DB
        query = f"exercises for {target} muscle"
        if injury:
            query += f" safe for {injury} injury"
        results = self.vs.search_exercises(query, n_results=top_k*3)
        candidates = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            candidates.append({
                "name": meta["name"],
                "body_parts": meta["body_parts"],
                "target": meta["target_muscles"],
                "secondary": meta["secondary_muscles"],
                "equipment": meta["equipments"],
                "instructions": doc,
                "safety_info": meta.get("safety_info", "")
            })
        if not injury:
            return candidates[:top_k]

        # LLM filtering against injury guidelines
        injury_guidelines = self.vs.search_injury_guidelines(injury, n_results=2)
        guidelines = "\n".join(injury_guidelines["documents"][0]) if injury_guidelines["documents"] else ""

        candidate_desc = "\n".join(
            [f"- {c['name']} ({c['target']}, equipment: {c['equipment']})" for c in candidates]
        )
        filter_prompt = (
            f"From the list below, pick up to {top_k} exercises that are SAFE for a person with {injury} injury.\n"
            f"Safety guidelines:\n{guidelines}\n\n"
            f"Exercises:\n{candidate_desc}\n\n"
            f"Return only the exercise names, one per line."
        )
        resp = CLIENT.chat.completions.create(
            model=AI_MODEL,
            messages=[{"role": "user", "content": filter_prompt}],
            max_tokens=100,
            temperature=0.2
        )
        chosen = [line.strip("- ").strip() for line in resp.choices[0].message.content.split("\n") if line.strip()]
        return [c for c in candidates if c["name"] in chosen][:top_k]

    def generate_response(self, user_msg: str, exercises: list, history: list) -> str:
        # Build full prompt with history and system instructions
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_msg})

        # Add retrieved exercises as context (hidden from user, but assistant will use)
        if exercises:
            context_text = "Retrieved exercises (use these to answer, but do not list them unless asked):\n"
            for ex in exercises:
                context_text += f"- {ex['name']} [{ex['target']}]: {ex['instructions'][:200]}...\n"
            messages.append({"role": "system", "content": context_text})

        # Add instruction to be concise
        messages.append({"role": "system", "content": "Now answer concisely, without greetings."})

        resp = CLIENT.chat.completions.create(
            model=AI_MODEL,
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        return resp.choices[0].message.content.strip()

    def handle_message(self, user_msg: str, history: list) -> str:
        if is_medical_query(user_msg):
            return SAFETY_MESSAGE  # no history update here – caller will append
        intent = self.extract_intent(user_msg, history)
        target = intent.get("target_muscle", "unknown")
        injury = intent.get("injury")
        exercises = []
        if target != "unknown" or injury:
            exercises = self.get_safe_exercises(target if target != "unknown" else "general", injury)
        answer = self.generate_response(user_msg, exercises, history)
        return answer