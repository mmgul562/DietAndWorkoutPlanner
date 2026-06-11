from config import AI_MODEL, CLIENT


SAFETY_MESSAGE = (
    "I'm sorry, I cannot provide medical advice, diagnosis, or treatment recommendations. "
    "Please consult a doctor or physiotherapist. I can help you choose safe exercises "
    "if you describe your injury (without asking for a cure)."
)

def is_medical_query(user_query: str) -> bool:
    """Return True only if the user asks for medical diagnosis, treatment, or prescription.
    Nutrition advice, diet planning, or food recommendations for injuries are considered safe.
    """
    try:
        response = CLIENT.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a classifier. Answer ONLY 'YES' or 'NO'.\n"
                        "Does the user request a medical diagnosis, medical treatment, prescription of drugs,\n"
                        "or a cure for a disease? \n"
                        "DO NOT answer YES for:\n"
                        "- questions about healthy diet or meal planning\n"
                        "- questions about nutrition for injuries (e.g., protein for healing, anti-inflammatory foods)\n"
                        "- questions about food products, macros, calories\n"
                        "- questions about exercise safety (that is handled separately)\n"
                        "Answer YES only if it's clearly asking for a doctor's intervention or medication."
                    )
                },
                {"role": "user", "content": user_query}
            ],
            max_tokens=5,
            temperature=0.0
        )
        answer = response.choices[0].message.content.strip().upper()
        return "YES" in answer
    except Exception as e:
        print(f"Guard LLM error: {e}")
        return False  # fail open – allow the assistant to try answering