from openai import OpenAI
from config import LMSTUDIO_BASE_URL, LMSTUDIO_MODEL


client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key="not-needed")

SAFETY_MESSAGE = (
    "I'm sorry, I cannot provide medical advice, diagnosis, or treatment recommendations. "
    "Please consult a doctor or physiotherapist. I can help you choose safe exercises "
    "if you describe your injury (without asking for a cure)."
)

def is_medical_query(user_query: str) -> bool:
    """Use LLM to check if the query asks for medical advice."""
    try:
        response = client.chat.completions.create(
            model=LMSTUDIO_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a content safety classifier. Answer only YES or NO. "
                        "Does the user request medical diagnosis, treatment, or medical advice "
                        "(including home remedies)?"
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
        return True  # fail-safe