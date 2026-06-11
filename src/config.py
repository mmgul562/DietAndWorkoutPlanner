import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DB_DIR = "database/chroma_db"
FOOD_DB_PATH = "database/food.db"

# AI_MODEL = "llama3.3-8b-instruct"
AI_MODEL = "gpt-4.1-mini"
# CLIENT = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key="not-needed")  # <-- local model llama
CLIENT = OpenAI(api_key=OPENAI_API_KEY)  # <-- chat gpt
