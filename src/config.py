import os
from dotenv import load_dotenv

load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
LMSTUDIO_MODEL = "llama3.3-8b-instruct"  # nazwa modelu w LM Studio

SQLITE_DB_PATH = "database/nutrition.db"
# SQLITE_DB_PATH = os.path.join(os.path.dirname(__file__), "nutrition.db")
CHROMA_DB_DIR = "database/chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"