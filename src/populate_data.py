import json
from vector_store import VectorStore

INJURIES_PATH = "sources/injuries.json"
EXERCISES_PATH = "sources/exercises.json"

def main():
    try:
        with open(INJURIES_PATH, 'r') as injuries_file:
            injuries_knowledge = json.load(injuries_file)
    except FileNotFoundError:
        print(f"Could not find file with injuries knowledge ({INJURIES_PATH}).")
        return
    try:
        with open(EXERCISES_PATH, "r") as exercises_file:
            exercises_knowledge = json.load(exercises_file)
    except FileNotFoundError:
        print(f"Could not find file with exercise knowledge ({INJURIES_PATH}).")
        return

    vs = VectorStore()
    print("Indexing in ChromaDB...")
    vs.index_exercises(exercises_knowledge)
    vs.index_injury_knowledge(injuries_knowledge)
    print("Database ready.")


if __name__ == "__main__":
    main()