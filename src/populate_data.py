import json
from vector_store import VectorStore


INJURIES_PATH = "sources/injuries.json"
EXERCISES_PATH = "sources/exercises.json"
DIET_PATH = "sources/diet_knowledge.json"

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
        print(f"Could not find file with exercise knowledge ({EXERCISES_PATH}).")
        return

    try:
        with open(DIET_PATH, "r") as f:
            diet_knowledge = json.load(f)
    except FileNotFoundError:
        print(f"Could not find file with diet knowledge ({DIET_PATH}).")
        return

    vs = VectorStore()
    print("Indexing in ChromaDB...")
    vs.index_exercises(exercises_knowledge)
    vs.index_injury_knowledge(injuries_knowledge)
    vs.index_diet_knowledge(diet_knowledge)
    print("Database ready.")


if __name__ == "__main__":
    main()