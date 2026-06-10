from vector_store import VectorStore
from nutrition_db import NutritionDB
from agent import TrainingAssistant


SENTENCE_THRESHOLD = 12

def main():
    vs = VectorStore()
    ndb = NutritionDB()
    assistant = TrainingAssistant(vs, ndb)

    print("Workout assistant ready. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        while len(user_input) < SENTENCE_THRESHOLD:
            user_input += input("\t> ")

        print("\nAssistant is thinking...", end="")
        response = assistant.handle_message(user_input)
        print(f"\rAssistant: {response}\n")

if __name__ == "__main__":
    main()