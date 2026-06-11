from main_agent import UnifiedAssistant
from vector_store import VectorStore
import os


SENTENCE_THRESHOLD = 12

HELP_MESSAGE = f"""{'=' * 60}
Welcome to Diet & Training Assistant!

Commands:
\t!exit  -  exit the program
\t!reset -  reset the conversation
\t!clear -  clear the screen
{'=' * 60}
"""

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')
    print(HELP_MESSAGE)

def main():
    vs = VectorStore()
    assistant = UnifiedAssistant(vs)

    print(HELP_MESSAGE)
    while True:
        user_input = input("You: ")
        input_lower = user_input.lower()
        if input_lower == "!exit":
            break
        elif input_lower == "!clear":
            clear_screen()
            continue
        elif input_lower == "!reset":
            assistant.clear_history()
            clear_screen()
            continue
        while len(user_input) < SENTENCE_THRESHOLD:
            user_input += input("\t> ")

        print("\nAssistant is thinking...", end="")
        response = assistant.handle_message(user_input)
        print(f"\rAssistant: {response}\n")

if __name__ == "__main__":
    main()