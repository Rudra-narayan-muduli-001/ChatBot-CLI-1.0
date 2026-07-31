import requests
import json
from datetime import datetime
from uuid import uuid4

try:
    from config import ASSISTANT_NAME
except ImportError:
    ASSISTANT_NAME = "Assistant"

BASE_URL = "http://localhost:8000"
SESSION_ID = None
CURRENT_MODE = None


def print_header():
    print("\n" + "="*60)
    print("ChatBot CLI - General & Realtime Chat")
    print("="*60)
    print("\nModes:")
    print("  1 = General Chat (pure LLM, no web search)")
    print("  2 = Realtime Chat (with Tavily search)")
    print("\nCommands:")
    print("  /history - See chat history")
    print("  /clear - Start new session")
    print("  /quit - Exit")
    print("="*60 + "\n")


def get_user_input():
    try:
        choice = input("\nYou: ").strip()
        return choice
    except (KeyboardInterrupt, EOFError):
        return None


def send_message(message, mode):
    global SESSION_ID

    if not SESSION_ID:
        SESSION_ID = str(uuid4())

    endpoint = "/chat/realtime" if mode == "realtime" else "/chat"

    try:
        response = requests.post(
            f"{BASE_URL}{endpoint}",
            json={
                "message": message,
                "session_id": SESSION_ID
            },
            timeout=60 if mode == "realtime" else 30
        )

        if response.status_code == 200:
            data = response.json()
            SESSION_ID = data.get("session_id", SESSION_ID)
            return data.get("response", "No response")
        else:
            try:
                err = response.json()
                if isinstance(err.get("detail"), str):
                    return f"Error: {err['detail']}"
            except Exception:
                pass
            return f"Error: {response.status_code} - {response.text}"

    except requests.exceptions.ConnectionError:
        return "Cannot connect to backend. Start it with: python run.py"
    except requests.exceptions.Timeout:
        return "Request timed out. Try a simpler query."
    except Exception as e:
        return f"Error: {str(e)}"


def get_chat_history():
    if not SESSION_ID:
        return "No active session"

    try:
        response = requests.get(
            f"{BASE_URL}/chat/history/{SESSION_ID}",
            timeout=10
        )

        if response.status_code == 200:
            history = response.json()
            messages = history.get("messages", [])

            if not messages:
                return "No messages in this session"

            output = f"\nChat History ({len(messages)} messages):\n"
            output += "-" * 60 + "\n"

            for i, msg in enumerate(messages, 1):
                role = "You" if msg.get("role") == "user" else ASSISTANT_NAME
                content = msg.get("content", "")
                output += f"{i}. {role}: {content}\n"

            output += "-" * 60 + "\n"
            return output
        else:
            return "Could not retrieve history"

    except Exception as e:
        return f"Error retrieving history: {str(e)}"


def main():
    print_header()

    global SESSION_ID, CURRENT_MODE

    print("Tip: Select a mode (1 or 2) then type your messages.")
    print("    Both modes share the same session until you clear it.\n")
    print("Select mode first (1=General, 2=Realtime):\n")

    while True:
        try:
            user_input = get_user_input()
            if user_input is None:
                print("\nGoodbye!")
                break

            if user_input == "1":
                CURRENT_MODE = "general"
                print("Switched to GENERAL chat (pure LLM, no web search)\n")
                continue

            elif user_input == "2":
                CURRENT_MODE = "realtime"
                print("Switched to REALTIME chat (with Tavily web search)\n")
                continue

            elif user_input == "/history":
                print(get_chat_history())
                continue

            elif user_input == "/clear":
                SESSION_ID = None
                CURRENT_MODE = None
                print("\nSession cleared. Starting fresh!")
                print("Select mode again (1=General, 2=Realtime):\n")
                continue

            elif user_input in ["/quit", "/exit"]:
                print("\nGoodbye!")
                break

            elif user_input.startswith("/"):
                print(f"Unknown command: {user_input}")
                continue

            if not CURRENT_MODE:
                print("Please select a mode first (1=General or 2=Realtime)")
                continue

            message = user_input
            mode_label = "General" if CURRENT_MODE == "general" else "Realtime"
            print(f"{ASSISTANT_NAME} ({mode_label}): ", end="", flush=True)
            response = send_message(message, CURRENT_MODE)
            print(response)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
