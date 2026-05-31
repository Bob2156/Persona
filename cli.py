import json
import sys
import threading
import time

from harness import DynamicChatHarness
from lm_studio_client import LMStudioClient, check_server_status


class Spinner:
    def __init__(self, message="Thinking..."):
        self.message = message
        self.stop_event = threading.Event()
        self.thread = None

    def _spin(self):
        chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        idx = 0
        while not self.stop_event.is_set():
            sys.stdout.write(f"\r\033[93m{chars[idx % len(chars)]}\033[0m {self.message}")
            sys.stdout.flush()
            idx += 1
            time.sleep(0.08)
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def start(self):
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def stop(self):
        if self.thread:
            self.stop_event.set()
            self.thread.join()


def print_help():
    print("\nCommands:")
    print("  /help          Show this command list")
    print("  /profile       Show the current adaptation profile")
    print("  /ads on        Enable sponsored suggestions")
    print("  /ads off       Disable sponsored suggestions")
    print("  /reset         Clear conversation and profile state")
    print("  /exit          End the session\n")


def handle_command(user_input, harness):
    command = user_input.strip().lower()
    if command in {"/help", "help"}:
        print_help()
        return True
    if command in {"/exit", "/quit", "exit", "quit"}:
        print("\nGoodbye!")
        raise SystemExit(0)
    if command == "/profile":
        print("\nCurrent adaptation profile:")
        print(json.dumps(harness.profile_snapshot(), indent=2))
        return True
    if command == "/ads on":
        harness.set_sponsored_suggestions(True)
        print("\nSponsored suggestions enabled.")
        return True
    if command == "/ads off":
        harness.set_sponsored_suggestions(False)
        print("\nSponsored suggestions disabled.")
        return True
    if command == "/reset":
        harness.reset_session()
        print("\nSession profile and conversation history reset.")
        return True
    if command.startswith("/"):
        print("\nUnknown command. Type /help for available commands.")
        return True
    return False


def run_cli():
    print("=" * 60)
    print("      REAL-TIME ADAPTIVE CHATBOT HARNESS      ")
    print("=" * 60)
    print("[Diagnostics] Connecting to LM Studio server...")

    online, result = check_server_status()
    if not online:
        print("\nSERVER CONNECTION ERROR:")
        print(f"   {result}\n")
        print("Please verify:")
        print("  1. Open LM Studio.")
        print("  2. Load a model.")
        print("  3. Start the Developer server on port 1234.")
        print("\nExiting. Please start the server and run the script again.")
        sys.exit(1)

    model_name = result
    print("[Connected] Local LM Studio server detected!")
    print(f"[Model Loaded] Active model: '{model_name}'")
    print(
        "\nPrivacy notice: this harness locally analyzes message style and topics to adapt responses and optional product suggestions."
    )
    try:
        consent = input("Type 'yes' to continue: ").strip().lower()
    except EOFError:
        consent = ""
    if consent != "yes":
        print("Consent not provided. Exiting.")
        sys.exit(0)
    print("Type /help for commands, or /exit to end the session.\n")

    client = LMStudioClient(model_name, spinner_factory=Spinner)
    harness = DynamicChatHarness(client)
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if handle_command(user_input, harness):
                continue

            bot_reply = harness.send_chat_message(user_input)
            profile = harness.profile_snapshot()
            print(
                f"\n[Engine Stats: Style={profile['style_code']} | "
                f"Interests={profile['interests']} | Ads={'on' if profile['sponsored_suggestions_enabled'] else 'off'}]"
            )
            print(f"Chatbot: {bot_reply}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
