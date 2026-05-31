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


def run_cli():
    print("=" * 60)
    print("      REAL-TIME PSYCHOMETRIC CHATBOT HARNESS      ")
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
    print("Type 'exit' or 'quit' to end the session.\n")

    client = LMStudioClient(model_name, spinner_factory=Spinner)
    harness = DynamicChatHarness(client)
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                print("\nGoodbye!")
                break

            bot_reply = harness.send_chat_message(user_input)
            print(f"\n[Engine Stats: MBTI={harness.current_mbti} | Interests={harness.interests}]")
            print(f"Chatbot: {bot_reply}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
