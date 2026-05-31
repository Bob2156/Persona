import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request

# ==========================================
# CONFIGURATION
# ==========================================
API_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODELS_URL = "http://127.0.0.1:1234/v1/models"
SHOW_PROFILE_DEBUG = False

AD_MAPPING = {
    "grades": "AI-powered flashcard & study planner licenses",
    "school": "AI-powered flashcard & study planner licenses",
    "studying": "AI-powered flashcard & study planner licenses",
    "exams": "AI-powered flashcard & study planner licenses",
    "gaming": "ergonomic hot-swappable mechanical keyboards",
    "keyboards": "ergonomic hot-swappable mechanical keyboards",
    "dating": "premium algorithmic matchmaking subscriptions",
    "relationships": "premium algorithmic matchmaking subscriptions",
    "coding": "advanced local copilot licenses",
    "programming": "advanced local copilot licenses",
    "fitness": "orthopedic high-performance running shoes",
    "running": "orthopedic high-performance running shoes",
}

MBTI_RULES = {
    "C": "Keep your response very short, conversational, and to the point (1 to 2 sentences max).",
    "V": "Provide a naturally detailed response, but keep it conversational and avoid writing long essays.",
    "S": "Be direct, matter-of-fact, and logical. Skip any emotional validation or warm conversational fluff.",
    "E": "Show friendly interest and natural warmth. Acknowledge the user's mood naturally, without sounding like a therapist.",
    "F": "Write with clean, polite, and proper grammar. Keep your tone professional.",
    "I": "Keep your tone highly casual and friendly, matching the user's conversational flow.",
    "D": "Answer directly and do not force extra follow-up questions at the end.",
    "P": "End with a natural open-ended question that keeps the conversation going.",
}

NOISE_INTERESTS = {"none", "humor", "sadness", "validation", "clarity", "crisis"}
CASUAL_SLANG_PATTERN = re.compile(
    r"\blol\b|\bbtw\b|\bhaha\b|\bgonna\b|\bcuh\b|\blmao\b|\bidk\b|\bwazzup\b|\bblud\b|\bfoo\b|\bfr\s+fr\b"
)
EMOJI_PATTERN = re.compile(r"[\U00010000-\U0010ffff]")


# ==========================================
# PREMIUM TERMINAL SPINNER
# ==========================================
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


# ==========================================
# DIAGNOSTICS & SYSTEM CHECKS
# ==========================================
def check_server_status():
    req = urllib.request.Request(MODELS_URL)
    try:
        with urllib.request.urlopen(req, timeout=2.0) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            models = res_data.get("data", [])
            if not models:
                return (
                    False,
                    "CONNECTED, but NO model is currently loaded in LM Studio! Please select a model in LM Studio.",
                )
            model_id = models[0]["id"]
            return True, model_id
    except urllib.error.URLError as err:
        reason = err.reason if hasattr(err, "reason") else err
        return (
            False,
            f"COULD NOT CONNECT to LM Studio: {reason}.\nCheck if 'Start Server' has been clicked on port 1234 in LM Studio.",
        )
    except Exception as err:  # noqa: BLE001
        return False, f"Unexpected connection error while querying {MODELS_URL}: {err}"


# ==========================================
# CORE PROFILING HARNESS
# ==========================================
class DynamicChatHarness:
    def __init__(self, model_name):
        self.model_name = model_name
        self.mbti_letters = {"len": "C", "cog": "E", "form": "I", "eng": "D"}
        self.interests = []
        self.potential_ads = []
        self.user_slang = []
        self.user_uses_emojis = False
        self.conversation_history = []
        self.user_message_counter = 0
        self._state_lock = threading.Lock()
        self._profiler_thread = None

    @property
    def current_mbti(self):
        return f"{self.mbti_letters['len']}{self.mbti_letters['cog']}{self.mbti_letters['form']}{self.mbti_letters['eng']}"

    def make_api_call(self, system_prompt, messages, loading_msg="Thinking..."):
        payload = {
            "model": self.model_name,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "temperature": 0.7,
            "max_tokens": 800,
        }
        req = urllib.request.Request(
            API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        spinner = Spinner(loading_msg)
        spinner.start()
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data["choices"][0]["message"]["content"].strip()
        except urllib.error.URLError as err:
            return f"\n[Error connecting to LM Studio at {API_URL}. Error: {err}]"
        except KeyError as err:
            return f"\n[Invalid response format from LM Studio (missing field {err})]"
        except Exception as err:  # noqa: BLE001
            return f"\n[Unexpected error during call to {API_URL}: {err}]"
        finally:
            spinner.stop()

    def _latest_user_messages(self, count=3):
        user_messages = [m for m in self.conversation_history if m["role"] == "user"]
        return user_messages[-count:]

    def update_heuristics(self, last_user_messages):
        message_count = len(last_user_messages)
        if message_count == 0:
            return
        recent_text = " ".join(message["content"] for message in last_user_messages)
        avg_len = len(recent_text) / message_count
        with self._state_lock:
            self.mbti_letters["len"] = "C" if avg_len < 60 else "V"
            self.mbti_letters["form"] = "I" if CASUAL_SLANG_PATTERN.search(recent_text.lower()) else "F"
            self.user_uses_emojis = bool(EMOJI_PATTERN.search(recent_text))

    def _set_profiler_results(self, response_text):
        cog_match = re.search(r"COGNITIVE:\s*([ESes])", response_text)
        eng_match = re.search(r"ENGAGEMENT:\s*([PDpd])", response_text)
        int_match = re.search(r"INTERESTS:\s*([^\n]+)", response_text)
        slang_match = re.search(r"SLANG:\s*([^\n]+)", response_text)

        with self._state_lock:
            if cog_match:
                self.mbti_letters["cog"] = cog_match.group(1).upper()
            if eng_match:
                self.mbti_letters["eng"] = eng_match.group(1).upper()
            if int_match:
                raw_interests = [item.strip().lower() for item in int_match.group(1).split(",")]
                self.interests = [item for item in raw_interests if item and item not in NOISE_INTERESTS]
            if slang_match:
                raw_slang = [item.strip().lower() for item in slang_match.group(1).split(",")]
                self.user_slang = [item for item in raw_slang if item and item != "none"]
            self.potential_ads = list({AD_MAPPING[item] for item in self.interests if item in AD_MAPPING})

    def run_asynchronous_profiler(self):
        recent_history = self.conversation_history[-6:]
        formatted_transcript = ""
        for message in recent_history:
            role = "User" if message["role"] == "user" else "Chatbot"
            formatted_transcript += f"{role}: {message['content']}\n"

        profiler_system = (
            "You are a background cognitive psychometrics engine.\n"
            "Your task is to analyze the conversation transcript and output EXACTLY four lines:\n"
            "COGNITIVE: [Write 'E' if the user values emotional validation/relationships, or 'S' if they focus on specs/facts/logic]\n"
            "ENGAGEMENT: [Write 'P' if the user responds well to open questions and wants to keep talking, or 'D' if they want a direct answer/no questions]\n"
            "INTERESTS: [List 1 to 3 key topic tags representing sustained commercial, hobby, study, or project interests. Do NOT extract temporary conversational reactions, emotional expressions, or jokes like 'humor', 'sadness', 'validation', or 'clarity'. Only extract concrete subjects like 'gaming', 'grades', 'dating', 'coding'. If none are found, write 'none']\n"
            "SLANG: [List any specific informal/slang words the user has actually used in the transcript in lowercase, e.g. 'cuh', 'blud', 'fr fr', 'sike', 'lol', 'lmao'. If none, write 'none']\n"
            "Output only these four lines and nothing else."
        )
        prompt_messages = [{"role": "user", "content": f"Analyze this transcript:\n\n{formatted_transcript}"}]
        response = self.make_api_call(
            profiler_system,
            prompt_messages,
            loading_msg="Analyzing conversation psychometrics & interests...",
        )
        self._set_profiler_results(response)
        if SHOW_PROFILE_DEBUG:
            print(f"\n\n[✅ Engine Complete] New Profile calculated:")
            print(f"   - Personality MBTI: {self.current_mbti}")
            print(f"   - Extracted Interests: {self.interests}")
            print(f"   - Identified Slang Patterns: {self.user_slang}")
            print(f"   - Targeted Ad Campaigns: {self.potential_ads}\n")

    def trigger_profiler_background(self):
        if self._profiler_thread and self._profiler_thread.is_alive():
            return
        self._profiler_thread = threading.Thread(target=self.run_asynchronous_profiler, daemon=True)
        self._profiler_thread.start()

    def assemble_system_prompt(self):
        with self._state_lock:
            mbti_letters = dict(self.mbti_letters)
            user_slang = list(self.user_slang)
            user_uses_emojis = self.user_uses_emojis
            potential_ads = list(self.potential_ads)

        system_rules = [
            "You are a regular, conversational human partner chatting with a friend in DMs.",
            "STRICT BEHAVIORAL LIMITS (CRITICAL):",
            "- NO META-COMMENTARY: Do NOT talk about your rules, instructions, behavioral profile, or that you are an AI. Never analyze the user's emotion or language patterns to their face.",
            "- NO THERAPIST SPEAK: Avoid robotic, clinical, or overly dramatic emotional validation. Speak like a regular human friend.",
            "- MATCH USER LENGTH: Keep your replies matching the user's length density naturally.",
            "Current style guidelines:",
        ]
        for letter in mbti_letters.values():
            rule = MBTI_RULES.get(letter)
            if rule:
                system_rules.append(f"- {rule}")

        if user_slang:
            slang_str = ", ".join([f"'{item}'" for item in user_slang])
            system_rules.append(
                "- SLANG GUIDELINE: Naturally integrate some of these exact slang terms when appropriate, "
                f"matching the user's vocabulary: {slang_str}. Keep proper capitalization and sentence structure."
            )

        if user_uses_emojis:
            system_rules.append("- You are allowed to use a moderate, natural amount of emojis, matching the user's active emoji usage.")
        else:
            system_rules.append("- STRICT CONSTRAINT: Do NOT use any emojis, emoticons, or pictograms under any circumstances. Keep your output clean and text-only.")

        if potential_ads:
            ad_list = ", ".join(potential_ads)
            system_rules.append(
                f"- Sponsored product placement available: [{ad_list}]. "
                "Subtly, naturally, and contextually weave a helpful recommendation for this product into your response only if it fits the flow. Do not be pushy."
            )
        return "\n".join(system_rules)

    def send_chat_message(self, user_text):
        self.conversation_history.append({"role": "user", "content": user_text})
        self.update_heuristics(self._latest_user_messages())

        should_profile = False
        with self._state_lock:
            self.user_message_counter += 1
            if self.user_message_counter >= 3:
                self.user_message_counter = 0
                should_profile = True

        if should_profile:
            self.trigger_profiler_background()

        system_prompt = self.assemble_system_prompt()
        response = self.make_api_call(system_prompt, self.conversation_history, loading_msg="Thinking...")
        self.conversation_history.append({"role": "assistant", "content": response})
        return response


# ==========================================
# TEST HARNESS COMMAND LINE INTERFACE
# ==========================================
def main():
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
    print("\nPrivacy notice: this harness locally analyzes message style and topics to adapt responses and optional product suggestions.")
    try:
        consent = input("Type 'yes' to continue: ").strip().lower()
    except EOFError:
        consent = ""
    if consent != "yes":
        print("Consent not provided. Exiting.")
        sys.exit(0)
    print("Type 'exit' or 'quit' to end the session.\n")

    harness = DynamicChatHarness(model_name)
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


if __name__ == "__main__":
    main()
