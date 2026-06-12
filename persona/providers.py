"""
LLM provider abstraction for the Persona profiler.

The profiler needs to make LLM calls to analyze conversations. This module
provides pluggable backends so the profiler can use any LLM service.

The main chat does NOT go through Persona — customers keep their own LLM
integration. Only the background profiler needs an LLM.
"""

import json
import urllib.error
import urllib.request


class ProfilerProvider:
    """Abstract base for profiler LLM backends.

    Subclass this and implement `complete()` for custom providers.
    """

    def complete(self, system_prompt, user_prompt):
        """Send a completion request and return the response text.

        Args:
            system_prompt: The profiler system instructions.
            user_prompt: The transcript to analyze.

        Returns:
            Response text string.
        """
        raise NotImplementedError


class OpenAICompatibleProvider(ProfilerProvider):
    """Works with any OpenAI-compatible API.

    Supports: OpenAI, LM Studio, Ollama, Together, Groq, vLLM,
    Azure OpenAI, Anyscale, Fireworks, and any other endpoint
    that implements the /v1/chat/completions spec.
    """

    def __init__(self, base_url="http://127.0.0.1:1234/v1",
                 model=None, api_key=None, timeout=30):
        """
        Args:
            base_url: API base URL (e.g., "https://api.openai.com/v1").
            model: Model identifier. If None, auto-detects from /v1/models.
            api_key: API key (optional for local servers).
            timeout: Request timeout in seconds.
        """
        # Normalize: strip trailing slash
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._model = model

    @property
    def model(self):
        if self._model is None:
            self._model = self._detect_model()
        return self._model

    def _detect_model(self):
        """Auto-detect the loaded model from the /v1/models endpoint."""
        url = f"{self.base_url}/models"
        req = urllib.request.Request(url)
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = data.get("data", [])
                if models:
                    return models[0]["id"]
        except Exception:
            pass
        return "default"

    def _build_headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def complete(self, system_prompt, user_prompt):
        """Make a chat completion request."""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,   # Low temp for consistent profiling
            "max_tokens": 200,    # Profiler output is very short
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._build_headers(),
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip()
        except (urllib.error.URLError, KeyError, Exception) as err:
            return f"[Profiler error: {err}]"

    def check_health(self):
        """Check if the provider is reachable and has a model loaded.

        Returns:
            (ok: bool, detail: str)
        """
        url = f"{self.base_url}/models"
        req = urllib.request.Request(url)
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = data.get("data", [])
                if not models:
                    return False, "Connected but no model loaded."
                return True, models[0]["id"]
        except urllib.error.URLError as err:
            reason = err.reason if hasattr(err, "reason") else str(err)
            return False, f"Cannot connect: {reason}"
        except Exception as err:
            return False, f"Unexpected error: {err}"


class NullProvider(ProfilerProvider):
    """No-op provider — skips all profiler LLM calls.

    Useful when the customer only wants heuristic-based profiling
    (message length, slang detection, emoji detection) without the
    deep LLM-based analysis.
    """

    def complete(self, system_prompt, user_prompt):
        return "COGNITIVE: E\nENGAGEMENT: D\nINTERESTS: none\nSLANG: none"

    def check_health(self):
        return True, "NullProvider (heuristics only)"
