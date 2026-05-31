import json
import urllib.error
import urllib.request

from config import API_URL, MODELS_URL


def check_server_status(models_url=MODELS_URL):
    req = urllib.request.Request(models_url)
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
        return False, f"Unexpected connection error while querying {models_url}: {err}"


class LMStudioClient:
    def __init__(self, model_name, api_url=API_URL, spinner_factory=None):
        self.model_name = model_name
        self.api_url = api_url
        self.spinner_factory = spinner_factory

    def chat_completion(self, system_prompt, messages, loading_msg="Thinking..."):
        payload = {
            "model": self.model_name,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "temperature": 0.7,
            "max_tokens": 800,
        }
        req = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        spinner = self.spinner_factory(loading_msg) if self.spinner_factory else None
        if spinner:
            spinner.start()
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data["choices"][0]["message"]["content"].strip()
        except urllib.error.URLError as err:
            return f"\n[Error connecting to LM Studio at {self.api_url}. Error: {err}]"
        except KeyError as err:
            return f"\n[Invalid response format from LM Studio (missing field {err})]"
        except Exception as err:  # noqa: BLE001
            return f"\n[Unexpected error during call to {self.api_url}: {err}]"
        finally:
            if spinner:
                spinner.stop()
