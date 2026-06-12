# Persona

Real-time user personality profiling middleware for any LLM.

Persona passively builds a behavioral profile as users chat — tracking communication style, interests, slang, and emoji preferences — then generates a compact personalization supplement for any system prompt. **2 lines of code** to make any AI feel like it knows the user.

## Quick Start

```python
from persona import Persona
from persona.providers import OpenAICompatibleProvider

# Point the profiler at any OpenAI-compatible endpoint
provider = OpenAICompatibleProvider(
    base_url="https://api.openai.com/v1",
    model="gpt-4o-mini",
    api_key="sk-..."
)

# Create a persona for a user
p = Persona(user_id="user_123", provider=provider, storage_dir="./profiles")

# --- In your chat loop ---

# 1. Enhance your system prompt (adds ~2-3 lines of personalization)
enhanced_prompt = p.enhance("You are a helpful coding assistant.")

# 2. Make your normal LLM call with the enhanced prompt
response = your_llm_call(enhanced_prompt, messages)

# 3. Feed the turn back to Persona
p.observe(user_msg, response)
```

That's it. Persona handles profiling, smoothing, persistence, and prompt generation automatically.

## What It Does

After a few messages, Persona builds a profile like this:

```json
{
  "user_id": "user_123",
  "style_code": "CEIP",
  "style_axes": {
    "length": "concise",
    "cognitive": "emotion-aware",
    "formality": "casual",
    "engagement": "open-ended"
  },
  "confidence": {"len": 5, "cog": 2, "form": 5, "eng": 2},
  "interests": ["gaming", "keyboards"],
  "slang": ["fr fr", "lol", "cuh"],
  "uses_emojis": false
}
```

And injects a compact supplement into your system prompt:

```
Keep replies short (1-2 sentences). Keep it casual, match their vibe.
Mirror their slang: 'fr fr', 'lol', 'cuh'. No emojis.
```

## Features

- **Provider-agnostic** — works with OpenAI, LM Studio, Ollama, Together, Groq, any OpenAI-compatible API
- **Minimal prompt footprint** — 2-3 lines injected, your AI's personality stays intact
- **Zero dependencies** — pure Python stdlib, nothing to install
- **Persistent profiles** — saved as JSON, improve over time across sessions
- **Confidence-gated** — only injects rules when it has enough data
- **Axis smoothing** — prevents single-message flip-flops via majority-vote windows
- **Enrichment hooks** — pluggable extensions for ads, tone, safety, etc.
- **Privacy-first** — all analysis runs locally, profiles stay on your infrastructure

## Enrichment Hooks

Hooks let you inject custom content based on the user's profile:

```python
from persona import Persona, SponsoredSuggestionsHook
from persona.hooks import ToneOverrideHook, SafetyFilterHook

# Contextual product placement
ads = SponsoredSuggestionsHook({
    "gaming": "mechanical keyboards",
    "coding": "AI copilot subscriptions",
})

# Brand voice
tone = ToneOverrideHook("Always be encouraging and supportive.")

# Safety
safety = SafetyFilterHook()

p = Persona(user_id="u1", provider=provider, hooks=[ads, tone, safety])
```

## Heuristic-Only Mode

Don't want to use an LLM for profiling? Use the `NullProvider`:

```python
from persona.providers import NullProvider

p = Persona(user_id="u1", provider=NullProvider())
# Still gets: message length, slang detection, emoji detection
# Skips: cognitive/engagement axis analysis, interest extraction
```

## Architecture

```
persona/
  __init__.py       # Public API: Persona, hooks
  engine.py         # Core engine — enhance(), observe(), profile
  profiling.py      # Heuristics and LLM profiler prompts
  providers.py      # Pluggable LLM backends for the profiler
  hooks.py          # Enrichment hook system
  defaults.py       # Configuration defaults
  cli.py            # Interactive demo
  tests.py          # Test suite
```

## Demo

```bash
# Start a local LLM server (LM Studio, Ollama, etc.) on port 1234
# Or use the included mock server for testing:
python mock_server.py

# Run the interactive demo:
python -m persona.cli
```

## Development

```bash
# Start the mock server
python mock_server.py

# Run the SDK test suite (19 checks)
python -m persona.tests

# Run the legacy harness tests (13 checks)
python test_harness.py
```
