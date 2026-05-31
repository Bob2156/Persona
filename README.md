# Persona

Adaptive ad-inserting chatbot harness for local LM Studio models.

## Run

```bash
python3 chatbot_harness.py
```

The CLI will ask for a local profiling consent confirmation before starting chat.

## Architecture

- `config.py`: central configuration, mappings, and regex patterns.
- `profiling.py`: heuristic updates and parsing of profiler output.
- `lm_studio_client.py`: LM Studio connectivity and chat completion calls.
- `harness.py`: core conversation state machine and prompt assembly.
- `cli.py`: terminal UX, consent flow, and main chat loop.
- `chatbot_harness.py`: entrypoint that wires the CLI together.

Runtime flow: CLI checks LM Studio availability → builds a client and harness → reads user input → harness updates heuristics, optionally profiles in the background, and returns a response.