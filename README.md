# SpecAgent

SpecAgent is a minimal **no‑code agent builder**. The Streamlit UI lets you define an agent as a JSON **AgentSpec** (validated by Pydantic), save it, and run it via a sandboxed executor process.

## Features (MVP)

- **Single SDK choice**: Schema-first design (Pydantic-focused). The sample executor stubs the model call so it runs without API keys.
- **Spec, not code**: UI saves an `AgentSpec` JSON that the executor interprets at runtime.
- **Sandboxed execution**: Separate subprocess with CPU/memory limits (POSIX `resource`), timeout, temp working dir, and basic network restrictions.
- **Artifacts**: Runs produce `result.json` and `logs.jsonl` under `runs/<agent_id>/<run_id>/`.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Launch UI
streamlit run streamlit_app.py
```

## Project Layout

```
SpecAgent/
├─ streamlit_app.py          # Streamlit UI (create/edit agents, run them)
├─ sandbox_executor.py       # Sandboxed interpreter (subprocess)
├─ models.py                 # Pydantic schemas
├─ agents/                   # Saved agent specs (JSON)
├─ runs/                     # Run outputs (logs + results)
├─ requirements.txt
└─ README.md
```

## How it works

1. **Create** an agent in the UI → it writes `agents/<id>/spec.json` (the single source of truth).
2. **Run** the agent → UI writes your input as `runs/<agent_id>/<run_id>/input.json` and launches the executor:

   ```bash
   python sandbox_executor.py --agent <id> --input runs/<agent_id>/<run_id>/input.json --out runs/<agent_id>/<run_id>/
   ```

3. **View results** → The UI tails logs and renders `result.json`.

## Notes

- The executor includes a tiny tool registry (`math_eval`, `string_template`, `kv_memory`, `http_get` with allowlist). Tools must be enabled in the AgentSpec to be callable.
- The "model" call is **stubbed** so you can run without API keys. Swap the stub with a real model (e.g., PydanticAI) if desired.
- The sandbox uses `resource.setrlimit` which is POSIX-specific (Linux/macOS). On Windows, limits are skipped.

## Next Steps

- Replace the stubbed model response with a PydanticAI assistant call.
- Add Docker/Firecracker sandbox backends.
- Add authentication and a background worker queue for runs.
