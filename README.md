# SpecAgent

SpecAgent is a **no‑code agent builder** that lets you create and run AI agents through a simple web interface. The Streamlit UI allows you to define agents using a JSON **AgentSpec** (validated by Pydantic), save them, and execute them in a sandboxed environment.

## Features

- **Schema-first Design**: Built with Pydantic for robust data validation
- **Visual Agent Creation**: Intuitive UI for defining agent behavior and parameters
- **Sandboxed Execution**: Runs agents in isolated subprocesses with resource limits
- **Comprehensive Logging**: Detailed execution logs and results for each run
- **Built-in Tools**: Includes useful tools like `math_eval`, `string_template`, `kv_memory`, and `http_get`

## Quickstart

1. **Set up the environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure your API key**

   Create a `.env` file in the project root with your OpenAI API key:

   ```env
   OPENAI_API_KEY=your-api-key-here
   ```

3. **Launch the UI**
   ```bash
   streamlit run streamlit_app.py
   ```

## Project Structure

```
SpecAgent/
├─ streamlit_app.py          # Streamlit web interface
├─ sandbox_executor.py       # Agent execution environment
├─ models.py                 # Pydantic models and schemas
├─ agents/                   # Agent specifications (JSON)
├─ runs/                     # Execution logs and results
├─ output/                   # Additional output files
├─ requirements.txt          # Python dependencies
└─ README.md
```

## How It Works

1. **Create an Agent**
   - Use the UI to define your agent's behavior, inputs, and tools
   - The agent configuration is saved as `agents/<agent_id>/spec.json`

2. **Run an Agent**
   - Provide input values through the UI
   - The system creates a unique run directory: `runs/<agent_id>/<run_id>/`
   - Inputs are saved to `input.json` in the run directory

3. **View Results**
   - The UI displays the agent's output message
   - Full execution details and logs are available in the run directory
   - Results are saved to `result.json`

## Debugging

Use the Debug Mode dropdown in the Run Agent tab to view detailed execution information:

- **Off**: Only show essential output
- **Basic**: Show input values and basic execution info
- **Verbose**: Show full agent spec, input files, and execution logs

## Notes

- The executor includes a tool registry with `math_eval`, `string_template`, `kv_memory`, and `http_get`
- Tools must be explicitly enabled in the agent's configuration
- Network access is restricted for security - only explicitly allowed domains can be accessed via `http_get`
- The sandbox uses `resource.setrlimit` which is POSIX-specific (Linux/macOS). On Windows, limits are skipped.

## Next Steps

- Replace the stubbed model response with a PydanticAI assistant call.
- Add Docker/Firecracker sandbox backends.
- Add authentication and a background worker queue for runs.
