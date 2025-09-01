# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development Environment Setup
```bash
# Create virtual environment (required - always use virtual env)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Create .env file with OPENAI_API_KEY=your-key-here
```

### Running the Application
```bash
# Launch Streamlit web interface
streamlit run streamlit_app.py

# Run agent in standalone mode (for testing)
python sandbox_executor.py --agent_id <agent_id> --input '{"key": "value"}'
```

### Testing
```bash
# Run unit tests (using unittest framework)
python -m unittest discover -s . -p "test_*.py"

# Run specific test file
python -m unittest test_kv_memory.py

# Run single test method
python -m unittest test_kv_memory.TestKVMemory.test_set_get
```

### Development Tools
No linting or type checking tools are currently configured in the repository.

## Architecture Overview

### Core Components

1. **models.py**: Pydantic schemas defining the data structure
   - `AgentSpec`: Main agent configuration schema
   - `InputField`: Defines agent input parameters
   - `ToolRef`: Tool configuration and parameters
   - `RunRecord`: Execution tracking and metadata

2. **streamlit_app.py**: Web interface for creating, editing, and running agents
   - Three main tabs: Create/Edit Agent, My Agents, Run Agent
   - Handles agent spec creation and validation
   - Provides UI for agent execution with debug modes

3. **sandbox_executor.py**: Core execution engine
   - Runs agents in isolated subprocess with resource limits (POSIX only)
   - Contains built-in tool registry: `math_eval`, `string_template`, `kv_memory`, `http_get`, `web_search`, `conversational_memory`
   - Handles PydanticAI agent creation and execution
   - Supports multiple model backends (OpenAI, local Ollama)

4. **memory.py**: Simple JSON file-based storage for conversation history

### Agent System

- **Agent Specs**: JSON configurations stored in `agents/<agent_id>/spec.json`
- **Run Isolation**: Each execution creates unique run in `runs/<agent_id>/<run_id>/`
- **Tool System**: Modular tools that can be enabled per agent
- **Resource Limits**: CPU and memory constraints applied during execution

### File Organization

```
agents/           # Agent specifications (JSON files)
├── <agent_id>/
│   └── spec.json

runs/             # Execution logs and results  
├── <agent_id>/
│   └── <run_id>/
│       ├── input.json
│       ├── logs.jsonl
│       └── result.json

output/           # Additional output files
```

### Built-in Tools

- `math_eval`: Safe mathematical expression evaluation
- `string_template`: String formatting with variable substitution
- `kv_memory`: Key-value storage with agent/run namespacing
- `http_get`: HTTP requests with domain whitelist restrictions
- `web_search`: DuckDuckGo search integration
- `conversational_memory`: JSON file-based conversation storage

### Model Configuration

Supports multiple backends via `SDKConfig`:
- OpenAI models (default: gpt-4o-mini)
- Local Ollama models (set `use_local: true`)
- Configurable temperature settings

### Security Features

- Resource limits applied to agent execution (POSIX systems)
- HTTP requests restricted to explicitly allowed domains
- Safe mathematical expression evaluation with limited scope
- Subprocess isolation for agent execution