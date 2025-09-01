import argparse, json, os, sys, pathlib, signal, time, platform, asyncio, traceback
from models import AgentSpec, SDKConfig
from memory import JsonFileMemory
import tempfile
from pydantic_ai import Agent, TextOutput
from pydantic_ai.tools import Tool
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from openai import OpenAI
from dotenv import load_dotenv
import yaml
from typing import Optional, Dict, Any

# Try to apply resource limits on POSIX
def apply_rlimits(cpu_s: int = 10, mem_mb: int = 256):
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb*1024*1024, mem_mb*1024*1024))
    except Exception:
        pass  # ignore on non-POSIX or if not permitted

# --- Minimal "tool" registry ---
def math_eval(expr: str) -> str:
    import re
    if not re.fullmatch(r"[0-9\.\+\-\*\/\(\) ]{1,100}", expr or ""):
        return "Invalid expression"
    try:
        return str(eval(expr, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"

def string_template(template: str, **kwargs):
    try:
        return (template or "").format(**kwargs)
    except Exception as e:
        return f"Template error: {e}"

class KVMemory:
    def __init__(self):
        self.store = {}
    
    def get(self, agent_id: str, run_id: str, key: str):
        """Get a value from the key-value store."""
        try:
            return self.store.get(f"{agent_id}:{run_id}", {}).get(key)
        except Exception as e:
            return f"Error getting key '{key}': {str(e)}"
    
    def set(self, agent_id: str, run_id: str, key: str, value: str):
        """Set a value in the key-value store."""
        try:
            namespace = f"{agent_id}:{run_id}"
            if namespace not in self.store:
                self.store[namespace] = {}
            self.store[namespace][key] = value
            return "OK"
        except Exception as e:
            return f"Error setting key '{key}': {str(e)}"
    
    def delete(self, agent_id: str, run_id: str, key: str):
        """Delete a key from the key-value store."""
        try:
            namespace = f"{agent_id}:{run_id}"
            if namespace in self.store and key in self.store[namespace]:
                del self.store[namespace][key]
                return "OK"
            return f"Key '{key}' not found"
        except Exception as e:
            return f"Error deleting key '{key}': {str(e)}"
    
    def list_keys(self, agent_id: str, run_id: str):
        """List all keys in the namespace."""
        try:
            namespace = f"{agent_id}:{run_id}"
            return list(self.store.get(namespace, {}).keys())
        except Exception as e:
            return f"Error listing keys: {str(e)}"

# Global KV memory store with namespacing
kv_store = KVMemory()

def kv_memory_get(agent_id: str, run_id: str, key: str):
    """Get a value from the key-value store."""
    return kv_store.get(agent_id, run_id, key)

def kv_memory_set(agent_id: str, run_id: str, key: str, value: str):
    """Set a value in the key-value store."""
    return kv_store.set(agent_id, run_id, key, value)

def kv_memory_delete(agent_id: str, run_id: str, key: str):
    """Delete a key from the key-value store."""
    return kv_store.delete(agent_id, run_id, key)

def kv_memory_list(agent_id: str, run_id: str):
    """List all keys in the namespace."""
    return kv_store.list_keys(agent_id, run_id)

def http_get(url: str, allow: list[str] = None):
    import urllib.request
    allow = allow or []
    if not any(url.startswith(p) for p in allow):
        return "Blocked"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.read(2000).decode("utf-8","ignore")
    except Exception as e:
        return f"HTTP error: {e}"

def kv_memory_tool(agent_id: str = "", run_id: str = "", key: str = "", value: str = "", action: str = "get") -> str:
    """
    Key-Value memory tool for storing and retrieving data.
    
    Args:
        agent_id: ID of the agent
        run_id: ID of the current run
        key: Key to operate on
        value: Value to set (for set operations)
        action: Action to perform (get, set, delete, list)
            
    Returns:
        Result of the operation as a string
    """
    try:
        if value:
            # Set operation
            return kv_memory_set(agent_id, run_id, key, value)
        elif action == "delete":
            # Delete operation
            return kv_memory_delete(agent_id, run_id, key)
        elif action == "list":
            # List operation
            return str(kv_memory_list(agent_id, run_id))
        else:
            # Get operation (default)
            return str(kv_memory_get(agent_id, run_id, key) or "")
    except Exception as e:
        return f"Error in kv_memory_tool: {str(e)}"


def conversational_memory(cmd: str, entry: Optional[Dict[str, Any]] = None, agent_id: str = None, run_id: str = None):
    if not agent_id or not run_id:
        return "Error: agent_id and run_id are required"

    memory_dir = pathlib.Path("runs") / agent_id / run_id / "memory"
    memory_file = memory_dir / "conversation.json"
    memory = JsonFileMemory(memory_file)

    if cmd == "read":
        return memory.read()
    elif cmd == "append" and entry is not None:
        memory.append(entry)
        return "OK"
    else:
        return "Invalid command"

def create_kv_memory_tool():
    async def kv_memory_func(agent_id: str = "", run_id: str = "", key: str = "", value: str = "", action: str = "get") -> str:
        """
        Key-Value memory tool for storing and retrieving data.
        
        Args:
            agent_id: ID of the agent
            run_id: ID of the current run
            key: Key to operate on
            value: Value to set (for set operations)
            action: Action to perform (get, set, delete, list)
                
        Returns:
            Result of the operation as a string
        """
        try:
            if value:
                # Set operation
                return kv_memory_set(agent_id, run_id, key, value)
            elif action == "delete":
                # Delete operation
                return kv_memory_delete(agent_id, run_id, key)
            elif action == "list":
                # List operation
                return str(kv_memory_list(agent_id, run_id))
            else:
                # Get operation (default)
                return str(kv_memory_get(agent_id, run_id, key) or "")
        except Exception as e:
            return f"Error in kv_memory_tool: {str(e)}"
    
    return Tool(
        function=kv_memory_func,
        name="kv_memory",
        description="Key-Value memory tool for storing and retrieving data. Use 'set' action with 'value' to store data, 'get' to retrieve, 'delete' to remove, and 'list' to list all keys.",
        takes_ctx=False
    )

TOOLS = {
    "math_eval": lambda args: math_eval(args.get("expr","")),
    "string_template": lambda args: string_template(args.get("template",""), **args.get("vars",{})),
    "kv_memory": create_kv_memory_tool(),
    "http_get": lambda args: http_get(args.get("url",""), allow=args.get("allow",[])),
    "web_search": duckduckgo_search_tool(),
    "conversational_memory": lambda args: conversational_memory(
        cmd=args.get("cmd"),
        entry=args.get("entry"),
        agent_id=args.get("agent_id"),
        run_id=args.get("run_id")
    ),
}

async def run_agent_async(spec: AgentSpec, inputs: dict, out_dir: pathlib.Path, run_id: str):
    """Run an agent asynchronously with the given spec and inputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = []
    log_file = out_dir / "logs.jsonl"
    result_file = out_dir / "result.json"
    
    def log(ev, **kw): 
        logs.append({"ts": time.time(), "ev": ev, **kw})
        log_event(log_file, ev, **kw)

    try:
        model_name = spec.sdk_config.local_model if spec.sdk_config.use_local else spec.sdk_config.model
        log("model_call", 
            model=model_name, 
            is_local=spec.sdk_config.use_local,
            settings=spec.sdk_config.model_dump()
        )
        
        if spec.sdk_config.use_local:
            # Use Ollama local model
            import ollama
            
            # Ensure the model is available
            try:
                await asyncio.to_thread(ollama.show, model_name)
            except Exception as e:
                log("error", message=f"Model {model_name} not found, attempting to pull")
                await asyncio.to_thread(ollama.pull, model_name)
            
            # Create a simple agent that uses Ollama
            class OllamaAgent:
                def __init__(self, model, system, temperature):
                    self.model = model
                    self.system = system
                    self.temperature = temperature
                
                async def run(self, prompt):
                    response = await asyncio.to_thread(
                        ollama.chat,
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self.system},
                            {"role": "user", "content": prompt}
                        ],
                        options={
                            "temperature": self.temperature,
                            "num_predict": spec.run_limits.max_output_chars
                        }
                    )
                    return response['message']['content']
            
            agent = OllamaAgent(
                model=model_name,
                system=spec.system_instructions,
                temperature=spec.sdk_config.temperature if hasattr(spec.sdk_config, 'temperature') else None
            )
        else:
            # Use Pydantic AI with OpenAI
            model_settings = {
                "max_tokens": spec.run_limits.max_output_chars
            }
            
            # Only add temperature if it's explicitly set in the config and model is not gpt-5
            if (hasattr(spec.sdk_config, 'temperature') and 
                spec.sdk_config.temperature is not None and
                spec.sdk_config.model != "gpt-5"):
                model_settings["temperature"] = spec.sdk_config.temperature
            
            # Dynamically create tools based on the spec
            enabled_tools = []
            if spec.tools:
                for tool_ref in spec.tools:
                    if tool_ref.name == "kv_memory":
                        # Special handling for kv_memory tool
                        tool = Tool(
                            function=kv_memory_tool,
                            name="kv_memory",
                            description="Key-Value memory tool for storing and retrieving data. Use 'set' action with 'value' to store data, 'get' to retrieve, 'delete' to remove, and 'list' to list all keys.",
                            takes_ctx=False
                        )
                        enabled_tools.append(tool)
                    elif tool_ref.name in TOOLS:
                        # For other tools, use the existing TOOLS dictionary
                        tool_func = TOOLS[tool_ref.name]
                        if callable(tool_func):
                            enabled_tools.append(tool_func)

            agent = Agent(
                model=spec.sdk_config.model,
                model_settings=model_settings,
                instructions=spec.system_instructions,
                name=spec.name,
                tools=enabled_tools,
            )
        
        # Format the user message
        user_message = f"""Task:
{spec.task_prompt.strip()}

Inputs:
{json.dumps(inputs, indent=2)}"""
        
        # Set system instructions
        agent.instructions = spec.system_instructions.strip()
        
        # Call the model asynchronously
        log("calling_agent_run", user_message=user_message)
        response = await agent.run(user_message)
        log("agent_run_completed", response_type=type(response).__name__)
        
        # Debug: Print the full response object
        import pprint
        response_str = pprint.pformat(response, width=120)
        log("raw_response", response=response_str)
        
        # Extract response text
        response_text = ""
        if hasattr(response, 'content') and response.content:
            response_text = response.content
        elif isinstance(response, str):
            response_text = response
        elif hasattr(response, 'choices') and response.choices:
            response_text = response.choices[0].message.content
        elif hasattr(response, 'output') and response.output:
            response_text = str(response.output)
        
        # Log the response
        log("model_output", response=response_text, response_type=type(response).__name__)
        
        # Create result object
        result = {
            "message": response_text,
            "tool_results": [],
            "inputs": inputs,
            "model_metadata": {
                "model": "gpt-5",
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
        }
        
        # Write the result to a file
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        log("agent_completed", result_path=str(result_file))
        return 0  # Success
        
    except Exception as e:
        error_msg = str(e)
        log("agent_error", error=error_msg, stacktrace=traceback.format_exc())
        
        # Write error result
        with open(result_file, 'w') as f:
            json.dump({
                "error": error_msg,
                "inputs": inputs,
                "stacktrace": traceback.format_exc()
            }, f, indent=2)
        
        return 1  # Error

def run_agent(spec: AgentSpec, inputs: dict, out_dir: pathlib.Path, run_id: str) -> int:
    """Synchronous wrapper for running the agent asynchronously.
    
    Args:
        spec: The agent specification
        inputs: Input dictionary for the agent
        out_dir: Directory to write output files to
        run_id: The ID of the current run
        
    Returns:
        int: 0 on success, 1 on error
    """
    return asyncio.run(run_agent_async(spec, inputs, out_dir, run_id))

def log_event(log_file, event, **kwargs):
    """Append a log entry to the log file.
    
    Args:
        log_file: Path to the log file
        event: Event name
        **kwargs: Additional event data
    """
    log_entry = json.dumps({"timestamp": time.time(), "event": event, **kwargs}, ensure_ascii=False)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{log_entry}\n")

def load_agent_spec(agent_path: str) -> Optional[AgentSpec]:
    """Load agent specification from a JSON or YAML file."""
    try:
        path = pathlib.Path(agent_path)
        content = path.read_text()
        
        if path.suffix.lower() in ('.yaml', '.yml'):
            data = yaml.safe_load(content)
            return AgentSpec.model_validate(data)
        else:
            return AgentSpec.model_validate_json(content)
    except Exception as e:
        print(f"Error loading agent spec: {e}")
        return None

def load_inputs(input_path: str) -> Optional[Dict[str, Any]]:
    """Load input data from a JSON or YAML file."""
    try:
        path = pathlib.Path(input_path)
        content = path.read_text()
        
        if path.suffix.lower() in ('.yaml', '.yml'):
            return yaml.safe_load(content)
        else:
            return json.loads(content)
    except Exception as e:
        print(f"Error loading inputs: {e}")
        return None

def create_agent_spec() -> Optional[AgentSpec]:
    """Interactively create a new agent specification."""
    try:
        print("\nCreate a new agent specification")
        print("-" * 30)
        
        name = input("Agent name: ").strip()
        description = input("Agent description: ").strip()
        system_instructions = input("System instructions (press Enter twice to finish):\n")
        
        # Read multi-line system instructions
        while True:
            line = input()
            if not line:
                break
            system_instructions += "\n" + line
            
        task_prompt = input("\nTask prompt: ").strip()
        
        # Create a basic agent spec
        spec = AgentSpec(
            name=name,
            description=description,
            system_instructions=system_instructions,
            task_prompt=task_prompt,
            input_schema=[],  # Can be extended to collect input schema
            tools=[],         # Can be extended to add tools
        )
        
        # Save to file
        save = input("\nSave agent spec? (y/n): ").strip().lower()
        if save == 'y':
            filename = input("Filename (without extension): ").strip()
            if not filename:
                filename = name.lower().replace(' ', '_')
            
            format_choice = input("Save as JSON or YAML? (json/yaml, default: json): ").strip().lower()
            if format_choice == 'yaml':
                filepath = f"agents/{filename}.yaml"
                pathlib.Path("agents").mkdir(exist_ok=True)
                with open(filepath, 'w') as f:
                    yaml.dump(spec.model_dump(), f, default_flow_style=False)
            else:
                filepath = f"agents/{filename}.json"
                pathlib.Path("agents").mkdir(exist_ok=True)
                with open(filepath, 'w') as f:
                    f.write(spec.model_dump_json(indent=2))
            
            print(f"Agent spec saved to {filepath}")
        
        return spec
    
    except Exception as e:
        print(f"Error creating agent spec: {e}")
        return None

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Run or create AI agents")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run an agent')
    run_parser.add_argument("--agent", required=True, help="Path to agent spec file (JSON or YAML)")
    run_parser.add_argument("--input_path", required=False, help="Path to input file (JSON or YAML)")
    run_parser.add_argument("--input", required=False, help="Input data (JSON or YAML)")
    run_parser.add_argument("--out", default="./output", help="Output directory")
    run_parser.add_argument("--run_id", required=True, help="ID of the run")
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create a new agent')
    
    args = parser.parse_args()
    
    if args.command == 'run':
        spec = load_agent_spec(args.agent)
        if not spec:
            return 1
        
        try:
            if args.input_path:
                print(f"Loading inputs from file: {args.input_path}")
                inputs = load_inputs(args.input_path)
            else:
                print(f"Loading inputs from command line: {args.input}")
                inputs = json.loads(args.input)
            
            if not inputs:
                print("Error: No inputs provided")
                return 1
                
            print(f"Successfully loaded inputs: {inputs}")
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON input: {e}")
            return 1
        except Exception as e:
            print(f"Error loading inputs: {e}")
            return 1
            
        out_dir = pathlib.Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            import signal
            def on_timeout(signum, frame):
                print("TIMEOUT", file=sys.stderr)
                sys.exit(124)
            signal.signal(signal.SIGALRM, on_timeout)
            signal.alarm(spec.run_limits.timeout_s)
        except Exception as e:
            print(f"Error setting up timeout: {e}")
            
        try:
            print(f"Starting agent execution with spec: {spec.name}")
            print(f"Model: {spec.sdk_config.model}, Temperature: {spec.sdk_config.temperature}")
            print(f"System instructions: {spec.system_instructions[:100]}...")
            print(f"Task prompt: {spec.task_prompt[:100]}...")
            
            result = run_agent(spec, inputs, out_dir, args.run_id)
            print(f"Agent execution completed with result: {result}")
            
            # Check output files
            result_file = out_dir / "result.json"
            if result_file.exists():
                print(f"Result file created at: {result_file}")
                print("Result content:")
                print(result_file.read_text())
            else:
                print("Warning: No result.json file was created")
                
            return result
            
        except Exception as e:
            print(f"Error running agent: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1
        
    elif args.command == 'create':
        if create_agent_spec():
            return 0
        return 1
    else:
        parser.print_help()
        return 1
    return 0

def smoke_test():
    """Run a simple smoke test to verify the executor works
    """
    use_local = True
    print("🔍 Running smoke test...")
    print(f"Using {'local Ollama model' if use_local else 'OpenAI model'}")
    
    # Configure the model settings
    sdk_config = {
        "use_local": use_local,
        "local_model": "gemma3n"
    }
    
    if not use_local:
        sdk_config.update({
            "model": "openai:gpt-4o-mini",
            # Use default temperature for GPT-5
            "max_tokens": 1000,
            "max_output_chars": 1000
        })
    
    # Create a simple agent spec
    spec = AgentSpec(
        name="Test Agent",
        description="A test agent for smoke testing",
        system_instructions="You are a helpful assistant.",
        task_prompt="Say hello and tell me what time it is.",
        input_schema=[],
        sdk_config=SDKConfig(use_local=use_local)
    )
    print("✅ Created test agent spec")
    
    # Create a test input
    inputs = {"test": "test"}
    print(f"✅ Created test inputs: {inputs}")
    
    # Create a temporary directory for output
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = pathlib.Path(temp_dir)
        print(f"📁 Created temporary directory: {temp_dir}")
        
        print("🚀 Running agent...")
        try:
            run_id = os.urandom(6).hex()
            result = run_agent(spec, inputs, temp_dir, run_id)
            print(f"✅ Agent completed with result: {result}")
        except Exception as e:
            print(f"❌ Agent execution failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
        # Check if the output files were created
        result_file = temp_dir / "result.json"
        logs_file = temp_dir / "logs.jsonl"
        
        print(f"📄 Looking for output files in: {temp_dir}")
        print(f"   - result.json exists: {result_file.exists()}")
        print(f"   - logs.jsonl exists: {logs_file.exists()}")
        
        if not result_file.exists():
            print("❌ result.json not found")
            print(f"    Directory contents: {list(temp_dir.glob('*'))}")
            return False
            
        if not logs_file.exists():
            print("❌ logs.jsonl not found")
            return False
            
        # Check the result content
        try:
            print(f"📝 Reading result file: {result_file}")
            result_content = result_file.read_text()
            print(f"   Raw result content: {result_content[:500]}...")
            
            result_data = json.loads(result_content)
            print(f"✅ Successfully parsed result JSON")
            
            if "message" not in result_data:
                print("❌ 'message' key not found in result")
                print(f"    Result keys: {list(result_data.keys())}")
                if "error" in result_data:
                    print(f"    Error: {result_data['error']}")
                return False
                
            message = result_data["message"]
            print("✅ Found message in result")
            print(f"📄 Message content: {message[:200]}...")
            
            if not message or message == "An error occurred while processing your request.":
                print("❌ Empty or error message in result")
                if "error" in result_data:
                    print(f"    Error: {result_data['error']}")
                return False
                
            print("✅ Smoke test passed!")
            return True
            
        except Exception as e:
            print(f"❌ Smoke test failed: {str(e)}")
            return False

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    
    # Check if OPENAI_API_KEY is set when not using local model
    if not os.getenv("OPENAI_API_KEY") and "--local" not in sys.argv:
        print("Warning: OPENAI_API_KEY not found in environment variables or .env file")
    
    # If --smoke-test flag is passed, run the smoke test
    if "--smoke-test" in sys.argv:
        sys.exit(0 if smoke_test() else 1)
    else:
        sys.exit(main())
