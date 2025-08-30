import argparse, json, os, sys, pathlib, signal, time, platform, asyncio, traceback
from models import AgentSpec
import tempfile
from pydantic_ai import Agent, TextOutput
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

KV = {}
def kv_memory_get(key: str): return KV.get(key)
def kv_memory_set(key: str, value): KV[key] = value; return "OK"

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

TOOLS = {
    "math_eval": lambda args: math_eval(args.get("expr","")),
    "string_template": lambda args: string_template(args.get("template",""), **args.get("vars",{})),
    "kv_memory": lambda args: (kv_memory_set(args["key"], args.get("value")) if "value" in args else kv_memory_get(args["key"])),
    "http_get": lambda args: http_get(args.get("url",""), allow=args.get("allow",[])),
}

async def run_agent_async(spec: AgentSpec, inputs: dict, out_dir: pathlib.Path):
    """Run an agent asynchronously with the given spec and inputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = []
    log_file = out_dir / "logs.jsonl"
    result_file = out_dir / "result.json"
    
    def log(ev, **kw): 
        logs.append({"ts": time.time(), "ev": ev, **kw})
        log_event(log_file, ev, **kw)

    try:
        # Create the agent instance with GPT-5
        agent = Agent(
            model="gpt-5",
            model_settings={
                "max_tokens": spec.run_limits.max_output_chars
            },
            instructions=spec.system_instructions,
            name=spec.name
        )
        
        # Log model call
        log("model_call", model="gpt-5", settings=spec.sdk_config.model_dump())
        
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

def run_agent(spec: AgentSpec, inputs: dict, out_dir: pathlib.Path) -> int:
    """Synchronous wrapper for running the agent asynchronously.
    
    Args:
        spec: The agent specification
        inputs: Input dictionary for the agent
        out_dir: Directory to write output files to
        
    Returns:
        int: 0 on success, 1 on error
    """
    return asyncio.run(run_agent_async(spec, inputs, out_dir))

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
            
            result = run_agent(spec, inputs, out_dir)
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
    """Run a simple smoke test to verify the executor works"""
    print("🔍 Running smoke test...")
    
    # Create a simple agent spec
    spec = AgentSpec(
        name="Test Agent",
        description="A test agent for smoke testing",
        system_instructions="You are a helpful assistant.",
        task_prompt="Say hello and tell me what time it is.",
        input_schema=[]
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
            result = run_agent(spec, inputs, temp_dir)
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
    
    # Check if OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found in environment variables or .env file")
    
    # If --smoke-test flag is passed, run the smoke test
    if "--smoke-test" in sys.argv:
        sys.exit(0 if smoke_test() else 1)
    else:
        sys.exit(main())
