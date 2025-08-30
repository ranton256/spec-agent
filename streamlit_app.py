import json, os, pathlib, subprocess, time
import streamlit as st
from models import AgentSpec, InputField, ToolRef, RunRecord
from dotenv import load_dotenv

BASE = pathlib.Path(".")
AGENTS_DIR = BASE / "agents"
RUNS_DIR = BASE / "runs"
AGENTS_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="SpecAgent", layout="wide")

# Load environment variables
load_dotenv()

# Sidebar for API key and settings
with st.sidebar:
    st.header("Settings")
    
    # Check if OPENAI_API_KEY is already set in environment
    env_api_key = os.getenv("OPENAI_API_KEY")
    
    if env_api_key:
        st.info("OpenAI API key is set in environment")
        api_key = st.text_input(
            "OpenAI API Key",
            value="•" * 20,  # Show placeholder dots if key is set
            type="password",
            disabled=True,
            help="API key is set in environment variables"
        )
    else:
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key (stored in session state only)"
        )
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.success("API key set for this session")

tabs = st.tabs(["Create/Edit Agent", "My Agents", "Run Agent"])

# --- Tab 1: Create/Edit Agent ---
with tabs[0]:
    st.header("Create / Edit Agent")
    name = st.text_input("Agent name")
    desc = st.text_area("Description")
    system = st.text_area("System instructions (persona / rules)", height=160, placeholder="You are a helpful assistant...")
    task = st.text_area("Task prompt (what the agent should do)", height=160)

    st.subheader("Inputs")
    if "input_rows" not in st.session_state:
        st.session_state["input_rows"] = []
    if st.button("Add input field"):
        st.session_state["input_rows"].append({"name":"","type":"str","required":True,"default":None,"label":"","help":""})

    for i, row in enumerate(st.session_state["input_rows"]):
        with st.container(border=True):
            cols = st.columns([2,1,1,2,2])
            row["name"] = cols[0].text_input("name", key=f"name_{i}", value=row["name"])
            row["type"] = cols[1].selectbox("type", ["str","int","float","bool","json"], key=f"type_{i}", index=["str","int","float","bool","json"].index(row["type"]))
            row["required"] = cols[2].checkbox("required", key=f"req_{i}", value=row["required"])
            row["label"] = cols[3].text_input("label", key=f"label_{i}", value=row.get("label",""))
            row["help"] = cols[4].text_input("help", key=f"help_{i}", value=row.get("help",""))
            row["default"] = st.text_input("default (string or JSON)", key=f"default_{i}", value="" if row.get("default") in [None,"None"] else str(row.get("default")))

    st.subheader("Tools")
    tool_names = ["math_eval","string_template","kv_memory","http_get", "web_search"]
    selected_tools = st.multiselect("Enable tools", tool_names, default=[])
    tools = [ToolRef(name=t) for t in selected_tools]

    st.subheader("Model & Limits")
    
    # Model source selection
    model_source = st.radio("Model Source", ["OpenAI", "Local (Ollama)"], horizontal=True)
    use_local = model_source == "Local (Ollama)"
    
    if use_local:
        model = st.text_input("Local model name", value="gemma3n", help="Name of the Ollama model to use")
    else:
        model = st.text_input("OpenAI model", value="gpt-4o-mini", help="OpenAI model identifier")
    
    # Common settings
    temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    
    # Limits
    col1, col2 = st.columns(2)
    with col1:
        timeout = st.number_input("Timeout (s)", 1, 600, 30)
    with col2:
        max_out = st.number_input("Max output chars", 100, 50000, 8000)

    if st.button("Save Agent", type="primary"):
        spec = AgentSpec(
            name=name, description=desc,
            system_instructions=system, task_prompt=task,
            input_schema=[
                InputField(
                    **{
                        **r,
                        "default": None if (r.get("default") in [None,"","None"]) else (
                            json.loads(r["default"]) if r["type"]=="json" else r["default"]
                        )
                    }
                ) for r in st.session_state["input_rows"]
            ],
            tools=tools,
            run_limits={"timeout_s": timeout, "max_output_chars": max_out},
            sdk_config={
                "model": model, 
                "temperature": temp,
                "use_local": use_local,
                "local_model": "gemma3n" if use_local else ""
            },
        )
        agent_dir = AGENTS_DIR / spec.id
        agent_dir.mkdir(exist_ok=True, parents=True)
        (agent_dir / "spec.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        st.success(f"Saved: agents/{spec.id}/spec.json")

    st.subheader("Preview JSON")
    st.code(json.dumps({
        "name": name, 
        "description": desc,
        "system_instructions": system, 
        "task_prompt": task,
        "input_schema": st.session_state["input_rows"],
        "tools": [t for t in selected_tools],
        "run_limits": {
            "timeout_s": timeout,
            "max_output_chars": max_out
        },
        "sdk_config": {
            "model": model,
            "temperature": temp,
            "use_local": use_local,
            "local_model": "gemma3n" if use_local else ""
        }
    }, indent=2), language="json")

# --- Tab 2: My Agents ---
with tabs[1]:
    st.header("My Agents")
    any_agents = False
    for d in AGENTS_DIR.glob("*/spec.json"):
        any_agents = True
        spec = AgentSpec.model_validate_json(d.read_text())
        with st.container(border=True):
            st.subheader(spec.name)
            st.caption(spec.description)
            st.text(f"agents/{spec.id}/spec.json")
    if not any_agents:
        st.info("No agents yet. Create one in the first tab.")

# --- Tab 3: Run Agent ---
with tabs[2]:
    st.header("Run Agent")
    specs = list(AGENTS_DIR.glob("*/spec.json"))
    if not specs:
        st.info("Create an agent first in the previous tab.")
    else:
        def fmt(p: pathlib.Path):
            s = AgentSpec.model_validate_json(p.read_text())
            return f"{s.name} — {s.id[:8]}"
        choice = st.selectbox("Choose agent", specs, format_func=fmt)
        spec = AgentSpec.model_validate_json(choice.read_text())

        st.subheader("Inputs")
        values = {}
        for f in spec.input_schema:
            label = f.label or f.name
            if f.type == "str":
                values[f.name] = st.text_input(label, value=(f.default or ""))
            elif f.type == "int":
                values[f.name] = st.number_input(label, value=int(f.default or 0), step=1)
            elif f.type == "float":
                values[f.name] = st.number_input(label, value=float(f.default or 0.0))
            elif f.type == "bool":
                values[f.name] = st.checkbox(label, value=bool(f.default or False))
            else:
                values[f.name] = st.text_area(label, value=json.dumps(f.default) if f.default is not None else "")

        # Add debug mode dropdown
        debug_mode = st.selectbox(
            "Debug Mode",
            ["Off", "Basic", "Verbose"],
            index=0,
            help="Control the visibility of diagnostic messages and logs"
        )
        
        if st.button("Run in Sandbox", type="primary"):
            run_id = os.urandom(6).hex()
            run_dir = RUNS_DIR / spec.id / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Show debug info based on debug mode
            if debug_mode != "Off":
                with st.expander("Debug Information", expanded=False):
                    st.write("### Input Values")
                    st.json(values)
                    
                    # Ensure values are JSON serializable
                    serializable_values = {}
                    for k, v in values.items():
                        if hasattr(v, 'model_dump_json'):  # Handle Pydantic models
                            serializable_values[k] = json.loads(v.model_dump_json())
                        else:
                            serializable_values[k] = v
                    
                    # Write the input file
                    input_file = run_dir / "input.json"
                    input_file.write_text(json.dumps(serializable_values, indent=2), encoding="utf-8")
                    
                    st.write("### Input File")
                    st.write(f"Path: `{input_file}`")
                    st.code(input_file.read_text(), language="json")
                    
                    # Show agent spec info in verbose mode
                    if debug_mode == "Verbose":
                        agent_spec_path = AGENTS_DIR / spec.id / "spec.json"
                        st.write("### Agent Spec")
                        st.write(f"Path: `{agent_spec_path}`")
                        if agent_spec_path.exists():
                            st.code(agent_spec_path.read_text(), language="json")
                        else:
                            st.error("Agent spec file not found")

            # Use the full path to the agent's spec file
            agent_spec_path = AGENTS_DIR / spec.id / "spec.json"
            
            # Create input file with the form values
            input_file = run_dir / "input.json"
            input_file.write_text(json.dumps(values, indent=2), encoding="utf-8")
            
            # Build the command
            cmd = ["python", "sandbox_executor.py", "run", "--agent", str(agent_spec_path), "--input_path", str(input_file), "--out", str(run_dir)]
            
            # Show command in debug mode
            if debug_mode != "Off":
                with st.expander("Execution Details", expanded=False):
                    st.write("### Agent Spec Location")
                    st.code(f"{agent_spec_path}", language="text")
                    st.write("### Input File")
                    st.code(input_file.read_text(), language="json")
                    st.write("### Command")
                    st.code(" ".join(cmd), language="bash")
            
            # Show execution status
            status_text = st.empty()
            status_text.info("Running agent...")
            
            # Run the agent
            if debug_mode != "Off":
                # In debug mode, show live logs
                log_expander = st.expander("Show Execution Logs", expanded=False)
                with log_expander:
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
                    logbox = st.empty()
                    buf = ""
                    while True:
                        line = proc.stdout.readline()
                        if not line:
                            if proc.poll() is not None:
                                break
                            time.sleep(0.05)
                            continue
                        buf += line
                        logbox.code(buf)
                    code = proc.wait()
            else:
                # In non-debug mode, just run without showing output
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                code = proc.wait()
            
            # Update status
            if code == 0:
                status_text.success("Agent execution completed successfully!")
            else:
                status_text.error(f"Agent execution failed with code {code}")
                
                # Show error details if available
                if debug_mode != "Off":
                    st.error("Error details:")
                    st.code(proc.stderr.read() if proc.stderr else "No error details available")
            
            # Show results
            st.subheader("Result")
            res_path = run_dir / "result.json"
            if res_path.exists():
                result = json.loads(res_path.read_text())
                if "message" in result:
                    st.write(result["message"])
                
                # Show full result in debug mode
                if debug_mode != "Off":
                    with st.expander("View Full Result", expanded=False):
                        st.json(result)
            else:
                st.error("No result was generated.")
                if debug_mode != "Off" and (run_dir / "logs.jsonl").exists():
                    with st.expander("View Error Logs", expanded=True):
                        st.code((run_dir / "logs.jsonl").read_text(), language="json")
