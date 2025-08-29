import argparse, json, os, sys, pathlib, signal, time, platform
from models import AgentSpec
import tempfile

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

def run_agent(spec: AgentSpec, inputs: dict, out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = []
    def log(ev, **kw): logs.append({"ts": time.time(), "ev": ev, **kw})

    # Build prompt (schema-first; no codegen)
    prompt = f"""{spec.system_instructions.strip()}

Task:
{spec.task_prompt.strip()}

Inputs:
{json.dumps(inputs, indent=2)}
"""
    log("prompt_built", prompt_preview=prompt[:800])

    # Demo tool calls: optional list in inputs["__tool_calls__"]
    tool_calls = inputs.get("__tool_calls__", [])
    tool_results = []
    enabled = {t.name for t in spec.tools}
    for call in tool_calls:
        name = call.get("name"); args = call.get("args",{})
        if name in enabled and name in TOOLS:
            res = TOOLS[name](args)
            tool_results.append({"name": name, "result": res})
            log("tool_result", name=name, result_preview=str(res)[:200])
        else:
            msg = "not enabled"
            tool_results.append({"name": name, "error": msg})
            log("tool_blocked", name=name, reason=msg)

    # Stub "model" response so the repo runs without API keys
    result = {
        "message": "Agent run completed (stub model). Replace with real PydanticAI call in your environment.",
        "tool_results": tool_results,
        "inputs": inputs,
    }

    (out_dir/"result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir/"logs.jsonl").write_text("\n".join(json.dumps(l) for l in logs), encoding="utf-8")
    print("OK")
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec_path = pathlib.Path("agents")/args.agent/"spec.json"
    out_dir = pathlib.Path(args.out)
    spec = AgentSpec.model_validate_json(spec_path.read_text())
    inputs = json.loads(pathlib.Path(args.input).read_text())

    # Sandbox: resource limits + timeout
    apply_rlimits(cpu_s=min(10, spec.run_limits.timeout_s), mem_mb=256)

    # Best-effort no-network (clear common proxy env vars)
    for k in ["HTTP_PROXY","HTTPS_PROXY","NO_PROXY"]:
        os.environ.pop(k, None)

    # Enforce timeout using alarm where available
    try:
        import signal
        def on_timeout(signum, frame):
            print("TIMEOUT", file=sys.stderr); sys.exit(124)
        signal.signal(signal.SIGALRM, on_timeout)
        signal.alarm(spec.run_limits.timeout_s)
    except Exception:
        pass

    code = run_agent(spec, inputs, out_dir)
    sys.exit(code)

if __name__ == "__main__":
    main()
