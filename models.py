from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any
import uuid, datetime as dt

class InputField(BaseModel):
    name: str
    type: Literal["str","int","float","bool","json"] = "str"
    required: bool = True
    default: Optional[Any] = None
    label: Optional[str] = None
    help: Optional[str] = None

class ToolRef(BaseModel):
    name: Literal["math_eval","string_template","kv_memory","http_get", "web_search"]
    params: Dict[str, Any] = Field(default_factory=dict)

class RunLimits(BaseModel):
    timeout_s: int = 30
    max_output_chars: int = 8000

class SDKConfig(BaseModel):
    model: str = "gpt-4o-mini"
    temperature: Optional[float] = None  # Optional, will use model's default if not set
    use_local: bool = False
    local_model: str = "gemma3n"

class AgentSpec(BaseModel):
    version: str = "v0"
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str
    description: str
    system_instructions: str
    task_prompt: str
    input_schema: List[InputField] = Field(default_factory=list)
    tools: List[ToolRef] = Field(default_factory=list)
    run_limits: RunLimits = Field(default_factory=RunLimits)
    sdk_config: SDKConfig = Field(default_factory=SDKConfig)

class RunRecord(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    agent_id: str
    created_at: str = Field(default_factory=lambda: dt.datetime.utcnow().isoformat()+"Z")
    status: Literal["queued","running","succeeded","failed","timeout"] = "queued"
    result_path: Optional[str] = None
    logs_path: Optional[str] = None
    stderr_path: Optional[str] = None
