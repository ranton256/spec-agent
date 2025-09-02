from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any, Union
import uuid, datetime as dt

class InputField(BaseModel):
    name: str
    type: Literal["str","int","float","bool","json"] = "str"
    required: bool = True
    default: Optional[Any] = None
    label: Optional[str] = None
    help: Optional[str] = None

class ToolRef(BaseModel):
    name: Literal["math_eval","string_template","kv_memory","http_get", "web_search", "conversational_memory"]
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

# Workflow-specific models (Phase 1)
class NodeIOSpec(BaseModel):
    """Defines input/output specification for a workflow node"""
    inputs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    outputs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

class DataConnection(BaseModel):
    """Defines data flow between workflow nodes"""
    source_node: str
    source_output_field: str
    target_node: str  
    target_input_field: str
    transform_function: Optional[str] = None

class WorkflowNodeSpec(BaseModel):
    """Configuration for a workflow node"""
    id: str
    type: Literal["agent", "condition", "loop", "parallel", "merge"]
    name: Optional[str] = None
    agent_spec_id: Optional[str] = None  # Reference to AgentSpec for agent nodes
    condition_expression: Optional[str] = None  # For condition nodes
    max_iterations: Optional[int] = 100  # For loop nodes
    is_start: bool = False
    is_end: bool = False
    config: Dict[str, Any] = Field(default_factory=dict)
    io_spec: NodeIOSpec = Field(default_factory=NodeIOSpec)

class WorkflowEdgeSpec(BaseModel):
    """Configuration for workflow edges"""
    from_node: str
    to_node: str
    condition: Optional[str] = None
    data_mapping: Optional[Dict[str, str]] = None

class ExecutionLimits(BaseModel):
    """Execution limits for workflow runs"""
    timeout_s: int = 300  # 5 minutes default for workflows
    max_nodes: int = 50
    max_parallel_nodes: int = 10

class WorkflowSpec(BaseModel):
    """Extended spec for workflow-based agent systems"""
    version: str = "v1"
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str
    description: str
    
    # Workflow Definition
    nodes: List[WorkflowNodeSpec] = Field(default_factory=list)
    edges: List[WorkflowEdgeSpec] = Field(default_factory=list)
    data_connections: List[DataConnection] = Field(default_factory=list)
    
    # Referenced Agents (for agent nodes)
    agent_specs: List[AgentSpec] = Field(default_factory=list)
    
    # Execution Configuration
    execution_limits: ExecutionLimits = Field(default_factory=ExecutionLimits)
    
    # Metadata
    created_at: str = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)

class WorkflowExecutionResult(BaseModel):
    """Result from executing a workflow"""
    workflow_id: str
    execution_id: str
    status: Literal["queued", "running", "completed", "failed", "timeout"]
    start_time: str = Field(default_factory=lambda: dt.datetime.utcnow().isoformat()+"Z")
    end_time: Optional[str] = None
    execution_time_seconds: Optional[float] = None
    node_results: Dict[str, Any] = Field(default_factory=dict)
    final_outputs: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    run_directory: Optional[str] = None

# Enhanced AgentSpec with workflow compatibility
class EnhancedAgentSpec(AgentSpec):
    """AgentSpec with workflow integration capabilities"""
    io_spec: NodeIOSpec = Field(default_factory=NodeIOSpec)
    
    def get_input_schema_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert input schema to dictionary format for workflow integration"""
        schema = {}
        for field in self.input_schema:
            schema[field.name] = {
                "type": field.type,
                "required": field.required,
                "default": field.default,
                "label": field.label,
                "help": field.help
            }
        return schema
    
    def get_output_schema_dict(self) -> Dict[str, Dict[str, Any]]:
        """Define expected output schema for workflow integration"""
        return {
            "message": {"type": "str", "required": True, "description": "Agent response message"},
            "tool_results": {"type": "list", "required": False, "description": "Results from tool calls"},
            "metadata": {"type": "dict", "required": False, "description": "Additional metadata"}
        }
    
    def to_agent_spec(self) -> AgentSpec:
        """Convert to base AgentSpec for backward compatibility"""
        return AgentSpec(
            version=self.version,
            id=self.id,
            name=self.name,
            description=self.description,
            system_instructions=self.system_instructions,
            task_prompt=self.task_prompt,
            input_schema=self.input_schema,
            tools=self.tools,
            run_limits=self.run_limits,
            sdk_config=self.sdk_config
        )
