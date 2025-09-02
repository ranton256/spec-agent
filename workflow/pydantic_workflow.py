from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable, Set, TypeVar, Generic
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
from dataclasses import dataclass
import uuid
from collections import defaultdict, deque

# Core Types and Enums
class NodeType(Enum):
    AGENT = "agent"
    CONDITION = "condition"
    LOOP = "loop"
    PARALLEL = "parallel"
    MERGE = "merge"

class ExecutionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

# Base Models
class WorkflowContext(BaseModel):
    """Shared context that flows through the workflow"""
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class NodeResult(BaseModel):
    """Result from executing a node"""
    node_id: str
    status: ExecutionStatus
    data: Optional[Any] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None

# Abstract Base Classes
class WorkflowNode(ABC):
    """Base class for all workflow nodes"""
    
    def __init__(self, node_id: str, name: Optional[str] = None):
        self.node_id = node_id
        self.name = name or node_id
        self.input_schema: Optional[BaseModel] = None
        self.output_schema: Optional[BaseModel] = None
    
    @abstractmethod
    async def execute(self, context: WorkflowContext) -> NodeResult:
        """Execute the node with given context"""
        pass
    
    def validate_input(self, data: Any) -> Any:
        """Validate input data against schema"""
        if self.input_schema and data is not None:
            return self.input_schema.model_validate(data)
        return data
    
    def validate_output(self, data: Any) -> Any:
        """Validate output data against schema"""
        if self.output_schema and data is not None:
            return self.output_schema.model_validate(data)
        return data

# Concrete Node Implementations
class AgentNode(WorkflowNode):
    """Node that wraps a PydanticAI agent"""
    
    def __init__(self, node_id: str, agent, input_schema=None, output_schema=None, name=None):
        super().__init__(node_id, name)
        self.agent = agent
        self.input_schema = input_schema
        self.output_schema = output_schema
    
    async def execute(self, context: WorkflowContext) -> NodeResult:
        """Execute the PydanticAI agent"""
        try:
            # Extract input for this agent from context
            input_data = context.data.get(self.node_id + "_input")
            validated_input = self.validate_input(input_data)
            
            # Run the agent
            result = await self.agent.run_async(validated_input)
            validated_output = self.validate_output(result.data)
            
            # Store result in context
            context.data[self.node_id + "_output"] = validated_output
            
            return NodeResult(
                node_id=self.node_id,
                status=ExecutionStatus.COMPLETED,
                data=validated_output
            )
            
        except Exception as e:
            return NodeResult(
                node_id=self.node_id,
                status=ExecutionStatus.FAILED,
                error=str(e)
            )

class ConditionNode(WorkflowNode):
    """Node for conditional branching"""
    
    def __init__(self, node_id: str, condition_fn: Callable[[WorkflowContext], bool], name=None):
        super().__init__(node_id, name)
        self.condition_fn = condition_fn
    
    async def execute(self, context: WorkflowContext) -> NodeResult:
        """Evaluate condition"""
        try:
            result = self.condition_fn(context)
            context.data[self.node_id + "_result"] = result
            
            return NodeResult(
                node_id=self.node_id,
                status=ExecutionStatus.COMPLETED,
                data=result
            )
        except Exception as e:
            return NodeResult(
                node_id=self.node_id,
                status=ExecutionStatus.FAILED,
                error=str(e)
            )

class LoopNode(WorkflowNode):
    """Node for loop control"""
    
    def __init__(self, node_id: str, condition_fn: Callable[[WorkflowContext], bool], 
                 max_iterations: int = 100, name=None):
        super().__init__(node_id, name)
        self.condition_fn = condition_fn
        self.max_iterations = max_iterations
        self.current_iteration = 0
    
    async def execute(self, context: WorkflowContext) -> NodeResult:
        """Check loop condition"""
        try:
            should_continue = (self.current_iteration < self.max_iterations and 
                             self.condition_fn(context))
            
            if should_continue:
                self.current_iteration += 1
            else:
                self.current_iteration = 0  # Reset for next workflow run
            
            context.data[self.node_id + "_continue"] = should_continue
            context.data[self.node_id + "_iteration"] = self.current_iteration
            
            return NodeResult(
                node_id=self.node_id,
                status=ExecutionStatus.COMPLETED,
                data={"continue": should_continue, "iteration": self.current_iteration}
            )
        except Exception as e:
            return NodeResult(
                node_id=self.node_id,
                status=ExecutionStatus.FAILED,
                error=str(e)
            )

# Edge and Connection Types
class Edge(BaseModel):
    """Represents a connection between nodes"""
    from_node: str
    to_node: str
    condition: Optional[str] = None  # For conditional edges
    data_mapping: Optional[Dict[str, str]] = None  # Map output fields to input fields

class DataMapping(BaseModel):
    """Defines how data flows between nodes"""
    source_node: str
    source_field: str
    target_node: str
    target_field: str
    transform_fn: Optional[Callable] = None

# Main Workflow Class
class WorkflowGraph:
    """Main workflow orchestrator"""
    
    def __init__(self, name: str):
        self.name = name
        self.nodes: Dict[str, WorkflowNode] = {}
        self.edges: List[Edge] = []
        self.data_mappings: List[DataMapping] = []
        self.start_nodes: Set[str] = set()
        self.end_nodes: Set[str] = set()
    
    def add_node(self, node: WorkflowNode, is_start: bool = False, is_end: bool = False):
        """Add a node to the workflow"""
        self.nodes[node.node_id] = node
        if is_start:
            self.start_nodes.add(node.node_id)
        if is_end:
            self.end_nodes.add(node.node_id)
    
    def add_edge(self, from_node: str, to_node: str, condition: Optional[str] = None,
                 data_mapping: Optional[Dict[str, str]] = None):
        """Add an edge between nodes"""
        edge = Edge(from_node=from_node, to_node=to_node, 
                   condition=condition, data_mapping=data_mapping)
        self.edges.append(edge)
    
    def add_data_mapping(self, source_node: str, source_field: str,
                        target_node: str, target_field: str,
                        transform_fn: Optional[Callable] = None):
        """Add data mapping between nodes"""
        mapping = DataMapping(
            source_node=source_node, source_field=source_field,
            target_node=target_node, target_field=target_field,
            transform_fn=transform_fn
        )
        self.data_mappings.append(mapping)
    
    def get_next_nodes(self, node_id: str, context: WorkflowContext) -> List[str]:
        """Get next nodes to execute based on current node and context"""
        next_nodes = []
        
        for edge in self.edges:
            if edge.from_node == node_id:
                # Check condition if exists
                if edge.condition:
                    condition_result = context.data.get(edge.condition, True)
                    if not condition_result:
                        continue
                
                next_nodes.append(edge.to_node)
        
        return next_nodes
    
    def apply_data_mappings(self, context: WorkflowContext, completed_node: str):
        """Apply data mappings after a node completes"""
        for mapping in self.data_mappings:
            if mapping.source_node == completed_node:
                source_data = context.data.get(f"{mapping.source_node}_output")
                if source_data and hasattr(source_data, mapping.source_field):
                    value = getattr(source_data, mapping.source_field)
                    
                    if mapping.transform_fn:
                        value = mapping.transform_fn(value)
                    
                    context.data[f"{mapping.target_node}_input"] = value
    
    async def execute(self, initial_context: Optional[WorkflowContext] = None) -> WorkflowContext:
        """Execute the workflow"""
        context = initial_context or WorkflowContext()
        executed_nodes = set()
        node_queue = deque(self.start_nodes)
        results = {}
        
        while node_queue:
            current_node_id = node_queue.popleft()
            
            # Skip if already executed (avoid infinite loops)
            if current_node_id in executed_nodes:
                continue
            
            node = self.nodes[current_node_id]
            
            # Execute the node
            result = await node.execute(context)
            results[current_node_id] = result
            executed_nodes.add(current_node_id)
            
            # Apply data mappings
            if result.status == ExecutionStatus.COMPLETED:
                self.apply_data_mappings(context, current_node_id)
            
            # Get next nodes and add to queue
            next_nodes = self.get_next_nodes(current_node_id, context)
            for next_node in next_nodes:
                if next_node not in executed_nodes:
                    node_queue.append(next_node)
        
        context.data["_execution_results"] = results
        return context

# Builder Pattern for Easy Workflow Creation
class WorkflowBuilder:
    """Builder for creating workflows easily"""
    
    def __init__(self, name: str):
        self.workflow = WorkflowGraph(name)
    
    def add_agent(self, node_id: str, agent, input_schema=None, output_schema=None,
                  is_start=False, is_end=False) -> 'WorkflowBuilder':
        """Add an agent node"""
        node = AgentNode(node_id, agent, input_schema, output_schema)
        self.workflow.add_node(node, is_start, is_end)
        return self
    
    def add_condition(self, node_id: str, condition_fn: Callable[[WorkflowContext], bool],
                     is_start=False, is_end=False) -> 'WorkflowBuilder':
        """Add a condition node"""
        node = ConditionNode(node_id, condition_fn)
        self.workflow.add_node(node, is_start, is_end)
        return self
    
    def add_loop(self, node_id: str, condition_fn: Callable[[WorkflowContext], bool],
                max_iterations=100, is_start=False, is_end=False) -> 'WorkflowBuilder':
        """Add a loop node"""
        node = LoopNode(node_id, condition_fn, max_iterations)
        self.workflow.add_node(node, is_start, is_end)
        return self
    
    def connect(self, from_node: str, to_node: str, condition: Optional[str] = None) -> 'WorkflowBuilder':
        """Connect two nodes"""
        self.workflow.add_edge(from_node, to_node, condition)
        return self
    
    def map_data(self, source_node: str, source_field: str,
                target_node: str, target_field: str,
                transform_fn: Optional[Callable] = None) -> 'WorkflowBuilder':
        """Map data between nodes"""
        self.workflow.add_data_mapping(source_node, source_field, target_node, target_field, transform_fn)
        return self
    
    def build(self) -> WorkflowGraph:
        """Build the workflow"""
        return self.workflow

# Example Usage and Testing
async def example_usage():
    """Example of how to use the workflow system"""
    
    # Mock PydanticAI agents (replace with real agents)
    class MockAgent:
        def __init__(self, name):
            self.name = name
        
        async def run_async(self, input_data):
            # Mock response structure
            class MockResult:
                def __init__(self, data):
                    self.data = data
            
            return MockResult(f"Result from {self.name}: {input_data}")
    
    # Create agents
    agent1 = MockAgent("DataProcessor")
    agent2 = MockAgent("Analyzer")
    agent3 = MockAgent("Reporter")
    
    # Build workflow
    workflow = (WorkflowBuilder("Data Processing Pipeline")
                .add_agent("process", agent1, is_start=True)
                .add_condition("check_quality", lambda ctx: len(str(ctx.data.get("process_output", ""))) > 10)
                .add_agent("analyze", agent2)
                .add_agent("report", agent3, is_end=True)
                .connect("process", "check_quality")
                .connect("check_quality", "analyze", condition="check_quality_result")
                .connect("analyze", "report")
                .build())
    
    # Execute workflow
    initial_context = WorkflowContext()
    initial_context.data["process_input"] = "Sample data to process"
    
    result_context = await workflow.execute(initial_context)
    print(f"Workflow completed. Final context: {result_context.data}")

# Run example
if __name__ == "__main__":
    asyncio.run(example_usage())