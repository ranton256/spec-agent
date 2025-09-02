import json
import yaml
from typing import Dict, Any, List, Optional, Callable
from pydantic import BaseModel, Field
from enum import Enum

# Configuration Models for JSON/YAML Definition
class NodeConfig(BaseModel):
    """Configuration for a workflow node"""
    id: str
    type: str  # agent, condition, loop, parallel, merge
    name: Optional[str] = None
    agent_id: Optional[str] = None  # Reference to registered agent
    condition_expression: Optional[str] = None  # For condition nodes
    max_iterations: Optional[int] = 100  # For loop nodes
    input_schema: Optional[str] = None  # Schema class name
    output_schema: Optional[str] = None  # Schema class name
    is_start: bool = False
    is_end: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EdgeConfig(BaseModel):
    """Configuration for workflow edges"""
    from_node: str
    to_node: str
    condition: Optional[str] = None
    data_mapping: Optional[Dict[str, str]] = None

class DataMappingConfig(BaseModel):
    """Configuration for data mappings"""
    source_node: str
    source_field: str
    target_node: str
    target_field: str
    transform_function: Optional[str] = None  # Reference to registered transform

class WorkflowConfig(BaseModel):
    """Complete workflow configuration"""
    name: str
    description: Optional[str] = None
    version: str = "1.0"
    nodes: List[NodeConfig]
    edges: List[EdgeConfig]
    data_mappings: List[DataMappingConfig] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

# Registry System for Agents and Functions
class ComponentRegistry:
    """Registry for agents, conditions, and transform functions"""
    
    def __init__(self):
        self.agents: Dict[str, Any] = {}
        self.conditions: Dict[str, Callable] = {}
        self.transforms: Dict[str, Callable] = {}
        self.schemas: Dict[str, BaseModel] = {}
    
    def register_agent(self, agent_id: str, agent):
        """Register a PydanticAI agent"""
        self.agents[agent_id] = agent
        return self
    
    def register_condition(self, condition_id: str, condition_fn: Callable):
        """Register a condition function"""
        self.conditions[condition_id] = condition_fn
        return self
    
    def register_transform(self, transform_id: str, transform_fn: Callable):
        """Register a data transform function"""
        self.transforms[transform_id] = transform_fn
        return self
    
    def register_schema(self, schema_name: str, schema_class: BaseModel):
        """Register a Pydantic schema"""
        self.schemas[schema_name] = schema_class
        return self
    
    def get_agent(self, agent_id: str):
        """Get registered agent"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent '{agent_id}' not found in registry")
        return self.agents[agent_id]
    
    def get_condition(self, condition_id: str):
        """Get registered condition function"""
        if condition_id not in self.conditions:
            raise ValueError(f"Condition '{condition_id}' not found in registry")
        return self.conditions[condition_id]
    
    def get_transform(self, transform_id: str):
        """Get registered transform function"""
        if transform_id not in self.transforms:
            raise ValueError(f"Transform '{transform_id}' not found in registry")
        return self.transforms[transform_id]
    
    def get_schema(self, schema_name: str):
        """Get registered schema class"""
        if schema_name not in self.schemas:
            raise ValueError(f"Schema '{schema_name}' not found in registry")
        return self.schemas[schema_name]

# Workflow Factory
class WorkflowFactory:
    """Factory for creating workflows from configuration"""
    
    def __init__(self, registry: ComponentRegistry):
        self.registry = registry
    
    def create_from_config(self, config: WorkflowConfig) -> WorkflowGraph:
        """Create workflow from configuration"""
        workflow = WorkflowGraph(config.name)
        
        # Create nodes
        for node_config in config.nodes:
            node = self._create_node(node_config)
            workflow.add_node(node, node_config.is_start, node_config.is_end)
        
        # Create edges
        for edge_config in config.edges:
            workflow.add_edge(
                edge_config.from_node,
                edge_config.to_node,
                edge_config.condition,
                edge_config.data_mapping
            )
        
        # Create data mappings
        for mapping_config in config.data_mappings:
            transform_fn = None
            if mapping_config.transform_function:
                transform_fn = self.registry.get_transform(mapping_config.transform_function)
            
            workflow.add_data_mapping(
                mapping_config.source_node,
                mapping_config.source_field,
                mapping_config.target_node,
                mapping_config.target_field,
                transform_fn
            )
        
        return workflow
    
    def _create_node(self, config: NodeConfig) -> WorkflowNode:
        """Create a node based on configuration"""
        if config.type == "agent":
            if not config.agent_id:
                raise ValueError(f"Agent node '{config.id}' requires agent_id")
            
            agent = self.registry.get_agent(config.agent_id)
            input_schema = self.registry.get_schema(config.input_schema) if config.input_schema else None
            output_schema = self.registry.get_schema(config.output_schema) if config.output_schema else None
            
            return AgentNode(config.id, agent, input_schema, output_schema, config.name)
        
        elif config.type == "condition":
            if not config.condition_expression:
                raise ValueError(f"Condition node '{config.id}' requires condition_expression")
            
            condition_fn = self.registry.get_condition(config.condition_expression)
            return ConditionNode(config.id, condition_fn, config.name)
        
        elif config.type == "loop":
            if not config.condition_expression:
                raise ValueError(f"Loop node '{config.id}' requires condition_expression")
            
            condition_fn = self.registry.get_condition(config.condition_expression)
            return LoopNode(config.id, condition_fn, config.max_iterations, config.name)
        
        elif config.type == "parallel":
            # Implementation would depend on your parallel node requirements
            raise NotImplementedError("Parallel nodes not yet implemented in config system")
        
        elif config.type == "merge":
            # Implementation would depend on your merge node requirements
            raise NotImplementedError("Merge nodes not yet implemented in config system")
        
        else:
            raise ValueError(f"Unknown node type: {config.type}")
    
    def create_from_json(self, json_str: str) -> WorkflowGraph:
        """Create workflow from JSON string"""
        config_dict = json.loads(json_str)
        config = WorkflowConfig(**config_dict)
        return self.create_from_config(config)
    
    def create_from_yaml(self, yaml_str: str) -> WorkflowGraph:
        """Create workflow from YAML string"""
        config_dict = yaml.safe_load(yaml_str)
        config = WorkflowConfig(**config_dict)
        return self.create_from_config(config)
    
    def create_from_file(self, file_path: str) -> WorkflowGraph:
        """Create workflow from JSON or YAML file"""
        with open(file_path, 'r') as f:
            if file_path.endswith(('.yml', '.yaml')):
                return self.create_from_yaml(f.read())
            else:
                return self.create_from_json(f.read())

# Configuration Templates
def get_sample_workflow_configs():
    """Get sample workflow configurations"""
    
    # Simple Sequential Workflow
    simple_config = {
        "name": "Simple Data Processing",
        "description": "Basic data processing pipeline",
        "version": "1.0",
        "nodes": [
            {
                "id": "input_processor",
                "type": "agent",
                "agent_id": "data_processor_agent",
                "input_schema": "RawDataSchema",
                "output_schema": "ProcessedDataSchema",
                "is_start": True
            },
            {
                "id": "quality_check",
                "type": "condition",
                "condition_expression": "data_quality_check"
            },
            {
                "id": "analyzer",
                "type": "agent",
                "agent_id": "analysis_agent",
                "input_schema": "ProcessedDataSchema",
                "output_schema": "AnalysisResultSchema"
            },
            {
                "id": "reporter",
                "type": "agent",
                "agent_id": "report_agent",
                "input_schema": "AnalysisResultSchema",
                "output_schema": "ReportSchema",
                "is_end": True
            }
        ],
        "edges": [
            {"from_node": "input_processor", "to_node": "quality_check"},
            {"from_node": "quality_check", "to_node": "analyzer", "condition": "quality_check_result"},
            {"from_node": "analyzer", "to_node": "reporter"}
        ],
        "data_mappings": [
            {
                "source_node": "input_processor",
                "source_field": "processed_data",
                "target_node": "analyzer",
                "target_field": "data_input"
            }
        ]
    }
    
    # Branching Workflow with Loops
    complex_config = {
        "name": "Complex Processing Pipeline",
        "description": "Advanced pipeline with branching and loops",
        "version": "1.0",
        "nodes": [
            {
                "id": "initial_processor",
                "type": "agent",
                "agent_id": "initial_processor_agent",
                "is_start": True
            },
            {
                "id": "complexity_check",
                "type": "condition",
                "condition_expression": "is_complex_data"
            },
            {
                "id": "simple_processor",
                "type": "agent",
                "agent_id": "simple_processor_agent"
            },
            {
                "id": "complex_processor",
                "type": "agent",
                "agent_id": "complex_processor_agent"
            },
            {
                "id": "refinement_loop",
                "type": "loop",
                "condition_expression": "needs_refinement",
                "max_iterations": 3
            },
            {
                "id": "refiner",
                "type": "agent",
                "agent_id": "refiner_agent"
            },
            {
                "id": "final_processor",
                "type": "agent",
                "agent_id": "final_processor_agent",
                "is_end": True
            }
        ],
        "edges": [
            {"from_node": "initial_processor", "to_node": "complexity_check"},
            {"from_node": "complexity_check", "to_node": "simple_processor", "condition": "not complexity_check_result"},
            {"from_node": "complexity_check", "to_node": "complex_processor", "condition": "complexity_check_result"},
            {"from_node": "complex_processor", "to_node": "refinement_loop"},
            {"from_node": "refinement_loop", "to_node": "refiner", "condition": "refinement_loop_continue"},
            {"from_node": "refiner", "to_node": "refinement_loop"},
            {"from_node": "simple_processor", "to_node": "final_processor"},
            {"from_node": "refinement_loop", "to_node": "final_processor", "condition": "not refinement_loop_continue"}
        ]
    }
    
    return {
        "simple": simple_config,
        "complex": complex_config
    }

# YAML Templates
SIMPLE_WORKFLOW_YAML = """
name: "Customer Support Pipeline"
description: "Automated customer support workflow"
version: "1.0"

nodes:
  - id: "ticket_processor"
    type: "agent"
    agent_id: "ticket_processor_agent"
    input_schema: "TicketSchema"
    output_schema: "ProcessedTicketSchema"
    is_start: true
    
  - id: "priority_classifier"
    type: "condition"
    condition_expression: "is_high_priority"
    
  - id: "auto_responder"
    type: "agent"
    agent_id: "auto_response_agent"
    
  - id: "human_escalation"
    type: "agent"
    agent_id: "escalation_agent"
    is_end: true

edges:
  - from_node: "ticket_processor"
    to_node: "priority_classifier"
    
  - from_node: "priority_classifier"
    to_node: "auto_responder"
    condition: "not priority_classifier_result"
    
  - from_node: "priority_classifier"
    to_node: "human_escalation"
    condition: "priority_classifier_result"

data_mappings:
  - source_node: "ticket_processor"
    source_field: "processed_ticket"
    target_node: "auto_responder"
    target_field: "ticket_data"
    transform_function: "format_for_response"
"""

# Usage Example
async def demonstrate_config_workflow():
    """Demonstrate configuration-driven workflow creation"""
    
    # Create registry and register components
    registry = ComponentRegistry()
    
    # Mock agents
    class MockTicketProcessor:
        async def run_async(self, data):
            class Result:
                data = {"processed_ticket": f"Processed: {data}", "priority": "normal"}
            return Result()
    
    class MockAutoResponder:
        async def run_async(self, data):
            class Result:
                data = {"response": f"Auto response for: {data}"}
            return Result()
    
    # Register components
    registry.register_agent("ticket_processor_agent", MockTicketProcessor())
    registry.register_agent("auto_response_agent", MockAutoResponder())
    registry.register_condition("is_high_priority", lambda ctx: ctx.data.get("ticket_processor_output", {}).get("priority") == "high")
    registry.register_transform("format_for_response", lambda data: f"Formatted: {data}")
    
    # Create workflow factory
    factory = WorkflowFactory(registry)
    
    # Create workflow from YAML
    workflow = factory.create_from_yaml(SIMPLE_WORKFLOW_YAML)
    
    print("Created workflow from YAML configuration:")
    WorkflowVisualizer.print_workflow_structure(workflow)
    
    return workflow

# Configuration Validator
class WorkflowConfigValidator:
    """Validator for workflow configurations"""
    
    def __init__(self, registry: ComponentRegistry):
        self.registry = registry
    
    def validate_config(self, config: WorkflowConfig) -> List[str]:
        """Validate workflow configuration and return list of errors"""
        errors = []
        
        # Check for duplicate node IDs
        node_ids = [node.id for node in config.nodes]
        if len(node_ids) != len(set(node_ids)):
            errors.append("Duplicate node IDs found")
        
        # Check edge references
        for edge in config.edges:
            if edge.from_node not in node_ids:
                errors.append(f"Edge references unknown from_node: {edge.from_node}")
            if edge.to_node not in node_ids:
                errors.append(f"Edge references unknown to_node: {edge.to_node}")
        
        # Check agent references
        for node in config.nodes:
            if node.type == "agent" and node.agent_id:
                if node.agent_id not in self.registry.agents:
                    errors.append(f"Node '{node.id}' references unknown agent: {node.agent_id}")
        
        # Check condition references
        for node in config.nodes:
            if node.type in ["condition", "loop"] and node.condition_expression:
                if node.condition_expression not in self.registry.conditions:
                    errors.append(f"Node '{node.id}' references unknown condition: {node.condition_expression}")
        
        # Check for at least one start node
        start_nodes = [node for node in config.nodes if node.is_start]
        if not start_nodes:
            errors.append("Workflow must have at least one start node")
        
        return errors

# Export/Import utilities
class WorkflowSerializer:
    """Utilities for serializing/deserializing workflows"""
    
    @staticmethod
    def workflow_to_config(workflow: WorkflowGraph) -> WorkflowConfig:
        """Convert workflow object to configuration"""
        # This would be implemented to reverse-engineer a workflow into config
        # Useful for visual workflow builders that need to save configurations
        pass
    
    @staticmethod
    def export_to_file(config: WorkflowConfig, file_path: str):
        """Export configuration to file"""
        if file_path.endswith(('.yml', '.yaml')):
            with open(file_path, 'w') as f:
                yaml.dump(config.model_dump(), f, default_flow_style=False)
        else:
            with open(file_path, 'w') as f:
                json.dump(config.model_dump(), f, indent=2)

# Main demonstration
if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_config_workflow())