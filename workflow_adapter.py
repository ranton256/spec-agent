"""
SpecAgent Workflow Integration Adapter

This module provides the adapter layer between SpecAgent's AgentSpec system
and the workflow orchestration engine. It handles:
- Converting AgentSpecs to workflow-compatible agents
- Managing the component registry
- Creating workflows from WorkflowSpecs
- Executing workflows using SpecAgent's sandbox execution
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import uuid

from models import (
    AgentSpec, WorkflowSpec, WorkflowNodeSpec, WorkflowEdgeSpec, 
    DataConnection, WorkflowExecutionResult, EnhancedAgentSpec
)
from workflow.config_driven_workflows import ComponentRegistry, WorkflowFactory, WorkflowConfig, NodeConfig, EdgeConfig
from workflow.pydantic_workflow import WorkflowGraph, WorkflowContext, AgentNode, ConditionNode, LoopNode
import sandbox_executor


class SpecAgentWrapper:
    """Wrapper that adapts a SpecAgent AgentSpec to work with the workflow system"""
    
    def __init__(self, agent_spec: AgentSpec):
        self.agent_spec = agent_spec
        self.name = agent_spec.name
    
    async def run_async(self, input_data: Any):
        """Execute the SpecAgent using sandbox_executor"""
        
        class MockResult:
            def __init__(self, data: Any):
                self.data = data
        
        # Convert input_data to format expected by sandbox_executor
        if isinstance(input_data, dict):
            inputs = input_data
        else:
            # Create a default input mapping
            if self.agent_spec.input_schema:
                first_field = self.agent_spec.input_schema[0].name
                inputs = {first_field: str(input_data)}
            else:
                inputs = {"input": str(input_data)}
        
        # Execute using sandbox_executor
        try:
            # Create a temporary run context
            run_id = f"workflow_run_{uuid.uuid4().hex[:8]}"
            result = await sandbox_executor.run_agent_async(
                self.agent_spec, 
                inputs,
                Path("output"),  # Use default output directory
                run_id
            )
            
            # Extract the message from the result
            if hasattr(result, 'data') and 'message' in result.data:
                return MockResult(result.data['message'])
            elif isinstance(result, dict) and 'message' in result:
                return MockResult(result['message'])
            else:
                return MockResult(str(result))
                
        except Exception as e:
            return MockResult(f"Error executing agent {self.agent_spec.name}: {str(e)}")


class SpecAgentWorkflowAdapter:
    """Main adapter class for integrating SpecAgent with workflow system"""
    
    def __init__(self):
        self.registry = ComponentRegistry()
        self.spec_cache: Dict[str, AgentSpec] = {}
        self.workflow_factory = WorkflowFactory(self.registry)
    
    def register_agent_spec(self, agent_spec: AgentSpec) -> str:
        """Register a SpecAgent AgentSpec as a workflow component"""
        agent_id = f"spec_agent_{agent_spec.id}"
        workflow_agent = SpecAgentWrapper(agent_spec)
        self.registry.register_agent(agent_id, workflow_agent)
        self.spec_cache[agent_id] = agent_spec
        return agent_id
    
    def register_condition(self, condition_id: str, condition_fn: Callable[[Any], bool]):
        """Register a condition function for workflow nodes"""
        self.registry.register_condition(condition_id, condition_fn)
    
    def register_transform(self, transform_id: str, transform_fn: Callable[[Any], Any]):
        """Register a data transform function"""
        self.registry.register_transform(transform_id, transform_fn)
    
    def create_workflow_from_spec(self, workflow_spec: WorkflowSpec) -> WorkflowGraph:
        """Convert a SpecAgent WorkflowSpec to an executable WorkflowGraph"""
        
        # Register all agent specs referenced in the workflow
        agent_id_map = {}
        for agent_spec in workflow_spec.agent_specs:
            registered_id = self.register_agent_spec(agent_spec)
            agent_id_map[agent_spec.id] = registered_id
        
        # Convert to workflow config format
        config = self._spec_to_workflow_config(workflow_spec, agent_id_map)
        
        # Create and return the workflow
        return self.workflow_factory.create_from_config(config)
    
    def _spec_to_workflow_config(self, workflow_spec: WorkflowSpec, agent_id_map: Dict[str, str]) -> WorkflowConfig:
        """Convert WorkflowSpec to WorkflowConfig"""
        
        # Convert nodes
        node_configs = []
        for node_spec in workflow_spec.nodes:
            node_config = NodeConfig(
                id=node_spec.id,
                type=node_spec.type,
                name=node_spec.name,
                is_start=node_spec.is_start,
                is_end=node_spec.is_end,
                metadata=node_spec.config
            )
            
            # Set type-specific fields
            if node_spec.type == "agent" and node_spec.agent_spec_id:
                node_config.agent_id = agent_id_map.get(node_spec.agent_spec_id)
            elif node_spec.type in ["condition", "loop"] and node_spec.condition_expression:
                node_config.condition_expression = node_spec.condition_expression
            elif node_spec.type == "loop" and node_spec.max_iterations:
                node_config.max_iterations = node_spec.max_iterations
            
            node_configs.append(node_config)
        
        # Convert edges
        edge_configs = []
        for edge_spec in workflow_spec.edges:
            edge_config = EdgeConfig(
                from_node=edge_spec.from_node,
                to_node=edge_spec.to_node,
                condition=edge_spec.condition,
                data_mapping=edge_spec.data_mapping
            )
            edge_configs.append(edge_config)
        
        return WorkflowConfig(
            name=workflow_spec.name,
            description=workflow_spec.description,
            version=workflow_spec.version,
            nodes=node_configs,
            edges=edge_configs,
            metadata=workflow_spec.metadata
        )
    
    def validate_workflow_spec(self, workflow_spec: WorkflowSpec) -> List[str]:
        """Validate a workflow specification and return list of errors"""
        errors = []
        
        # Check for at least one start node
        start_nodes = [node for node in workflow_spec.nodes if node.is_start]
        if not start_nodes:
            errors.append("Workflow must have at least one start node")
        
        # Check for at least one end node
        end_nodes = [node for node in workflow_spec.nodes if node.is_end]
        if not end_nodes:
            errors.append("Workflow must have at least one end node")
        
        # Check node ID uniqueness
        node_ids = [node.id for node in workflow_spec.nodes]
        if len(node_ids) != len(set(node_ids)):
            errors.append("Duplicate node IDs found")
        
        # Check edge references
        for edge in workflow_spec.edges:
            if edge.from_node not in node_ids:
                errors.append(f"Edge references unknown from_node: {edge.from_node}")
            if edge.to_node not in node_ids:
                errors.append(f"Edge references unknown to_node: {edge.to_node}")
        
        # Check agent spec references
        agent_spec_ids = {spec.id for spec in workflow_spec.agent_specs}
        for node in workflow_spec.nodes:
            if node.type == "agent" and node.agent_spec_id:
                if node.agent_spec_id not in agent_spec_ids:
                    errors.append(f"Node '{node.id}' references unknown agent_spec_id: {node.agent_spec_id}")
        
        return errors


class SpecAgentWorkflowExecutor:
    """Executes workflows using SpecAgent's execution model"""
    
    def __init__(self, workflow_spec: WorkflowSpec):
        self.workflow_spec = workflow_spec
        self.adapter = SpecAgentWorkflowAdapter()
        self._setup_default_conditions()
        
        # Validate workflow before creating
        validation_errors = self.adapter.validate_workflow_spec(workflow_spec)
        if validation_errors:
            raise ValueError(f"Workflow validation failed: {'; '.join(validation_errors)}")
        
        self.workflow = self.adapter.create_workflow_from_spec(workflow_spec)
    
    def _setup_default_conditions(self):
        """Register default condition functions"""
        # Basic condition that always returns True
        self.adapter.register_condition("always_true", lambda ctx: True)
        
        # Basic condition that always returns False  
        self.adapter.register_condition("always_false", lambda ctx: False)
        
        # Condition to check if a field exists and is not empty
        def has_value(field_name: str):
            def condition_fn(ctx):
                value = ctx.data.get(field_name)
                return value is not None and value != ""
            return condition_fn
        
        self.adapter.register_condition("has_value", has_value)
    
    async def execute(self, initial_inputs: Dict[str, Any]) -> WorkflowExecutionResult:
        """Execute the workflow with comprehensive result tracking"""
        execution_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Create execution directory
        run_dir = Path(f"runs/workflows/{self.workflow_spec.id}/{execution_id}")
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save initial inputs
        with open(run_dir / "inputs.json", 'w') as f:
            json.dump(initial_inputs, f, indent=2, default=str)
        
        try:
            # Create workflow context with initial inputs
            context = WorkflowContext(data=initial_inputs.copy())
            context.metadata['execution_id'] = execution_id
            context.metadata['workflow_id'] = self.workflow_spec.id
            
            # Execute the workflow
            result_context = await self.workflow.execute(context)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Create result object
            result = WorkflowExecutionResult(
                workflow_id=self.workflow_spec.id,
                execution_id=execution_id,
                status="completed",
                start_time=context.metadata.get('start_time', 
                                             time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(start_time))),
                end_time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(end_time)),
                execution_time_seconds=execution_time,
                node_results=result_context.data.get('_execution_results', {}),
                final_outputs={k: v for k, v in result_context.data.items() 
                             if not k.startswith('_') and not k.endswith('_input')},
                run_directory=str(run_dir)
            )
            
            # Save results
            with open(run_dir / "result.json", 'w') as f:
                json.dump(result.model_dump(), f, indent=2, default=str)
            
            with open(run_dir / "context.json", 'w') as f:
                json.dump(result_context.data, f, indent=2, default=str)
            
            return result
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Create error result
            result = WorkflowExecutionResult(
                workflow_id=self.workflow_spec.id,
                execution_id=execution_id,
                status="failed",
                start_time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(start_time)),
                end_time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(end_time)),
                execution_time_seconds=execution_time,
                error_message=str(e),
                run_directory=str(run_dir)
            )
            
            # Save error result
            with open(run_dir / "result.json", 'w') as f:
                json.dump(result.model_dump(), f, indent=2, default=str)
            
            return result


# Utility functions for creating common workflow patterns
def create_simple_sequential_workflow(agent_specs: List[AgentSpec], workflow_name: str) -> WorkflowSpec:
    """Create a simple sequential workflow from a list of agent specs"""
    
    workflow_nodes = []
    workflow_edges = []
    
    # Create nodes
    for i, agent_spec in enumerate(agent_specs):
        node = WorkflowNodeSpec(
            id=f"agent_{i}",
            type="agent",
            name=agent_spec.name,
            agent_spec_id=agent_spec.id,
            is_start=(i == 0),
            is_end=(i == len(agent_specs) - 1)
        )
        workflow_nodes.append(node)
        
        # Create edges (connect sequential nodes)
        if i > 0:
            edge = WorkflowEdgeSpec(
                from_node=f"agent_{i-1}",
                to_node=f"agent_{i}"
            )
            workflow_edges.append(edge)
    
    return WorkflowSpec(
        name=workflow_name,
        description=f"Sequential workflow with {len(agent_specs)} agents",
        nodes=workflow_nodes,
        edges=workflow_edges,
        agent_specs=agent_specs
    )


def create_conditional_workflow(condition_agent: AgentSpec, 
                               true_agent: AgentSpec, 
                               false_agent: AgentSpec,
                               workflow_name: str) -> WorkflowSpec:
    """Create a conditional workflow with branching logic"""
    
    workflow_nodes = [
        WorkflowNodeSpec(
            id="condition_agent",
            type="agent", 
            name=condition_agent.name,
            agent_spec_id=condition_agent.id,
            is_start=True
        ),
        WorkflowNodeSpec(
            id="condition_check",
            type="condition",
            name="Condition Check",
            condition_expression="always_true"  # Simplified for Phase 1
        ),
        WorkflowNodeSpec(
            id="true_branch",
            type="agent",
            name=true_agent.name,
            agent_spec_id=true_agent.id,
            is_end=True
        ),
        WorkflowNodeSpec(
            id="false_branch", 
            type="agent",
            name=false_agent.name,
            agent_spec_id=false_agent.id,
            is_end=True
        )
    ]
    
    workflow_edges = [
        WorkflowEdgeSpec(from_node="condition_agent", to_node="condition_check"),
        WorkflowEdgeSpec(from_node="condition_check", to_node="true_branch", condition="condition_check_result"),
        WorkflowEdgeSpec(from_node="condition_check", to_node="false_branch", condition="not condition_check_result")
    ]
    
    return WorkflowSpec(
        name=workflow_name,
        description="Conditional workflow with branching",
        nodes=workflow_nodes,
        edges=workflow_edges,
        agent_specs=[condition_agent, true_agent, false_agent]
    )