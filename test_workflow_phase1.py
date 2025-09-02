"""
Comprehensive test suite for Workflow Integration Phase 1

Tests cover:
- Model schema validation and serialization
- Workflow adapter functionality  
- Agent spec to workflow conversion
- Basic workflow execution
- Backward compatibility
- Error handling and validation
"""

import unittest
import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

from models import (
    AgentSpec, InputField, ToolRef, RunLimits, SDKConfig,
    WorkflowSpec, WorkflowNodeSpec, WorkflowEdgeSpec, DataConnection,
    ExecutionLimits, WorkflowExecutionResult, EnhancedAgentSpec, NodeIOSpec
)
from workflow_adapter import (
    SpecAgentWorkflowAdapter, SpecAgentWorkflowExecutor, SpecAgentWrapper,
    create_simple_sequential_workflow, create_conditional_workflow
)
import sandbox_executor


class TestWorkflowModels(unittest.TestCase):
    """Test workflow model schemas and validation"""
    
    def test_node_io_spec_creation(self):
        """Test NodeIOSpec model creation and validation"""
        io_spec = NodeIOSpec(
            inputs={"text": {"type": "str", "required": True}},
            outputs={"result": {"type": "str", "required": True}}
        )
        
        self.assertEqual(io_spec.inputs["text"]["type"], "str")
        self.assertEqual(io_spec.outputs["result"]["required"], True)
        
        # Test empty default
        empty_spec = NodeIOSpec()
        self.assertEqual(len(empty_spec.inputs), 0)
        self.assertEqual(len(empty_spec.outputs), 0)
    
    def test_data_connection_validation(self):
        """Test DataConnection model validation"""
        connection = DataConnection(
            source_node="node1",
            source_output_field="output",
            target_node="node2", 
            target_input_field="input",
            transform_function="uppercase"
        )
        
        self.assertEqual(connection.source_node, "node1")
        self.assertEqual(connection.target_node, "node2")
        self.assertEqual(connection.transform_function, "uppercase")
    
    def test_workflow_node_spec_types(self):
        """Test WorkflowNodeSpec supports all node types"""
        # Agent node
        agent_node = WorkflowNodeSpec(
            id="agent1",
            type="agent",
            agent_spec_id="test_agent_id",
            is_start=True
        )
        self.assertEqual(agent_node.type, "agent")
        self.assertTrue(agent_node.is_start)
        
        # Condition node
        condition_node = WorkflowNodeSpec(
            id="condition1",
            type="condition",
            condition_expression="always_true"
        )
        self.assertEqual(condition_node.type, "condition")
        self.assertEqual(condition_node.condition_expression, "always_true")
        
        # Loop node
        loop_node = WorkflowNodeSpec(
            id="loop1",
            type="loop",
            condition_expression="continue_loop",
            max_iterations=5
        )
        self.assertEqual(loop_node.type, "loop")
        self.assertEqual(loop_node.max_iterations, 5)
    
    def test_workflow_spec_creation(self):
        """Test WorkflowSpec creation and validation"""
        agent_spec = AgentSpec(
            name="Test Agent",
            description="Test description", 
            system_instructions="Test instructions",
            task_prompt="Test prompt"
        )
        
        workflow_spec = WorkflowSpec(
            name="Test Workflow",
            description="Test workflow description",
            nodes=[
                WorkflowNodeSpec(
                    id="start_node",
                    type="agent",
                    agent_spec_id=agent_spec.id,
                    is_start=True,
                    is_end=True
                )
            ],
            agent_specs=[agent_spec]
        )
        
        self.assertEqual(workflow_spec.name, "Test Workflow")
        self.assertEqual(len(workflow_spec.nodes), 1)
        self.assertEqual(len(workflow_spec.agent_specs), 1)
        self.assertEqual(workflow_spec.nodes[0].agent_spec_id, agent_spec.id)
    
    def test_enhanced_agent_spec_compatibility(self):
        """Test EnhancedAgentSpec backward compatibility"""
        enhanced_spec = EnhancedAgentSpec(
            name="Enhanced Agent",
            description="Enhanced description",
            system_instructions="Enhanced instructions", 
            task_prompt="Enhanced prompt",
            input_schema=[
                InputField(name="input1", type="str"),
                InputField(name="input2", type="int", required=False)
            ]
        )
        
        # Test schema conversion
        input_dict = enhanced_spec.get_input_schema_dict()
        self.assertIn("input1", input_dict)
        self.assertEqual(input_dict["input1"]["type"], "str")
        self.assertTrue(input_dict["input1"]["required"])
        self.assertFalse(input_dict["input2"]["required"])
        
        # Test output schema
        output_dict = enhanced_spec.get_output_schema_dict()
        self.assertIn("message", output_dict)
        self.assertIn("tool_results", output_dict)
        
        # Test backward compatibility
        base_spec = enhanced_spec.to_agent_spec()
        self.assertIsInstance(base_spec, AgentSpec)
        self.assertEqual(base_spec.name, enhanced_spec.name)


class TestSpecAgentWrapper(unittest.TestCase):
    """Test SpecAgentWrapper functionality"""
    
    def setUp(self):
        self.test_agent = AgentSpec(
            name="Test Wrapper Agent",
            description="Test agent for wrapper testing",
            system_instructions="You are a test agent",
            task_prompt="Respond with a greeting",
            input_schema=[
                InputField(name="name", type="str")
            ]
        )
        self.wrapper = SpecAgentWrapper(self.test_agent)
    
    def test_wrapper_initialization(self):
        """Test wrapper initialization"""
        self.assertEqual(self.wrapper.name, "Test Wrapper Agent")
        self.assertEqual(self.wrapper.agent_spec.id, self.test_agent.id)
    
    async def test_wrapper_execution_with_dict_input(self):
        """Test wrapper execution with dictionary input"""
        # Mock the sandbox_executor.run_agent_async function for testing
        original_run_agent_async = None
        if hasattr(sandbox_executor, 'run_agent_async'):
            original_run_agent_async = sandbox_executor.run_agent_async
        
        async def mock_run_agent_async(spec, inputs, output_dir, run_id):
            return {"data": {"message": f"Hello {inputs.get('name', 'World')}!"}}
        
        sandbox_executor.run_agent_async = mock_run_agent_async
        
        try:
            result = await self.wrapper.run_async({"name": "Test"})
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.data)
        finally:
            # Restore original function
            if original_run_agent_async:
                sandbox_executor.run_agent_async = original_run_agent_async
    
    async def test_wrapper_execution_with_string_input(self):
        """Test wrapper execution with string input"""
        # Mock the sandbox_executor.run_agent_async function for testing
        async def mock_run_agent_async(spec, inputs, output_dir, run_id):
            return {"data": {"message": f"Processed: {inputs.get('name', inputs)}"}}
        
        original_run_agent_async = getattr(sandbox_executor, 'run_agent_async', None)
        sandbox_executor.run_agent_async = mock_run_agent_async
        
        try:
            result = await self.wrapper.run_async("test string")
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.data)
        finally:
            if original_run_agent_async:
                sandbox_executor.run_agent_async = original_run_agent_async


class TestWorkflowAdapter(unittest.TestCase):
    """Test SpecAgentWorkflowAdapter functionality"""
    
    def setUp(self):
        self.adapter = SpecAgentWorkflowAdapter()
        self.test_agent = AgentSpec(
            name="Test Agent",
            description="Test agent for adapter testing",
            system_instructions="Test instructions",
            task_prompt="Test prompt"
        )
    
    def test_agent_registration(self):
        """Test agent spec registration"""
        agent_id = self.adapter.register_agent_spec(self.test_agent)
        
        self.assertIsNotNone(agent_id)
        self.assertTrue(agent_id.startswith("spec_agent_"))
        self.assertIn(agent_id, self.adapter.registry.agents)
        self.assertIn(agent_id, self.adapter.spec_cache)
    
    def test_condition_registration(self):
        """Test condition function registration"""
        def test_condition(ctx):
            return True
        
        self.adapter.register_condition("test_condition", test_condition)
        self.assertIn("test_condition", self.adapter.registry.conditions)
    
    def test_transform_registration(self):
        """Test transform function registration"""
        def test_transform(data):
            return data.upper() if isinstance(data, str) else str(data)
        
        self.adapter.register_transform("test_transform", test_transform)
        self.assertIn("test_transform", self.adapter.registry.transforms)
    
    def test_workflow_spec_validation(self):
        """Test workflow specification validation"""
        # Valid workflow
        valid_workflow = WorkflowSpec(
            name="Valid Workflow",
            description="A valid workflow",
            nodes=[
                WorkflowNodeSpec(
                    id="start",
                    type="agent",
                    agent_spec_id=self.test_agent.id,
                    is_start=True,
                    is_end=True
                )
            ],
            agent_specs=[self.test_agent]
        )
        
        errors = self.adapter.validate_workflow_spec(valid_workflow)
        self.assertEqual(len(errors), 0)
        
        # Invalid workflow - no start node
        invalid_workflow = WorkflowSpec(
            name="Invalid Workflow",
            description="An invalid workflow",
            nodes=[
                WorkflowNodeSpec(
                    id="middle",
                    type="agent",
                    agent_spec_id=self.test_agent.id
                )
            ],
            agent_specs=[self.test_agent]
        )
        
        errors = self.adapter.validate_workflow_spec(invalid_workflow)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("start node" in error for error in errors))
    
    def test_workflow_creation_from_spec(self):
        """Test workflow creation from specification"""
        workflow_spec = WorkflowSpec(
            name="Test Workflow",
            description="Test workflow creation",
            nodes=[
                WorkflowNodeSpec(
                    id="agent1",
                    type="agent",
                    agent_spec_id=self.test_agent.id,
                    is_start=True,
                    is_end=True
                )
            ],
            agent_specs=[self.test_agent]
        )
        
        workflow = self.adapter.create_workflow_from_spec(workflow_spec)
        self.assertIsNotNone(workflow)
        self.assertEqual(workflow.name, "Test Workflow")
        self.assertGreater(len(workflow.nodes), 0)


class TestWorkflowUtilityFunctions(unittest.TestCase):
    """Test workflow utility functions"""
    
    def setUp(self):
        self.agent1 = AgentSpec(
            name="Agent 1",
            description="First agent",
            system_instructions="Instructions 1",
            task_prompt="Prompt 1"
        )
        self.agent2 = AgentSpec(
            name="Agent 2", 
            description="Second agent",
            system_instructions="Instructions 2",
            task_prompt="Prompt 2"
        )
    
    def test_simple_sequential_workflow_creation(self):
        """Test creation of simple sequential workflows"""
        workflow = create_simple_sequential_workflow(
            [self.agent1, self.agent2],
            "Sequential Test Workflow"
        )
        
        self.assertEqual(workflow.name, "Sequential Test Workflow")
        self.assertEqual(len(workflow.nodes), 2)
        self.assertEqual(len(workflow.edges), 1)
        self.assertEqual(len(workflow.agent_specs), 2)
        
        # Check start and end nodes
        start_nodes = [node for node in workflow.nodes if node.is_start]
        end_nodes = [node for node in workflow.nodes if node.is_end]
        self.assertEqual(len(start_nodes), 1)
        self.assertEqual(len(end_nodes), 1)
        
        # Check edge connection
        edge = workflow.edges[0]
        self.assertEqual(edge.from_node, "agent_0")
        self.assertEqual(edge.to_node, "agent_1")
    
    def test_conditional_workflow_creation(self):
        """Test creation of conditional workflows"""
        agent3 = AgentSpec(
            name="Agent 3",
            description="Third agent",
            system_instructions="Instructions 3", 
            task_prompt="Prompt 3"
        )
        
        workflow = create_conditional_workflow(
            self.agent1, self.agent2, agent3,
            "Conditional Test Workflow"
        )
        
        self.assertEqual(workflow.name, "Conditional Test Workflow")
        self.assertEqual(len(workflow.nodes), 4)  # condition agent + condition + 2 branches
        self.assertEqual(len(workflow.edges), 3)   # condition->check, check->true, check->false
        self.assertEqual(len(workflow.agent_specs), 3)


class TestWorkflowExecution(unittest.TestCase):
    """Test workflow execution functionality"""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create simple test agent
        self.test_agent = AgentSpec(
            name="Echo Agent",
            description="Simple echo agent for testing",
            system_instructions="You echo back what you receive",
            task_prompt="Echo the input back to the user",
            input_schema=[
                InputField(name="message", type="str")
            ]
        )
        
        # Create simple workflow
        self.workflow_spec = WorkflowSpec(
            name="Echo Workflow",
            description="Simple echo workflow for testing",
            nodes=[
                WorkflowNodeSpec(
                    id="echo_node",
                    type="agent",
                    agent_spec_id=self.test_agent.id,
                    is_start=True,
                    is_end=True
                )
            ],
            agent_specs=[self.test_agent]
        )
    
    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_workflow_executor_initialization(self):
        """Test workflow executor initialization"""
        executor = SpecAgentWorkflowExecutor(self.workflow_spec)
        self.assertIsNotNone(executor.workflow_spec)
        self.assertIsNotNone(executor.adapter)
        self.assertIsNotNone(executor.workflow)
    
    def test_workflow_executor_validation(self):
        """Test workflow executor validation"""
        # Valid workflow should initialize successfully
        try:
            executor = SpecAgentWorkflowExecutor(self.workflow_spec)
            self.assertIsNotNone(executor)
        except ValueError:
            self.fail("Valid workflow should not raise ValueError")
        
        # Invalid workflow should raise ValueError
        invalid_workflow = WorkflowSpec(
            name="Invalid Workflow",
            description="Workflow with no start node",
            nodes=[
                WorkflowNodeSpec(
                    id="middle_node",
                    type="agent",
                    agent_spec_id=self.test_agent.id
                )
            ],
            agent_specs=[self.test_agent]
        )
        
        with self.assertRaises(ValueError):
            SpecAgentWorkflowExecutor(invalid_workflow)
    
    async def test_workflow_execution_success(self):
        """Test successful workflow execution"""
        # Mock the actual agent execution for testing
        original_run_agent_async = getattr(sandbox_executor, 'run_agent_async', None)
        
        async def mock_run_agent_async(spec, inputs, output_dir, run_id):
            return {
                "data": {
                    "message": f"Echo: {inputs.get('message', 'default')}",
                    "tool_results": [],
                    "metadata": {}
                }
            }
        
        sandbox_executor.run_agent_async = mock_run_agent_async
        
        try:
            executor = SpecAgentWorkflowExecutor(self.workflow_spec)
            result = await executor.execute({"message": "Hello World"})
            
            self.assertIsInstance(result, WorkflowExecutionResult)
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.workflow_id, self.workflow_spec.id)
            self.assertIsNotNone(result.execution_id)
            self.assertIsNotNone(result.execution_time_seconds)
            
        finally:
            if original_run_agent_async:
                sandbox_executor.run_agent_async = original_run_agent_async
    
    async def test_workflow_execution_error(self):
        """Test workflow execution error handling"""
        # Mock agent execution to raise error
        async def mock_run_agent_async_error(spec, inputs, output_dir, run_id):
            raise Exception("Test execution error")
        
        original_run_agent_async = getattr(sandbox_executor, 'run_agent_async', None) 
        sandbox_executor.run_agent_async = mock_run_agent_async_error
        
        try:
            executor = SpecAgentWorkflowExecutor(self.workflow_spec)
            result = await executor.execute({"message": "Hello World"})
            
            self.assertIsInstance(result, WorkflowExecutionResult)
            # The workflow completes, but the agent error is captured in the result
            self.assertEqual(result.status, "completed")
            
            # Check that the error is captured in the final outputs
            self.assertIsNotNone(result.final_outputs)
            
            # Find the agent output and verify it contains the error message
            found_error = False
            for key, value in result.final_outputs.items():
                if isinstance(value, str) and "Error executing agent" in value and "Test execution error" in value:
                    found_error = True
                    break
            
            self.assertTrue(found_error, "Expected to find error message in workflow outputs")
            
        finally:
            if original_run_agent_async:
                sandbox_executor.run_agent_async = original_run_agent_async


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with existing AgentSpec functionality"""
    
    def test_agent_spec_unchanged(self):
        """Test that AgentSpec model is unchanged"""
        # Create AgentSpec using original signature
        agent_spec = AgentSpec(
            name="Compatible Agent",
            description="Test backward compatibility",
            system_instructions="Original instructions",
            task_prompt="Original prompt",
            input_schema=[
                InputField(name="input", type="str")
            ],
            tools=[
                ToolRef(name="math_eval")
            ],
            run_limits=RunLimits(timeout_s=60),
            sdk_config=SDKConfig(model="gpt-4o-mini")
        )
        
        # Verify all original fields are preserved
        self.assertEqual(agent_spec.name, "Compatible Agent")
        self.assertEqual(len(agent_spec.input_schema), 1)
        self.assertEqual(len(agent_spec.tools), 1)
        self.assertEqual(agent_spec.run_limits.timeout_s, 60)
        self.assertEqual(agent_spec.sdk_config.model, "gpt-4o-mini")
    
    def test_enhanced_agent_spec_conversion(self):
        """Test conversion between enhanced and base AgentSpec"""
        enhanced = EnhancedAgentSpec(
            name="Enhanced Agent",
            description="Enhanced description",
            system_instructions="Enhanced instructions",
            task_prompt="Enhanced prompt"
        )
        
        # Convert to base AgentSpec
        base = enhanced.to_agent_spec()
        
        # Verify conversion preserves all base fields
        self.assertEqual(base.name, enhanced.name)
        self.assertEqual(base.description, enhanced.description)
        self.assertEqual(base.system_instructions, enhanced.system_instructions)
        self.assertEqual(base.task_prompt, enhanced.task_prompt)
        self.assertIsInstance(base, AgentSpec)


class TestWorkflowIntegrationEndToEnd(unittest.TestCase):
    """End-to-end integration tests"""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    async def test_complete_workflow_lifecycle(self):
        """Test complete workflow creation, validation, and execution"""
        # Create agents
        agent1 = AgentSpec(
            name="Input Processor",
            description="Processes input data",
            system_instructions="Process and format input data",
            task_prompt="Format the input data nicely"
        )
        
        agent2 = AgentSpec(
            name="Output Formatter", 
            description="Formats output data",
            system_instructions="Format output data for display",
            task_prompt="Create a nice output format"
        )
        
        # Create workflow using utility function
        workflow_spec = create_simple_sequential_workflow(
            [agent1, agent2],
            "Integration Test Workflow"
        )
        
        # Validate workflow
        adapter = SpecAgentWorkflowAdapter()
        validation_errors = adapter.validate_workflow_spec(workflow_spec)
        self.assertEqual(len(validation_errors), 0)
        
        # Mock agent execution
        async def mock_run_agent_async(spec, inputs, output_dir, run_id):
            return {
                "data": {
                    "message": f"Processed by {spec.name}: {list(inputs.values())[0]}",
                    "tool_results": [],
                    "metadata": {"agent": spec.name}
                }
            }
        
        original_run_agent_async = getattr(sandbox_executor, 'run_agent_async', None)
        sandbox_executor.run_agent_async = mock_run_agent_async
        
        try:
            # Execute workflow
            executor = SpecAgentWorkflowExecutor(workflow_spec)
            result = await executor.execute({"input_data": "test data"})
            
            # Verify results
            self.assertEqual(result.status, "completed")
            self.assertIsNotNone(result.execution_time_seconds)
            self.assertGreater(len(result.node_results), 0)
            
        finally:
            if original_run_agent_async:
                sandbox_executor.run_agent_async = original_run_agent_async


def run_async_test(coro):
    """Helper function to run async tests"""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# Convert async test methods to sync for unittest
for test_class in [TestSpecAgentWrapper, TestWorkflowExecution, TestWorkflowIntegrationEndToEnd]:
    for attr_name in dir(test_class):
        attr = getattr(test_class, attr_name)
        if callable(attr) and attr_name.startswith('test_') and asyncio.iscoroutinefunction(attr):
            # Create sync wrapper
            def make_sync_test(async_method):
                def sync_test(self):
                    return run_async_test(async_method(self))
                return sync_test
            
            setattr(test_class, attr_name, make_sync_test(attr))


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)