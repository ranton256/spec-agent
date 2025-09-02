"""
Backward Compatibility Tests for Workflow Integration

This test suite ensures that all existing SpecAgent functionality 
remains intact after the workflow integration changes.
"""

import unittest
import json
import asyncio
from pathlib import Path

# Import existing models and functionality
from models import AgentSpec, InputField, ToolRef, RunLimits, SDKConfig
import sandbox_executor


class TestExistingAgentSpecFunctionality(unittest.TestCase):
    """Test that existing AgentSpec functionality is preserved"""
    
    def setUp(self):
        """Set up test agent specs using original patterns"""
        self.simple_agent = AgentSpec(
            name="Simple Test Agent",
            description="A simple test agent", 
            system_instructions="You are a helpful assistant",
            task_prompt="Help the user with their request"
        )
        
        self.complex_agent = AgentSpec(
            name="Complex Test Agent",
            description="A complex test agent with all features",
            system_instructions="You are a specialized assistant",
            task_prompt="Process the input according to the schema",
            input_schema=[
                InputField(name="text", type="str", required=True),
                InputField(name="count", type="int", required=False, default=1),
                InputField(name="options", type="json", required=False)
            ],
            tools=[
                ToolRef(name="math_eval", params={}),
                ToolRef(name="string_template", params={})
            ],
            run_limits=RunLimits(timeout_s=60, max_output_chars=2000),
            sdk_config=SDKConfig(
                model="gpt-4o-mini",
                temperature=0.7,
                use_local=False
            )
        )
    
    def test_agent_spec_serialization(self):
        """Test AgentSpec can still be serialized to JSON"""
        # Simple agent
        simple_dict = self.simple_agent.model_dump()
        self.assertIn("name", simple_dict)
        self.assertIn("description", simple_dict)
        self.assertIn("system_instructions", simple_dict)
        
        # Complex agent  
        complex_dict = self.complex_agent.model_dump()
        self.assertEqual(len(complex_dict["input_schema"]), 3)
        self.assertEqual(len(complex_dict["tools"]), 2)
        self.assertEqual(complex_dict["sdk_config"]["model"], "gpt-4o-mini")
    
    def test_agent_spec_deserialization(self):
        """Test AgentSpec can be recreated from JSON"""
        # Serialize then deserialize
        serialized = self.complex_agent.model_dump()
        deserialized = AgentSpec(**serialized)
        
        # Verify all fields preserved
        self.assertEqual(deserialized.name, self.complex_agent.name)
        self.assertEqual(deserialized.description, self.complex_agent.description)
        self.assertEqual(len(deserialized.input_schema), 3)
        self.assertEqual(len(deserialized.tools), 2)
        self.assertEqual(deserialized.sdk_config.model, "gpt-4o-mini")
    
    def test_input_field_types(self):
        """Test all InputField types are still supported"""
        fields = [
            InputField(name="str_field", type="str"),
            InputField(name="int_field", type="int"),
            InputField(name="float_field", type="float"),
            InputField(name="bool_field", type="bool"),
            InputField(name="json_field", type="json")
        ]
        
        for field in fields:
            self.assertIn(field.type, ["str", "int", "float", "bool", "json"])
    
    def test_tool_references(self):
        """Test all existing tool references are supported"""
        tools = [
            ToolRef(name="math_eval"),
            ToolRef(name="string_template"), 
            ToolRef(name="kv_memory"),
            ToolRef(name="http_get"),
            ToolRef(name="web_search"),
            ToolRef(name="conversational_memory")
        ]
        
        for tool in tools:
            self.assertIn(tool.name, [
                "math_eval", "string_template", "kv_memory", 
                "http_get", "web_search", "conversational_memory"
            ])
    
    def test_run_limits_configuration(self):
        """Test RunLimits configuration still works"""
        limits = RunLimits(
            timeout_s=120,
            max_output_chars=5000
        )
        
        self.assertEqual(limits.timeout_s, 120)
        self.assertEqual(limits.max_output_chars, 5000)
    
    def test_sdk_config_options(self):
        """Test SDKConfig options are preserved"""
        config = SDKConfig(
            model="gpt-4o-mini",
            temperature=0.5,
            use_local=True,
            local_model="custom_model"
        )
        
        self.assertEqual(config.model, "gpt-4o-mini")
        self.assertEqual(config.temperature, 0.5)
        self.assertTrue(config.use_local)
        self.assertEqual(config.local_model, "custom_model")


class TestExistingSandboxExecutorFunctionality(unittest.TestCase):
    """Test that sandbox_executor functions are preserved"""
    
    def setUp(self):
        self.test_agent = AgentSpec(
            name="Executor Test Agent",
            description="Agent for testing executor functionality",
            system_instructions="You are a test assistant",
            task_prompt="Echo back the input",
            input_schema=[
                InputField(name="message", type="str")
            ]
        )
    
    def test_run_agent_async_signature(self):
        """Test that run_agent_async function signature is preserved"""
        # Check function exists
        self.assertTrue(hasattr(sandbox_executor, 'run_agent_async'))
        
        # Check it's callable
        self.assertTrue(callable(sandbox_executor.run_agent_async))
    
    def test_tool_functions_preserved(self):
        """Test that existing tool functions are still available"""
        # Math eval
        result = sandbox_executor.math_eval("2 + 2")
        self.assertEqual(result, "4")
        
        # String template
        result = sandbox_executor.string_template("Hello {name}", name="World")
        self.assertEqual(result, "Hello World")
        
        # KV memory functions
        self.assertTrue(hasattr(sandbox_executor, 'kv_memory_get'))
        self.assertTrue(hasattr(sandbox_executor, 'kv_memory_set'))
        self.assertTrue(hasattr(sandbox_executor, 'kv_memory_delete'))
        self.assertTrue(hasattr(sandbox_executor, 'kv_memory_list'))
    
    def test_workflow_functions_added(self):
        """Test that new workflow functions are added without breaking existing ones"""
        # Check new workflow functions exist
        self.assertTrue(hasattr(sandbox_executor, 'run_workflow_async'))
        self.assertTrue(hasattr(sandbox_executor, 'run_workflow'))
        self.assertTrue(hasattr(sandbox_executor, 'load_and_run_workflow'))
        
        # Check they're callable
        self.assertTrue(callable(sandbox_executor.run_workflow_async))
        self.assertTrue(callable(sandbox_executor.run_workflow))
        self.assertTrue(callable(sandbox_executor.load_and_run_workflow))


class TestExistingKVMemoryFunctionality(unittest.TestCase):
    """Test that KV memory functionality is preserved"""
    
    def setUp(self):
        # Clear KV store for testing
        sandbox_executor.kv_store.store.clear()
    
    def test_kv_memory_operations(self):
        """Test that KV memory operations work as before"""
        agent_id = "test_agent"
        run_id = "test_run"
        
        # Test set
        result = sandbox_executor.kv_memory_set(agent_id, run_id, "key1", "value1")
        self.assertEqual(result, "OK")
        
        # Test get
        result = sandbox_executor.kv_memory_get(agent_id, run_id, "key1")
        self.assertEqual(result, "value1")
        
        # Test list
        result = sandbox_executor.kv_memory_list(agent_id, run_id)
        self.assertEqual(result, ["key1"])
        
        # Test delete
        result = sandbox_executor.kv_memory_delete(agent_id, run_id, "key1")
        self.assertEqual(result, "OK")
        
        # Test list after delete
        result = sandbox_executor.kv_memory_list(agent_id, run_id)
        self.assertEqual(result, [])
    
    def test_kv_memory_namespacing(self):
        """Test that KV memory namespacing works correctly"""
        # Different namespaces should be isolated
        sandbox_executor.kv_memory_set("agent1", "run1", "key", "value1")
        sandbox_executor.kv_memory_set("agent2", "run1", "key", "value2")
        sandbox_executor.kv_memory_set("agent1", "run2", "key", "value3")
        
        # Check isolation
        self.assertEqual(
            sandbox_executor.kv_memory_get("agent1", "run1", "key"), 
            "value1"
        )
        self.assertEqual(
            sandbox_executor.kv_memory_get("agent2", "run1", "key"), 
            "value2"
        )
        self.assertEqual(
            sandbox_executor.kv_memory_get("agent1", "run2", "key"), 
            "value3"
        )


class TestExistingFileStructure(unittest.TestCase):
    """Test that existing file structure expectations are maintained"""
    
    def test_agents_directory_structure(self):
        """Test that agents directory structure is compatible"""
        # Check if agents directory exists
        agents_dir = Path("agents")
        if agents_dir.exists():
            # Check for existing agent spec files
            for agent_dir in agents_dir.iterdir():
                if agent_dir.is_dir():
                    spec_file = agent_dir / "spec.json"
                    if spec_file.exists():
                        # Verify existing spec can be loaded
                        try:
                            with open(spec_file) as f:
                                spec_data = json.load(f)
                            agent_spec = AgentSpec(**spec_data)
                            self.assertIsInstance(agent_spec, AgentSpec)
                        except Exception as e:
                            self.fail(f"Failed to load existing agent spec {spec_file}: {e}")
    
    def test_runs_directory_structure(self):
        """Test that runs directory structure is compatible"""
        # Check if runs directory exists
        runs_dir = Path("runs")
        if runs_dir.exists():
            # Verify structure is maintained
            self.assertTrue(runs_dir.is_dir())
    
    def test_output_directory_structure(self):
        """Test that output directory structure is compatible"""
        # Check if output directory exists  
        output_dir = Path("output")
        if output_dir.exists():
            self.assertTrue(output_dir.is_dir())


class TestWorkflowIntegrationNonBreaking(unittest.TestCase):
    """Test that workflow integration doesn't break existing functionality"""
    
    def test_models_import(self):
        """Test that models can be imported without breaking changes"""
        try:
            from models import AgentSpec, InputField, ToolRef, RunLimits, SDKConfig
            from models import WorkflowSpec, WorkflowNodeSpec  # New models should also import
        except ImportError as e:
            self.fail(f"Failed to import models: {e}")
    
    def test_existing_agent_execution_path(self):
        """Test that existing agent execution path is not broken"""
        # Create a simple agent
        agent = AgentSpec(
            name="Compatibility Test Agent",
            description="Test existing execution path",
            system_instructions="Echo the input",
            task_prompt="Repeat what the user says"
        )
        
        # Verify agent can be created and has expected attributes
        self.assertEqual(agent.name, "Compatibility Test Agent")
        self.assertIsNotNone(agent.id)
        self.assertEqual(agent.version, "v0")  # Should maintain existing version
    
    def test_no_required_changes_for_existing_users(self):
        """Test that existing users don't need to change their code"""
        # Existing pattern should still work
        agent_spec = AgentSpec(
            name="User Agent",
            description="User's existing agent",
            system_instructions="Help the user",
            task_prompt="Be helpful"
        )
        
        # Should be able to serialize/deserialize as before
        data = agent_spec.model_dump()
        recreated = AgentSpec(**data)
        
        self.assertEqual(agent_spec.name, recreated.name)
        self.assertEqual(agent_spec.description, recreated.description)


if __name__ == '__main__':
    # Run backward compatibility tests
    print("Running backward compatibility tests...")
    unittest.main(verbosity=2)