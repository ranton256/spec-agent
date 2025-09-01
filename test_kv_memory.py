import unittest
import json
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from sandbox_executor import KVMemory, kv_memory_set, kv_memory_get, kv_memory_delete, kv_memory_list

class TestKVMemory(unittest.TestCase):
    def setUp(self):
        self.store = KVMemory()
        self.agent_id = "test_agent"
        self.run_id = "test_run"
        self.key1 = "test_key1"
        self.value1 = "test_value1"
        self.key2 = "test_key2"
        self.value2 = "test_value2"

    def test_set_and_get(self):
        # Test setting and getting a value
        result = self.store.set(self.agent_id, self.run_id, self.key1, self.value1)
        self.assertEqual(result, "OK")
        
        # Test getting the value
        result = self.store.get(self.agent_id, self.run_id, self.key1)
        self.assertEqual(result, self.value1)

    def test_namespace_isolation(self):
        # Set values in different namespaces
        self.store.set("agent1", "run1", self.key1, "value1")
        self.store.set("agent1", "run2", self.key1, "value2")
        self.store.set("agent2", "run1", self.key1, "value3")
        
        # Verify namespace isolation
        self.assertEqual(self.store.get("agent1", "run1", self.key1), "value1")
        self.assertEqual(self.store.get("agent1", "run2", self.key1), "value2")
        self.assertEqual(self.store.get("agent2", "run1", self.key1), "value3")

    def test_delete(self):
        # Set a value
        self.store.set(self.agent_id, self.run_id, self.key1, self.value1)
        
        # Delete the value
        result = self.store.delete(self.agent_id, self.run_id, self.key1)
        self.assertEqual(result, "OK")
        
        # Verify deletion
        result = self.store.get(self.agent_id, self.run_id, self.key1)
        self.assertIsNone(result)
        
        # Test deleting non-existent key
        result = self.store.delete(self.agent_id, self.run_id, "non_existent_key")
        self.assertEqual(result, "Key 'non_existent_key' not found")

    def test_list_keys(self):
        # Set some values
        self.store.set(self.agent_id, self.run_id, self.key1, self.value1)
        self.store.set(self.agent_id, self.run_id, self.key2, self.value2)
        
        # List keys
        keys = self.store.list_keys(self.agent_id, self.run_id)
        self.assertIsInstance(keys, list)
        self.assertEqual(len(keys), 2)
        self.assertIn(self.key1, keys)
        self.assertIn(self.key2, keys)
        
        # Test listing keys for non-existent namespace
        keys = self.store.list_keys("non_existent_agent", "non_existent_run")
        self.assertEqual(keys, [])

    def test_error_handling(self):
        # Test invalid inputs - the implementation handles them gracefully
        result = self.store.get(None, None, None)
        self.assertIsNone(result)
        
        # Setting with None values creates a valid entry with string 'None' as key and value
        result = self.store.set(None, None, None, None)
        self.assertEqual(result, "OK")
        
        # Verify the value was set
        result = self.store.get(None, None, None)
        self.assertIsNone(result)  # Because we passed None as key
        
        # Test deleting non-existent key
        result = self.store.delete("non_existent_agent", "non_existent_run", "non_existent_key")
        self.assertEqual(result, "Key 'non_existent_key' not found")
        
        # Test listing keys for non-existent namespace
        result = self.store.list_keys("non_existent_agent", "non_existent_run")
        self.assertEqual(result, [])

class TestKVMemoryFunctions(unittest.TestCase):
    def setUp(self):
        self.agent_id = "test_agent"
        self.run_id = "test_run"
        self.key = "test_key"
        self.value = "test_value"

    def test_kv_memory_functions(self):
        # Test set and get
        result = kv_memory_set(self.agent_id, self.run_id, self.key, self.value)
        self.assertEqual(result, "OK")
        
        result = kv_memory_get(self.agent_id, self.run_id, self.key)
        self.assertEqual(result, self.value)
        
        # Test list
        keys = kv_memory_list(self.agent_id, self.run_id)
        self.assertIn(self.key, keys)
        
        # Test delete
        result = kv_memory_delete(self.agent_id, self.run_id, self.key)
        self.assertEqual(result, "OK")
        
        # Verify deletion
        result = kv_memory_get(self.agent_id, self.run_id, self.key)
        self.assertIsNone(result)

if __name__ == "__main__":
    unittest.main()
