from sandbox_executor import KVMemory

# Create a KV memory store
kv_store = KVMemory()

# Test data
agent_id = "test_agent"
run_id = "test_run_123"
key = "test_key"
value = "test_value"

# Test set
try:
    print(f"Setting key '{key}' to '{value}'...")
    result = kv_store.set(agent_id, run_id, key, value)
    print(f"Set result: {result}")
except Exception as e:
    print(f"Error in set: {e}")

# Test get
try:
    print(f"\nGetting key '{key}'...")
    result = kv_store.get(agent_id, run_id, key)
    print(f"Get result: {result}")
except Exception as e:
    print(f"Error in get: {e}")

# Test list keys
try:
    print(f"\nListing all keys...")
    result = kv_store.list_keys(agent_id, run_id)
    print(f"List result: {result}")
except Exception as e:
    print(f"Error in list: {e}")

# Test delete
try:
    print(f"\nDeleting key '{key}'...")
    result = kv_store.delete(agent_id, run_id, key)
    print(f"Delete result: {result}")
except Exception as e:
    print(f"Error in delete: {e}")

# Verify delete
try:
    print(f"\nVerifying delete...")
    result = kv_store.get(agent_id, run_id, key)
    print(f"Get after delete: {result}")
except Exception as e:
    print(f"Error in get after delete: {e}")
