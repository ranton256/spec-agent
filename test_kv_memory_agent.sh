#!/bin/bash

# Test script for KV Memory Example Agent
# Tests all KV memory operations: help, list, save, get, overwrite, delete

set -e  # Exit on error

AGENT_SPEC="agents/kv_memory_example/spec.json"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_OUTPUT_DIR="$SCRIPT_DIR/test_output"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create test output directory
mkdir -p "$TEST_OUTPUT_DIR"

echo -e "${BLUE}=== KV Memory Agent Integration Test ===${NC}"
echo "Agent Spec: $AGENT_SPEC"
echo "Test Output Directory: $TEST_OUTPUT_DIR"
echo ""

# Function to run agent and capture output
run_test() {
    local test_name="$1"
    local input_json="$2"
    local expected_pattern="$3"
    
    echo -e "${YELLOW}Test: $test_name${NC}"
    echo "Input: $input_json"
    
    # Run the agent and capture output
    local output_file="$TEST_OUTPUT_DIR/${test_name}.json"
    local result
    
    # Use shared run ID for all tests to maintain KV memory state
    local run_id="shared_test_run"
    
    if result=$(python sandbox_executor.py run --agent "$AGENT_SPEC" --input "$input_json" --run_id "$run_id" 2>&1); then
        echo "$result" > "$output_file"
        echo "✓ Agent executed successfully"
        
        # Check if expected pattern is found (if provided)
        if [ -n "$expected_pattern" ]; then
            if echo "$result" | grep -q "$expected_pattern"; then
                echo -e "${GREEN}✓ Expected pattern found: $expected_pattern${NC}"
            else
                echo -e "${RED}✗ Expected pattern not found: $expected_pattern${NC}"
                echo "Actual output: $result"
            fi
        fi
        
        # Check if tools were actually called
        if echo "$result" | grep -q '"tool_results": \[\]'; then
            echo -e "${YELLOW}⚠ No tools were called (tool_results is empty)${NC}"
        elif echo "$result" | grep -q '"tool_results":'; then
            echo -e "${GREEN}✓ Tools were called${NC}"
        fi
        
        echo "Output saved to: $output_file"
    else
        echo -e "${RED}✗ Agent execution failed${NC}"
        echo "Error: $result"
        echo "$result" > "$output_file"
    fi
    
    echo ""
}

# Test 1: Help command
run_test "help" \
    '{"command": "help"}' \
    "help"

# Test 2: List (should be empty initially)
run_test "list_empty" \
    '{"command": "list"}' \
    "list"

# Test 3: Save first key-value pair
run_test "save_name" \
    '{"command": "save", "key": "name", "value": "John Doe"}' \
    "save"

# Test 4: Get the saved value
run_test "get_name" \
    '{"command": "get", "key": "name"}' \
    "John Doe"

# Test 5: Save another key-value pair
run_test "save_age" \
    '{"command": "save", "key": "age", "value": "30"}' \
    "save"

# Test 6: List all keys (should show both keys)
run_test "list_with_keys" \
    '{"command": "list"}' \
    "name"

# Test 7: Overwrite existing value
run_test "save_name_overwrite" \
    '{"command": "save", "key": "name", "value": "Jane Smith"}' \
    "save"

# Test 8: Get the overwritten value
run_test "get_name_after_overwrite" \
    '{"command": "get", "key": "name"}' \
    "Jane Smith"

# Test 9: Delete a key
run_test "delete_age" \
    '{"command": "delete", "key": "age"}' \
    "delete"

# Test 10: List after deletion (should only show name)
run_test "list_after_delete" \
    '{"command": "list"}' \
    "name"

# Test 11: Try to get deleted key (should return empty or not found)
run_test "get_deleted_key" \
    '{"command": "get", "key": "age"}' \
    ""

# Test 12: Delete remaining key
run_test "delete_name" \
    '{"command": "delete", "key": "name"}' \
    "delete"

# Test 13: Final list (should be empty)
run_test "list_final_empty" \
    '{"command": "list"}' \
    "list"

echo -e "${BLUE}=== Test Summary ===${NC}"
echo "All tests completed. Check individual output files in $TEST_OUTPUT_DIR for detailed results."
echo ""

# Count successful and failed tests
success_count=0
fail_count=0

for file in "$TEST_OUTPUT_DIR"/*.json; do
    if [ -f "$file" ]; then
        if grep -q "Error\|Failed\|Exception" "$file"; then
            ((fail_count++))
            echo -e "${RED}✗ $(basename "$file" .json)${NC}"
        else
            ((success_count++))
            echo -e "${GREEN}✓ $(basename "$file" .json)${NC}"
        fi
    fi
done

echo ""
echo -e "${GREEN}Successful tests: $success_count${NC}"
echo -e "${RED}Failed tests: $fail_count${NC}"

# Return appropriate exit code
if [ $fail_count -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed. Check output files for details.${NC}"
    exit 1
fi