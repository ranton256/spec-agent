#!/bin/bash

# Phase 1 Workflow Integration Test Runner
# Runs all tests for Phase 1 implementation

set -e

echo "🧪 Running Phase 1 Workflow Integration Tests"
echo "=============================================="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

run_test_suite() {
    local test_file="$1"
    local test_name="$2"
    
    echo -e "${BLUE}Running $test_name...${NC}"
    
    if python -m unittest "$test_file" -v; then
        echo -e "${GREEN}✅ $test_name PASSED${NC}"
        ((PASSED_TESTS++))
    else
        echo -e "${RED}❌ $test_name FAILED${NC}"
        ((FAILED_TESTS++))
    fi
    
    ((TOTAL_TESTS++))
    echo
}

# Ensure we're in the right directory
if [[ ! -f "models.py" ]]; then
    echo "❌ Error: Please run this script from the SpecAgent root directory"
    exit 1
fi

# Check Python environment
echo "🔍 Checking Python environment..."
if ! python -c "import pydantic, pathlib" 2>/dev/null; then
    echo "❌ Error: Required Python packages not found. Please install requirements."
    exit 1
fi
echo "✅ Python environment OK"
echo

# Run backward compatibility tests first (most critical)
run_test_suite "test_backward_compatibility" "Backward Compatibility Tests"

# Run existing functionality tests  
echo -e "${BLUE}Running existing KV memory tests...${NC}"
if python -m unittest test_kv_memory -v; then
    echo -e "${GREEN}✅ Existing KV Memory Tests PASSED${NC}"
    ((PASSED_TESTS++))
else
    echo -e "${RED}❌ Existing KV Memory Tests FAILED${NC}"
    ((FAILED_TESTS++))
fi
((TOTAL_TESTS++))
echo

# Run comprehensive Phase 1 tests
run_test_suite "test_workflow_phase1" "Phase 1 Workflow Integration Tests"

# Test workflow files can be imported
echo -e "${BLUE}Testing workflow module imports...${NC}"
if python -c "
try:
    from workflow.pydantic_workflow import WorkflowGraph, WorkflowContext, AgentNode
    from workflow.config_driven_workflows import ComponentRegistry, WorkflowFactory
    from workflow_adapter import SpecAgentWorkflowAdapter, SpecAgentWorkflowExecutor
    from models import WorkflowSpec, WorkflowNodeSpec, WorkflowEdgeSpec
    print('✅ All workflow modules import successfully')
except Exception as e:
    print(f'❌ Import error: {e}')
    exit(1)
"; then
    echo -e "${GREEN}✅ Workflow Module Imports PASSED${NC}"
    ((PASSED_TESTS++))
else
    echo -e "${RED}❌ Workflow Module Imports FAILED${NC}"  
    ((FAILED_TESTS++))
fi
((TOTAL_TESTS++))
echo

# Test model serialization/deserialization
echo -e "${BLUE}Testing model serialization...${NC}"
if python -c "
import json
from models import AgentSpec, WorkflowSpec, WorkflowNodeSpec, InputField

# Test existing AgentSpec
agent = AgentSpec(
    name='Test Agent',
    description='Test description', 
    system_instructions='Test instructions',
    task_prompt='Test prompt',
    input_schema=[InputField(name='input', type='str')]
)

# Serialize and deserialize
data = agent.model_dump()
recreated = AgentSpec(**data)
assert agent.name == recreated.name

# Test new WorkflowSpec
workflow = WorkflowSpec(
    name='Test Workflow',
    description='Test workflow',
    nodes=[WorkflowNodeSpec(
        id='test_node',
        type='agent',
        is_start=True,
        is_end=True
    )],
    agent_specs=[agent]
)

# Serialize and deserialize
wf_data = workflow.model_dump()
recreated_wf = WorkflowSpec(**wf_data)
assert workflow.name == recreated_wf.name

print('✅ Model serialization/deserialization working')
"; then
    echo -e "${GREEN}✅ Model Serialization Tests PASSED${NC}"
    ((PASSED_TESTS++))
else
    echo -e "${RED}❌ Model Serialization Tests FAILED${NC}"
    ((FAILED_TESTS++))
fi
((TOTAL_TESTS++))
echo

# Summary
echo "=============================================="
echo -e "${BLUE}📊 Test Results Summary${NC}"
echo "=============================================="
echo -e "Total Test Suites: $TOTAL_TESTS"
echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
echo -e "${RED}Failed: $FAILED_TESTS${NC}"

if [[ $FAILED_TESTS -eq 0 ]]; then
    echo
    echo -e "${GREEN}🎉 All Phase 1 tests passed! Integration is ready.${NC}"
    echo
    echo "✅ Backward compatibility maintained"
    echo "✅ New workflow functionality working"
    echo "✅ Model schemas validated"
    echo "✅ Adapter layer functional"
    exit 0
else
    echo
    echo -e "${RED}❌ Some tests failed. Please review and fix before proceeding.${NC}"
    exit 1
fi