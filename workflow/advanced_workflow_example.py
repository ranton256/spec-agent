import asyncio
from typing import List, Any, Dict
from pydantic import BaseModel
from dataclasses import dataclass
import time

# Advanced Node Types for Parallelism and Complex Patterns

class ParallelNode(WorkflowNode):
    """Node that executes multiple child nodes in parallel"""
    
    def __init__(self, node_id: str, child_nodes: List[str], name=None):
        super().__init__(node_id, name)
        self.child_nodes = child_nodes
    
    async def execute(self, context: WorkflowContext) -> NodeResult:
        """Execute child nodes in parallel"""
        try:
            # This would be implemented in the workflow executor
            # For now, just mark as completed
            return NodeResult(
                node_id=self.node_id,
                status=ExecutionStatus.COMPLETED,
                data={"parallel_nodes": self.child_nodes}
            )
        except Exception as e:
            return NodeResult(
                node_id=self.node_id,
                status=ExecutionStatus.FAILED,
                error=str(e)
            )

class MergeNode(WorkflowNode):
    """Node that waits for multiple inputs and merges them"""
    
    def __init__(self, node_id: str, input_nodes: List[str], merge_fn=None, name=None):
        super().__init__(node_id, name)
        self.input_nodes = input_nodes
        self.merge_fn = merge_fn or self.default_merge
        self.received_inputs = {}
    
    def default_merge(self, inputs: Dict[str, Any]) -> Any:
        """Default merge strategy - combine all inputs into a list"""
        return list(inputs.values())
    
    async def execute(self, context: WorkflowContext) -> NodeResult:
        """Check if all inputs are ready and merge them"""
        try:
            # Check if all required inputs are available
            for input_node in self.input_nodes:
                output_key = f"{input_node}_output"
                if output_key in context.data:
                    self.received_inputs[input_node] = context.data[output_key]
            
            if len(self.received_inputs) == len(self.input_nodes):
                # All inputs received, merge them
                merged_result = self.merge_fn(self.received_inputs)
                context.data[f"{self.node_id}_output"] = merged_result
                
                return NodeResult(
                    node_id=self.node_id,
                    status=ExecutionStatus.COMPLETED,
                    data=merged_result
                )
            else:
                # Not all inputs ready yet
                return NodeResult(
                    node_id=self.node_id,
                    status=ExecutionStatus.PENDING,
                    data=None
                )
        except Exception as e:
            return NodeResult(
                node_id=self.node_id,
                status=ExecutionStatus.FAILED,
                error=str(e)
            )

# Enhanced Workflow Executor with Parallel Support
class AdvancedWorkflowExecutor:
    """Advanced executor with parallel execution support"""
    
    def __init__(self, workflow: WorkflowGraph):
        self.workflow = workflow
        self.execution_state = {}
        self.pending_nodes = set()
        self.completed_nodes = set()
        
    async def execute_parallel_batch(self, node_ids: List[str], context: WorkflowContext) -> Dict[str, NodeResult]:
        """Execute multiple nodes in parallel"""
        tasks = []
        for node_id in node_ids:
            node = self.workflow.nodes[node_id]
            tasks.append(self._execute_node_with_timing(node, context))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        result_dict = {}
        for i, result in enumerate(results):
            node_id = node_ids[i]
            if isinstance(result, Exception):
                result_dict[node_id] = NodeResult(
                    node_id=node_id,
                    status=ExecutionStatus.FAILED,
                    error=str(result)
                )
            else:
                result_dict[node_id] = result
        
        return result_dict
    
    async def _execute_node_with_timing(self, node: WorkflowNode, context: WorkflowContext) -> NodeResult:
        """Execute a node and track timing"""
        start_time = time.time()
        result = await node.execute(context)
        end_time = time.time()
        result.execution_time = end_time - start_time
        return result
    
    def find_ready_nodes(self, context: WorkflowContext) -> List[str]:
        """Find nodes that are ready to execute"""
        ready_nodes = []
        
        for node_id, node in self.workflow.nodes.items():
            if (node_id not in self.completed_nodes and 
                node_id not in self.pending_nodes and
                self._are_dependencies_met(node_id, context)):
                ready_nodes.append(node_id)
        
        return ready_nodes
    
    def _are_dependencies_met(self, node_id: str, context: WorkflowContext) -> bool:
        """Check if all dependencies for a node are met"""
        # Find all nodes that feed into this node
        dependent_nodes = []
        for edge in self.workflow.edges:
            if edge.to_node == node_id:
                dependent_nodes.append(edge.from_node)
        
        # Check if all dependencies are completed
        for dep_node in dependent_nodes:
            if dep_node not in self.completed_nodes:
                return False
        
        return True
    
    async def execute(self, initial_context: Optional[WorkflowContext] = None) -> WorkflowContext:
        """Execute workflow with parallel support"""
        context = initial_context or WorkflowContext()
        results = {}
        
        # Start with start nodes
        ready_nodes = list(self.workflow.start_nodes) if self.workflow.start_nodes else [list(self.workflow.nodes.keys())[0]]
        
        while ready_nodes or self.pending_nodes:
            if ready_nodes:
                # Execute ready nodes in parallel
                batch_results = await self.execute_parallel_batch(ready_nodes, context)
                results.update(batch_results)
                
                # Process results
                for node_id, result in batch_results.items():
                    if result.status == ExecutionStatus.COMPLETED:
                        self.completed_nodes.add(node_id)
                        self.workflow.apply_data_mappings(context, node_id)
                    elif result.status == ExecutionStatus.PENDING:
                        self.pending_nodes.add(node_id)
                    else:  # FAILED
                        # Handle failure - could implement retry logic here
                        self.completed_nodes.add(node_id)  # Mark as completed to avoid infinite loop
            
            # Find next ready nodes
            ready_nodes = self.find_ready_nodes(context)
            
            # Remove nodes that are no longer pending
            self.pending_nodes = {node for node in self.pending_nodes if node not in self.completed_nodes}
        
        context.data["_execution_results"] = results
        return context

# Schema Examples for Typed Data Flow
class UserInputSchema(BaseModel):
    user_id: str
    query: str
    preferences: Dict[str, Any] = {}

class ProcessedDataSchema(BaseModel):
    processed_query: str
    intent: str
    entities: List[str]
    confidence: float

class AnalysisResultSchema(BaseModel):
    sentiment: str
    categories: List[str]
    priority: int
    recommendations: List[str]

class FinalResponseSchema(BaseModel):
    response_text: str
    metadata: Dict[str, Any]
    processing_time: float

# Complex Workflow Example
async def create_complex_workflow():
    """Create a complex workflow with branching, loops, and parallelism"""
    
    # Mock agents with different purposes
    class InputProcessor:
        async def run_async(self, data):
            class Result:
                data = ProcessedDataSchema(
                    processed_query=f"Processed: {data.query}",
                    intent="information_request",
                    entities=["user", "query"],
                    confidence=0.95
                )
            return Result()
    
    class SentimentAnalyzer:
        async def run_async(self, data):
            class Result:
                data = {"sentiment": "positive", "confidence": 0.87}
            return Result()
    
    class CategoryClassifier:
        async def run_async(self, data):
            class Result:
                data = {"categories": ["general", "support"], "confidence": 0.92}
            return Result()
    
    class ResponseGenerator:
        async def run_async(self, data):
            class Result:
                data = FinalResponseSchema(
                    response_text="Here's your response based on the analysis",
                    metadata={"generated_at": "2024-01-01T12:00:00Z"},
                    processing_time=0.5
                )
            return Result()
    
    # Create workflow with parallel analysis
    workflow = (WorkflowBuilder("Complex AI Pipeline")
                
                # Input processing
                .add_agent("input_processor", InputProcessor(), 
                          input_schema=UserInputSchema,
                          output_schema=ProcessedDataSchema,
                          is_start=True)
                
                # Parallel analysis branches
                .add_agent("sentiment_analyzer", SentimentAnalyzer())
                .add_agent("category_classifier", CategoryClassifier())
                
                # Quality check condition
                .add_condition("quality_check", 
                             lambda ctx: ctx.data.get("input_processor_output", {}).confidence > 0.8)
                
                # Merge results
                .add_node(MergeNode("analysis_merger", 
                                  ["sentiment_analyzer", "category_classifier"],
                                  merge_fn=lambda inputs: {**inputs["sentiment_analyzer"], **inputs["category_classifier"]}))
                
                # Final response generation
                .add_agent("response_generator", ResponseGenerator(),
                          input_schema=Dict,
                          output_schema=FinalResponseSchema,
                          is_end=True)
                
                # Connect the workflow
                .connect("input_processor", "quality_check")
                .connect("quality_check", "sentiment_analyzer", condition="quality_check_result")
                .connect("quality_check", "category_classifier", condition="quality_check_result")
                .connect("sentiment_analyzer", "analysis_merger")
                .connect("category_classifier", "analysis_merger")
                .connect("analysis_merger", "response_generator")
                
                # Data mappings
                .map_data("input_processor", "processed_query", "sentiment_analyzer", "text")
                .map_data("input_processor", "processed_query", "category_classifier", "text")
                .map_data("analysis_merger", "merged_data", "response_generator", "analysis_results")
                
                .build())
    
    return workflow

# Loop Example - Iterative Refinement
async def create_iterative_workflow():
    """Create a workflow with loops for iterative refinement"""
    
    class InitialProcessor:
        async def run_async(self, data):
            class Result:
                data = {"text": data, "quality_score": 0.6, "iteration": 0}
            return Result()
    
    class Refiner:
        async def run_async(self, data):
            class Result:
                current_score = data.get("quality_score", 0.6)
                new_score = min(current_score + 0.2, 1.0)  # Improve quality each iteration
                data = {
                    "text": f"Refined: {data.get('text', '')}",
                    "quality_score": new_score,
                    "iteration": data.get("iteration", 0) + 1
                }
            return Result()
    
    workflow = (WorkflowBuilder("Iterative Refinement Pipeline")
                .add_agent("initial_processor", InitialProcessor(), is_start=True)
                .add_loop("quality_loop", 
                         lambda ctx: ctx.data.get("refiner_output", {}).get("quality_score", 0) < 0.9,
                         max_iterations=5)
                .add_agent("refiner", Refiner())
                .add_condition("final_check", 
                             lambda ctx: ctx.data.get("refiner_output", {}).get("quality_score", 0) >= 0.9,
                             is_end=True)
                
                .connect("initial_processor", "quality_loop")
                .connect("quality_loop", "refiner", condition="quality_loop_continue")
                .connect("refiner", "quality_loop")
                .connect("quality_loop", "final_check", condition="not quality_loop_continue")
                
                .build())
    
    return workflow

# Utility Functions
class WorkflowVisualizer:
    """Utility to visualize workflow structure"""
    
    @staticmethod
    def print_workflow_structure(workflow: WorkflowGraph):
        """Print a text representation of the workflow"""
        print(f"Workflow: {workflow.name}")
        print("=" * 50)
        
        print(f"Nodes ({len(workflow.nodes)}):")
        for node_id, node in workflow.nodes.items():
            node_type = type(node).__name__
            print(f"  - {node_id} ({node_type})")
        
        print(f"\nEdges ({len(workflow.edges)}):")
        for edge in workflow.edges:
            condition_str = f" [condition: {edge.condition}]" if edge.condition else ""
            print(f"  - {edge.from_node} → {edge.to_node}{condition_str}")
        
        print(f"\nStart nodes: {workflow.start_nodes}")
        print(f"End nodes: {workflow.end_nodes}")

# Example Usage
async def main():
    """Run examples"""
    print("Creating complex workflow...")
    complex_workflow = await create_complex_workflow()
    WorkflowVisualizer.print_workflow_structure(complex_workflow)
    
    print("\n" + "="*50 + "\n")
    
    print("Creating iterative workflow...")
    iterative_workflow = await create_iterative_workflow()
    WorkflowVisualizer.print_workflow_structure(iterative_workflow)
    
    # Execute complex workflow
    print("\n" + "="*50 + "\n")
    print("Executing complex workflow...")
    
    initial_context = WorkflowContext()
    initial_context.data["input_processor_input"] = UserInputSchema(
        user_id="user123",
        query="Help me understand machine learning",
        preferences={"detail_level": "intermediate"}
    )
    
    executor = AdvancedWorkflowExecutor(complex_workflow)
    result = await executor.execute(initial_context)
    
    print("Execution completed!")
    print(f"Results: {list(result.data.keys())}")

if __name__ == "__main__":
    asyncio.run(main())