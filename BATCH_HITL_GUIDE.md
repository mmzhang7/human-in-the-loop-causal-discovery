# Batch Verification & Human-in-the-Loop (HITL) Architecture

## Overview

This project now implements a **Batch & Verify** architecture that reduces API quota usage by **5-10x** while setting up checkpoints for Human-in-the-Loop (HITL) research.

## Key Improvements

### 1. Batch API Calls (5-10x Cost Reduction)

Instead of making one API call per edge, the system now processes multiple edges in a single batch request.

#### Before (One Call Per Edge):
```python
# For 50 edges, this makes 50 API calls
for src, dst in edges:
    verdict = llm.verify_edge_direct(src, dst, nodes)
```

#### After (Batch Processing):
```python
# For 50 edges, this makes 1 API call
verdicts = llm.batch_verify_edges_direct(edges, nodes)
```

### 2. Confidence Scoring

All LLM responses now include a `confidence` field (0.0-1.0) that enables:
- Identifying uncertain predictions for human review
- Filtering decisions by certainty threshold
- Simulating HITL scenarios without actual human input

### 3. Transitive Reduction Support

The `DiGraph` class now includes methods to detect potentially redundant edges:
```python
graph.has_path(u, v)              # Check if path exists
graph.has_indirect_path(u, v)     # Check if indirect path exists (u->...->v)
graph.get_transitive_edges()      # Find potentially redundant edges
```

## New Batch Methods

### In `llm_interface.py`:

#### `batch_get_children(parent_node, candidate_nodes)`
Query multiple children in one call instead of sequential queries.

#### `batch_verify_edges_direct(edges, nodes)`
Verify directness of multiple edges simultaneously.
```python
verdicts = llm.batch_verify_edges_direct(
    edges=[(A, B), (B, C), (C, D)],
    nodes=all_nodes
)
# Returns: {(A,B): {keep: True, confidence: 0.9, ...}, ...}
```

#### `batch_verify_edges_direction(edges, nodes)`
Check direction correctness for multiple edges at once.
```python
verdicts = llm.batch_verify_edges_direction(
    edges=[(A, B), (B, C)],
    nodes=all_nodes
)
# Returns: {(A,B): {action: "flip", confidence: 0.85, ...}, ...}
```

### In `intervention.py`:

#### `batch_prune_indirect_edges(edges, nodes, llm, confidence_threshold=0.0)`
Batch version of pruning with optional confidence filtering.

#### `batch_correct_edge_directions(edges, nodes, llm, confidence_threshold=0.0)`
Batch version of direction correction with confidence tracking.

## HITL Tiers

### Level 0: Pure LLM Baseline
```python
result = run_hitl_level_0_pure_llm(dataset)
```
- No human intervention
- Uses batch processing for efficiency
- Fully autonomous operation

### Level 1: Human-Guided Rooting
```python
# Interactive mode - prompts user for root selection
result = run_hitl_level_1_human_rooting(dataset)

# Pre-specified roots (for controlled experiments)
result = run_hitl_level_1_human_rooting(
    dataset, 
    human_selected_roots=["VisitAsia", "Smoking"]
)
```
- LLM proposes 5 root nodes
- Human selects the correct 2-3
- **Hypothesis:** Correcting the source prevents error propagation

### Level 2: Uncertainty-Triggered HITL
```python
# Simulation mode - flags uncertain edges but auto-decides
result = run_hitl_level_2_uncertainty_triggered(
    dataset,
    confidence_threshold=0.7,
    enable_human_input=False
)

# Interactive mode - prompts human for uncertain edges
result = run_hitl_level_2_uncertainty_triggered(
    dataset,
    confidence_threshold=0.7,
    enable_human_input=True
)
```
- Edges with `confidence < threshold` trigger intervention
- **Research Value:** Quantify accuracy gain per human decision

## Metrics Tracking

All results now include:
```json
{
  "llm_calls": 15,           // Total API calls made
  "cache_hits": 8,           // Cached responses reused
  "human_interventions": 3,  // Human decisions made
  "total_api_calls": 15,     // Same as llm_calls (for clarity)
  "f1": 0.85,
  "shd": 2
}
```

This enables calculating:
- **Accuracy per API dollar:** `f1 / llm_calls`
- **Accuracy per human hour:** `f1 / human_interventions`
- **Human efficiency:** `(f1_with_human - f1_without) / human_interventions`

## Running Experiments

Edit `src/run_experiment.py` and uncomment the desired method:

### Original Methods (One Call Per Edge):
```python
result = run_standard_baseline(dataset)
result = run_standard_plus_intervention(dataset)
result = run_standard_plus_two_interventions(dataset)
result = run_standard_plus_three_interventions(dataset)
```

### Batch Methods (5-10x Cost Reduction):
```python
result = run_batch_baseline_plus_two_interventions(dataset)
result = run_batch_baseline_plus_three_interventions(dataset)
```

### HITL Methods:
```python
# Level 0 (baseline)
result = run_hitl_level_0_pure_llm(dataset)

# Level 1 (human rooting)
result = run_hitl_level_1_human_rooting(dataset)

# Level 2 (uncertainty-triggered)
result = run_hitl_level_2_uncertainty_triggered(
    dataset,
    confidence_threshold=0.7,
    enable_human_input=True
)
```

## Expected API Savings

### Example: Asia Network with 8 nodes, 8 true edges

**Original Approach:**
- PC algorithm produces ~15 candidate edges
- Pruning: 15 API calls (one per edge)
- Direction: 12 API calls (after pruning)
- **Total: ~27 API calls**

**Batch Approach:**
- Pruning: 1 API call (batch of 15 edges)
- Direction: 1 API call (batch of 12 edges)
- **Total: ~2 API calls**

**Savings: ~13.5x reduction**

## Research Workflow

1. **Establish Baseline:**
   ```python
   result_pure = run_hitl_level_0_pure_llm(dataset)
   ```

2. **Test HITL Level 1:**
   ```python
   result_l1 = run_hitl_level_1_human_rooting(
       dataset,
       human_selected_roots=true_roots  # Ground truth
   )
   ```

3. **Test HITL Level 2:**
   ```python
   for threshold in [0.5, 0.6, 0.7, 0.8]:
       result = run_hitl_level_2_uncertainty_triggered(
           dataset,
           confidence_threshold=threshold,
           enable_human_input=False  # Simulation
       )
       # Analyze: accuracy vs. number of uncertain edges
   ```

4. **Compare Results:**
   ```python
   metrics = {
       "Pure LLM": {"f1": result_pure["f1"], "interventions": 0},
       "L1 Rooting": {"f1": result_l1["f1"], "interventions": 1},
       "L2 Uncertain": {"f1": result_l2["f1"], "interventions": result_l2["human_interventions"]},
   }
   ```

## Graph Utility Functions

### Check for Transitive Edges Before Querying:
```python
from src.graph import DiGraph

graph = DiGraph(nodes)
for edge in candidate_edges:
    graph.add_edge(*edge)

# Find potentially redundant edges
transitive = graph.get_transitive_edges()
print(f"High-priority edges to verify: {transitive}")
```

### Path Existence Check:
```python
if graph.has_path("Smoking", "Cancer"):
    print("Path exists (might be direct or indirect)")

if graph.has_indirect_path("Smoking", "Cancer"):
    print("Indirect path exists - direct edge might be redundant")
```

## Next Steps for Your Research

1. **Compare Batch vs. Sequential:** Run the same dataset with both approaches and compare API usage.

2. **Ablation Study:** Test each HITL level systematically:
   - Level 0 (baseline)
   - Level 1 (human rooting)
   - Level 2 with varying confidence thresholds

3. **Simulate Human Decisions:** Use ground truth to simulate perfect human intervention and measure upper bounds.

4. **Cost-Accuracy Tradeoff:** Plot F1 score vs. (API calls + human_interventions) to find optimal balance.

5. **Error Analysis:** Examine which edges are flagged as uncertain and whether they are actually incorrect.

## Implementation Notes

- **Caching:** All LLM calls are cached on disk (`.cache_gemini/`) to avoid repeated charges
- **Retry Logic:** Built-in retry with exponential backoff for rate limits
- **JSON Parsing:** Robust extraction handles markdown fences and partial responses
- **Type Safety:** All JSON responses are validated and cleaned

## Configuration

Set your API key:
```bash
export GEMINI_API_KEY="your-key-here"
```

Or create a `.env` file:
```
GEMINI_API_KEY=your-key-here
```

## File Structure

```
src/
  llm_interface.py     # Batch LLM methods + confidence scoring
  intervention.py      # Batch intervention functions
  graph.py             # DiGraph with transitive reduction  
  run_experiment.py    # HITL tiers + batch experiments
  datasets.py          # Dataset loaders
  causal_baseline.py   # PC algorithm baseline
  metrics.py           # F1, SHD evaluation
```

## Citation

If you use this batch verification or HITL architecture in your research, please cite the appropriate methodology papers and acknowledge the efficiency improvements.
