"""
Example script demonstrating the Batch & HITL architecture.

This shows how to:
1. Run batch processing for API efficiency
2. Use HITL tiers for different intervention levels
3. Compare results across methods
"""

import json
import os
from dotenv import load_dotenv

from src.datasets import load_asia_generated
from src.run_experiment import (
    run_standard_plus_two_interventions,
    run_batch_baseline_plus_two_interventions,
    run_hitl_level_0_pure_llm,
    run_hitl_level_2_uncertainty_triggered,
)

load_dotenv()


def compare_api_efficiency():
    """
    Compare API usage between sequential and batch methods.
    """
    print("=" * 60)
    print("COMPARING API EFFICIENCY")
    print("=" * 60)
    
    dataset = load_asia_generated(n_samples=2000)
    
    print("\n[1/2] Running SEQUENTIAL method (one call per edge)...")
    result_sequential = run_standard_plus_two_interventions(dataset)
    
    print("\n[2/2] Running BATCH method (batch processing)...")
    result_batch = run_batch_baseline_plus_two_interventions(dataset)
    
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    
    print(f"\nSEQUENTIAL Method:")
    print(f"  F1 Score:    {result_sequential['f1']:.3f}")
    print(f"  SHD:         {result_sequential['shd']}")
    print(f"  API Calls:   {result_sequential['llm_calls']}")
    print(f"  Cache Hits:  {result_sequential['cache_hits']}")
    
    print(f"\nBATCH Method:")
    print(f"  F1 Score:    {result_batch['f1']:.3f}")
    print(f"  SHD:         {result_batch['shd']}")
    print(f"  API Calls:   {result_batch['llm_calls']}")
    print(f"  Cache Hits:  {result_batch['cache_hits']}")
    
    if result_sequential['llm_calls'] > 0 and result_batch['llm_calls'] > 0:
        reduction = result_sequential['llm_calls'] / result_batch['llm_calls']
        print(f"\n✓ API Call Reduction: {reduction:.1f}x")
        print(f"  (Sequential: {result_sequential['llm_calls']} → Batch: {result_batch['llm_calls']})")


def test_hitl_tiers():
    """
    Test different HITL intervention levels.
    """
    print("\n" + "=" * 60)
    print("TESTING HITL TIERS")
    print("=" * 60)
    
    dataset = load_asia_generated(n_samples=2000)
    
    print("\n[1/3] HITL Level 0: Pure LLM (no human intervention)...")
    result_l0 = run_hitl_level_0_pure_llm(dataset)
    
    print("\n[2/3] HITL Level 2: Uncertainty threshold 0.8 (high certainty)...")
    result_l2_high = run_hitl_level_2_uncertainty_triggered(
        dataset,
        confidence_threshold=0.8,
        enable_human_input=False
    )
    
    print("\n[3/3] HITL Level 2: Uncertainty threshold 0.6 (medium certainty)...")
    result_l2_med = run_hitl_level_2_uncertainty_triggered(
        dataset,
        confidence_threshold=0.6,
        enable_human_input=False
    )
    
    print("\n" + "=" * 60)
    print("HITL TIER COMPARISON")
    print("=" * 60)
    
    results = [
        ("Level 0 (Pure LLM)", result_l0),
        ("Level 2 (threshold=0.8)", result_l2_high),
        ("Level 2 (threshold=0.6)", result_l2_med),
    ]
    
    for name, result in results:
        print(f"\n{name}:")
        print(f"  F1 Score:            {result['f1']:.3f}")
        print(f"  SHD:                 {result['shd']}")
        print(f"  API Calls:           {result['llm_calls']}")
        print(f"  Human Interventions: {result.get('human_interventions', 0)}")
        
        # Calculate edges that would need review (if threshold was used)
        # This is approximated from the method name
        if 'threshold' in result['method']:
            print(f"  Method:              {result['method']}")


def demonstrate_confidence_tracking():
    """
    Show how confidence scores are tracked.
    """
    print("\n" + "=" * 60)
    print("CONFIDENCE TRACKING DEMO")
    print("=" * 60)
    
    from src.datasets import load_toy
    from src.llm_interface import GeminiLLM
    
    dataset = load_toy()
    llm = GeminiLLM(model="gemini-2.5-flash")
    
    # Example: Batch verify some edges
    edges = [
        ("Smoking", "Tar"),
        ("Tar", "Cancer"),
        ("Pollution", "Cancer"),
        ("Smoking", "Cancer"),  # Potentially indirect
    ]
    
    print(f"\nVerifying {len(edges)} edges with confidence tracking...")
    verdicts = llm.batch_verify_edges_direct(
        edges=edges,
        nodes=dataset["nodes"],
        descriptions=dataset.get("descriptions"),
    )
    
    print("\nResults:")
    for (src, dst), verdict in verdicts.items():
        keep = verdict['keep']
        conf = verdict['confidence']
        reason = verdict['reason']
        
        status = "✓ KEEP" if keep else "✗ REMOVE"
        conf_level = "HIGH" if conf > 0.8 else "MEDIUM" if conf > 0.6 else "LOW"
        
        print(f"\n  {src} → {dst}: {status}")
        print(f"    Confidence: {conf:.2f} ({conf_level})")
        print(f"    Reason: {reason}")
        
        if verdict.get('mediators'):
            print(f"    Mediators: {verdict['mediators']}")


def main():
    """
    Run all examples.
    """
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: Please set GEMINI_API_KEY environment variable")
        return
    
    print("\n" + "=" * 60)
    print("BATCH & HITL ARCHITECTURE EXAMPLES")
    print("=" * 60)
    
    # Example 1: API efficiency comparison
    # Uncomment to run (requires API calls):
    # compare_api_efficiency()
    
    # Example 2: HITL tiers
    # Uncomment to run (requires API calls):
    # test_hitl_tiers()
    
    # Example 3: Confidence tracking (cheaper, uses toy dataset)
    demonstrate_confidence_tracking()
    
    print("\n" + "=" * 60)
    print("EXAMPLES COMPLETE")
    print("=" * 60)
    print("\nTo run the full examples, uncomment the desired functions in main()")


if __name__ == "__main__":
    main()
