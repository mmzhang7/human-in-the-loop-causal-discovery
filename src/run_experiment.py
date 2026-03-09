import json
import os
from datetime import datetime
from typing import Dict, Set, Tuple

from dotenv import load_dotenv

from .baseline_bfs import build_graph_bfs
from .causal_baseline import run_causal_baseline
from .datasets import load_asia_generated, load_asia_placeholder, load_toy
from .intervention import (
    prune_indirect_edges,
    correct_edge_directions,
    suggest_missing_edges,
    batch_prune_indirect_edges,
    batch_correct_edge_directions,
)
from .llm_interface import GeminiLLM, OpenRouterLLM
from .logger import RunLogger
from .metrics import edge_f1, shd_directed

load_dotenv()

Edge = Tuple[str, str]


def summarize_result(
    dataset_name: str,
    method_name: str,
    pred_edges: Set[Edge],
    true_edges: Set[Edge],
    llm_calls: int = 0,
    cache_hits: int = 0,
    human_interventions: int = 0,
) -> Dict:
    pred_edges_set = set(pred_edges)
    true_edges_set = set(true_edges)

    fp_edges = sorted(list(pred_edges_set - true_edges_set))
    fn_edges = sorted(list(true_edges_set - pred_edges_set))

    return {
        "dataset": dataset_name,
        "method": method_name,
        "n_true_edges": len(true_edges_set),
        "n_pred_edges": len(pred_edges_set),
        "pred_edges": sorted(list(pred_edges_set)),
        "true_edges": sorted(list(true_edges_set)),
        "fp_edges": fp_edges,
        "fn_edges": fn_edges,
        "f1": edge_f1(pred_edges_set, true_edges_set),
        "shd": shd_directed(pred_edges_set, true_edges_set),
        "llm_calls": llm_calls,
        "cache_hits": cache_hits,
        "human_interventions": human_interventions,
        "total_api_calls": llm_calls,
    }


def run_bfs_baseline(dataset: Dict) -> Dict:
    llm = OpenRouterLLM(model="deepseek/deepseek-r1:free")
    logger = RunLogger(verbose=True)

    nodes = dataset["nodes"]
    descriptions = dataset.get("descriptions")
    true_edges = set(dataset.get("true_edges", set()))

    graph = build_graph_bfs(
        nodes=nodes,
        llm=llm,
        descriptions=descriptions,
        logger=logger,
    )

    pred_edges = graph.edges()

    return summarize_result(
        dataset_name=dataset["name"],
        method_name="llm_bfs_baseline",
        pred_edges=pred_edges,
        true_edges=true_edges,
        llm_calls=llm.usage.calls,
        cache_hits=llm.usage.cache_hits,
        human_interventions=llm.usage.human_interventions,
    )


def run_standard_baseline(dataset: Dict) -> Dict:
    true_edges = set(dataset.get("true_edges", set()))
    pred_edges = run_causal_baseline(dataset)

    return summarize_result(
        dataset_name=dataset["name"],
        method_name="causal_baseline",
        pred_edges=pred_edges,
        true_edges=true_edges,
        human_interventions=0,
    )


def run_standard_plus_intervention(dataset: Dict) -> Dict:
    llm = GeminiLLM(model="gemini-2.5-flash")

    nodes = dataset["nodes"]
    descriptions = dataset.get("descriptions")
    true_edges = set(dataset.get("true_edges", set()))

    baseline_edges = run_causal_baseline(dataset)
    pruned_edges = prune_indirect_edges(
        edges=baseline_edges,
        nodes=nodes,
        llm=llm,
        descriptions=descriptions,
        verbose=True,
    )

    return summarize_result(
        dataset_name=dataset["name"],
        method_name="causal_baseline_plus_pruning",
        pred_edges=pruned_edges,
        true_edges=true_edges,
        llm_calls=llm.usage.calls,
        cache_hits=llm.usage.cache_hits,
        human_interventions=llm.usage.human_interventions,
    )


def run_standard_plus_two_interventions(dataset: Dict) -> Dict:
    llm = OpenRouterLLM(model="deepseek/deepseek-r1:free")

    nodes = dataset["nodes"]
    descriptions = dataset.get("descriptions")
    true_edges = set(dataset.get("true_edges", set()))

    baseline_edges = run_causal_baseline(dataset)

    pruned_edges = prune_indirect_edges(
        edges=baseline_edges,
        nodes=nodes,
        llm=llm,
        descriptions=descriptions,
        verbose=True,
    )

    corrected_edges = correct_edge_directions(
        edges=pruned_edges,
        nodes=nodes,
        llm=llm,
        descriptions=descriptions,
        verbose=True,
    )

    return summarize_result(
        dataset_name=dataset["name"],
        method_name="causal_baseline_plus_pruning_plus_direction",
        pred_edges=corrected_edges,
        true_edges=true_edges,
        llm_calls=llm.usage.calls,
        cache_hits=llm.usage.cache_hits,
        human_interventions=llm.usage.human_interventions,
    )


def run_standard_plus_three_interventions(dataset: Dict) -> Dict:
    llm = OpenRouterLLM(model="deepseek/deepseek-r1:free")

    nodes = dataset["nodes"]
    descriptions = dataset.get("descriptions")
    true_edges = set(dataset.get("true_edges", set()))

    baseline_edges = run_causal_baseline(dataset)

    pruned_edges = prune_indirect_edges(
        edges=baseline_edges,
        nodes=nodes,
        llm=llm,
        descriptions=descriptions,
        verbose=True,
    )

    corrected_edges = correct_edge_directions(
        edges=pruned_edges,
        nodes=nodes,
        llm=llm,
        descriptions=descriptions,
        verbose=True,
    )

    completed_edges = suggest_missing_edges(
        edges=corrected_edges,
        nodes=nodes,
        llm=llm,
        descriptions=descriptions,
        max_edges=5,
        verbose=True,
    )

    return summarize_result(
        dataset_name=dataset["name"],
        method_name="causal_baseline_plus_pruning_plus_direction_plus_missing",
        pred_edges=completed_edges,
        true_edges=true_edges,
        llm_calls=llm.usage.calls,
        cache_hits=llm.usage.cache_hits,
        human_interventions=llm.usage.human_interventions,
    )


# ========== BATCH PROCESSING METHODS (5-10x API REDUCTION) ==========


def run_batch_baseline_plus_two_interventions(dataset: Dict) -> Dict:
    """
    Uses batch API calls to reduce LLM costs by 5-10x.
    Processes pruning and direction correction in batch mode.
    """
    llm = OpenRouterLLM(model="deepseek/deepseek-r1:free")

    nodes = dataset["nodes"]
    descriptions = dataset.get("descriptions")
    true_edges = set(dataset.get("true_edges", set()))

    baseline_edges = run_causal_baseline(dataset)

    # Batch pruning: Process all edges in one API call
    pruned_edges = batch_prune_indirect_edges(
        edges=baseline_edges,
        nodes=nodes,
        llm=llm,
        descriptions=descriptions,
        verbose=True,
    )

    # Batch direction correction: Process all edges in one API call
    corrected_edges = batch_correct_edge_directions(
        edges=pruned_edges,
        nodes=nodes,
        llm=llm,
        descriptions=descriptions,
        verbose=True,
    )

    return summarize_result(
        dataset_name=dataset["name"],
        method_name="batch_causal_baseline_plus_pruning_plus_direction",
        pred_edges=corrected_edges,
        true_edges=true_edges,
        llm_calls=llm.usage.calls,
        cache_hits=llm.usage.cache_hits,
        human_interventions=llm.usage.human_interventions,
    )


def run_batch_baseline_plus_three_interventions(dataset: Dict) -> Dict:
    """
    Full batch pipeline: baseline + batch pruning + batch direction + missing edges.
    """
    llm = OpenRouterLLM(model="deepseek/deepseek-r1:free")

    nodes = dataset["nodes"]
    descriptions = dataset.get("descriptions")
    true_edges = set(dataset.get("true_edges", set()))

    baseline_edges = run_causal_baseline(dataset)

    pruned_edges = batch_prune_indirect_edges(
        edges=baseline_edges,
        nodes=nodes,
        llm=llm,
        descriptions=descriptions,
        verbose=True,
    )

    corrected_edges = batch_correct_edge_directions(
        edges=pruned_edges,
        nodes=nodes,
        llm=llm,
        descriptions=descriptions,
        verbose=True,
    )

    completed_edges = suggest_missing_edges(
        edges=corrected_edges,
        nodes=nodes,
        llm=llm,
        descriptions=descriptions,
        max_edges=5,
        verbose=True,
    )

    return summarize_result(
        dataset_name=dataset["name"],
        method_name="batch_causal_baseline_plus_pruning_plus_direction_plus_missing",
        pred_edges=completed_edges,
        true_edges=true_edges,
        llm_calls=llm.usage.calls,
        cache_hits=llm.usage.cache_hits,
        human_interventions=llm.usage.human_interventions,
    )


# ========== HITL TIERS ==========


def run_hitl_level_0_pure_llm(dataset: Dict) -> Dict:
    """
    HITL Level 0: Pure LLM baseline (no human intervention).
    Uses batch processing for efficiency.
    """
    return run_batch_baseline_plus_three_interventions(dataset)


def run_hitl_level_1_human_rooting(
    dataset: Dict, human_selected_roots: list[str] | None = None
) -> Dict:
    """
    HITL Level 1: Human-Guided Rooting.

    The LLM proposes root nodes, but a human selects the correct ones.
    This prevents error propagation from the start of BFS.

    Args:
        human_selected_roots: If None, prompts user for input. Otherwise uses provided roots.
    """
    llm = OpenRouterLLM(model="deepseek/deepseek-r1:free")

    nodes = dataset["nodes"]
    descriptions = dataset.get("descriptions")
    true_edges = set(dataset.get("true_edges", set()))

    # LLM proposes root nodes
    proposed_roots = llm.get_root_nodes(nodes, descriptions)
    print(f"\n[HITL-L1] LLM proposed roots: {proposed_roots}")

    # Human intervention: select actual roots
    if human_selected_roots is None:
        print("\n[HUMAN INPUT REQUIRED]")
        print(f"Available nodes: {nodes}")
        print(f"LLM proposed roots: {proposed_roots}")
        user_input = input("Enter correct root nodes (comma-separated): ")
        selected_roots = [
            r.strip() for r in user_input.split(",") if r.strip() in nodes
        ]
    else:
        selected_roots = [r for r in human_selected_roots if r in nodes]

    print(f"[HITL-L1] Human selected roots: {selected_roots}")
    llm.usage.human_interventions += 1

    # Build graph using selected roots (would need BFS implementation)
    # For now, fall back to baseline + interventions
    baseline_edges = run_causal_baseline(dataset)

    pruned_edges = batch_prune_indirect_edges(
        edges=baseline_edges,
        nodes=nodes,
        llm=llm,
        descriptions=descriptions,
        verbose=True,
    )

    corrected_edges = batch_correct_edge_directions(
        edges=pruned_edges,
        nodes=nodes,
        llm=llm,
        descriptions=descriptions,
        verbose=True,
    )

    return summarize_result(
        dataset_name=dataset["name"],
        method_name="hitl_level_1_human_rooting",
        pred_edges=corrected_edges,
        true_edges=true_edges,
        llm_calls=llm.usage.calls,
        cache_hits=llm.usage.cache_hits,
        human_interventions=llm.usage.human_interventions,
    )


def run_hitl_level_2_uncertainty_triggered(
    dataset: Dict,
    confidence_threshold: float = 0.7,
    enable_human_input: bool = False,
) -> Dict:
    """
    HITL Level 2: Uncertainty-Triggered Human Intervention.

    When LLM confidence is below threshold, the system can pause for human input.

    Args:
        confidence_threshold: Edges with confidence below this trigger intervention.
        enable_human_input: If True, actually prompts human. If False, just flags them.
    """
    llm = OpenRouterLLM(model="deepseek/deepseek-r1:free")

    nodes = dataset["nodes"]
    descriptions = dataset.get("descriptions")
    true_edges = set(dataset.get("true_edges", set()))

    baseline_edges = run_causal_baseline(dataset)

    # Batch pruning with confidence tracking
    print(f"\n[HITL-L2] Using confidence threshold: {confidence_threshold}")

    edges_list = sorted(list(baseline_edges))
    verdicts = llm.batch_verify_edges_direct(
        edges=edges_list,
        nodes=nodes,
        descriptions=descriptions,
    )

    pruned_edges = set()
    uncertain_count = 0

    print(f"\n[DEBUG] Baseline returned {len(edges_list)} edges to evaluate.")
    print(f"[DEBUG] Using confidence threshold: {confidence_threshold}")

    for src, dst in edges_list:
        verdict = verdicts.get((src, dst))
        if verdict is None:
            print(f"[DEBUG-ERROR] No verdict returned for {src} -> {dst}")
            continue

        keep = verdict.get("keep", False)
        confidence = verdict.get("confidence", 0.5)
        reason = verdict.get("reason", "No reason provided")

        # ==========================================
        # NEW DEBUG LINE: Print EVERYTHING!
        # ==========================================
        print(f"\n[EVAL] {src} -> {dst}")
        print(f"       LLM says Keep = {keep}")
        print(f"       LLM Confidence = {confidence:.2f}")
        print(f"       Reason: {reason}")
        # ==========================================

        if confidence < confidence_threshold:
            uncertain_count += 1
            print(
                f"       *** TRIGGERING HUMAN INPUT (Conf < {confidence_threshold}) ***"
            )

            if enable_human_input:
                user_input = (
                    input(f"       Human decision - Keep this edge? (y/n): ")
                    .strip()
                    .lower()
                )
                llm.usage.human_interventions += 1

                if user_input == "y":
                    pruned_edges.add((src, dst))
                    print(f"       [HUMAN-KEEP] {src} -> {dst}")
                else:
                    print(f"       [HUMAN-REMOVE] {src} -> {dst}")
            else:
                # Auto-decide based on LLM (simulated HITL)
                if keep:
                    pruned_edges.add((src, dst))
                print(f"       [AUTO-DECIDE] {src} -> {dst} (keep={keep})")
        else:
            if keep:
                pruned_edges.add((src, dst))
                print(
                    f"       [AUTO-KEEP] Confidence {confidence:.2f} >= {confidence_threshold}"
                )
            else:
                print(
                    f"       [AUTO-REMOVE] Confidence {confidence:.2f} >= {confidence_threshold}"
                )

    print(f"\n[HITL-L2] Found {uncertain_count} uncertain edges")

    # Continue with batch direction correction
    corrected_edges = batch_correct_edge_directions(
        edges=pruned_edges,
        nodes=nodes,
        llm=llm,
        descriptions=descriptions,
        verbose=True,
        confidence_threshold=confidence_threshold,
    )

    # Suggest missing edges to recover false negatives and boost recall
    completed_edges = suggest_missing_edges(
        edges=corrected_edges,
        nodes=nodes,
        llm=llm,
        descriptions=descriptions,
        max_edges=5,
        verbose=True,
    )

    return summarize_result(
        dataset_name=dataset["name"],
        method_name=f"hitl_level_2_uncertainty_threshold_{confidence_threshold}",
        pred_edges=completed_edges,  # <--- FIXED!
        true_edges=true_edges,
        llm_calls=llm.usage.calls,
        cache_hits=llm.usage.cache_hits,
        human_interventions=llm.usage.human_interventions,
    )


def save_result(result: Dict) -> None:
    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"results/{result['dataset']}_{result['method']}_{ts}.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nsaved -> {path}")


def main() -> None:
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("set GEMINI_API_KEY before running.")

    # dataset = load_toy()
    # dataset = load_asia_placeholder()
    dataset = load_asia_generated(n_samples=2000)

    # ========== ORIGINAL METHODS (ONE CALL PER EDGE) ==========
    # result = run_bfs_baseline(dataset)
    # result = run_standard_baseline(dataset)
    # result = run_standard_plus_intervention(dataset)
    # result = run_standard_plus_two_interventions(dataset)
    # result = run_standard_plus_three_interventions(dataset)

    # ========== BATCH METHODS (5-10x API REDUCTION) ==========
    # result = run_batch_baseline_plus_two_interventions(dataset)
    result = run_batch_baseline_plus_three_interventions(dataset)

    # ========== HITL TIERS ==========
    # Level 0: Pure LLM (same as batch baseline)
    # result = run_hitl_level_0_pure_llm(dataset)

    # Level 1: Human-Guided Rooting (requires human input or pre-selected roots)
    # result = run_hitl_level_1_human_rooting(dataset)
    # result = run_hitl_level_1_human_rooting(dataset, human_selected_roots=["VisitAsia", "Smoking"])

    # Level 2: Uncertainty-Triggered HITL (flags low-confidence edges)
    # result = run_hitl_level_2_uncertainty_triggered(dataset, confidence_threshold=0.7, enable_human_input=False)
    result = run_hitl_level_2_uncertainty_triggered(
        dataset, confidence_threshold=0.99, enable_human_input=True
    )  # Interactive

    print("\n=== summary ===")
    print(json.dumps(result, indent=2))

    save_result(result)


if __name__ == "__main__":
    main()
