from typing import Any, Dict, List, Set, Tuple

from .llm_interface import GeminiLLM

Edge = Tuple[str, str]


def prune_indirect_edges(
    edges: Set[Edge],
    nodes: List[str],
    llm: GeminiLLM,
    descriptions: Dict[str, str] | None = None,
    verbose: bool = True,
) -> Set[Edge]:
    """
    Remove edges that are indirect or implausible.
    """
    pruned_edges = set()

    for src, dst in edges:
        verdict = llm.verify_edge_direct(
            src,
            dst,
            nodes,
            descriptions=descriptions,
        )

        if verdict.get("keep", False):
            pruned_edges.add((src, dst))
            if verbose:
                print(f"[keep] {src} -> {dst}")
        else:
            mediators = verdict.get("mediators", [])
            if verbose:
                if mediators:
                    print(f"[remove] {src} -> {dst} (mediators={mediators})")
                else:
                    print(f"[remove] {src} -> {dst}")

    return pruned_edges


def correct_edge_directions(
    edges: Set[Edge],
    nodes: List[str],
    llm: GeminiLLM,
    descriptions: Dict[str, str] | None = None,
    verbose: bool = True,
) -> Set[Edge]:
    """
    Post-processing intervention step that checks whether each edge
    should be kept, flipped, or removed.

    This is useful for correcting direction mistakes after the baseline
    causal graph has already been constructed.
    """
    corrected_edges = set()

    for src, dst in edges:
        verdict = llm.verify_edge_direction(
            src,
            dst,
            nodes,
            descriptions=descriptions,
        )

        action = verdict.get("action", "keep")
        reason = verdict.get("reason", "")

        if action == "flip":
            corrected_edges.add((dst, src))
            if verbose:
                if reason:
                    print(f"[flip] {src} -> {dst}  =>  {dst} -> {src} ({reason})")
                else:
                    print(f"[flip] {src} -> {dst}  =>  {dst} -> {src}")

        elif action == "remove":
            if verbose:
                if reason:
                    print(f"[remove] {src} -> {dst} ({reason})")
                else:
                    print(f"[remove] {src} -> {dst}")

        else:
            corrected_edges.add((src, dst))
            if verbose:
                if reason:
                    print(f"[keep] {src} -> {dst} ({reason})")
                else:
                    print(f"[keep] {src} -> {dst}")

    return corrected_edges

def suggest_missing_edges(
    edges: Set[Edge],
    nodes: List[str],
    llm: GeminiLLM,
    descriptions: Dict[str, str] | None = None,
    max_edges: int = 5,
    verbose: bool = True,
) -> Set[Edge]:
    """
    Ask the LLM for a small number of missing direct causal edges
    and add them to the current graph.
    """
    current_edges = sorted(list(edges))

    verdict = llm.suggest_missing_edges(
        current_edges=current_edges,
        nodes=nodes,
        descriptions=descriptions,
        max_edges=max_edges,
    )

    suggested_edges = verdict.get("suggested_edges", [])
    updated_edges = set(edges)

    for src, dst in suggested_edges:
        if (src, dst) not in updated_edges:
            updated_edges.add((src, dst))
            if verbose:
                print(f"[add-missing] {src} -> {dst}")

    return updated_edges


# ========== BATCH INTERVENTION FUNCTIONS ==========

def batch_prune_indirect_edges(
    edges: Set[Edge],
    nodes: List[str],
    llm: GeminiLLM,
    descriptions: Dict[str, str] | None = None,
    verbose: bool = True,
    confidence_threshold: float = 0.0,
) -> Set[Edge]:
    """
    Batch version: Remove edges that are indirect or implausible.
    Processes all edges in one API call, reducing cost significantly.
    
    Args:
        confidence_threshold: If > 0, edges with confidence below this are flagged
                             for potential human review (currently just logged).
    """
    edges_list = list(edges)
    
    if not edges_list:
        return set()
    
    # Batch verify all edges at once
    verdicts = llm.batch_verify_edges_direct(
        edges=edges_list,
        nodes=nodes,
        descriptions=descriptions,
    )
    
    pruned_edges = set()
    uncertain_edges = []
    
    for src, dst in edges_list:
        verdict = verdicts.get((src, dst))
        
        if verdict is None:
            # Edge wasn't in results, skip it
            if verbose:
                print(f"[warning] {src} -> {dst} not in batch results")
            continue
        
        keep = verdict.get("keep", False)
        confidence = verdict.get("confidence", 0.5)
        
        if keep:
            pruned_edges.add((src, dst))
            if verbose:
                conf_str = f" (conf={confidence:.2f})" if confidence < 1.0 else ""
                print(f"[keep] {src} -> {dst}{conf_str}")
            
            if confidence < confidence_threshold:
                uncertain_edges.append((src, dst, confidence))
        else:
            mediators = verdict.get("mediators", [])
            if verbose:
                if mediators:
                    print(f"[remove] {src} -> {dst} (mediators={mediators})")
                else:
                    print(f"[remove] {src} -> {dst}")
    
    if uncertain_edges and verbose:
        print(f"\n[info] {len(uncertain_edges)} edges have low confidence:")
        for src, dst, conf in uncertain_edges:
            print(f"  {src} -> {dst} (confidence={conf:.2f})")
    
    return pruned_edges


def batch_correct_edge_directions(
    edges: Set[Edge],
    nodes: List[str],
    llm: GeminiLLM,
    descriptions: Dict[str, str] | None = None,
    verbose: bool = True,
    confidence_threshold: float = 0.0,
) -> Set[Edge]:
    """
    Batch version: Check direction for multiple edges at once.
    Processes all edges in one API call.
    
    Args:
        confidence_threshold: If > 0, edges with confidence below this are flagged
                             for potential human review.
    """
    edges_list = list(edges)
    
    if not edges_list:
        return set()
    
    verdicts = llm.batch_verify_edges_direction(
        edges=edges_list,
        nodes=nodes,
        descriptions=descriptions,
    )
    
    corrected_edges = set()
    uncertain_edges = []
    
    for src, dst in edges_list:
        verdict = verdicts.get((src, dst))
        
        if verdict is None:
            if verbose:
                print(f"[warning] {src} -> {dst} not in batch results")
            continue
        
        action = verdict.get("action", "keep")
        reason = verdict.get("reason", "")
        confidence = verdict.get("confidence", 0.5)
        
        if action == "flip":
            corrected_edges.add((dst, src))
            if verbose:
                conf_str = f" conf={confidence:.2f}" if confidence < 1.0 else ""
                if reason:
                    print(f"[flip] {src} -> {dst} => {dst} -> {src} ({reason},{conf_str})")
                else:
                    print(f"[flip] {src} -> {dst} => {dst} -> {src} ({conf_str})")
        
        elif action == "remove":
            if verbose:
                conf_str = f" conf={confidence:.2f}" if confidence < 1.0 else ""
                if reason:
                    print(f"[remove] {src} -> {dst} ({reason},{conf_str})")
                else:
                    print(f"[remove] {src} -> {dst} ({conf_str})")
        
        else:  # keep
            corrected_edges.add((src, dst))
            if verbose:
                conf_str = f" conf={confidence:.2f}" if confidence < 1.0 else ""
                if reason:
                    print(f"[keep] {src} -> {dst} ({reason},{conf_str})")
                else:
                    print(f"[keep] {src} -> {dst} ({conf_str})")
        
        if confidence < confidence_threshold:
            uncertain_edges.append((src, dst, action, confidence))
    
    if uncertain_edges and verbose:
        print(f"\n[info] {len(uncertain_edges)} edges have low confidence:")
        for src, dst, action, conf in uncertain_edges:
            print(f"  {src} -> {dst} action={action} (confidence={conf:.2f})")
    
    return corrected_edges