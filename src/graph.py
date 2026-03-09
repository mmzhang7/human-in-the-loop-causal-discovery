from collections import defaultdict, deque
from typing import Dict, Iterable, List, Set, Tuple

Edge = Tuple[str, str]


class DiGraph:
    """
    Simple directed graph for causal edges.
    Provides cycle check to enforce DAG constraint.
    """

    def __init__(self, nodes: Iterable[str]):
        self.nodes: List[str] = list(nodes)
        self._adj: Dict[str, Set[str]] = defaultdict(set)
        self._edges: Set[Edge] = set()

    def add_edge(self, u: str, v: str) -> None:
        """Add edge u -> v."""
        self._adj[u].add(v)
        self._edges.add((u, v))

    def has_edge(self, u: str, v: str) -> bool:
        return (u, v) in self._edges

    def edges(self) -> Set[Edge]:
        return set(self._edges)

    def would_create_cycle(self, u: str, v: str) -> bool:
        """
        Adding u -> v creates a cycle if v can already reach u.
        """
        if u == v:
            return True

        q = deque([v])
        seen = {v}

        while q:
            x = q.popleft()
            if x == u:
                return True
            for nxt in self._adj.get(x, set()):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)

        return False

    def has_path(self, u: str, v: str) -> bool:
        """
        Check if there's a directed path from u to v.
        Useful for transitive reduction: if u->...->v exists via other nodes,
        then u->v might be redundant.
        """
        if u == v:
            return True

        q = deque([u])
        seen = {u}

        while q:
            x = q.popleft()
            if x == v:
                return True
            for nxt in self._adj.get(x, set()):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)

        return False

    def has_indirect_path(self, u: str, v: str) -> bool:
        """
        Check if there's a path from u to v that goes through at least one intermediate node.
        Returns True if u can reach v without using the direct edge u->v.
        """
        if u == v:
            return False

        # BFS from u, but skip the direct edge u->v
        q = deque([u])
        seen = {u}

        while q:
            x = q.popleft()
            for nxt in self._adj.get(x, set()):
                # Skip direct edge u->v
                if x == u and nxt == v:
                    continue
                
                if nxt == v:
                    return True
                
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)

        return False

    def get_transitive_edges(self) -> Set[Edge]:
        """
        Find edges that might be redundant due to transitive paths.
        An edge (u, v) is potentially transitive if there exists a path
        u -> ... -> v through other nodes.
        
        This doesn't remove edges automatically, but flags them for verification.
        """
        transitive = set()
        
        for u, v in self._edges:
            if self.has_indirect_path(u, v):
                transitive.add((u, v))
        
        return transitive