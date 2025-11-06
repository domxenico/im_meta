from typing import List, Set
import networkx as nx
import random


class QueryNodeSelector:
    def __init__(self, alpha: float = 1.0, k: int = 5):
        self.alpha = alpha
        self.k = k  # seed nodes
    
    def select_next_query(self, explored_graph: nx.Graph, 
                         reinforced_graph: nx.Graph,
                         explored_nodes: Set[int]) -> int:
        
        # ------------------------------------------------------------
        # Step 1: Degree Discount Heuristic on reinforced graph
        # ------------------------------------------------------------
        potential_seeds = self._degree_discount_heuristic(reinforced_graph, self.k)

        # we don't want seeds that are already explored
        potential_seeds = [s for s in potential_seeds if s not in explored_nodes]
        
        # fallback: if all potential seeds already explored → pick random unexplored
        if not potential_seeds:
            candidates = [n for n in reinforced_graph.nodes() if n not in explored_nodes]
            return random.choice(candidates) if candidates else None
        
        # ------------------------------------------------------------
        # Step 2: Candidate nodes for next query
        # According to IM-META, the query node must be UNEXPLORED
        # ------------------------------------------------------------
        query_candidates = [n for n in reinforced_graph.nodes() 
                            if n not in explored_nodes]

        if not query_candidates:
            return None
        
        # ------------------------------------------------------------
        # Step 3: Ranking function (Equation 3 of the paper)
        # rank(u) = residual_degree(u) – α * Σ shortest_paths(u, seeds)
        # ------------------------------------------------------------
        best_node = None
        best_rank = float('-inf')
        
        for u in query_candidates:
            # estimated degree in the reinforced graph
            estimated_degree = reinforced_graph.degree(u, weight='edge_prob')

            # observed degree in explored graph (0 if node not present)
            observed_degree = explored_graph.degree(u) if u in explored_graph else 0

            residual_degree = estimated_degree - observed_degree

            # compute sum of geodesic distances from u to all potential seeds
            sum_geodesic = 0
            for v in potential_seeds:
                try:
                    distance = nx.shortest_path_length(reinforced_graph, u, v)
                except nx.NetworkXNoPath:
                    distance = 1000  # penalty
                sum_geodesic += distance
            
            # final IM-META ranking equation
            rank = residual_degree - self.alpha * sum_geodesic
            
            if rank > best_rank:
                best_rank = rank
                best_node = u
        
        return best_node
    

    def _degree_discount_heuristic(self, G: nx.Graph, k: int) -> List[int]:
        """Degree Discount Seed Selection (Algorithm 1 in IM literature)"""
        degrees = dict(G.degree(weight='edge_prob'))
        seeds = []
        
        for _ in range(k):
            if not degrees:
                break
            
            v = max(degrees, key=degrees.get)
            seeds.append(v)
            
            # discount neighbors
            for u in G.neighbors(v):
                if u in degrees:
                    degrees[u] -= 1
            
            del degrees[v]
        
        return seeds
