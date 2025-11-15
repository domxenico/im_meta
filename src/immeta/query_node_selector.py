from typing import Dict, List, Tuple, Set
import networkx as nx
import random


class QueryNodeSelector:
    def __init__(self, alpha: float = 1.0, k: int = 5):
        self.alpha = alpha
        self.k = k  # seed nodes
    
    def select_next_query(self, explored_graph: nx.Graph, 
                         reinforced_graph: nx.Graph,
                         explored_nodes: Set[int],
                         queried_nodes: Set[int]) -> int:
        
        # step 1: find k potentially influential nodes using degree discount heuristics
        potential_seeds = self._degree_discount_heuristic(reinforced_graph, self.k)
        
        # filter out already explored potential seeds
        potential_seeds = [s for s in potential_seeds if s not in explored_nodes]
        
        if not potential_seeds:
            # fallback: random unqueried node with edges in reinforced graph
            candidates = [n for n in reinforced_graph.nodes() if n not in queried_nodes]
            return random.choice(candidates) if candidates else None
        
        # step 2: compute ranking for each explored node
        best_node = None
        best_rank = float('-inf')
        
        for u in (explored_nodes - queried_nodes):
            # Compute residual degree
            estimated_degree = reinforced_graph.degree(u, weight='edge_prob')
            observed_degree = explored_graph.degree(u)
            residual_degree = estimated_degree - observed_degree
            
            # Compute sum of shortest paths to potential seeds
            sum_geodesic = 0
            for v in potential_seeds:
                try:
                    distance = nx.shortest_path_length(reinforced_graph, u, v)
                    sum_geodesic += distance
                except nx.NetworkXNoPath:
                    sum_geodesic += 1000  # large penalty for unreachable nodes
            
            # topology-aware ranking (Equation 3)
            rank = residual_degree - self.alpha * sum_geodesic
            
            if rank > best_rank:
                best_rank = rank
                best_node = u
        
        return best_node
    
    def _degree_discount_heuristic(self, G: nx.Graph, k: int) -> List[int]:
        """Degree discount heuristics for fast seed selection"""
        degrees = dict(G.degree(weight='edge_prob'))
        seeds = []
        
        for _ in range(k):
            if not degrees:
                break
            
            # select node with highest degree
            v = max(degrees, key=degrees.get)
            seeds.append(v)
            
            # discount degrees of neighbors
            for u in G.neighbors(v):
                if u in degrees:
                    degrees[u] -= 1
            
            del degrees[v]
        
        return seeds
