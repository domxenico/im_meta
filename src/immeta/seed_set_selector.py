from typing import Dict, List, Tuple, Set
import networkx as nx
import random

class SeedSetSelector:
    def __init__(self, k: int, num_simulations: int = 100, ic_diff_prob: float = 0.1):
        self.k = k
        self.num_simulations = num_simulations
        self.ic_diff_prob = ic_diff_prob
    
    def select_seeds(self, G: nx.Graph, explored_nodes: Set[int]) -> List[int]:
        """modified greedy"""
        seeds = []
        
        for i in range(self.k):
            best_node = None
            best_marginal_spread = 0
            
            for v in explored_nodes:
                if v not in seeds:
                    # sigma({v} U S) - sigma(S) [marginal spread]
                    marginal = self._compute_influence_spread(G, seeds + [v]) - \
                              self._compute_influence_spread(G, seeds)
                    
                    if marginal > best_marginal_spread:
                        best_marginal_spread = marginal
                        best_node = v
            
            if best_node is not None:
                seeds.append(best_node)
        
        sigma = self._compute_influence_spread(G, seeds)
        return seeds, sigma
    
    def _compute_influence_spread(self, G: nx.Graph, seed_set: List[int]) -> float:
        """estimated influence spread via Monte Carlo 
        simulation with independent cascade [sigma(.)]"""

        total_influenced = 0
        
        for _ in range(self.num_simulations):
            influenced = set(seed_set)
            active = list(seed_set)
            
            while active:
                new_active = []
                for u in active:
                    for v in G.neighbors(u):
                        if v not in influenced:
                            # activation probability = theta_uv * IC diffusion probability
                            if random.random() < (G[u][v]['weight']):
                                influenced.add(v)
                                new_active.append(v)
                active = new_active
            
            total_influenced += len(influenced)
        
        return total_influenced / self.num_simulations
