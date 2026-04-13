import heapq
from typing import Dict, List, Tuple, Set
import networkx as nx
import random

class SeedSetSelector:
    def __init__(self, k: int, num_simulations: int = 100, ic_diff_prob: float = 0.1, real_graph: nx.Graph = nx.Graph()):
        self.k = k
        self.num_simulations = num_simulations
        self.ic_diff_prob = ic_diff_prob
        self.real_graph = real_graph
    
    def select_seeds(self, G: nx.Graph, explored_nodes: Set[int]) -> Tuple[List[int], float, float]:
        """
        select seeds using celf optimization (lazy greedy) on reinforced graph G
        returns:
            - seeds: selected nodes list
            - est_sigma: estimated influence spread on reinforced graph
            - real_sigma: real influence spread on true graph (ground truth)
        """
        candidates = list(explored_nodes)
        
        # celf phase 1: initial computation
        gains = [] 
        base_spread = 0.0
        
        for node in candidates:
            # compute spread on reinforced graph G
            spread = self._compute_influence_spread(G, [node])
            marginal_gain = spread - base_spread
            # max-heap simulation using negative values
            heapq.heappush(gains, (-marginal_gain, node))
            
        # celf phase 2: iterative selection
        seeds = []
        est_sigma = 0.0 # estimated accumulated influence
        
        while len(seeds) < self.k:
            matched = False
            while not matched and gains:
                gain, best_node = heapq.heappop(gains)
                gain = -gain
                
                if len(seeds) == 0:
                    matched = True
                    seeds.append(best_node)
                    est_sigma += gain
                else:
                    # recompute marginal gain on G
                    new_spread = self._compute_influence_spread(G, seeds + [best_node])
                    marginal_gain = new_spread - est_sigma
                    
                    if not gains:
                        matched = True
                        seeds.append(best_node)
                        est_sigma = new_spread
                    else:
                        next_best_gain = -gains[0][0]
                        if marginal_gain >= next_best_gain:
                            matched = True
                            seeds.append(best_node)
                            est_sigma = new_spread
                        else:
                            heapq.heappush(gains, (-marginal_gain, best_node))
        
        # compute real sigma for evaluation using ground truth graph
        real_sigma = self._compute_real_influence_spread(self.real_graph, seeds)

        return seeds, est_sigma, real_sigma
    
    def _compute_influence_spread(self, G: nx.Graph, seed_set: List[int]) -> float:
        """estimated influence spread via monte carlo simulation with independent cascade"""
        total_influenced = 0
        
        for _ in range(self.num_simulations):
            influenced = set(seed_set)
            active = list(seed_set)
            
            while active:
                new_active = []
                for u in active:
                    for v in G.neighbors(u):
                        if v not in influenced:
                            # activation via theta_uv probability
                            if random.random() < (G[u][v]['weight']):
                                influenced.add(v)
                                new_active.append(v)
                active = new_active
            
            total_influenced += len(influenced)
        
        return total_influenced / self.num_simulations
    
    def _compute_real_influence_spread(self, G: nx.Graph, seed_set: List[int]):
        """real influence spread via monte carlo simulation with independent cascade"""
        total_influenced = 0
        
        for _ in range(self.num_simulations):
            influenced = set(seed_set)
            active = list(seed_set)
            
            while active:
                new_active = []
                for u in active:
                    for v in G.neighbors(u):
                        if v not in influenced:
                            if random.random() < self.ic_diff_prob:
                                influenced.add(v)
                                new_active.append(v)
                active = new_active
            
            total_influenced += len(influenced)
        
        return total_influenced / self.num_simulations
