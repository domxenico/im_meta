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
        Seleziona i seed usando l'ottimizzazione CELF (Lazy Greedy) sul grafo rinforzato G.
        Ritorna:
            - seeds: La lista dei nodi selezionati.
            - est_sigma: L'influenza stimata sul grafo rinforzato (quella che l'algoritmo crede di avere).
            - real_sigma: L'influenza reale sul grafo vero (ground truth).
        """
        candidates = list(explored_nodes)
        
        # --- [CELF] Fase 1: Calcolo iniziale ---
        gains = [] 
        base_spread = 0.0
        
        for node in candidates:
            # Calcoliamo lo spread su G (grafo rinforzato/inferito)
            spread = self._compute_influence_spread(G, [node])
            marginal_gain = spread - base_spread
            # Usiamo un min-heap con valori negativi per simulare un max-heap
            heapq.heappush(gains, (-marginal_gain, node))
            
        # --- [CELF] Fase 2: Selezione iterativa ---
        seeds = []
        est_sigma = 0.0 # Questa è l'influenza stimata (accumulata)
        
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
                    # Ricalcoliamo il guadagno marginale su G
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
        
        # --- CALCOLO DELLA SIGMA REALE (VALIDAZIONE) ---
        # Usiamo self.real_graph per vedere quanto valgono davvero questi seed
        real_sigma = self._compute_real_influence_spread(self.real_graph, seeds)

        return seeds, est_sigma, real_sigma
    
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
    
    def _compute_real_influence_spread(self, G: nx.Graph, seed_set: List[int]):
        """real influence spread via Monte Carlo 
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
                            if random.random() < self.ic_diff_prob:
                                influenced.add(v)
                                new_active.append(v)
                active = new_active
            
            total_influenced += len(influenced)
        
        return total_influenced / self.num_simulations
