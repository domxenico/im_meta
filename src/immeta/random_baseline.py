from typing import Dict, List, Tuple, Set
import networkx as nx
import numpy as np
import random
from .seed_set_selector import SeedSetSelector

class RandomBaseline:
    def __init__(self, k: int = 5, T: int = 60, real_graph: nx.Graph = nx.Graph()):
        """
        random baseline:
        1. query: random node from explored border
        2. inference: none, observable graph only
        3. seed selection: greedy on observable graph
        """
        self.k = k
        self.T = T
        
        self.seed_selector = SeedSetSelector(k, real_graph=real_graph)
        self.real_graph = real_graph
    
    def run(self, G_full: nx.Graph, initial_nodes: List[int] = None) -> Tuple[List[int], nx.Graph, float]:
        
        if initial_nodes is None:
            initial_nodes = random.sample(list(G_full.nodes()), 4)
        
        explored_nodes = set(initial_nodes)
        queried_nodes = set()
        
        print(f"[random] starting with {len(initial_nodes)} initial nodes")
        
        for t in range(self.T):
            candidates = list(explored_nodes - queried_nodes)
            
            if not candidates:
                print("[random] no more candidates to query.")
                break
            
            next_query = random.choice(candidates)
            queried_nodes.add(next_query)
            
            neighbors = set(G_full.neighbors(next_query))
            new_nodes = neighbors - explored_nodes
            
            explored_nodes.update(new_nodes)
            explored_nodes.add(next_query)

        explored_graph = G_full.subgraph(explored_nodes).copy()
        
        # assign weights; standard 0.1 uniform probability for ic
        for u, v in explored_graph.edges():
            explored_graph[u][v]['weight'] = 0.1  

        print(f"[random] exploration finished. explored graph size: {len(explored_graph.nodes())} nodes.")
        print("[random] selecting seeds on observable graph...")
        
        seeds, est_sigma, real_sigma = self.seed_selector.select_seeds(explored_graph, explored_nodes)
        
        print(f"[random] selected seeds: {seeds}")
        print(f"[random] real sigma: {real_sigma}")
        
        return explored_graph, real_sigma