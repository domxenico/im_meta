from collections import defaultdict
from typing import Dict, List, Tuple, Set
import networkx as nx
import time

class ReinforcedGraphGenerator:
    def __init__(self, threshold: float = 0.5, diffusion_model: str = 'IC'):
        self.threshold = threshold
        self.diffusion_model = diffusion_model  # 'IC' or 'WC'
    
    def generate(self, explored_graph: nx.Graph, edge_probabilities: Dict[Tuple[int, int], float],
                 all_nodes: Set[int], queried_nodes: Set[int]) -> nx.Graph:
        """generate reinforced weighted graph G_gen-prun"""
        
        # create weighted graph starting from explored edges
        G_gen_prun = nx.Graph()
        G_gen_prun.add_nodes_from(all_nodes)
        
        for u, v in explored_graph.edges():
            G_gen_prun.add_edge(u, v, weight=0.1, edge_prob=1.0)

        # select confident edges 
        confident_edges = {pair: prob for pair, prob in edge_probabilities.items() 
                          if prob >= self.threshold}

        # add confident inferred edges
        for (u, v), theta in confident_edges.items():
            if not G_gen_prun.has_edge(u, v):
                G_gen_prun.add_edge(u, v, edge_prob=theta, weight=0.0)  # weight computed below
        
        # compute diffusion probabilities based on model
        if self.diffusion_model == 'IC':
            # IC model: uniform diffusion probability
            p_uv = 0.1
            for u, v in G_gen_prun.edges():
                theta = G_gen_prun[u][v]['edge_prob']
                G_gen_prun[u][v]['weight'] = theta * p_uv
        
        elif self.diffusion_model == 'WC':
            # WC model: weight = 1/degree
            # estimate degrees using edge probabilities
            estimated_degrees = defaultdict(float)
            for u, v in G_gen_prun.edges():
                theta = G_gen_prun[u][v]['edge_prob']
                estimated_degrees[u] += theta
                estimated_degrees[v] += theta
            
            for u, v in G_gen_prun.edges():
                theta = G_gen_prun[u][v]['edge_prob']
                d_v = estimated_degrees[v]
                if d_v > 0:
                    G_gen_prun[u][v]['weight'] = theta / d_v
                else:
                    G_gen_prun[u][v]['weight'] = 0.0
        
        return G_gen_prun
