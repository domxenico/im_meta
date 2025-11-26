from typing import Dict, List, Tuple, Set
import networkx as nx
import numpy as np
import random
from .seed_set_selector import SeedSetSelector

class RandomBaseline:
    def __init__(self, k: int = 5, T: int = 60, real_graph: nx.Graph = nx.Graph()):
        """
        Baseline Random:
        1. Query: Seleziona casualmente un nodo dalla frontiera esplorata.
        2. Inference: Nessuna. Usa solo il grafo osservato.
        3. Seed Selection: Greedy sul grafo osservato.
        """
        self.k = k  # n seeds
        self.T = T  # n queries
        
        self.seed_selector = SeedSetSelector(k, real_graph=real_graph)
        self.real_graph = real_graph
    
    def run(self, G_full: nx.Graph, initial_nodes: List[int] = None) -> Tuple[List[int], nx.Graph, float]:
        
        if initial_nodes is None:
            initial_nodes = random.sample(list(G_full.nodes()), 4)
        
        # Insiemi per tenere traccia dello stato
        explored_nodes = set(initial_nodes)
        queried_nodes = set()
        
        print(f"[Random] Starting with {len(initial_nodes)} initial nodes")
        
        # --- QUERY PHASE ---
        for t in range(self.T):
            candidates = list(explored_nodes - queried_nodes)
            
            if not candidates:
                print("[Random] No more candidates to query.")
                break
            
            # Selezione Casuale
            next_query = random.choice(candidates)
            queried_nodes.add(next_query)
            
            # Espansione del grafo
            neighbors = set(G_full.neighbors(next_query))
            new_nodes = neighbors - explored_nodes
            
            explored_nodes.update(new_nodes)
            explored_nodes.add(next_query)

        # Costruiamo il grafo finale osservato
        explored_graph = G_full.subgraph(explored_nodes).copy()
        
        # --- FIX: Assegnazione dei pesi ---
        # Il SeedSetSelector richiede l'attributo 'weight' per calcolare lo spread.
        # Per la baseline Random (su modello IC), assegniamo una probabilità uniforme standard (es. 0.1).
        for u, v in explored_graph.edges():
            explored_graph[u][v]['weight'] = 0.1  
            # Nota: se usassi il modello WC, dovresti calcolare 1/in_degree qui.
            # Ma 0.1 è lo standard per i confronti IC.

        print(f"[random] exploration finished. explored graph size: {len(explored_graph.nodes())} nodes.")
        print("[random] selecting seeds on observable graph...")
        
        # --- SEED SELECTION PHASE ---
        
        seeds, est_sigma, real_sigma = self.seed_selector.select_seeds(explored_graph, explored_nodes)
        
        print(f"[random] selected seeds: {seeds}")
        print(f"[random] real sigma: {real_sigma}")
        
        return explored_graph, real_sigma