import random
import numpy as np
import networkx as nx
from collections import deque

def forest_fire_sample(G_full, target_size=3000, p_forward=0.7):
    """
    Implementazione esatta del Forest Fire come descritto in 'Sampling from Large Graphs', 
    Appendice A.1 (Leskovec et al., KDD'06).
    """
    if len(G_full) <= target_size:
        return G_full.copy()
    
    visited = set()
    
    # Loop principale per i restart (se il fuoco muore) [cite: 438]
    while len(visited) < target_size:
        
        # Selezione nuovo seed se necessario [cite: 438]
        remaining_nodes = list(set(G_full.nodes()) - visited)
        if not remaining_nodes:
            break
        
        seed = random.choice(remaining_nodes)
        visited.add(seed)
        queue = deque([seed])
        
        while queue and len(visited) < target_size:
            u = queue.popleft()
            
            # 1. Identifica i vicini non visitati ("links incident to nodes not yet visited") 
            unvisited_neighbors = [n for n in G_full.neighbors(u) if n not in visited]
            
            # 2. Genera numero casuale x geometricamente distribuito 
            # La media deve essere p / (1 - p). 
            # In numpy, geometric(q) ha media 1/q. 
            # Se poniamo q = 1 - p_forward, la media di (geometric(1-p) - 1) è:
            # (1 / (1-p)) - 1 = (1 - (1-p)) / (1-p) = p / (1-p). Corretto.
            if p_forward >= 1.0:
                x = len(unvisited_neighbors) # Brucia tutto se p=1
            else:
                # np.random.geometric prende il parametro di successo (1 - p_forward)
                # Sottraiamo 1 perché numpy genera valori >= 1, ma noi possiamo bruciare 0 vicini.
                x = np.random.geometric(p=1.0 - p_forward) - 1
            
            # 3. Il nodo seleziona x vicini 
            # Se x è maggiore dei vicini disponibili, li prende tutti.
            count_to_burn = min(x, len(unvisited_neighbors))
            
            if count_to_burn > 0:
                # La selezione è casuale ("Node v selects x out-links") 
                burned_neighbors = random.sample(unvisited_neighbors, count_to_burn)
                
                for v in burned_neighbors:
                    if len(visited) >= target_size:
                        break
                    visited.add(v)
                    queue.append(v) # Ricorsione [cite: 436]

    return G_full.subgraph(visited).copy()