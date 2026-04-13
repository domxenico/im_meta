import random
import numpy as np
import networkx as nx
from collections import deque

def forest_fire_sample(G_full, target_size=3000, p_forward=0.7):
    """exact forest fire implementation as described in 'sampling from large graphs' (leskovec et al., kdd'06)"""
    if len(G_full) <= target_size:
        return G_full.copy()
    
    visited = set()
    
    # main loop for restarts if fire dies
    while len(visited) < target_size:
        
        # choose new seed from remaining nodes
        remaining_nodes = list(set(G_full.nodes()) - visited)
        if not remaining_nodes:
            break
        
        seed = random.choice(remaining_nodes)
        visited.add(seed)
        queue = deque([seed])
        
        while queue and len(visited) < target_size:
            u = queue.popleft()
            
            # identify unvisited neighbors
            unvisited_neighbors = [n for n in G_full.neighbors(u) if n not in visited]
            
            # generate x from geometric distribution 
            if p_forward >= 1.0:
                x = len(unvisited_neighbors)
            else:
                x = np.random.geometric(p=1.0 - p_forward) - 1
            
            # burn up to x neighbors
            count_to_burn = min(x, len(unvisited_neighbors))
            
            if count_to_burn > 0:
                burned_neighbors = random.sample(unvisited_neighbors, count_to_burn)
                
                for v in burned_neighbors:
                    if len(visited) >= target_size:
                        break
                    visited.add(v)
                    queue.append(v)

    return G_full.subgraph(visited).copy()