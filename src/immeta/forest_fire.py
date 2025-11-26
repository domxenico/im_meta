import random
from collections import deque
import networkx as nx

def forest_fire_sample(G_full, target_size=3000, p_forward=0.7):
    """
    forest fire sampling for undirected graphs
    """
    
    if len(G_full) <= target_size:
        return G_full.copy()
    
    visited = set()
    
    # until we don't reach the target dimension
    while len(visited) < target_size:
        
        # restart mechanism
        # if queue empty but we didn't finish, we choose a new seed
        # (into nonvisited nodes)
        remaining_nodes = list(set(G_full.nodes()) - visited)
        if not remaining_nodes:
            break
        
        seed = random.choice(remaining_nodes)
        visited.add(seed)
        queue = deque([seed])
        
        # burn process
        while queue and len(visited) < target_size:
            u = queue.popleft()
            
            # non visited neighbors
            neighbors = [n for n in G_full.neighbors(u) if n not in visited]
            random.shuffle(neighbors)
            
            # Bruciamo geometricamente o probabilisticamente
            # Qui usiamo la logica probabilistica per arco (simil-percolazione)
            # che è più semplice e spesso usata come approssimazione
            for v in neighbors:
                if len(visited) >= target_size:
                    break
                
                if random.random() < p_forward:
                    visited.add(v)
                    queue.append(v)
    
    return G_full.subgraph(visited).copy()