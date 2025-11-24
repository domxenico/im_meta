import random
from collections import deque
import networkx as nx

def forest_fire_sample(G_full, target_size=3000, p_forward=0.7):
    """
    Forest Fire sampling corretto con meccanismo di restart.
    Nota: Per grafi non diretti, p_backward è ridondante, usiamo solo p_forward.
    """
    # Se il grafo è più piccolo del target, restituiscilo tutto
    if len(G_full) <= target_size:
        return G_full.copy()
    
    # Insieme dei nodi visitati
    visited = set()
    
    # Loop principale: continua finché non raggiungiamo la dimensione target
    while len(visited) < target_size:
        
        # --- MECCANISMO DI RESTART ---
        # Se la coda è vuota ma non abbiamo finito, scegliamo un nuovo seed
        # (cercando tra i nodi non ancora visitati)
        remaining_nodes = list(set(G_full.nodes()) - visited)
        if not remaining_nodes:
            break # Abbiamo esaurito l'intero grafo
            
        seed = random.choice(remaining_nodes)
        visited.add(seed)
        queue = deque([seed])
        
        # --- PROCESSO DI BRUCIATURA (Burn) ---
        while queue and len(visited) < target_size:
            u = queue.popleft()
            
            # Ottieni i vicini non ancora visitati
            # (Ottimizzazione: filtriamo subito quelli già visitati)
            neighbors = [n for n in G_full.neighbors(u) if n not in visited]
            random.shuffle(neighbors) # Mischiamo per evitare bias di ordine
            
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