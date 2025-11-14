import random
from collections import deque

def forest_fire_sample(G_full, target_size=3000, p_forward=0.7, p_backward=0.7):
    """ Forest Fire sampling... (invariato) """
    if len(G_full) <= target_size:
        return G_full.copy()
    seed = random.choice(list(G_full.nodes()))
    visited = set([seed])
    queue = deque([seed])
    while len(visited) < target_size and queue:
        u = queue.popleft()
        neighbors = list(G_full.neighbors(u))
        random.shuffle(neighbors)
        burn_forward = [v for v in neighbors if v not in visited and random.random() < p_forward]
        burn_backward = [v for v in neighbors if v in visited and random.random() < p_backward]
        to_burn = burn_forward + burn_backward
        for v in to_burn:
            if len(visited) >= target_size:
                break
            if v not in visited:
                visited.add(v)
                queue.append(v)
    return G_full.subgraph(visited).copy()
