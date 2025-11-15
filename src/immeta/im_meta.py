from typing import Dict, List, Tuple, Set
import networkx as nx
import numpy as np
import random
import time

from .query_node_selector import QueryNodeSelector
from .reinforced_graph_generator import ReinforcedGraphGenerator
from .seed_set_selector import SeedSetSelector
from .network_inference import NetworkInference


class IMMETA:
    def __init__(self, feature_dim: int, k: int = 5, T: int = 60,
                 alpha: float = 1.0, threshold: float = 0.5,
                 diffusion_model: str = 'IC', real_graph: nx.Graph = nx.Graph()):
        
        self.k = k  # n seeds
        self.T = T  # n queries
        self.feature_dim = feature_dim
        
        self.network_inference = NetworkInference(feature_dim)
        self.graph_generator = ReinforcedGraphGenerator(threshold, diffusion_model)
        self.query_selector = QueryNodeSelector(alpha, k)
        self.seed_selector = SeedSetSelector(k, real_graph=real_graph)
        self.real_graph = real_graph
    
    def run(self, G_full: nx.Graph, node_features: Dict[int, np.ndarray],
            initial_nodes: List[int] = None) -> Tuple[List[int], nx.Graph]:
        """
        full IM-META pipeline
        
        returns:
            seeds: selected seed nodes
            explored_graph: final explored subgraph
            sigma
        """
        
        if initial_nodes is None:
            # random initial node to build initial subgraph
            initial_nodes = random.sample(list(G_full.nodes()), 4)
        
        # explored != queried
        explored_nodes = set(initial_nodes)
        explored_edges = set()
        
        queried_nodes = set()

        explored_graph = nx.Graph()
        explored_graph.add_nodes_from(explored_nodes)
        explored_graph.add_edges_from(explored_edges)
        
        print(f"starting with {len(initial_nodes)} initial node/s")
        print(f"query budget: {self.T}, seed budget: {self.k}")
        
        # NDP
        for t in range(self.T):
            print(f"\n--- query {t+1}/{self.T} ---")
            
            # NDP1: network inference
            # print("training Siamese network...")
            self.network_inference.train(node_features, explored_graph, epochs=10)
            
            # uncertain edges
            uncertain_pairs = []
            all_nodes_set = set(G_full.nodes())
            unqueried_nodes = all_nodes_set - queried_nodes
            
            for u in (explored_nodes - queried_nodes):
                for v in unqueried_nodes:
                    uncertain_pairs.append((u, v))
            
            print(f"predicting {len(uncertain_pairs)} uncertain edges...")
            edge_probs = self.network_inference.predict_edge_probabilities(
                node_features, uncertain_pairs
            )
            
            # NDP2: reinforced graph generation
            print("generating reinforced graph...")
            G_gen_prun = self.graph_generator.generate(
                explored_graph, edge_probs, all_nodes_set, queried_nodes
            )
            
            # NDP3: query node selection
            print("selecting next query node...")
            next_query = self.query_selector.select_next_query(
                explored_graph, G_gen_prun, explored_nodes, queried_nodes
            )
            
            if next_query is None:
                print("no more nodes to query")
                break
            queried_nodes.add(next_query)

            # query execution
            neighbors = set(G_full.neighbors(next_query))
            new_nodes = neighbors - explored_nodes
            
            explored_nodes.update(new_nodes)
            explored_nodes.add(next_query)
            
            explored_graph = G_full.subgraph(explored_nodes).copy()
            
            print(f"queried node with id:{next_query}, discovered {len(new_nodes)} new nodes")
            print(f"explored graph now has {len(explored_nodes)} nodes")

        # self.network_inference.save_model_checkpoint()
        
        print("\n--- seed selection phase ---")
        
        # final inference and reinforced graph generation
        self.network_inference.train(node_features, explored_graph, epochs=20)
        
        uncertain_pairs = []
        for u in explored_nodes:
            for v in explored_nodes:
                if u < v and not explored_graph.has_edge(u, v):
                    uncertain_pairs.append((u, v))
        
        edge_probs = self.network_inference.predict_edge_probabilities(
            node_features, uncertain_pairs
        )
        
        unexplored_nodes = all_nodes_set - explored_nodes
        G_final = self.graph_generator.generate(
            explored_graph, edge_probs, unexplored_nodes, queried_nodes
        )
        
        print("selecting seed nodes...")
        seeds, est_sigma = self.seed_selector.select_seeds(G_final, explored_nodes)
        print("computing influence spread on real graph")
        sigma = self.seed_selector._compute_real_influence_spread(self.real_graph, seed_set=seeds)
        
        print(f"est_sigma: {est_sigma}\nreal_sigma: {sigma}")
        print(f"\nselected seeds: {seeds}")
        
        # theoretically this should be the way we test the seeds and we plot sigma but it is not made in this way
        
        
        return seeds, explored_graph, sigma
