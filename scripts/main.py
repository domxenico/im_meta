from immeta import IMMETA, coauthor_data

MC_SIM = 1
COAUTHOR_DATASET = "CS"
NUM_QUERIES = 20

def main():
    
    print("loading dataset...")
    G_full, node_features = coauthor_data(COAUTHOR_DATASET)
    
    print(f"graph: {len(G_full.nodes())} nodes, {len(G_full.edges())} edges")
    print(f"feature dimension: {len(next(iter(node_features.values())))}")
    
    feature_dim = len(next(iter(node_features.values())))
    
    nodes_sum = 0
    edges_sum = 0
    for mc in range(MC_SIM):

        im_meta = IMMETA(
            feature_dim=feature_dim,
            k=5,           # n seeds
            T=NUM_QUERIES,          # n queries
            alpha=1.0,     # balance parameter
            threshold=0.5, # confident edge threshold (epsilon)
            diffusion_model='IC'
        )
        
        seeds, explored_graph = im_meta.run(G_full, node_features)
        nodes_sum += len(explored_graph.nodes())
        edges_sum += len(explored_graph.edges())
        print(f"\nMC {mc}\nexplored graph: {len(explored_graph.nodes())}/{len(G_full.nodes())} nodes , {len(explored_graph.edges())}/{len(G_full.edges())} edges")
    
    print("\n","-"*10)
    print(f"average explored graph: {nodes_sum/MC_SIM} nodes , {edges_sum/MC_SIM} edges")
    print(f"last iteration seed nodes: {seeds}")

if __name__ == "__main__":
    main()