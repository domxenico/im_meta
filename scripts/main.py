from immeta import IMMETA, coauthor_data


mc_sim = 1
coauthor_dataset = "CS"

budgets_to_test = [60]
results_file_name = "results_log.txt"

def run_experiment(G_full, node_features, feature_dim, num_queries):
   
    print(f"  starting simulation (budget={num_queries})...")
    
    nodes_sum = 0
    edges_sum = 0
    sigma_sum = 0
    
    for mc in range(mc_sim):
        im_meta = IMMETA(
            feature_dim=feature_dim,
            k=5,           # n seeds
            T=num_queries, # uses the dynamic budget
            alpha=1.0,     # balance parameter
            threshold=0.5, # confident edge threshold (epsilon)
            diffusion_model='IC',
            real_graph=G_full
        )
        
        seeds, explored_graph, sigma = im_meta.run(G_full, node_features)
        
        nodes_sum += len(explored_graph.nodes())
        edges_sum += len(explored_graph.edges())
        sigma_sum += sigma
    
    avg_nodes = nodes_sum / mc_sim
    avg_sigma = sigma_sum / mc_sim
    
    print(f"  average discovered nodes: {avg_nodes}")
    print(f"  average sigma: {avg_sigma}")
    
    return avg_nodes, avg_sigma


def main():
    
    print("starting influence maximization experiments...")
    
    print("\nloading dataset...")
    g_full, node_features = coauthor_data(coauthor_dataset)
    
    feature_dim = len(next(iter(node_features.values())))
    
    try:
        with open(results_file_name, 'w', encoding='utf-8') as f:
            f.write("influence maximization experiment results\n")
            f.write("=========================================\n\n")
            
            # --- main loop over query budgets ---
            for budget in budgets_to_test:
                
                print(f"\n--- executing budget: {budget} queries ---")
                
                # run the experiment for the current budget
                discovered_nodes, obtained_sigma = run_experiment(
                    g_full, 
                    node_features, 
                    feature_dim, 
                    budget
                )
                
                # write results to the text file
                f.write(f"[budget = {budget}]\n")
                f.write(f"  discovered nodes (avg): {discovered_nodes}\n")
                f.write(f"  obtained sigma (avg): {obtained_sigma}\n\n")
            
            print(f"\n--- experiments completed ---")
            print(f"all results saved to '{results_file_name}'.")

    except IOError as e:
        print(f"error: could not write to file '{results_file_name}'. details: {e}")
    except Exception as e:
        print(f"an unexpected error occurred: {e}")


if __name__ == "__main__":
    main()