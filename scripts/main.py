import numpy as np
import torch
import os 

from immeta import IMMETA, coauthor_data
from immeta.gsm import AutoencoderGSM
from immeta.gsm import train_gsm_model
from immeta.random_baseline import RandomBaseline

from immeta.feature_utils import create_dirty_features
from immeta.feature_utils import reconstruct_features
from immeta.forest_fire import forest_fire_sample

MC_SIM = 1
COAUTHOR_DATASET = "CS"
budgets_to_test = [20] # in case of various budget test add [20, desired_budget, another_desired_budget]
results_file_name = "results_log.txt"

FEATURE_DIM = 6805 if COAUTHOR_DATASET == 'CS' else 8415 # Physics

GSM_LATENT_DIM = 256 # generative-surrogate-model latent dimension
GSM_MODEL_PATH = f"gsm_autoencoder_{COAUTHOR_DATASET}.pth"
CORRUPTION_RATE = 0.3
GSM_EPOCHS = 10
GSM_BATCH_SIZE = 64


def run_experiment(model_type, G_full, node_features, feature_dim, num_queries):
    """
    model_type: "IMMETA" oppure "RAND"
    """
    print(f"starting simulation [{model_type}] (budget={num_queries})...")
    nodes_sum = 0
    sigma_sum = 0
    
    for mc in range(MC_SIM):
        if model_type == "IMMETA":
            model = IMMETA(
                feature_dim=feature_dim,
                k=5, T=num_queries, alpha=1.0, threshold=0.5,
                diffusion_model='IC', real_graph=G_full
            )
            seeds, explored_graph, sigma = model.run(G_full, node_features)
            
        elif model_type == "RAND":
            model = RandomBaseline(
                k=5, T=num_queries, real_graph=G_full
            )
            seeds, explored_graph, sigma = model.run(G_full)

        nodes_sum += len(explored_graph.nodes())
        sigma_sum += sigma
    
    avg_nodes = nodes_sum / MC_SIM
    avg_sigma = sigma_sum / MC_SIM
    
    return avg_nodes, avg_sigma

def main():
    
    print("starting influence maximization experiments...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device: {device}")

    print("\nloading dataset...")
    G_full, real_node_features = coauthor_data(COAUTHOR_DATASET)
    
    # NODE METADATA IMPUTATION
    try:
        # matrix N_nodes x M_features 
        all_features_np = np.stack([real_node_features[i] for i in range(len(real_node_features))])
        all_features_tensor = torch.from_numpy(all_features_np).float()
    except Exception as e:
        print(f"error: cannot stack features {e}")
        print("check that 'coauthor_data' returns an ordered dict 0..N-1.")
        return

    gsm_model = AutoencoderGSM(FEATURE_DIM, GSM_LATENT_DIM).to(device)
    
    if not os.path.exists(GSM_MODEL_PATH):
        print(f"cannot find pre-trained GSM at '{GSM_MODEL_PATH}'.")
        # GSM training
        train_gsm_model(
            full_features=all_features_tensor,
            input_dim=FEATURE_DIM,
            latent_dim=GSM_LATENT_DIM,
            epochs=GSM_EPOCHS,
            batch_size=GSM_BATCH_SIZE,
            corruption_rate=CORRUPTION_RATE,
            device=device,
            save_path=GSM_MODEL_PATH
        )
    else:
        print(f"found pre-trained GSM, loading from '{GSM_MODEL_PATH}'.")
        gsm_model.load_state_dict(torch.load(GSM_MODEL_PATH, map_location=device))

    # imputation
    node_features_DIRTY, node_MASKS = create_dirty_features(real_node_features, CORRUPTION_RATE)
    node_features_RECONSTRUCTED = reconstruct_features(
        node_features_DIRTY,
        node_MASKS,
        gsm_model, 
        device
    )

    
    # MONTECARLO experiment
    G_real = forest_fire_sample(G_full, target_size=3000, p_forward=0.25)
    print(f"forest fire: {len(G_real.nodes())} nodes and {len(G_real.edges())} edges")
    
    try:
        with open(results_file_name, 'w', encoding='utf-8') as f:
            f.write("influence maximization experiment results\n")
            f.write("=========================================\n\n")
            
            for budget in budgets_to_test:
                print(f"\n--- executing budget: {budget} queries ---")
                
                # IM-META
                nodes_immeta, sigma_immeta = run_experiment(
                    "IMMETA",
                    G_real, 
                    node_features_RECONSTRUCTED, 
                    FEATURE_DIM, 
                    budget
                )
                
                # RAND
                nodes_rand, sigma_rand = run_experiment(
                    "RAND",
                    G_real, 
                    None,
                    FEATURE_DIM, 
                    budget
                )
                
                f.write(f"[budget = {budget}]\n")
                f.write(f"  IM-META -> Nodes: {nodes_immeta}, Sigma: {sigma_immeta}\n")
                f.write(f"  RAND    -> Nodes: {nodes_rand}, Sigma: {sigma_rand}\n\n")
                
                print(f"Result: IM-META Sigma: {sigma_immeta} vs RAND Sigma: {sigma_rand}")

    except IOError as e:
        print(f"error: could not write to file '{results_file_name}'. details: {e}")
    except Exception as e:
        print(f"an unexpected error occurred: {e}")


if __name__ == "__main__":
    main()