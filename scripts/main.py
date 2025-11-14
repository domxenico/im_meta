import numpy as np
import torch
import os 

from immeta import IMMETA, coauthor_data
from immeta.gsm import AutoencoderGSM
from immeta.gsm import train_gsm_model

from immeta.feature_utils import create_dirty_features
from immeta.feature_utils import reconstruct_features
from immeta.forest_fire import forest_fire_sample

MC_SIM = 1
COAUTHOR_DATASET = "CS"
budgets_to_test = [20] # in case of various budget test add [20, desired_budget, another_desired_budget]
results_file_name = "results_log.txt"

if COAUTHOR_DATASET == 'CS':
    FEATURE_DIM = 6805
else:
    FEATURE_DIM = 8415 # Physics
    
GSM_LATENT_DIM = 256 # generative surrogate model latent dimension

GSM_MODEL_PATH = f"gsm_autoencoder_{COAUTHOR_DATASET}.pth"
CORRUPTION_RATE = 0.3 
GSM_EPOCHS = 10
GSM_BATCH_SIZE = 64

def run_experiment(G_full, node_features, feature_dim, num_queries):
    
    print(f"starting simulation (budget={num_queries})...")
    nodes_sum = 0
    sigma_sum = 0
    
    for mc in range(MC_SIM):
        im_meta = IMMETA(
            feature_dim=feature_dim,
            k=5, T=num_queries, alpha=1.0, threshold=0.5,
            diffusion_model='IC', real_graph=G_full
        )
        seeds, explored_graph, sigma = im_meta.run(G_full, node_features)
        nodes_sum += len(explored_graph.nodes())
        sigma_sum += sigma
        print(f"sigma {sigma}")
    
    avg_nodes = nodes_sum / MC_SIM
    avg_sigma = sigma_sum / MC_SIM
    
    print(f"  average discovered nodes: {avg_nodes}")
    print(f"  average sigma: {avg_sigma}")
    
    return avg_nodes, avg_sigma

def main():
    
    print("starting influence maximization experiments...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\nloading dataset...")
    g_real, node_features_CLEAN = coauthor_data(COAUTHOR_DATASET)
    
    # --- MODIFICA: Estraggo le features in un tensore per l'addestramento
    # (Assumo che node_features_CLEAN sia un dict {id: np.array})
    # (Assumo che gli ID dei nodi siano 0, 1, ..., N-1 in ordine)
    try:
        all_features_np = np.stack([node_features_CLEAN[i] for i in range(len(node_features_CLEAN))])
        all_features_tensor = torch.from_numpy(all_features_np).float()
    except Exception as e:
        print(f"Errore: non è stato possibile impilare le features. {e}")
        print("Assicurati che 'coauthor_data' restituisca un dict ordinato 0..N-1.")
        return

    # --- MODIFICA: Addestramento o Caricamento del GSM ---
    gsm_model = AutoencoderGSM(FEATURE_DIM, GSM_LATENT_DIM).to(device)
    
    if not os.path.exists(GSM_MODEL_PATH):
        print(f"Modello GSM non trovato in '{GSM_MODEL_PATH}'.")
        # Addestra il modello
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
        # Il modello è già in memoria e addestrato
    else:
        print(f"Modello GSM pre-addestrato trovato. Caricamento da '{GSM_MODEL_PATH}'.")
        # Carica il modello
        gsm_model.load_state_dict(torch.load(GSM_MODEL_PATH, map_location=device))

    # --- Fase di Imputazione (come prima) ---
    node_features_DIRTY = create_dirty_features(node_features_CLEAN, CORRUPTION_RATE)
    node_features_RECONSTRUCTED = reconstruct_features(
        node_features_DIRTY, 
        gsm_model, 
        device
    )
    
    # --- Esecuzione Esperimento (come prima) ---
    g_full = forest_fire_sample(g_real, target_size=3000, p_forward=0.35, p_backward=0.3)
    print(f"forest fire: {len(g_full.nodes())} nodes and {len(g_full.edges())} edges")
    
    try:
        with open(results_file_name, 'w', encoding='utf-8') as f:
            f.write("influence maximization experiment results (CON IMPUTAZIONE GSM)\n")
            f.write("=========================================\n\n")
            
            for budget in budgets_to_test:
                print(f"\n--- executing budget: {budget} queries ---")
                
                discovered_nodes, obtained_sigma = run_experiment(
                    g_full, 
                    node_features_RECONSTRUCTED, # Usa i dati ricostruiti
                    FEATURE_DIM, 
                    budget
                )
                
                f.write(f"[budget = {budget}]\n")
                f.write(f"  discovered nodes (avg): {discovered_nodes}\n")
                f.write(f"  obtained sigma (avg): {obtained_sigma}\n\n")
                print(f"sigma: {obtained_sigma}")
                
            print(f"\n--- experiments completed ---")
            print(f"all results saved to '{results_file_name}'.")

    except IOError as e:
        print(f"error: could not write to file '{results_file_name}'. details: {e}")
    except Exception as e:
        print(f"an unexpected error occurred: {e}")


if __name__ == "__main__":
    main()