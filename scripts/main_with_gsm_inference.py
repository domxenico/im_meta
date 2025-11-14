import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx
from typing import Dict
from collections import deque
import random
import torch
import torch.nn as nn

# --- Import del tuo framework ---
# (Assumo che 'immeta' sia installato e 'coauthor_data' sia lì)
from immeta import IMMETA, coauthor_data

# --- MODIFICA: Definizione del modello GSM (Autoencoder) ---
# Questa è la stessa classe Autoencoder che abbiamo addestrato
class AutoencoderGSM(nn.Module):
    """
    Un semplice Autoencoder MLP che funge da Generative Surrogate Model (GSM)
    per l'imputazione di features.
    """
    def __init__(self, input_dim: int, latent_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        
        # Encoder: riduce la dimensionalità
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.ReLU()
        )
        
        # Decoder: ricostruisce l'input originale
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# --- Parametri dell'esperimento ---
mc_sim = 1
coauthor_dataset = "CS"
budgets_to_test = [20]
results_file_name = "results_log.txt"

# --- MODIFICA: Parametri del GSM ---
# Devono corrispondere a quelli usati per l'addestramento
FEATURE_DIM_CS = 6805
LATENT_DIM = 256
GSM_MODEL_PATH = f"gsm_autoencoder_{coauthor_dataset}.pth"
CORRUPTION_RATE = 0.3 # Percentuale di dati da "sporcare"


def forest_fire_sample(G_full, target_size=3000, p_forward=0.7, p_backward=0.7):
    """
    Forest Fire sampling... (invariato)
    """
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

# --- MODIFICA: Funzione per "sporcare" i dati ---
def create_dirty_features(
    original_features: Dict[int, np.ndarray], 
    corruption_rate: float
) -> Dict[int, np.ndarray]:
    """
    Crea una copia dei metadati con feature mancanti (impostate a 0).
    """
    print(f"  Simulazione dati parziali (corruption rate: {corruption_rate})")
    dirty_features = {}
    for node_id, features_np in original_features.items():
        x_clean = torch.from_numpy(features_np).float()
        
        # 1.0 = "osservato", 0.0 = "mancante"
        mask = (torch.rand_like(x_clean) > corruption_rate).float()
        
        x_dirty = x_clean * mask  # Applica la maschera
        dirty_features[node_id] = x_dirty.numpy()
        
    return dirty_features

# --- MODIFICA: Funzione per ricostruire i dati con il GSM ---
def reconstruct_features(
    dirty_features: Dict[int, np.ndarray], 
    model: AutoencoderGSM, 
    device: torch.device
) -> Dict[int, np.ndarray]:
    """
    Usa il modello GSM addestrato per imputare i metadati mancanti.
    """
    print("  Ricostruzione metadati con GSM (Autoencoder)...")
    reconstructed_features = {}
    model.eval() # Modalità inferenza
    
    with torch.no_grad():
        for node_id, features_np in dirty_features.items():
            # Prepara il tensore di input
            x_dirty_tensor = torch.from_numpy(features_np).float()
            x_dirty_tensor = x_dirty_tensor.unsqueeze(0).to(device) # Aggiungi batch dim
            
            # Passaggio nel modello
            x_reconstructed_logits = model(x_dirty_tensor)
            
            # Ottieni le probabilità (il nostro metadato "soft" ricostruito)
            x_reconstructed_probs = torch.sigmoid(x_reconstructed_logits)
            
            # Riporta a NumPy
            reconstructed_array = x_reconstructed_probs.squeeze(0).cpu().numpy()
            reconstructed_features[node_id] = reconstructed_array
            
    print("  Ricostruzione completata.")
    return reconstructed_features


def run_experiment(G_full, node_features, feature_dim, num_queries):
    """
    Esegue la simulazione (invariata)
    """
    print(f"  starting simulation (budget={num_queries})...")
    
    nodes_sum = 0
    edges_sum = 0
    sigma_sum = 0
    
    for mc in range(mc_sim):
        im_meta = IMMETA(
            feature_dim=feature_dim,
            k=5,           
            T=num_queries, 
            alpha=1.0,     
            threshold=0.5, 
            diffusion_model='IC',
            real_graph=G_full
        )
        
        # im_meta.run ora riceverà i dati ricostruiti
        seeds, explored_graph, sigma = im_meta.run(G_full, node_features)
        
        nodes_sum += len(explored_graph.nodes())
        edges_sum += len(explored_graph.edges())
        sigma_sum += sigma
        print(f"sigma {sigma}")
    
    avg_nodes = nodes_sum / mc_sim
    avg_sigma = sigma_sum / mc_sim
    
    print(f"  average discovered nodes: {avg_nodes}")
    print(f"  average sigma: {avg_sigma}")
    
    return avg_nodes, avg_sigma

def main():
    
    print("starting influence maximization experiments...")
    
    # --- MODIFICA: Setup device e dimensioni ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if coauthor_dataset == 'CS':
        feature_dim = FEATURE_DIM_CS
    else:
        # Aggiungi 'Physics' se necessario
        raise ValueError("Dataset non supportato per il GSM")
    
    print("\nloading dataset...")
    g_real, node_features_CLEAN = coauthor_data(coauthor_dataset) # <--- MODIFICA
    
    # --- MODIFICA: Fase di Imputazione ---
    # 1. "Sporca" i dati per simulare la realtà
    node_features_DIRTY = create_dirty_features(node_features_CLEAN, CORRUPTION_RATE)
    
    # 2. Carica il modello GSM addestrato
    try:
        gsm_model = AutoencoderGSM(feature_dim, LATENT_DIM)
        gsm_model.load_state_dict(torch.load(GSM_MODEL_PATH, map_location=device))
        gsm_model.to(device)
        print(f"Modello GSM caricato da '{GSM_MODEL_PATH}'")
    except FileNotFoundError:
        print(f"ERRORE: File modello non trovato: '{GSM_MODEL_PATH}'")
        print("Esegui prima lo script di addestramento del GSM.")
        return
    except Exception as e:
        print(f"Errore durante il caricamento del modello: {e}")
        return

    # 3. Ricostruisci i dati
    node_features_RECONSTRUCTED = reconstruct_features(
        node_features_DIRTY, 
        gsm_model, 
        device
    )
    # --- Fine Fase Imputazione ---

    g_full = forest_fire_sample(g_real, target_size=3000, p_forward=0.35, p_backward=0.3)

    print(f"forest fire: {len(g_full.nodes())} nodes and {len(g_full.edges())} edges")
    # feature_dim è già definita sopra
    
    try:
        with open(results_file_name, 'w', encoding='utf-8') as f:
            f.write("influence maximization experiment results (CON IMPUTAZIONE GSM)\n") # <--- MODIFICA
            f.write("=========================================\n\n")
            
            for budget in budgets_to_test:
                print(f"\n--- executing budget: {budget} queries ---")
                
                # <--- MODIFICA: Passa i dati ricostruiti al framework
                discovered_nodes, obtained_sigma = run_experiment(
                    g_full, 
                    node_features_RECONSTRUCTED, # Non più node_features_CLEAN
                    feature_dim, 
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