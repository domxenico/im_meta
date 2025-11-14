import numpy as np
import random
import networkx as nx
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import torch
import csv

# --- Assunzioni sul Codice Esterno ---
# Assumiamo che queste classi siano definite e accessibili, 
# e che NetworkInference.train addestri la rete siamese.
from immeta.coauthor_data import coauthor_data
from immeta.network_inference import NetworkInference

# --- Parametri di Test ---
DATASET = "CS" # o "Physics"
EPOCHS = 5 # Ridotto per velocizzare il test, aumentare per risultati più accurati
TRAINING_SIZES = [0.01] # Percentuali di archi da usare per il training
OUTPUT_FILE = "results_siamese.txt" # File specifico per i risultati della rete siamese

# --- Funzione di Esecuzione per Rete Siamese ---

def run_experiment_siamese(G_full, node_features, training_split):
    """Esegue l'addestramento della rete siamese e la valutazione AUC sul test set."""
    print(f"\n--- Running Siamese Experiment with {training_split*100:.4f}% of edges ---")

    # 1. Split degli archi
    all_edges = list(G_full.edges())
    random.shuffle(all_edges)
    split_idx = int(training_split * len(all_edges))
    train_edges = all_edges[:split_idx]
    test_edges = all_edges[split_idx:]

    if not train_edges or not test_edges:
        print("Errore: La dimensione del training/test set è zero. Saltando.")
        return None, None

    G_train = nx.Graph()
    G_train.add_nodes_from(G_full.nodes())
    G_train.add_edges_from(train_edges)

    print(f"Training edges: {len(train_edges)}, Test edges: {len(test_edges)}")

    # 2. Rete Siamese Declaration and Training
    feature_dim = len(next(iter(node_features.values())))
    network_inference = NetworkInference(feature_dim)
    # L'oggetto network_inference contiene la rete siamese
    network_inference.train(node_features, G_train, epochs=EPOCHS)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Preparazione delle coppie negative (uguale al test set positivo)
    positive_pairs = test_edges
    all_nodes = list(G_full.nodes())
    negative_pairs = set()
    
    while len(negative_pairs) < len(positive_pairs):
        u, v = random.sample(all_nodes, 2)
        if not G_full.has_edge(u, v):
            negative_pairs.add(tuple(sorted((u, v))))
    negative_pairs = list(negative_pairs)
    
    test_couples = positive_pairs + negative_pairs
    y_true = [1] * len(positive_pairs) + [0] * len(negative_pairs)

    # 4. Calcolo degli Score usando il Modello Siamese
    y_score = []
    
    # Disabilita il calcolo del gradiente durante la valutazione
    with torch.no_grad():
        for u, v in test_couples:
            # Prepara le feature dei nodi come richiesto dal modello siamese
            u_features = torch.tensor(node_features[u], dtype=torch.float32).to(device)
            v_features = torch.tensor(node_features[v], dtype=torch.float32).to(device)
            
            # La rete siamese fornisce direttamente lo score di similarità (pred)
            pred_tensor = network_inference.model(u_features, v_features)
            y_score.append(pred_tensor.item())

    # 5. Calcolo AUC
    auc = roc_auc_score(y_true, y_score)
    print(f"AUC Result: {auc:.4f}")

    return len(train_edges), auc

# --- Esecuzione Principale ---
print("Caricamento dataset...")
G_full, node_features = coauthor_data(DATASET)
print(f"Dataset {DATASET} caricato. {len(G_full.nodes())} nodi e {len(G_full.edges())} archi totali.")

results = []
for split in TRAINING_SIZES:
    num_edges, auc_score = run_experiment_siamese(G_full, node_features, split)
    if num_edges is not None:
        results.append((num_edges, auc_score))

# --- Salvataggio dei Risultati ---
if results:
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["Num_Training_Edges", "AUC_Score"]) # Header
        writer.writerows(results)
    print(f"\nRisultati salvati in: {OUTPUT_FILE}")