import torch
import numpy as np
import networkx as nx
import random
import copy
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

from immeta.network_inference import NetworkInference
from immeta.coauthor_data import coauthor_data

# --- CONFIGURAZIONE ---
TRAINING = True
MODEL_NAME = "gsm_autoencoder_CS.pth" 
DATASET_NAME = "Physics" 
THRESHOLD = 0.5
N_TRAIN_EDGES = 1000  
N_TEST_NEGATIVES = 20000 
RANDOM_SEED = 42

# 1. SETUP E RIPRODUCIBILITÀ
# Impostando il seed, garantiamo che lo split sia identico ogni volta 
# senza dover salvare liste di indici su file.
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. CARICAMENTO DATI
print(f"Loading dataset {DATASET_NAME}...")
G_full, node_features = coauthor_data(DATASET_NAME)
full_edges = list(G_full.edges())
print(f"G_full: {len(G_full.nodes())} nodes, {len(full_edges)} edges")

feature_dim = len(node_features[0])
network_inference = NetworkInference(feature_dim=feature_dim)

# 3. CREAZIONE DATASET (Split)
# Mischiamo gli archi
random.shuffle(full_edges)

if TRAINING:
    # A. Training Set: Prendiamo SOLO pochi archi positivi
    train_edges = full_edges[:N_TRAIN_EDGES]
    
    # Costruiamo il grafo di training parziale
    G_train = nx.Graph()
    G_train.add_nodes_from(G_full.nodes()) # I nodi ci sono tutti (per le feature)
    G_train.add_edges_from(train_edges)    # Ma conosciamo solo pochi archi
    
    print(f"\n--- TRAINING PHASE ---")
    print(f"Training on strict subset of {len(train_edges)} edges (simulating exploration)...")
    
    # La classe NetworkInference internamente gestisce già il bilanciamento 
    # creando coppie negative (non-link) basate sul grafo passato (G_train).
    network_inference.train(node_features, G_train, epochs=10, batch_size=64)
    network_inference.save_model_checkpoint()
    print("Training complete & model saved.")

else:
    print(f"\n--- INFERENCE PHASE ---")
    # Carica l'ultimo checkpoint disponibile o uno specifico se necessario
    # Nota: la funzione load cerca nella cartella checkpoints
    # Modifica qui se hai un nome file specifico timestampato
    try:
        # Trova il file più recente se non specificato
        import os
        checkpoints = sorted(os.listdir("./checkpoints/"))
        if checkpoints:
            latest_model = checkpoints[-1].replace(".pth", "")
            print(f"Loading latest model: {latest_model}")
            network_inference.load_model_checkpoint(latest_model)
        else:
            print("No checkpoints found.")
    except Exception as e:
        print(f"Error loading model: {e}")

# 4. PREPARAZIONE TEST SET (Valutazione Globale)
print("\n--- EVALUATION PHASE ---")

# A. Test Positivi: TUTTI gli archi reali che NON erano nel training
# Se TRAINING=False, dobbiamo comunque rigenerare lo split col seed per sapere quali escludere
train_edges_set = set(full_edges[:N_TRAIN_EDGES])
test_edges_pos = [e for e in full_edges if e not in train_edges_set]

# B. Test Negativi: Generiamo archi che NON esistono nel grafo reale
# Ne generiamo un numero significativo per avere statistica
print(f"Generating {N_TEST_NEGATIVES} negative edges for testing...")
test_edges_neg = []
nodes_list = list(G_full.nodes())
while len(test_edges_neg) < N_TEST_NEGATIVES:
    u, v = random.sample(nodes_list, 2)
    if not G_full.has_edge(u, v):
        test_edges_neg.append((u, v))

print(f"Test set size -> Positive: {len(test_edges_pos)}, Negative: {len(test_edges_neg)}")

# 5. INFERENZA A BATCH (Molto più veloce)
print("Running batch prediction...")

# Uniamo tutto per fare un'unica passata (o due batch giganti)
# Nota: predict_edge_probabilities gestisce internamente il batching (es. a blocchi di 1024)
all_test_pairs = test_edges_pos + test_edges_neg
y_true = [1] * len(test_edges_pos) + [0] * len(test_edges_neg)

# Usiamo la funzione ottimizzata della classe
probs_dict = network_inference.predict_edge_probabilities(node_features, all_test_pairs)

# Estraiamo le predizioni ordinate
y_scores = [probs_dict.get(pair, 0.0) for pair in all_test_pairs]
y_pred = [1 if score >= THRESHOLD else 0 for score in y_scores]

# 6. CALCOLO METRICHE
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
try:
    auc = roc_auc_score(y_true, y_scores)
except:
    auc = 0.5

# Matrice di confusione manuale per debugging rapido
fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)

print("\nResults:")
print(f"Accuracy:  {acc:.4f}")
print(f"AUC:       {auc:.4f} (Ability to rank real edges higher than fake ones)")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print("-" * 20)
print(f"False Positives: {fp} / {len(test_edges_neg)} ({fp/len(test_edges_neg)*100:.2f}%)")
print(f"False Negatives: {fn} / {len(test_edges_pos)} ({fn/len(test_edges_pos)*100:.2f}%)")
print(f"True Positives:  {tp}")
print(f"True Negatives:  {tn}")