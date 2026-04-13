import torch
import numpy as np
import networkx as nx
import random
import copy
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

from immeta.network_inference import NetworkInference
from immeta.coauthor_data import coauthor_data

# config
TRAINING = True
MODEL_NAME = "gsm_autoencoder_CS.pth" 
DATASET_NAME = "Physics" 
THRESHOLD = 0.5
N_TRAIN_EDGES = 1000  
N_TEST_NEGATIVES = 20000 
RANDOM_SEED = 42

# setup reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

# load data
print(f"loading dataset {DATASET_NAME}...")
G_full, node_features = coauthor_data(DATASET_NAME)
full_edges = list(G_full.edges())
print(f"G_full: {len(G_full.nodes())} nodes, {len(full_edges)} edges")

feature_dim = len(node_features[0])
network_inference = NetworkInference(feature_dim=feature_dim)

# shuffle and split dataset
random.shuffle(full_edges)

if TRAINING:
    # use limited positive edges for training to simulate exploration
    train_edges = full_edges[:N_TRAIN_EDGES]
    
    G_train = nx.Graph()
    G_train.add_nodes_from(G_full.nodes())
    G_train.add_edges_from(train_edges)
    
    print(f"\n--- training phase ---")
    print(f"training on strict subset of {len(train_edges)} edges...")
    
    # unbalanced training data natively handled by the network inference class
    network_inference.train(node_features, G_train, epochs=10, batch_size=64)
    network_inference.save_model_checkpoint()
    print("training complete & model saved.")

else:
    print(f"\n--- inference phase ---")
    # load latest checkpoint
    try:
        import os
        checkpoints = sorted(os.listdir("./checkpoints/"))
        if checkpoints:
            latest_model = checkpoints[-1].replace(".pth", "")
            print(f"loading latest model: {latest_model}")
            network_inference.load_model_checkpoint(latest_model)
        else:
            print("no checkpoints found.")
    except Exception as e:
        print(f"error loading model: {e}")

# prepare test evaluation
print("\n--- evaluation phase ---")

# positive tests (excluding training set)
train_edges_set = set(full_edges[:N_TRAIN_EDGES])
test_edges_pos = [e for e in full_edges if e not in train_edges_set]

# negative tests (non-existent edges)
print(f"generating {N_TEST_NEGATIVES} negative edges for testing...")
test_edges_neg = []
nodes_list = list(G_full.nodes())
while len(test_edges_neg) < N_TEST_NEGATIVES:
    u, v = random.sample(nodes_list, 2)
    if not G_full.has_edge(u, v):
        test_edges_neg.append((u, v))

print(f"test set size -> positive: {len(test_edges_pos)}, negative: {len(test_edges_neg)}")

# batch prediction
print("running batch prediction...")

all_test_pairs = test_edges_pos + test_edges_neg
y_true = [1] * len(test_edges_pos) + [0] * len(test_edges_neg)

probs_dict = network_inference.predict_edge_probabilities(node_features, all_test_pairs)

y_scores = [probs_dict.get(pair, 0.0) for pair in all_test_pairs]
y_pred = [1 if score >= THRESHOLD else 0 for score in y_scores]

# metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
try:
    auc = roc_auc_score(y_true, y_scores)
except:
    auc = 0.5

fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)

print("\nresults:")
print(f"accuracy:  {acc:.4f}")
print(f"auc:       {auc:.4f}")
print(f"precision: {prec:.4f}")
print(f"recall:    {rec:.4f}")
print("-" * 20)
print(f"false positives: {fp} / {len(test_edges_neg)} ({fp/len(test_edges_neg)*100:.2f}%)")
print(f"false negatives: {fn} / {len(test_edges_pos)} ({fn/len(test_edges_pos)*100:.2f}%)")
print(f"true positives:  {tp}")
print(f"true negatives:  {tn}")