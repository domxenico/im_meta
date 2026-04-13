import numpy as np
import random
import networkx as nx
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import torch
import csv

from immeta.coauthor_data import coauthor_data
from immeta.network_inference import NetworkInference

# test parameters
DATASET = "CS" # or "Physics"
EPOCHS = 5 # reduced for speed
TRAINING_SIZES = [0.01] # edge percentages for training
OUTPUT_FILE = "results_siamese.txt"

def run_experiment_siamese(G_full, node_features, training_split):
    """runs siamese network training and computes auc on test set"""
    print(f"\n--- running siamese experiment with {training_split*100:.4f}% of edges ---")

    # split edges
    all_edges = list(G_full.edges())
    random.shuffle(all_edges)
    split_idx = int(training_split * len(all_edges))
    train_edges = all_edges[:split_idx]
    test_edges = all_edges[split_idx:]

    if not train_edges or not test_edges:
        print("error: training/test set size is zero. skipping.")
        return None, None

    G_train = nx.Graph()
    G_train.add_nodes_from(G_full.nodes())
    G_train.add_edges_from(train_edges)

    print(f"training edges: {len(train_edges)}, test edges: {len(test_edges)}")

    feature_dim = len(next(iter(node_features.values())))
    network_inference = NetworkInference(feature_dim)
    network_inference.train(node_features, G_train, epochs=EPOCHS)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare negative pairs
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

    y_score = []
    
    # disable autograd
    with torch.no_grad():
        for u, v in test_couples:
            # format features
            u_features = torch.tensor(node_features[u], dtype=torch.float32).to(device)
            v_features = torch.tensor(node_features[v], dtype=torch.float32).to(device)
            
            pred_tensor = network_inference.model(u_features, v_features)
            y_score.append(pred_tensor.item())

    # compute auc
    auc = roc_auc_score(y_true, y_score)
    print(f"auc result: {auc:.4f}")

    return len(train_edges), auc

print("loading dataset...")
G_full, node_features = coauthor_data(DATASET)
print(f"dataset {DATASET} loaded. {len(G_full.nodes())} nodes, {len(G_full.edges())} edges.")

results = []
for split in TRAINING_SIZES:
    num_edges, auc_score = run_experiment_siamese(G_full, node_features, split)
    if num_edges is not None:
        results.append((num_edges, auc_score))

# save results
if results:
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["num_training_edges", "auc_score"])
        writer.writerows(results)
    print(f"\nresults saved to: {OUTPUT_FILE}")