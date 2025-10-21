from immeta.network_inference import NetworkInference
from immeta.coauthor_data import coauthor_data

import torch
import random
import copy

# This script is used to test if the model is well posed
# observing if it understands the correlation between the nodes metadata
# and if this is significative

# Issue:
# if we do not save the indices of the randomly selected test nodes
# during the training run, the subsequent run (when importing the model)
# will likely perform its test evaluation on samples that were
# included in the training set.


TRAINING = True # if True we start a training on the whole G_full graph
                # otherwise we infer with the model indicated with MODEL_NAME  
MODEL_NAME = "20251007211746"
THRESHOLD = 0.55 # similarity threshold to determine two nodes connected


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("loading dataset...")
G_full, node_features = coauthor_data("Physics") # CS or Physics

feature_dim = (node_features[10]).__len__()
network_inference = NetworkInference(feature_dim=feature_dim)

# test set
test_nodes = random.sample(list(G_full.nodes()), 1000)
test_edges = G_full.edges(test_nodes) 

print(f"G_full has {len(G_full.nodes())} nodes and {len(G_full.edges())} edges")

if TRAINING:
    # training set
    G_train = copy.deepcopy(G_full)
    G_train.remove_nodes_from(test_nodes)
    print(f"G_train has {len(G_train.nodes())} nodes and {len(G_train.edges())} edges")

    print("starting training")
    network_inference.train(node_features, G_train, epochs=10)  # this is the whole training procedure
    network_inference.save_model_checkpoint()
    print("training complete, saved checkpoint")

else:
    network_inference.load_model_checkpoint(MODEL_NAME)
    print("model imported successfully")

# we take randomly some certain edges from the real graph
# and we check it with the model prediction

# we need to construct a false edges set, 
# a list of pairs of nodes
# idea: for each node we can take his neighborhood and 
# take all the nodes except his neighborhood


N_non_edges = len(test_edges)
non_edges = []

for u_ in test_nodes:
    neigborhood = G_full.neighbors(u_)

    for partial_iterations in range(0, N_non_edges//len(test_nodes)):
        v_ = random.choice(test_nodes)
        if G_full.has_edge(u_, v_) != True:
            non_edges.append((u_,v_))
# print(f"non edges test len: {len(non_edges)}")            


fn = 0
for pair in test_edges:
    u = pair[0]
    v = pair[1]

    u_features = torch.tensor(node_features[u], dtype=torch.float32).to(device)
    v_features = torch.tensor(node_features[v], dtype=torch.float32).to(device)

    pred = network_inference.model(u_features,v_features)
    if pred < THRESHOLD:
        fn += 1
    # print(f"edge ({u}, {v}), pred: {pred}")

fp = 0
for pair in non_edges:
    u = pair[0]
    v = pair[1]

    u_features = torch.tensor(node_features[u], dtype=torch.float32).to(device)
    v_features = torch.tensor(node_features[v], dtype=torch.float32).to(device)

    pred = network_inference.model(u_features,v_features)
    if pred > THRESHOLD:
        fp += 1
    # print(f"edge ({u}, {v}), pred: {pred}")

print(f"\nin this examples the model had {(fp/len(non_edges))*100}% false positives")
print(f"and {(fn/len(test_edges))*100}% false negatives")