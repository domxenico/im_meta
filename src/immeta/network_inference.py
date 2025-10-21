import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Set
import random
import time
import os


from .siamese_network import SiameseNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

CHECKPOINT_PATH = "./checkpoints/"


class NetworkInference:
    def __init__(self, feature_dim: int, embedding_dim: int = 256, learning_rate: float = 0.001):
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.model = SiameseNetwork(feature_dim, embedding_dim=embedding_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.criterion = nn.BCELoss()
        
    def create_training_pairs(self, node_features: Dict[int, np.ndarray], explored_graph: nx.Graph):
        """creates balanced training pairs"""
        
        positive_pairs = []
        for u, v in explored_graph.edges():
            if u in node_features and v in node_features:
                positive_pairs.append((u, v))
        
        # negative pairs (non-edges)
        explored_nodes = list(explored_graph.nodes())
        negative_pairs = []
        
        # max_negatives = min(len(positive_pairs) * 2, len(explored_nodes) * (len(explored_nodes) - 1) // 2 - len(positive_pairs))
        max_negatives = len(positive_pairs)
        attempts = 0
        max_attempts = max_negatives * 20 
        # print(f"max attempts {max_attempts} max negatives {max_negatives}")
        while len(negative_pairs) < max_negatives and attempts < max_attempts:
            u, v = random.sample(explored_nodes, 2)
            if not explored_graph.has_edge(u, v) and (u, v) not in negative_pairs and (v, u) not in negative_pairs:
                negative_pairs.append((u, v))
            attempts += 1
        
        if len(negative_pairs) > len(positive_pairs):
            negative_pairs = random.sample(negative_pairs, len(positive_pairs))
        
        return positive_pairs, negative_pairs
    
    def train(self, node_features: Dict[int, np.ndarray], explored_graph: nx.Graph, 
              epochs: int = 20, batch_size: int = 32):
        
        """training with current graph"""
        self.model.train()
        
        positive_pairs, negative_pairs = self.create_training_pairs(node_features, explored_graph)
        
        print(f"training with {len(positive_pairs)} edges and {len(negative_pairs)} non-edges")
        
        if len(positive_pairs) < 20:
            print("too few training pairs, skipping training")
            return
        
        for epoch in range(epochs):
            total_loss = 0
            total_batches = 0
            
            all_pairs = positive_pairs + negative_pairs
            labels = [1.0] * len(positive_pairs) + [0.0] * len(negative_pairs)
            combined = list(zip(all_pairs, labels))
            random.shuffle(combined)
            
            for i in range(0, len(combined), batch_size):
                batch = combined[i:i+batch_size]
                if len(batch) < 2:
                    continue
                
                batch_x1 = []
                batch_x2 = []
                batch_labels = []
                
                for (u, v), label in batch:
                    if u in node_features and v in node_features:
                        batch_x1.append(node_features[u])
                        batch_x2.append(node_features[v])
                        batch_labels.append(label)
                
                if not batch_x1:
                    continue
                
                # convert to tensors
                batch_x1_tensor = torch.tensor(np.array(batch_x1), dtype=torch.float32).to(device)
                batch_x2_tensor = torch.tensor(np.array(batch_x2), dtype=torch.float32).to(device)
                batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.float32).to(device)
                
                self.optimizer.zero_grad()
                predictions = self.model(batch_x1_tensor, batch_x2_tensor)
                loss = self.criterion(predictions, batch_labels_tensor)
                loss.backward()
                
                # gradient clipping to prevent explosions
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                total_batches += 1
            
            if total_batches > 0:
                avg_loss = total_loss / total_batches
                print(f"epoch {epoch+1}/{epochs}, loss: {avg_loss:.4f}")
                
                # early stopping if loss becomes NaN
                if np.isnan(avg_loss):
                    print("loss became NaN, stopping training")
                    break

    def predict_edge_probabilities(self, node_features: Dict[int, np.ndarray], 
                                  node_pairs: List[Tuple[int, int]]) -> Dict[Tuple[int, int], float]:
        """predict probabilities for node pairs"""
        self.model.eval()
        probabilities = {}
        
        with torch.no_grad():
            batch_size = 1024
            for i in range(0, len(node_pairs), batch_size):
                batch_pairs = node_pairs[i:i+batch_size]
                
                batch_x1 = []
                batch_x2 = []
                valid_pairs = []
                
                for u, v in batch_pairs:
                    if u in node_features and v in node_features:
                        batch_x1.append(node_features[u])
                        batch_x2.append(node_features[v])
                        valid_pairs.append((u, v))
                
                if batch_x1:
                    batch_x1_tensor = torch.tensor(np.array(batch_x1), dtype=torch.float32).to(device)
                    batch_x2_tensor = torch.tensor(np.array(batch_x2), dtype=torch.float32).to(device)
                    
                    batch_probs = self.model(batch_x1_tensor, batch_x2_tensor).cpu().numpy()
                    
                    for (u, v), prob in zip(valid_pairs, batch_probs):
                        probabilities[(u, v)] = float(prob)
        
        return probabilities
    
    def save_model_checkpoint(self):
        """saves a .pth model checkpoint to the CHECKPOINT_PATH"""

        time_string = time.strftime("%Y%m%d%H%M%S", time.localtime())
        torch.save(self.model.state_dict(), f"{CHECKPOINT_PATH}{time_string}.pth")

    def load_model_checkpoint(self, time_string):
        """loads a .pth model checkpoint from the CHECKPOINT_PATH"""

        file_path = f"{CHECKPOINT_PATH}{time_string}.pth" # Assumendo un'estensione .pth
    
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Checkpoint file not found: {file_path}")

        state_dict = torch.load(file_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)