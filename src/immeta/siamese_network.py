import torch
from torch import nn

class SiameseNetwork(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, embedding_dim),
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        emb_u = self.encoder(x1)
        emb_v = self.encoder(x2)
        
        hadamard_product = emb_u * emb_v
        edge_prob = self.predictor(hadamard_product)
        
        return edge_prob.squeeze(-1)