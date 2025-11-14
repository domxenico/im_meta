from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class AutoencoderGSM(nn.Module):
    """
    Un semplice Autoencoder MLP che funge da Generative Surrogate Model (GSM)
    per l'imputazione di features.
    """
    def __init__(self, input_dim: int, latent_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

class MaskedFeatureDataset(Dataset):
    """
    Dataset che prende le features pulite e restituisce
    (features_sporche, features_pulite) ad ogni iterazione.
    """
    def __init__(self, features: torch.Tensor, indices: torch.Tensor, corruption_rate: float = 0.3):
        self.features = features
        self.indices = indices
        self.corruption_rate = corruption_rate
        
        if not (0 < corruption_rate < 1):
            raise ValueError("Corruption rate must be between 0 and 1")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        node_idx = self.indices[idx]
        x_clean = self.features[node_idx]
        
        # 1.0 = "osservato", 0.0 = "mancante"
        mask = (torch.rand_like(x_clean) > self.corruption_rate).float()
        x_dirty = x_clean * mask  # Applica la maschera
        
        return x_dirty, x_clean

def create_splits(num_nodes: int, train_p: float = 0.7, val_p: float = 0.1):
    """Crea indici casuali per train, validation e test."""
    indices = torch.randperm(num_nodes)
    train_size = int(num_nodes * train_p)
    val_size = int(num_nodes * val_p)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    return train_idx, val_idx, test_idx

def train_gsm_model(
    full_features: torch.Tensor,
    input_dim: int,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    corruption_rate: float,
    device: torch.device,
    save_path: str
):
    """Funzione completa per addestrare e salvare il GSM."""
    
    print("\n--- Inizio Addestramento GSM (Autoencoder) ---")
    
    # 1. Splits
    num_nodes = full_features.shape[0]
    train_idx, val_idx, _ = create_splits(num_nodes)
    print(f"  GSM Splits -> Train: {len(train_idx)}, Val: {len(val_idx)}")
    
    # 2. DataLoaders
    train_dataset = MaskedFeatureDataset(full_features, train_idx, corruption_rate)
    val_dataset = MaskedFeatureDataset(full_features, val_idx, corruption_rate)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Modello, Loss, Optimizer
    model = AutoencoderGSM(input_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss() # Ideale per BoW (dati binari)

    # 4. Loop di Addestramento
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_dirty, x_clean in train_loader:
            x_dirty, x_clean = x_dirty.to(device), x_clean.to(device)
            
            x_reconstructed = model(x_dirty)
            loss = criterion(x_reconstructed, x_clean)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)

        # Validazione
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_dirty, x_clean in val_loader:
                x_dirty, x_clean = x_dirty.to(device), x_clean.to(device)
                x_reconstructed = model(x_dirty)
                loss = criterion(x_reconstructed, x_clean)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # 5. Salva Modello
    torch.save(model.state_dict(), save_path)
    print(f"  Addestramento completato. Modello salvato in '{save_path}'")
    print("--- Fine Addestramento GSM ---")
    return model
