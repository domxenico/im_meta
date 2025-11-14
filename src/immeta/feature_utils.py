import numpy as np
import torch
from typing import Dict

from immeta.gsm import AutoencoderGSM


def create_dirty_features( original_features: Dict[int, np.ndarray], 
                          corruption_rate: float) -> Dict[int, np.ndarray]:
    
    """ Crea una copia dei metadati con feature mancanti (impostate a 0). """

    print(f"  Simulazione dati parziali (corruption rate: {corruption_rate})")
    dirty_features = {}
    for node_id, features_np in original_features.items():
        x_clean = torch.from_numpy(features_np).float()
        mask = (torch.rand_like(x_clean) > corruption_rate).float()
        x_dirty = x_clean * mask
        dirty_features[node_id] = x_dirty.numpy()
    return dirty_features

def reconstruct_features(
    dirty_features: Dict[int, np.ndarray], 
    model: AutoencoderGSM, 
    device: torch.device
) -> Dict[int, np.ndarray]:
    """ Usa il modello GSM addestrato per imputare i metadati mancanti. """
    print("  Ricostruzione metadati con GSM (Autoencoder)...")
    reconstructed_features = {}
    model.eval() # Modalità inferenza
    
    with torch.no_grad():
        for node_id, features_np in dirty_features.items():
            x_dirty_tensor = torch.from_numpy(features_np).float()
            x_dirty_tensor = x_dirty_tensor.unsqueeze(0).to(device)
            x_reconstructed_logits = model(x_dirty_tensor)
            x_reconstructed_probs = torch.sigmoid(x_reconstructed_logits)
            reconstructed_array = x_reconstructed_probs.squeeze(0).cpu().numpy()
            reconstructed_features[node_id] = reconstructed_array
            
    print("  Ricostruzione completata.")
    return reconstructed_features
