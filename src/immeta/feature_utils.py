import numpy as np
import torch
from typing import Dict

from immeta.gsm import AutoencoderGSM


def create_dirty_features( original_features: Dict[int, np.ndarray], 
                          corruption_rate: float) -> Dict[int, np.ndarray]:
    
    """ creates a metadata copy with some missing features (at the momemnt missing means = 0) 
    idea: represent missing features with 0.5"""

    print(f"  simulating with partial data (corruption rate: {corruption_rate})")
    dirty_features = {}
    masks = {}
    for node_id, features_np in original_features.items():
        
        x_clean = torch.from_numpy(features_np).float()
        mask = (torch.rand_like(x_clean) > corruption_rate).float()
        
        dirty_features[node_id] = (x_clean * mask).numpy()
        masks[node_id] = mask.numpy()
    return dirty_features, masks

def reconstruct_features(dirty_features, masks, model, device):
    reconstructed_features = {}
    model.eval()
    with torch.no_grad():
        for node_id, features_np in dirty_features.items():
            # conversion in tensors
            
            # if the node has at least 1 masked feature, we reconstruct
            if (masks[node_id] == 0).any():
                
                x_dirty_tensor = torch.from_numpy(features_np).float().unsqueeze(0).to(device)
                mask_tensor = torch.from_numpy(masks[node_id]).float().unsqueeze(0).to(device) 
                
                #concatenate input and mask
                combined_input = torch.cat([x_dirty_tensor, mask_tensor], dim=1) 

                # gsm takes the combined input, returns reconstructed features
                x_reconstructed_logits = model(combined_input)
                x_reconstructed_probs = torch.sigmoid(x_reconstructed_logits)
                reconstructed_features[node_id] = x_reconstructed_probs.squeeze(0).cpu().numpy()
                
            else:
                print(f"found a non corrupted array")
                reconstructed_features[node_id] = features_np

    return reconstructed_features
