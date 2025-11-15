import numpy as np
import networkx as nx
from torch_geometric.datasets import Coauthor
from torch_geometric.utils import to_networkx
from typing import Dict


# edges indexes at:  data.edge_index
# node features at:  data.x

def coauthor_data(data_name):
    """
    returns a graph representing the selected dataset, node feature are 0 or 1
    """
    FEATURE_DIM = 8415 if data_name == 'Physics' else 6805

    dataset = Coauthor(root="data/Coauthor"+data_name, name=data_name)
    data = dataset[0]

    G: nx.Graph = to_networkx(
        data, 
        node_attrs=['x'], 
        to_undirected=True
    )

    is_symmetric = all( len(G.nodes[n]['x']) == FEATURE_DIM for n in G.nodes )
    if not is_symmetric:
        print("asymmetric features")

    # node features: Dict[int, np.ndarray]
    # data.x: pytorch tensor [num_nodes, num_features]

    node_features: Dict[int, np.ndarray] = {}
    features_np = data.x.cpu().numpy()
    
    for i in range(data.num_nodes):
        node_features[i] = features_np[i] 
    
    return G, node_features