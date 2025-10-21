# IM-META: Influence Maximization with Node Metadata

Implementation of the IM-META algorithm for influence maximization in social networks with limited network knowledge, using node features.

## Installation

```bash
pip install -e .
```

### Requirements
- Python ≥ 3.8
- PyTorch ≥ 2.0.0
- PyTorch Geometric ≥ 2.3.0
- NetworkX ≥ 3.0
- NumPy ≥ 1.24.0

## Quick Start

```python
from immeta import IMMETA, coauthor_data

# Load dataset
G_full, node_features = coauthor_data("Physics")  # or "CS"

# Initialize IM-META
im_meta = IMMETA(
    feature_dim=8415,      # Physics: 8415, CS: 6805
    k=5,                   # number of seed nodes
    T=20,                  # query budget
    threshold=0.5,         # edge confidence threshold
    diffusion_model='IC'   # Independent Cascade
)

# Run the algorithm
seeds, explored_graph, influence = im_meta.run(G_full, node_features)
```

## Run experiments

```bash
python scripts/main.py
```

Modify parameters in `scripts/main.py`:
- `COAUTHOR_DATASET`: "Physics" or "CS"
- `NUM_QUERIES`: query budget (default: 20)
- `MC_SIM`: Monte Carlo simulations (default: 1)

### Test model performance

```bash
python tests/model_test.py
```

This evaluates the Siamese network's ability to predict edges based on node features, reporting false positive and false negative rates.

## Project Structure

```
immeta/
├── src/immeta/
│   ├── im_meta.py                    # Main algorithm orchestration
│   ├── network_inference.py          # Siamese network training & inference
│   ├── query_node_selector.py        # Node query selection strategy
│   ├── reinforced_graph_generator.py # Graph generation with inferred edges
│   ├── seed_set_selector.py          # Seed selection with influence estimation
│   ├── siamese_network.py            # Neural network architecture
│   └── coauthor_data.py              # Dataset loader
├── scripts/
│   └── main.py                       # Main experiment script
├── tests/
│   └── model_test.py                 # Model validation
└── setup.py                          # Package configuration
```

## Algorithm Pipeline

1. **Network Discovery Process (NDP)** - Iteratively for T queries:
   - Train Siamese network on explored subgraph
   - Predict edge probabilities for uncertain pairs
   - Generate reinforced graph with confident edges
   - Select next node to query based on topology-aware ranking
   - Update explored subgraph

2. **Seed Selection**:
   - Final network inference on explored graph
   - Greedy seed selection maximizing influence spread (σ)

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `k` | Number of seed nodes | 5 |
| `T` | Query budget | 60 |
| `alpha` | Balance parameter for query selection | 1.0 |
| `threshold` | Edge confidence threshold (ε) | 0.5 |
| `diffusion_model` | 'IC' (Independent Cascade) or 'WC' (Weighted Cascade) | 'IC' |

## Datasets

PyTorch Geometric's Coauthor datasets:
- **Physics**: 34,493 nodes, 247,962 edges, 8,415 features
- **CS**: 18,333 nodes, 81,894 edges, 6,805 features

Data is automatically downloaded to `data/` on first run.

## Output

The algorithm returns:
- `seeds`: List of selected seed node IDs
- `explored_graph`: Final explored subgraph
- `sigma`: Expected influence spread (estimated via Monte Carlo)