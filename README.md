# IM-META: Influence Maximization with Node Metadata

An implementation of the IM-META algorithm for influence maximization in social networks with limited network knowledge, utilizing node features.

## About the Original Paper & Implementations
This codebase is inspired by and attempts to implement the methodology described in the original IM-META paper:
**[IM-META: Influence Maximization with Node Metadata](https://arxiv.org/abs/2106.02926)**.

While following the core architecture of the paper, several implementation details were not clearly explicitly defined in the original text, and thus are approximated or chosen experimentally here:
- **Siamese Network Parameters**: The architecture incorporates a 1-hidden-layer approach (1024 dimensions) with dropout (0.3) projecting to a 256-dimensional embedding, followed by a linear predictor and sigmoid function.
- **Forest Fire Sampling**: The paper does not clearly explain the sampling technique hyperparameters utilized for forest fire. We adopted an exact approach from Leskovec et al. (KDD'06), adjusting the probability forward (`p_forward`) experimentally.
- **Generative Surrogate Model (GSM)**: The GSM for imputing missing node features is not clearly explained in the paper. The autoencoder paradigm implemented in this codebase is a basic starting point but is observed to be not very effective with the very sparse arrays of features typical of these datasets.

## Project Context
This codebase was developed as part of a 6-month research fellowship at my university. It constitutes one component of the broader project:
*"Mechanism Design Online Learning Robust Optimization and Sentiment Extraction Tools for Adjustable Green-Aware Agents"*.

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

# load dataset
G_full, node_features = coauthor_data("Physics")  # or "CS"

# initialize im-meta
im_meta = IMMETA(
    feature_dim=8415,      # Physics: 8415, CS: 6805
    k=5,                   # number of seed nodes
    T=20,                  # query budget
    threshold=0.5,         # edge confidence threshold
    diffusion_model='IC'   # Independent Cascade
)

# run the algorithm
seeds, explored_graph, influence = im_meta.run(G_full, node_features)
```

## Run Experiments

```bash
python scripts/main.py
```

Modify parameters in `scripts/main.py`:
- `COAUTHOR_DATASET`: "Physics" or "CS"
- `NUM_QUERIES`: query budget (default: 40)
- `MC_SIM`: Monte Carlo simulations (default: 1)

### Test Model Performance

```bash
python tests/model_test.py
```
This evaluates the Siamese network's ability to predict edges based on node features, reporting false positive and false negative rates.

## Project Structure

```
immeta/
├── src/immeta/
│   ├── im_meta.py                    # main algorithm orchestration
│   ├── network_inference.py          # siamese network training & inference
│   ├── query_node_selector.py        # node query selection strategy
│   ├── reinforced_graph_generator.py # graph generation with inferred edges
│   ├── seed_set_selector.py          # seed selection with influence estimation
│   ├── siamese_network.py            # neural network architecture
│   ├── gsm.py                        # generative surrogate model (autoencoder)
│   └── coauthor_data.py              # dataset loader
├── scripts/
│   └── main.py                       # main experiment script
├── tests/
│   ├── model_test.py                 # model validation
│   └── creaz.py                      # siamese network testing
└── setup.py                          # package configuration
```

## Algorithm Pipeline

1. **Network Discovery Process (NDP)** - Iteratively for T queries:
   - train siamese network on explored subgraph
   - predict edge probabilities for uncertain pairs
   - generate reinforced graph with confident edges
   - select next node to query based on topology-aware ranking
   - update explored subgraph
2. **Seed Selection**:
   - final network inference on explored graph
   - greedy seed selection maximizing influence spread

## Datasets

PyTorch Geometric Coauthor datasets:
- **Physics**: 34,493 nodes, 247,962 edges, 8,415 features
- **CS**: 18,333 nodes, 81,894 edges, 6,805 features
Data is automatically downloaded to `data/` on first run.