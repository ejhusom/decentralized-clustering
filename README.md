# Algorithm-Agnostic Decentralized Clustering

A federated learning framework for clustering where multiple clients train local models (K-Means or MeanShift) and a central server aggregates their centroids into a global model.

## Features

- **Algorithm flexibility**: Clients can use different clustering algorithms (K-Means, MeanShift)
- **Centroid aggregation**: Server merges local centroids using pairwise distance thresholding or MeanShift
- **Dynamic data**: Supports distribution shifts and incremental data arrival
- **Evaluation**: Tracks ARI and Silhouette scores across iterations

## Usage

```bash
# Run directly:
python src/main.py

# Or using uv (recommended):
# Install uv if needed
pip install uv
uv run python3 src/main.py
```

Configure experiments in `src/config_parameters.py`:
- `n_clients`: Number of distributed clients
- `n_clusters`: Clusters per client (or `None` for MeanShift)
- `clustering_methods`: List of algorithms per client
- `merging_threshold`: Distance threshold for centroid merging
- `n_iterations`: Training rounds

## Project Structure

```
src/
├── main.py           # Experiment runner
├── client.py         # LocalClient: trains local clustering models
├── server.py         # ServerAggregator: merges centroids
├── utils.py          # Data generation, partitioning, evaluation
├── config_parameters.py
└── plotting.py       # Visualization utilities
```

## Requirements

- numpy
- scikit-learn
- matplotlib
- scipy
