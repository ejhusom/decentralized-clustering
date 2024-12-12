import config
import numpy as np

import matplotlib.pyplot as plt
from utils import generate_synthetic_data, create_base_dataset, partition_data, plot_data, sample_test_data, evaluate_global_model, mnist_clustering_experiment
from client import LocalClient
from server import ServerAggregator


if __name__ == "__main__":
    # Create base dataset
    base_data, base_labels = create_base_dataset(n_samples=1000, n_features=2, n_clusters=config.n_centers_generated, random_state=config.random_state)
    # Partition the data
    client_data, client_labels = partition_data(base_data, base_labels, n_clients=config.n_clients, max_clusters_per_client=config.n_centers_generated)
    # client_data, client_labels = mnist_clustering_experiment(n_clients=config.n_clients, n_features=config.n_features)

    # Sample test data
    test_data, test_labels = sample_test_data(base_data, base_labels, test_size=0.2)

    if config.visualize:
        # Plot the data
        plot_data(base_data, base_labels, client_data)

    if isinstance(config.n_clusters, int):
        n_clusters = [config.n_clusters] * config.n_clients
    else:
        n_clusters = config.n_clusters
        if len(n_clusters) != config.n_clients:
            raise ValueError("Length of n_clusters should be equal to n_clients")
    
    if isinstance(config.clustering_methods, str):
        clustering_methods = [config.clustering_methods] * config.n_clients
    else:
        clustering_methods = config.clustering_methods
        if len(clustering_methods) != config.n_clients:
            raise ValueError("Length of clustering_methods should be equal to n_clients")

    # Create clients
    clients = []
    for i in range(config.n_clients):
        client = LocalClient(i, client_data[i], n_clusters[i], clustering_method=clustering_methods[i], visualize=config.visualize)
        clients.append(client)

    # Train local models
    local_models = []
    for client in clients:
        client.train()
        local_models.append(client.get_model())

    # (Optional) Evaluate global labels on local data
    for client in clients:
        ari, silhouette = evaluate_global_model(client.centroids, client.data, client.labels)
        print(f"Client {client.client_id} - Local Labeling:")
        print(f"Adjusted Rand Index: {ari}")
        print(f"Silhouette Score: {silhouette}\n")

    # Aggregate at server
    server = ServerAggregator(merging_threshold=config.merging_threshold, visualize=False)
    server.aggregate(local_models)

    if config.visualize:
        # Plot all data, the local clusters, and the global clusters
        plt.figure(figsize=(12, 6))
        for i, client in enumerate(clients):
            plt.subplot(2, 3, i+1)
            plt.scatter(client.data[:, 0], client.data[:, 1], c='gray', alpha=0.5)
            plt.scatter(client.centroids[:, 0], client.centroids[:, 1], c='red', s=100, alpha=0.5)
            plt.title(f"Client {i} with {client.n_clusters} clusters using {client.clustering_method}")

        plt.subplot(2, 3, 6)
        colormap = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
        for i, client in enumerate(clients):
            plt.scatter(client.data[:, 0], client.data[:, 1], c='gray', alpha=0.5)
            plt.scatter(client.centroids[:, 0], client.centroids[:, 1], c=colormap[i], marker='x', s=30, alpha=0.7, label=f"Client {i}")
        plt.scatter(server.global_centroids[:, 0], server.global_centroids[:, 1], c='red', s=100, alpha=0.4, label="Global Centroids")
        plt.title("Global Centroids")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()

    # Evaluate the global model
    ari, silhouette = evaluate_global_model(server.unmerged_centroids, test_data, test_labels)
    print("Unmerged Centroids:")
    print(f"Adjusted Rand Index: {ari}")
    print(f"Silhouette Score: {silhouette}\n")
    ari, silhouette = evaluate_global_model(server.global_centroids, test_data, test_labels)
    print("Merged Centroids:")
    print(f"Adjusted Rand Index: {ari}")
    print(f"Silhouette Score: {silhouette}\n")

    # Distribute global centroids to clients
    global_centroids = server.global_centroids

    # Clients label data with global model and optionally retrain
    for client in clients:
        client.label_with_global_model(global_centroids)
        # if retrain_with_global:
        # client.retrain_with_global_centroids(global_centroids)

    # (Optional) Evaluate global labels on local data
    for client in clients:
        ari, silhouette = evaluate_global_model(global_centroids, client.data, client.global_labels)
        print(f"Client {client.client_id} - Global Labeling:")
        print(f"Adjusted Rand Index: {ari}")
        print(f"Silhouette Score: {silhouette}\n")
