import config
import numpy as np

import matplotlib.pyplot as plt
from utils import generate_synthetic_data, create_base_dataset, partition_data, plot_data, sample_test_data, evaluate_global_model, mnist_clustering_experiment, plot_data_after_aggregation, generate_synthetic_batch, append_client_data
from client import LocalClient
from server import ServerAggregator


if __name__ == "__main__":
    # Create base dataset
    base_data, base_labels = create_base_dataset(n_samples=config.n_samples, n_features=2, n_clusters=config.n_centers_generated, random_state=config.random_state)
    # Partition the data
    client_data, client_labels, cluster_distribution = partition_data(base_data, base_labels, n_clients=config.n_clients, max_clusters_per_client=config.n_centers_generated)
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

    n_iterations = config.n_iterations

    metrics = {"server": {"pre_aggregation": {"ari": [], "silhouette": []}, "post_aggregation": {"ari": [], "silhouette": []}}}
    for i in range(config.n_clients):
        metrics[f"client_{i}"] = {"local": {"ari": [], "silhouette": []}, "global": {"ari": [], "silhouette": []}}

    for i in range(n_iterations):

        # Train local models
        local_models = []
        for client in clients:
            if i == 0:
                client.train()
            else:
                client.retrain(client.global_centroids)

            local_models.append(client.get_model())

        # (Optional) Evaluate local labels on local data
        for client in clients:
            ari, silhouette = evaluate_global_model(client.centroids, client.data, client.labels)
            if config.verbose:
                print(f"Client {client.client_id} - Local Labeling:")
                print(f"Adjusted Rand Index: {ari}")
                print(f"Silhouette Score: {silhouette}\n")

            metrics[f"client_{client.client_id}"]["local"]["ari"].append(ari)
            metrics[f"client_{client.client_id}"]["local"]["silhouette"].append(silhouette)


        # Aggregate at server
        server = ServerAggregator(merging_threshold=config.merging_threshold, visualize=False)
        server.aggregate(local_models, method="pairwise")

        if config.visualize:
            plot_data_after_aggregation(clients, server)

        # Evaluate the global model
        ari, silhouette = evaluate_global_model(server.unmerged_centroids, test_data, test_labels)
        metrics["server"]["pre_aggregation"]["ari"].append(ari)
        metrics["server"]["pre_aggregation"]["silhouette"].append(silhouette)
        if config.verbose:
            print("Unmerged Centroids:")
            print(f"Adjusted Rand Index: {ari}")
            print(f"Silhouette Score: {silhouette}\n")

        ari, silhouette = evaluate_global_model(server.global_centroids, test_data, test_labels)
        metrics["server"]["post_aggregation"]["ari"].append(ari)
        metrics["server"]["post_aggregation"]["silhouette"].append(silhouette)
        if config.verbose:
            print("Merged Centroids:")
            print(f"Adjusted Rand Index: {ari}")
            print(f"Silhouette Score: {silhouette}\n")

        # Distribute global centroids to clients
        global_centroids = server.global_centroids

        # Clients label data with global model and optionally retrain
        for client in clients:
            client.global_centroids = global_centroids
            client.label_with_global_model(global_centroids)

        # (Optional) Evaluate global labels on local data
        for client in clients:
            ari, silhouette = evaluate_global_model(global_centroids, client.data, client.global_labels)
            metrics[f"client_{client.client_id}"]["global"]["ari"].append(ari)
            metrics[f"client_{client.client_id}"]["global"]["silhouette"].append(silhouette)
            if config.verbose:
                print(f"Client {client.client_id} - Global Labeling:")
                print(f"Adjusted Rand Index: {ari}")
                print(f"Silhouette Score: {silhouette}\n")

        # Clients receive new data
        for i, client in enumerate(clients):
            # Generate a new batch with a moderate distribution shift
            new_batch_data, new_batch_labels = generate_synthetic_batch(
                base_data=base_data,
                base_labels=base_labels,
                n_samples=int(config.n_samples_per_client * 0.1),
                cluster_distribution=cluster_distribution[i],
                distribution_shift_type='significant',
                random_state=42
            )

            # Plot old and new data for client i, showing the distribution shift, and also the combined data, in 3 subplots
            if config.visualize:
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].scatter(client.data[:, 0], client.data[:, 1], c=client.labels, cmap='viridis')
                ax[0].set_title(f"Client {i} - Old Data")
                ax[1].scatter(new_batch_data[:, 0], new_batch_data[:, 1], c=new_batch_labels, cmap='viridis')
                ax[1].set_title(f"Client {i} - New Batch")
                ax[2].scatter(np.concatenate([client.data, new_batch_data])[:, 0], np.concatenate([client.data, new_batch_data])[:, 1], c=np.concatenate([client.labels, new_batch_labels]), cmap='viridis')
                ax[2].set_title(f"Client {i} - Combined Data")
                plt.show()


            # Assume you have initial data and a new batch
            updated_data, updated_labels = append_client_data(
                current_data=client_data[i],
                current_labels=client_labels[i],
                new_batch_data=new_batch_data,
                new_batch_labels=new_batch_labels,
                max_data_size=None
            )

            client_data[i] = updated_data
            client_labels[i] = updated_labels

            # Update client data
            client.data = updated_data
            client.labels = updated_labels

    # Plot the silhouette scores (not the ARI). There should be three subplots: One for the local silhouette scores, one for the global silhouette scores, and one for the server silhouette scores (pre- and post-aggregation).
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # for client_id in range(config.n_clients):
    #     ax[0].plot(metrics[f"client_{client_id}"]["local"]["silhouette"], alpha=0.5)
    #     ax[1].plot(metrics[f"client_{client_id}"]["global"]["silhouette"], alpha=0.5)
    ax[2].plot(metrics["server"]["pre_aggregation"]["silhouette"], label="Server - Pre-aggregation", linestyle="--")
    ax[2].plot(metrics["server"]["post_aggregation"]["silhouette"], label="Server - Post-aggregation")

    # Plot average silhouette scores
    ax[0].plot(np.mean([metrics[f"client_{client_id}"]["local"]["silhouette"] for client_id in range(config.n_clients)], axis=0), label="Average Local", color="black")
    ax[1].plot(np.mean([metrics[f"client_{client_id}"]["global"]["silhouette"] for client_id in range(config.n_clients)], axis=0), label="Average Global", color="black")

    ax[0].set_title("Local Silhouette Scores")
    ax[1].set_title("Global Silhouette Scores")
    ax[2].set_title("Server Silhouette Scores")
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    # Set equal y-axis limits for better comparison
    ax[0].set_ylim([0, 1])
    ax[1].set_ylim([0, 1])
    ax[2].set_ylim([0, 1])

    plt.show()


    # Plot the ARI scores. There should be three subplots: One for the local ARI scores, one for the global ARI scores, and one for the server ARI scores (pre- and post-aggregation).
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # for client_id in range(config.n_clients):
    #     ax[0].plot(metrics[f"client_{client_id}"]["local"]["ari"], alpha=0.5)
    #     ax[1].plot(metrics[f"client_{client_id}"]["global"]["ari"], alpha=0.5)
    ax[2].plot(metrics["server"]["pre_aggregation"]["ari"], label="Server - Pre-aggregation", linestyle="--")
    ax[2].plot(metrics["server"]["post_aggregation"]["ari"], label="Server - Post-aggregation")

    # Plot average ARI scores
    ax[0].plot(np.mean([metrics[f"client_{client_id}"]["local"]["ari"] for client_id in range(config.n_clients)], axis=0), label="Average Local", color="black", linestyle="--")
    ax[1].plot(np.mean([metrics[f"client_{client_id}"]["global"]["ari"] for client_id in range(config.n_clients)], axis=0), label="Average Global", color="black", linestyle="--")

    ax[0].set_title("Local ARI Scores")
    ax[1].set_title("Global ARI Scores")
    ax[2].set_title("Server ARI Scores")
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    # Set equal y-axis limits for better comparison
    ax[0].set_ylim([0, 1])
    ax[1].set_ylim([0, 1])
    ax[2].set_ylim([0, 1])

    plt.show()
