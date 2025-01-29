import config
import numpy as np

import matplotlib.pyplot as plt
from utils import generate_synthetic_data, create_base_dataset, partition_data, plot_data, sample_test_data, evaluate_global_model, mnist_clustering_experiment, plot_data_after_aggregation, generate_synthetic_batch, append_client_data, cluster_recall, cluster_purity
from client import LocalClient
from server import ServerAggregator
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist

from plot_metrics import plot_metrics
from baselines import CentralizedBaseline
from server_fedkmeans import FedKMeansAggregator
from server_fedms import FedMeanShiftAggregator

def run_baseline_experiment(client_data, test_data, test_labels, true_centers, method='centralized'):
    """Run baseline experiments with proper parameter handling"""
    if method == 'centralized':
        # Centralized KMeans (oracle baseline)
        from sklearn.cluster import KMeans
        all_data = np.vstack([c_data for c_data in client_data])
        model = KMeans(n_clusters=len(true_centers), random_state=42)
        model.fit(all_data)
        centroids = model.cluster_centers_
        
    elif method == 'fedkmeans':
        # Federated KMeans with fixed cluster count
        from server import ServerAggregator
        server = ServerAggregator(merging_threshold=config.merging_threshold)
        clients = []
        
        # Create clients with true cluster count
        for i, data in enumerate(client_data):
            client = LocalClient(
                client_id=i,
                data=data,
                n_clusters=len(true_centers),  # Force true cluster count
                clustering_method="kmeans",
            )
            client.train()
            clients.append(client)
            
        # Aggregate centroids
        local_models = [client.get_model() for client in clients]
        centroids = server.aggregate(local_models, method='pairwise')
        
    elif method == 'fedms':
        # Federated MeanShift baseline
        from server import ServerAggregator
        server = ServerAggregator(merging_threshold=config.merging_threshold)
        clients = []
        
        for i, data in enumerate(client_data):
            client = LocalClient(
                client_id=i,
                data=data,
                n_clusters=None,  # MeanShift auto-detection
                clustering_method="meanshift",
            )
            client.train()
            clients.append(client)
            
        local_models = [client.get_model() for client in clients]
        centroids = server.aggregate(local_models, method='meanshift')
        
    else:
        raise ValueError(f"Unknown baseline method: {method}")

    # Calculate metrics
    from utils import cluster_recall, cluster_purity
    pred_labels = np.argmin(cdist(test_data, centroids), axis=1)
    
    return {
        'silhouette': silhouette_score(test_data, pred_labels),
        'recall': cluster_recall(centroids, true_centers, threshold=1.5),
        'purity': cluster_purity(centroids, test_data, test_labels)
    }

if __name__ == "__main__":
    # Define the metric to calculate
    metric_to_calculate = "silhouette"

    # Generate base data
    base_data, base_labels = create_base_dataset(
        n_samples=config.n_samples,
        n_features=config.n_features,
        n_clusters=config.n_centers_generated,
        random_state=config.random_state
    )

    # Split test data BEFORE client partitioning
    base_data, test_data, base_labels, test_labels = train_test_split(
        base_data, base_labels, 
        test_size=0.2,
        random_state=config.random_state
    )

    # Calculate true cluster centers from BASE DATA (not test data)
    true_centers = np.array([
        np.mean(base_data[base_labels == i], axis=0) 
        for i in np.unique(base_labels)
    ])

    # Now partition the remaining data to clients
    client_data, client_labels, cluster_distribution = partition_data(
        base_data, base_labels, 
        n_clients=config.n_clients,
        max_clusters_per_client=4
    )

    # Sample test data
    test_data, test_labels = sample_test_data(base_data, base_labels, test_size=0.2)

    true_centers = np.array([np.mean(base_data[base_labels == i], axis=0) 
                       for i in np.unique(base_labels)])

    # After test data generation
    baseline_results = {}

    # Centralized baseline
    baseline_results['centralized'] = run_baseline_experiment(
        client_data, test_data, test_labels, true_centers, method='centralized'
    )

    # Federated KMeans baseline
    baseline_results['fedkmeans'] = run_baseline_experiment(
        client_data, test_data, test_labels, true_centers, method='fedkmeans'
    )

    # Federated MeanShift baseline
    baseline_results['fedms'] = run_baseline_experiment(
        client_data, test_data, test_labels, true_centers, method='fedms'
    )

    # Your FALC results
    falcon_results = {
        'silhouette': [],
        'recall': [],
        'purity': []
    }

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

    # Create a dictionary with all the most common clustering metrics 
    metrics = {
        "server": {
            "pre_aggregation": {
                "silhouette": []
            },
            "post_aggregation": {
                "silhouette": []
            }
        }
    }

    for i in range(config.n_clients):
        metrics[f"client_{i}"] = {
            "local": {
                "silhouette": []
            },
            "global": {
                "silhouette": []
            }
        }

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
            results = evaluate_global_model(client.centroids, client.data, client.labels, metric=metric_to_calculate)
            if config.verbose:
                print(f"Client {client.client_id} - Local Labeling:")
                print(f"Silhouette Score: {results['silhouette']}\n")

            for metric, value in results.items():
                metrics[f"client_{client.client_id}"]["local"][metric].append(value)

        # Aggregate at server
        server = ServerAggregator(
            merging_threshold=4.0,
            visualize=False,
            use_dynamic_threshold=config.use_dynamic_threshold,
            use_weighted_merging=config.use_weighted_merging
        )
        # server.aggregate(local_models, method="pairwise")
        server.aggregate(local_models, method="meanshift")

        if config.visualize:
            plot_data_after_aggregation(clients, server)

        # Evaluate the global model
        results_pre = evaluate_global_model(server.unmerged_centroids, test_data, test_labels, metric=metric_to_calculate)
        for metric, value in results_pre.items():
            metrics["server"]["pre_aggregation"][metric].append(value)
        if config.verbose:
            print("Unmerged Centroids:")
            print(f"Silhouette Score: {results_pre['silhouette']}\n")

        results_post = evaluate_global_model(server.global_centroids, test_data, test_labels, metric=metric_to_calculate)
        for metric, value in results_post.items():
            metrics["server"]["post_aggregation"][metric].append(value)
        falcon_results['silhouette'].append(results_post['silhouette'])
        falcon_results['recall'].append(cluster_recall(server.global_centroids, true_centers))
        falcon_results['purity'].append(cluster_purity(server.global_centroids, test_data, test_labels))
        if config.verbose:
            print("Merged Centroids:")
            print(f"Silhouette Score: {results_post['silhouette']}\n")

        # Distribute global centroids to clients
        global_centroids = server.global_centroids

        # Clients label data with global model and optionally retrain
        for client in clients:
            client.global_centroids = global_centroids
            client.label_with_global_model(global_centroids)

        # (Optional) Evaluate global labels on local data
        for client in clients:
            results = evaluate_global_model(global_centroids, client.data, client.global_labels, metric=metric_to_calculate)
            for metric, value in results.items():
                metrics[f"client_{client.client_id}"]["global"][metric].append(value)

            if config.verbose:
                print(f"Client {client.client_id} - Global Labeling:")
                print(f"Silhouette Score: {results['silhouette']}\n")

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

    # Save all metrics to a file with a timestamp
    import json
    import os
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("metrics", exist_ok=True)
    with open(f"metrics/metrics_{timestamp}.json", "w") as f:
        json.dump(metrics, f)

    # Substitue None values with 0
    for key, value in metrics.items():
        for sub_key, sub_value in value.items():
            for metric, metric_values in sub_value.items():
                metrics[key][sub_key][metric] = [0 if v is None else v for v in metric_values]

    # Plot the silhouette scores. There should be three subplots: One for the local silhouette scores, one for the global silhouette scores, and one for the server silhouette scores (pre- and post-aggregation).
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

    # # Plot the ARI scores. There should be three subplots: One for the local ARI scores, one for the global ARI scores, and one for the server ARI scores (pre- and post-aggregation).
    # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # # for client_id in range(config.n_clients):
    # #     ax[0].plot(metrics[f"client_{client_id}"]["local"]["ari"], alpha=0.5)
    # #     ax[1].plot(metrics[f"client_{client_id}"]["global"]["ari"], alpha=0.5)
    # ax[2].plot(metrics["server"]["pre_aggregation"]["ari"], label="Server - Pre-aggregation", linestyle="--")
    # ax[2].plot(metrics["server"]["post_aggregation"]["ari"], label="Server - Post-aggregation")

    plot_metrics(baseline_results, falcon_results)