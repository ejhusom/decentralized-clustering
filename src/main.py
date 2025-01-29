import json
import os
import datetime
import parameters
import numpy as np

import matplotlib.pyplot as plt
from utils import generate_synthetic_data, create_base_dataset, partition_data, plot_data, sample_test_data, evaluate_global_model, mnist_clustering_experiment, plot_data_after_aggregation, generate_synthetic_batch, append_client_data
from client import LocalClient
from server import ServerAggregator

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = f"output/{timestamp}"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def run_experiment(parameters, experiment_name=""):
    # Create base dataset
    base_data, base_labels = create_base_dataset(n_samples=parameters.n_samples, n_features=2, n_clusters=parameters.n_centers_generated, random_state=parameters.random_state)
    # Partition the data
    client_data, client_labels, cluster_distribution = partition_data(base_data, base_labels, n_clients=parameters.n_clients, max_clusters_per_client=parameters.n_centers_generated)
    # client_data, client_labels = mnist_clustering_experiment(n_clients=parameters.n_clients, n_features=parameters.n_features)

    # Sample test data
    test_data, test_labels = sample_test_data(base_data, base_labels, test_size=0.2)

    if parameters.visualize:
        # Plot the data
        plot_data(base_data, base_labels, client_data)

    if isinstance(parameters.n_clusters, int):
        n_clusters = [parameters.n_clusters] * parameters.n_clients
    else:
        n_clusters = parameters.n_clusters
        if len(n_clusters) != parameters.n_clients:
            raise ValueError("Length of n_clusters should be equal to n_clients")
    
    if isinstance(parameters.clustering_methods, str):
        clustering_methods = [parameters.clustering_methods] * parameters.n_clients
    else:
        clustering_methods = parameters.clustering_methods
        if len(clustering_methods) != parameters.n_clients:
            raise ValueError("Length of clustering_methods should be equal to n_clients")

    # Create clients
    clients = []
    for i in range(parameters.n_clients):
        client = LocalClient(i, client_data[i], n_clusters[i], clustering_method=clustering_methods[i], visualize=parameters.visualize)
        clients.append(client)

    n_iterations = parameters.n_iterations

    metrics = {"server": {"pre_aggregation": {"ari": [], "silhouette": []}, "post_aggregation": {"ari": [], "silhouette": []}}}
    for i in range(parameters.n_clients):
        metrics[f"client_{i}"] = {"local": {"ari": [], "silhouette": []}, "global": {"ari": [], "silhouette": []}}

    for i in range(n_iterations):
        print(f"Iteration {i+1}/{n_iterations}")

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
            if parameters.verbose:
                print(f"Client {client.client_id} - Local Labeling:")
                print(f"Adjusted Rand Index: {ari}")
                print(f"Silhouette Score: {silhouette}\n")

            metrics[f"client_{client.client_id}"]["local"]["ari"].append(ari)
            metrics[f"client_{client.client_id}"]["local"]["silhouette"].append(silhouette)


        # Aggregate at server
        server = ServerAggregator(merging_threshold=parameters.merging_threshold, visualize=False)
        server.aggregate(local_models, method="pairwise")

        if parameters.visualize:
            plot_data_after_aggregation(clients, server)

        # Evaluate the global model on test data
        ari, silhouette = evaluate_global_model(server.unmerged_centroids, test_data, test_labels)
        metrics["server"]["pre_aggregation"]["ari"].append(ari)
        metrics["server"]["pre_aggregation"]["silhouette"].append(silhouette)
        if parameters.verbose:
            print("Unmerged Centroids:")
            print(f"Adjusted Rand Index: {ari}")
            print(f"Silhouette Score: {silhouette}\n")

        ari, silhouette = evaluate_global_model(server.global_centroids, test_data, test_labels)
        metrics["server"]["post_aggregation"]["ari"].append(ari)
        metrics["server"]["post_aggregation"]["silhouette"].append(silhouette)
        if parameters.verbose:
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
            if parameters.verbose:
                print(f"Client {client.client_id} - Global Labeling:")
                print(f"Adjusted Rand Index: {ari}")
                print(f"Silhouette Score: {silhouette}\n")

        # Clients receive new data
        for i, client in enumerate(clients):
            # Generate a new batch with a moderate distribution shift
            new_batch_data, new_batch_labels = generate_synthetic_batch(
                base_data=base_data,
                base_labels=base_labels,
                n_samples=max(int(parameters.n_samples_per_client * 0.05), 1),
                cluster_distribution=cluster_distribution[i],
                distribution_shift_type='significant',
                random_state=42
            )

            # Plot old and new data for client i, showing the distribution shift, and also the combined data, in 3 subplots
            if parameters.visualize:
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

    # Substitute None values in silhouette scores with 0
    for i in range(parameters.n_clients):
        # if metrics[f"client_{i}"]["local"]["silhouette"] == None:
        #     metrics[f"client_{i}"]["local"]["silhouette"] = [0] * n_iterations
        # if metrics[f"client_{i}"]["global"]["silhouette"] == None:
        #     metrics[f"client_{i}"]["global"]["silhouette"] = [0] * n_iterations
        for j, value in enumerate(metrics[f"client_{i}"]["local"]["silhouette"]):
            if value == None:
                metrics[f"client_{i}"]["local"]["silhouette"][j] = 0
        for j, value in enumerate(metrics[f"client_{i}"]["global"]["silhouette"]):
            if value == None:
                metrics[f"client_{i}"]["global"]["silhouette"][j] = 0


    # # Plot the silhouette scores (not the ARI). There should be three subplots: One for the local silhouette scores, one for the global silhouette scores, and one for the server silhouette scores (pre- and post-aggregation).
    # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # # for client_id in range(parameters.n_clients):
    # #     ax[0].plot(metrics[f"client_{client_id}"]["local"]["silhouette"], alpha=0.5)
    # #     ax[1].plot(metrics[f"client_{client_id}"]["global"]["silhouette"], alpha=0.5)
    # ax[2].plot(metrics["server"]["pre_aggregation"]["silhouette"], label="Server - Pre-aggregation", linestyle="--")
    # ax[2].plot(metrics["server"]["post_aggregation"]["silhouette"], label="Server - Post-aggregation")

    # # Plot average silhouette scores
    # ax[0].plot(np.mean([metrics[f"client_{client_id}"]["local"]["silhouette"] for client_id in range(parameters.n_clients)], axis=0), label="Average Local", color="black")
    # ax[1].plot(np.mean([metrics[f"client_{client_id}"]["global"]["silhouette"] for client_id in range(parameters.n_clients)], axis=0), label="Average Global", color="black")

    # ax[0].set_title("Local Silhouette Scores")
    # ax[1].set_title("Global Silhouette Scores")
    # ax[2].set_title("Server Silhouette Scores")
    # ax[0].legend()
    # ax[1].legend()
    # ax[2].legend()
    # # Set equal y-axis limits for better comparison
    # ax[0].set_ylim([0, 1])
    # ax[1].set_ylim([0, 1])
    # ax[2].set_ylim([0, 1])

    # plt.show()

    # Plot the silhouette scores. One plot shows both the average local and average global silhouette scores together, with a legend indicating which line is which.
    fig, ax = plt.subplots(1, 1, figsize=(4,3))
    ax.plot(np.mean([metrics[f"client_{client_id}"]["local"]["silhouette"] for client_id in range(parameters.n_clients)], axis=0), label="Average Local")
    ax.plot(np.mean([metrics[f"client_{client_id}"]["global"]["silhouette"] for client_id in range(parameters.n_clients)], axis=0), label="Average Global")
    ax.set_title("Average Silhouette Scores")
    ax.legend(loc='lower right')
    # Set equal y-axis limits for better comparison
    ax.set_ylim([0, 1])

    # Save plot with timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(OUTPUT_DIR + f"/silhouette_scores_{timestamp}.pdf")


    # plt.show()

    # Save the metrics to a JSON file with timestamp
    with open(OUTPUT_DIR + f"/metrics_{timestamp}.json", "w") as f:
        json.dump(metrics, f)

    


    # # Plot the ARI scores. There should be three subplots: One for the local ARI scores, one for the global ARI scores, and one for the server ARI scores (pre- and post-aggregation).
    # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # # for client_id in range(parameters.n_clients):
    # #     ax[0].plot(metrics[f"client_{client_id}"]["local"]["ari"], alpha=0.5)
    # #     ax[1].plot(metrics[f"client_{client_id}"]["global"]["ari"], alpha=0.5)
    # ax[2].plot(metrics["server"]["pre_aggregation"]["ari"], label="Server - Pre-aggregation", linestyle="--")
    # ax[2].plot(metrics["server"]["post_aggregation"]["ari"], label="Server - Post-aggregation")

    # # Plot average ARI scores
    # ax[0].plot(np.mean([metrics[f"client_{client_id}"]["local"]["ari"] for client_id in range(parameters.n_clients)], axis=0), label="Average Local", color="black", linestyle="--")
    # ax[1].plot(np.mean([metrics[f"client_{client_id}"]["global"]["ari"] for client_id in range(parameters.n_clients)], axis=0), label="Average Global", color="black", linestyle="--")

    # ax[0].set_title("Local ARI Scores")
    # ax[1].set_title("Global ARI Scores")
    # ax[2].set_title("Server ARI Scores")
    # ax[0].legend()
    # ax[1].legend()
    # ax[2].legend()
    # # Set equal y-axis limits for better comparison
    # ax[0].set_ylim([0, 1])
    # ax[1].set_ylim([0, 1])
    # ax[2].set_ylim([0, 1])

    # plt.show()

    return metrics

def plot_metrics(metrics_dict, param_name, param_values):
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for param_value in param_values:
        metrics = metrics_dict[param_value]
        if param_name == "n_clients":
            local_avg = np.mean([metrics[f"client_{client_id}"]["local"]["silhouette"] for client_id in range(param_value)], axis=0)
            global_avg = np.mean([metrics[f"client_{client_id}"]["global"]["silhouette"] for client_id in range(param_value)], axis=0)
        else:
            local_avg = np.mean([metrics[f"client_{client_id}"]["local"]["silhouette"] for client_id in range(parameters.n_clients)], axis=0)
            global_avg = np.mean([metrics[f"client_{client_id}"]["global"]["silhouette"] for client_id in range(parameters.n_clients)], axis=0)
        ax[0].plot(local_avg, label=f"{param_name}={param_value}")
        ax[1].plot(global_avg, label=f"{param_name}={param_value}")

    ax[0].set_title("Local Average Silhouette Scores")
    ax[1].set_title("Global Average Silhouette Scores")
    ax[0].legend(loc='lower right')
    ax[1].legend(loc='lower right')
    ax[0].set_ylim([0, 1])
    ax[1].set_ylim([0, 1])

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(OUTPUT_DIR + f"/silhouette_scores_{param_name}_{timestamp}.pdf")
    # plt.show()

if __name__ == "__main__":
    # run_experiment(parameters)
    # Save the original parameters from parameters.py module
    original_parameters = parameters.__dict__.copy()

    max_iterations_clustering_values = [1, 10, 100, 1000]
    n_clients_values = [2, 6, 10, 20]
    n_centers_generated_values = [5, 10, 15, 20]
    n_samples_per_client_values = [10, 20, 50, 100]
    merging_threshold_values = [0.1, 1.0, 2.0, 4.0]

    # Loop over different values of max_iterations_clustering and visualize the results
    max_iterations_clustering_metrics = {}
    for max_iterations_clustering in max_iterations_clustering_values:
        parameters.max_iterations_clustering = max_iterations_clustering
        print(f"Running experiment with max_iterations_clustering={max_iterations_clustering}")
        metrics = run_experiment(parameters)
        max_iterations_clustering_metrics[max_iterations_clustering] = metrics
    plot_metrics(max_iterations_clustering_metrics, "max_iterations_clustering", max_iterations_clustering_values)

    # Reset the parameters to the original values
    parameters.__dict__.update(original_parameters)


    # Loop over different values of n_clients and visualize the results
    n_clients_metrics = {}
    for n_clients in n_clients_values:
        parameters.n_clients = n_clients
        parameters.clustering_methods = ["meanshift"] * (n_clients // 2) + ["kmeans"] * (n_clients // 2)
        parameters.n_clusters = [None] * (n_clients // 2) + [int(parameters.n_centers_generated * 0.5)] * (n_clients // 2)
        print(f"Running experiment with n_clients={n_clients}")
        metrics = run_experiment(parameters)
        n_clients_metrics[n_clients] = metrics
    plot_metrics(n_clients_metrics, "n_clients", n_clients_values)
    
    parameters.__dict__.update(original_parameters)

    # Loop over different values of n_centers_generated and visualize the results
    n_centers_generated_metrics = {}
    for n_centers_generated in n_centers_generated_values:
        parameters.n_centers_generated = n_centers_generated
        parameters.n_clusters = [None] * (parameters.n_clients // 2) + [int(n_centers_generated * 0.5)] * (parameters.n_clients // 2)
        print(f"Running experiment with n_centers_generated={n_centers_generated}")
        metrics = run_experiment(parameters)
        n_centers_generated_metrics[n_centers_generated] = metrics
    plot_metrics(n_centers_generated_metrics, "n_clusters", n_centers_generated_values)

    parameters.__dict__.update(original_parameters)

    # Loop over different values of n_samples_per_client and visualize the results
    n_samples_per_client_metrics = {}
    for n_samples_per_client in n_samples_per_client_values:
        parameters.n_samples_per_client = n_samples_per_client
        print(f"Running experiment with n_samples_per_client={n_samples_per_client}")
        metrics = run_experiment(parameters)
        n_samples_per_client_metrics[n_samples_per_client] = metrics
    plot_metrics(n_samples_per_client_metrics, "n_samples_per_client", n_samples_per_client_values)

    parameters.__dict__.update(original_parameters)

    # Loop over different values of merging_threshold and visualize the results
    merging_threshold_metrics = {}
    for merging_threshold in merging_threshold_values:
        parameters.merging_threshold = merging_threshold
        print(f"Running experiment with merging_threshold={merging_threshold}")
        metrics = run_experiment(parameters)
        merging_threshold_metrics[merging_threshold] = metrics
    plot_metrics(merging_threshold_metrics, "merging_threshold", merging_threshold_values)
