from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.spatial.distance import cdist
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import config

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

def load_and_preprocess_mnist(n_samples=10000, n_components=50):
    """
    Load MNIST dataset and preprocess for clustering

    Parameters:
    -----------
    n_samples : int, optional
        Number of samples to use
    n_components : int, optional
        Number of PCA components to reduce to

    Returns:
    --------
    X : numpy.ndarray
        Preprocessed feature matrix
    y : numpy.ndarray
        Corresponding labels
    """
    # Load MNIST
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X = np.array(mnist.data.astype('float32'))
    y = np.array(mnist.target.astype('int'))

    # Subsample and normalize
    indices = np.random.choice(len(X), n_samples, replace=False)
    X = X[indices]
    y = y[indices]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reduce dimensionality with PCA
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)

    return X_reduced, y

def custom_mnist_partition(X, y, n_clients):
    """
    Custom partitioning strategy for MNIST

    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Labels
    n_clients : int
        Number of clients to distribute data to

    Returns:
    --------
    client_data : list
        List of data subsets for each client
    client_labels : list
        Corresponding labels for each client's data
    """
    # Group data by digit
    digit_groups = {digit: X[y == digit] for digit in range(10)}

    # Distribute digits across clients
    client_data = []
    client_labels = []

    for i in range(n_clients):
        # Each client gets a mix of digits
        client_subset = []
        client_subset_labels = []

        # Select a random subset of digits for this client
        selected_digits = np.random.choice(10, size=np.random.randint(3, 7), replace=False)

        for digit in selected_digits:
            # Get data for this digit
            digit_data = digit_groups[digit]

            # Randomly sample some percentage of the digit's data
            n_samples = int(len(digit_data) * np.random.uniform(0.1, 0.3))
            digit_indices = np.random.choice(len(digit_data), n_samples, replace=False)

            client_subset.append(digit_data[digit_indices])
            client_subset_labels.extend([digit] * n_samples)

        client_data.append(np.vstack(client_subset))
        client_labels.append(np.array(client_subset_labels))

    return client_data, client_labels

# Modify main script to use MNIST
def mnist_clustering_experiment(n_clients=5, n_features=50):
    # Load and preprocess MNIST
    X, y = load_and_preprocess_mnist(n_samples=5000, n_components=n_features)

    # Partition data across clients
    client_data, client_labels = custom_mnist_partition(X, y, n_clients=n_clients)

    # Visualize client data distributions
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 5))
    for i, (data, labels) in enumerate(zip(client_data, client_labels)):
        plt.subplot(1, len(client_data), i+1)
        plt.hist(labels, bins=range(11), alpha=0.7)
        plt.title(f'Client {i} Digit Distribution')
        plt.xlabel('Digit')
        plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    return client_data, client_labels

def generate_synthetic_data(n_clients=5, n_samples_per_client=100, n_centers=3, n_features=2, random_state=None):
    data = []
    for client_id in range(n_clients):
        X, _ = make_blobs(
                n_samples=n_samples_per_client, 
                centers=n_centers, 
                n_features=n_features,
                cluster_std=np.random.uniform(0.5, 1.5), 
                random_state=random_state+client_id if random_state else None)
        data.append(X)
    return data

def evaluate_global_model(global_centroids, test_data, test_labels):
    # Assign test points to nearest global centroids
    distances = cdist(test_data, global_centroids)
    predicted_labels = np.argmin(distances, axis=1)

    # Compute evaluation metrics
    ari = adjusted_rand_score(test_labels, predicted_labels)
    if len(set(predicted_labels)) < 2:
        silhouette = None
        print("Only one predicted label; not possible to compute silhouette score.")
    else:
        silhouette = silhouette_score(test_data, predicted_labels)

    return ari, silhouette

def create_base_dataset(n_samples, n_features, n_clusters, random_state=42):
    X, y = make_blobs(
        n_samples=n_samples,
        centers=n_clusters,
        n_features=n_features,
        random_state=random_state,
        cluster_std=1.0,
    )
    return X, y

def partition_data(base_data, base_labels, n_clients, 
                   cluster_distribution=None, 
                   random_state=None, 
                   max_clusters_per_client=5):
    """
    Partition data across clients with controllable and stochastic cluster distribution.

    Parameters:
    -----------
    base_data : numpy.ndarray
        Full dataset of features
    base_labels : numpy.ndarray
        Corresponding true labels
    n_clients : int
        Number of clients to distribute data to
    cluster_distribution : list of dict, optional
        Specifies distribution preferences for each client. 
    random_state : int, optional
        Seed for reproducible randomness
    max_clusters_per_client : int, optional
        Maximum number of clusters a client can use. 
        Default is 5, must be at least 2.

    Returns:
    --------
    list of numpy.ndarray
        List of data subsets for each client
    """
    # Validate max_clusters_per_client
    if max_clusters_per_client < 2:
        raise ValueError("max_clusters_per_client must be at least 2")

    # Set random seed for reproducibility
    rng = np.random.default_rng(random_state)
    
    # Get unique clusters
    unique_clusters = np.unique(base_labels)
    n_total_clusters = len(unique_clusters)
    
    # Create a mapping from cluster to data points
    clusters = {label: base_data[base_labels == label] for label in unique_clusters}
    cluster_labels = {label: label * np.ones(len(clusters[label]), dtype=base_labels.dtype) for label in unique_clusters}

    
    # If no distribution is specified, generate a random one
    if cluster_distribution is None:
        cluster_distribution = []
        for _ in range(n_clients):
            # Randomly select clusters
            n_selected_clusters = rng.integers(
                2, 
                min(n_total_clusters, max_clusters_per_client) + 1
            )
            selected_clusters = rng.choice(unique_clusters, n_selected_clusters, replace=False)
            
            # Generate random proportions that sum to 1
            proportions = rng.dirichlet(np.ones(n_selected_clusters))
            
            client_dist = {
                'clusters': selected_clusters,
                'proportions': proportions
            }
            cluster_distribution.append(client_dist)

    # Validate cluster distribution
    if len(cluster_distribution) != n_clients:
        raise ValueError(f"Cluster distribution must specify {n_clients} client distributions")

    client_data = []
    client_labels = []
    for client_dist in cluster_distribution:
        client_clusters = client_dist['clusters']
        client_proportions = client_dist['proportions']

        # Collect data for this client
        client_subset_data = []
        client_subset_labels = []
        for cluster, proportion in zip(client_clusters, client_proportions):
            cluster_data = clusters[cluster]
            cluster_client_labels = cluster_labels[cluster]

            # Determine number of samples to draw
            n_samples = int(len(cluster_data) * proportion)

            # Randomly sample with replacement if needed
            if n_samples > 0:
                sampled_indices = rng.choice(
                    len(cluster_data),
                    size=n_samples,
                    replace=False
                )
                client_subset_data.append(cluster_data[sampled_indices])
                client_subset_labels.append(cluster_client_labels[sampled_indices])

        # Combine data for this client
        client_subset = np.vstack(client_subset_data)
        client_subset_labels = np.concatenate(client_subset_labels)
        client_data.append(client_subset)
        client_labels.append(client_subset_labels)

    return client_data, client_labels, cluster_distribution

def generate_synthetic_batch(base_data, base_labels, 
                              n_samples=10,
                              cluster_distribution=None,
                              distribution_shift_type='mild',
                              random_state=None):
    """
    Generate a synthetic new batch of data with controlled distribution characteristics.

    Parameters:
    -----------
    base_data : numpy.ndarray
        Original dataset to use as a reference for synthetic data generation
    base_labels : numpy.ndarray
        Original labels corresponding to base_data
    n_samples : int, optional
    cluster_distribution : dict, optional
    distribution_shift_type : str, optional
        Type of distribution shift to simulate:
        - 'mild': Small, subtle changes in data distribution
        - 'moderate': Noticeable shifts in data characteristics
        - 'significant': Large, potentially disruptive changes
    random_state : int, optional
        Seed for reproducible randomness

    Returns:
    --------
    tuple: (new_batch_data, new_batch_labels)
        Synthetic new batch of data and corresponding labels
    """
    # Set random seed
    rng = np.random.default_rng(random_state)

    # Get unique labels and their current distribution
    unique_labels = np.unique(base_labels)

    # Create a mapping from cluster to data points
    clusters = {label: base_data[base_labels == label] for label in unique_labels}
    cluster_labels = {label: label * np.ones(len(clusters[label]), dtype=base_labels.dtype) for label in unique_labels}



    client_clusters = cluster_distribution['clusters']
    client_proportions = cluster_distribution['proportions']

    # Based on the distribution shift type, new clusters may be added to the client.
    if distribution_shift_type == 'mild':
        cluster_shift = rng.integers(1, max(1, int(len(unique_labels) * 0.2)))
    elif distribution_shift_type == 'moderate':
        cluster_shift = rng.integers(1, max(2, int(len(unique_labels) * 0.3)))
    elif distribution_shift_type == 'significant':
        cluster_shift = rng.integers(1, max(3, int(len(unique_labels) * 0.5)))
    else:
        raise ValueError("Invalid distribution shift type")

    # Add or remove clusters
    if cluster_shift < 1:
        cluster_shift = 1

    if cluster_shift > len(np.setdiff1d(unique_labels, client_clusters)):
        cluster_shift = len(np.setdiff1d(unique_labels, client_clusters))

    # Add new clusters
    new_clusters = rng.choice(
        np.setdiff1d(unique_labels, client_clusters),
        cluster_shift,
        replace=False
    )
    client_clusters = np.concatenate([client_clusters, new_clusters])
    client_proportions = np.concatenate([client_proportions, rng.dirichlet(np.ones(cluster_shift))])

    # Based on the distribution shift type, a shift is applied to the proportions of the clusters
    if distribution_shift_type == 'mild':
        # Apply a mild shift to the cluster proportions
        shift = rng.uniform(-0.1, 0.1, len(client_proportions))
        client_proportions += shift
        client_proportions /= np.sum(client_proportions)
    elif distribution_shift_type == 'moderate':
        # Apply a moderate shift to the cluster proportions
        shift = rng.uniform(-0.2, 0.2, len(client_proportions))
        client_proportions += shift
        client_proportions /= np.sum(client_proportions)
    elif distribution_shift_type == 'significant':
        # Apply a significant shift to the cluster proportions
        shift = rng.uniform(-0.5, 0.5, len(client_proportions))
        client_proportions += shift
        client_proportions /= np.sum(client_proportions)
    else:
        raise ValueError("Invalid distribution shift type")

    # Initialize new batch data and labels
    new_batch_data = []
    new_batch_labels = []

    for cluster, proportion in zip(client_clusters, client_proportions):
        cluster_data = clusters[cluster]
        cluster_client_labels = cluster_labels[cluster]

        # Randomly sample with replacement if needed
        if n_samples > 0:
            sampled_indices = rng.choice(
                len(cluster_data),
                size=min(n_samples, len(cluster_data)),
                replace=False
            )
            new_batch_data.append(cluster_data[sampled_indices])
            new_batch_labels.append(cluster_client_labels[sampled_indices])

    new_batch_data = np.vstack(new_batch_data)
    new_batch_labels = np.concatenate(new_batch_labels)

    return new_batch_data, new_batch_labels

def append_client_data(current_data, current_labels,
                       new_batch_data, new_batch_labels,
                       max_data_size=None):
    """
    Append new data to the existing client data.

    Parameters:
    -----------
    current_data : numpy.ndarray
        Current client's feature data
    current_labels : numpy.ndarray
        Current client's corresponding labels
    new_batch_data : numpy.ndarray
        New batch of feature data to be added
    new_batch_labels : numpy.ndarray
        Labels corresponding to the new batch
    max_data_size : int, optional
        Maximum number of data points to keep for the client
        If None, no size limit is applied

    Returns:
    --------
    tuple: (updated_data, updated_labels)
        Updated client data and corresponding labels
    """
    # Combine current and new data
    updated_data = np.vstack([current_data, new_batch_data])
    updated_labels = np.concatenate([current_labels, new_batch_labels])

    # Apply max_data_size if specified
    if max_data_size is not None and len(updated_data) > max_data_size:
        # Truncate to max_data_size, keeping the most recent data
        updated_data = updated_data[-max_data_size:]
        updated_labels = updated_labels[-max_data_size:]

    return updated_data, updated_labels

def sample_test_data(base_data, base_labels, test_size=0.2):
    n_test_samples = int(len(base_data) * test_size)
    indices = np.random.choice(len(base_data), n_test_samples, replace=False)
    return base_data[indices], base_labels[indices]

def example_generate_synthetic_data():
    data = generate_synthetic_data(n_clients=5)
    plt.figure(figsize=(12, 6))
    for i, X in enumerate(data):
        plt.subplot(2, 3, i+1)
        plt.scatter(X[:, 0], X[:, 1])
        plt.title(f"Client {i+1}")
    plt.tight_layout()
    plt.show()

def plot_data(base_data, base_labels, client_data):
    # Plot base dataset together with client data. Visualize the partitioned data for each client with different colors.
    lim = 13
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.scatter(base_data[:, 0], base_data[:, 1], c=base_labels, cmap="tab10", alpha=0.6)
    plt.title("Base Dataset with True Clusters")
    plt.subplot(1, 2, 2)
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    # colormap = plt.cm.get_cmap("tab10", len(client_data))
    colormap = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
    symbols = ["o", "s", "D", "v", "^", ">", "<", "p", "P", "*"]
    for i, data in enumerate(client_data):
        # print(f"Client {i} has {len(data)} samples, color: {colormap[i]}, marker: {symbols[i]}")
        plt.scatter(data[:, 0], data[:, 1], alpha=0.3, c=colormap[i], marker=symbols[i])
    plt.legend([f"Client {i}" for i in range(len(client_data))])
    plt.title("Partitioned Data for Clients")
    # Set same limits for both plots, based on the base dataset
    plt.tight_layout()
    plt.show()

def plot_data_after_aggregation(clients, server):
    colormap = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
    # Plot all data, the local clusters, and the global clusters
    plt.figure(figsize=(12, 6))
    for i, client in enumerate(clients):
        plt.subplot(2, 3, i+1)
        plt.scatter(client.data[:, 0], client.data[:, 1], c='gray', alpha=0.3)
        plt.scatter(client.centroids[:, 0], client.centroids[:, 1], c=colormap[i], s=100, alpha=0.5)
        plt.title(f"Client {i} with {client.n_clusters} clusters using {client.clustering_method}")

    plt.subplot(2, 3, 6)
    for i, client in enumerate(clients):
        plt.scatter(client.data[:, 0], client.data[:, 1], c='gray', alpha=0.3)
        plt.scatter(client.centroids[:, 0], client.centroids[:, 1], c=colormap[i], s=30, alpha=0.7, label=f"Client {i}")
    plt.scatter(server.global_centroids[:, 0], server.global_centroids[:, 1], marker="X", c='black', s=100, alpha=0.6, label="Global Centroids")
    plt.title("Global Centroids")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

def example_base_dataset():
    # Create base dataset
    base_data, base_labels = create_base_dataset(n_samples=1000, n_features=2, n_clusters=config.n_centers_generated)
    # Partition the data
    client_data = partition_data(base_data, base_labels, n_clients=3)
    # Sample test data
    test_data, test_labels = sample_test_data(base_data, base_labels)

    plot_data(base_data, base_labels, client_data)

if __name__ == "__main__":
    # example_generate_synthetic_data()
    example_base_dataset()
