#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-liner describing module.

Author:
    Erik Johannes Husom

Created:
    2021

"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.spatial.distance import cdist

class DynamicClusteringExperiment:
    def __init__(
        self, 
        n_clients: int = 5, 
        n_initial_samples: int = 200, 
        n_features: int = 2, 
        n_initial_clusters: int = 3, 
        n_iterations: int = 10,
        cluster_drift_magnitude: float = 0.5,
        noise_level: float = 0.1
    ):
        """
        Initialize a dynamic clustering experiment with evolving data streams.
        
        Parameters:
        -----------
        n_clients : int
            Number of clients in the federated learning setup
        n_initial_samples : int
            Initial number of samples per client
        n_features : int
            Number of features in the data
        n_initial_clusters : int
            Initial number of clusters per client
        n_iterations : int
            Number of federated learning iterations
        cluster_drift_magnitude : float
            Magnitude of cluster center shifts between iterations
        noise_level : float
            Level of noise to introduce in data generation
        """
        self.n_clients = n_clients
        self.n_initial_samples = n_initial_samples
        self.n_features = n_features
        self.n_initial_clusters = n_initial_clusters
        self.n_iterations = n_iterations
        self.cluster_drift_magnitude = cluster_drift_magnitude
        self.noise_level = noise_level
        
        # Tracking experiment results
        self.ari_history = []
        self.silhouette_history = []
        
        # Initialize experiment
        self._generate_initial_data()
    
    def _generate_initial_data(self):
        """
        Generate initial datasets for each client with controlled variability.
        """
        self.client_datasets = []
        self.client_labels = []
        
        for client_id in range(self.n_clients):
            # Create base clusters with some randomness
            centers = np.random.randn(self.n_initial_clusters, self.n_features)
            
            # Generate data for this client
            X, y = self._generate_clustered_data(
                centers, 
                n_samples=self.n_initial_samples,
                cluster_std=np.random.uniform(0.5, 1.5)
            )
            
            self.client_datasets.append(X)
            self.client_labels.append(y)
    
    def _generate_clustered_data(
        self, 
        centers: np.ndarray, 
        n_samples: int, 
        cluster_std: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate clustered data with controlled variance.
        
        Parameters:
        -----------
        centers : np.ndarray
            Cluster centers
        n_samples : int
            Total number of samples
        cluster_std : float
            Standard deviation for data generation
        
        Returns:
        --------
        X : np.ndarray
            Generated feature data
        y : np.ndarray
            Corresponding cluster labels
        """
        n_clusters = len(centers)
        samples_per_cluster = n_samples // n_clusters
        
        X, y = [], []
        for cluster_id, center in enumerate(centers):
            cluster_samples = samples_per_cluster + (n_samples % n_clusters if cluster_id < n_samples % n_clusters else 0)
            
            # Generate cluster data with noise
            cluster_data = np.random.normal(
                loc=center, 
                scale=cluster_std, 
                size=(cluster_samples, len(center))
            )
            
            X.append(cluster_data)
            y.append(np.full(cluster_samples, cluster_id))
        
        return np.vstack(X), np.concatenate(y)
    
    def evolve_client_data(self, global_centroids: np.ndarray = None):
        """
        Evolve client datasets, simulating data stream with controlled drift.
        
        Parameters:
        -----------
        global_centroids : np.ndarray, optional
            Global centroids from previous iteration to guide data evolution
        """
        new_datasets = []
        new_labels = []
        
        for client_id, (current_data, current_labels) in enumerate(zip(self.client_datasets, self.client_labels)):
            # Determine cluster centers
            if global_centroids is not None:
                # Adapt to global centroids with some local variation
                base_centers = global_centroids + np.random.normal(
                    scale=self.cluster_drift_magnitude, 
                    size=global_centroids.shape
                )
            else:
                # Use current cluster centers with drift
                unique_labels = np.unique(current_labels)
                base_centers = np.array([
                    current_data[current_labels == label].mean(axis=0) 
                    for label in unique_labels
                ])
                
                # Add drift
                base_centers += np.random.normal(
                    scale=self.cluster_drift_magnitude, 
                    size=base_centers.shape
                )
            
            # Generate new data
            new_samples = self.n_initial_samples // 2  # Half of initial samples
            X, y = self._generate_clustered_data(
                base_centers, 
                n_samples=new_samples,
                cluster_std=np.random.uniform(0.5, 1.5)
            )
            
            new_datasets.append(X)
            new_labels.append(y)
        
        # Update datasets
        self.client_datasets = new_datasets
        self.client_labels = new_labels
    
    def run_experiment(self):
        """
        Run the full dynamic clustering experiment.
        """
        # Initial global centroids (random initialization)
        global_centroids = np.random.randn(
            self.n_initial_clusters * self.n_clients, 
            self.n_features
        )
        
        for iteration in range(self.n_iterations):
            print(f"Iteration {iteration + 1}/{self.n_iterations}")
            
            # Simulate local clustering and global aggregation
            local_centroids = self.aggregate_local_models(global_centroids)
            
            # Global centroid aggregation (simplified)
            global_centroids = np.mean(local_centroids, axis=0)
            
            # Evaluate global model
            ari, silhouette = self.evaluate_global_model(global_centroids)
            self.ari_history.append(ari)
            self.silhouette_history.append(silhouette)
            
            # Evolve data for next iteration
            self.evolve_client_data(global_centroids)
    
    def aggregate_local_models(self, global_centroids):
        """
        Simulate local clustering and centroid generation.
        
        Parameters:
        -----------
        global_centroids : np.ndarray
            Global centroids from previous iteration
        
        Returns:
        --------
        local_centroids : np.ndarray
            Centroids generated by each client
        """
        local_centroids = []
        
        for client_id, (data, labels) in enumerate(zip(self.client_datasets, self.client_labels)):
            # Use global centroids as initialization guidance
            distances = cdist(data, global_centroids)
            predicted_labels = np.argmin(distances, axis=1)
            
            # Local centroid generation
            client_centroids = np.array([
                data[predicted_labels == label].mean(axis=0)
                for label in np.unique(predicted_labels)
            ])
            
            local_centroids.append(client_centroids)
        
        return local_centroids
    
    def evaluate_global_model(self, global_centroids):
        """
        Evaluate global model across all client data.
        
        Parameters:
        -----------
        global_centroids : np.ndarray
            Global centroids to evaluate
        
        Returns:
        --------
        ari : float
            Adjusted Rand Index
        silhouette : float
            Silhouette score
        """
        # Combine all client data for evaluation
        all_data = np.vstack(self.client_datasets)
        all_labels = np.concatenate(self.client_labels)
        
        # Assign points to nearest centroids
        distances = cdist(all_data, global_centroids)
        predicted_labels = np.argmin(distances, axis=1)
        
        # Compute metrics
        ari = adjusted_rand_score(all_labels, predicted_labels)
        
        # Handle silhouette score for single cluster case
        if len(np.unique(predicted_labels)) > 1:
            silhouette = silhouette_score(all_data, predicted_labels)
        else:
            silhouette = None
        
        return ari, silhouette
    
    def plot_experiment_results(self):
        """
        Visualize experiment results over iterations.
        """
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.ari_history, marker='o')
        plt.title('Adjusted Rand Index over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('ARI')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.silhouette_history, marker='o', color='green')
        plt.title('Silhouette Score over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Silhouette Score')
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    
    experiment = DynamicClusteringExperiment(
        n_clients=5,
        n_initial_samples=200,
        n_features=2,
        n_initial_clusters=3,
        n_iterations=10,
        cluster_drift_magnitude=0.5,
        noise_level=0.1
    )
    
    experiment.run_experiment()
    experiment.plot_experiment_results()


