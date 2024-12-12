import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, MeanShift
from scipy.spatial.distance import cdist

class LocalClient:
    def __init__(self, client_id, data, n_clusters=None, clustering_method="kmeans", visualize=False):
        self.client_id = client_id
        self.data = data
        self.n_clusters = n_clusters
        self.clustering_method = clustering_method
        self.centroids = None
        self.metadata = None
        self.model = None
        self.visualize = visualize

        if self.clustering_method not in ["kmeans", "meanshift"]:
            raise ValueError("Invalid clustering method. Supported methods: ['kmeans',, 'meanshift']")

        if self.clustering_method in ["meanshift"] and self.n_clusters is not None:
            raise ValueError("Meanshift does not require n_clusters. Please set n_clusters=None")

    def train(self):
        if self.clustering_method == "kmeans":
            model = self._train_kmeans()
        elif self.clustering_method == "meanshift":
            model = self._train_meanshift()

        model.fit(self.data)
        self.centroids = model.cluster_centers_
        self.labels = model.labels_
        self.n_clusters = len(self.centroids)
        self.metadata = {
            "weights": np.bincount(model.labels_),
            "variance": [np.var(self.data[model.labels_ == i], axis=0) for i in range(self.n_clusters)],
        }

        if self.visualize:
            plt.scatter(self.data[:, 0], self.data[:, 1], c=model.labels_, cmap='viridis')
            plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', s=100, alpha=0.5)
            plt.title(f"Client {self.client_id} with {self.n_clusters} clusters using {self.clustering_method}")
            plt.show()

        self.model = model
        
    def _train_kmeans(self):
        return KMeans(n_clusters=self.n_clusters, random_state=42)

    def _train_meanshift(self):
        return MeanShift()

    def get_model(self):
        return self.centroids, self.metadata, self.model

    def label_with_global_model(self, global_centroids):
        # Compute distances to global centroids
        distances = cdist(self.data, global_centroids)
        # Assign each point to the nearest global centroid
        global_labels = np.argmin(distances, axis=1)
        self.global_labels = global_labels

        if self.visualize:
            plt.scatter(self.data[:, 0], self.data[:, 1], c=global_labels, cmap='viridis')
            plt.scatter(global_centroids[:, 0], global_centroids[:, 1], c='red', s=100, alpha=0.5)
            plt.title(f"Client {self.client_id} with global labels")
            plt.show()

        return global_labels

    # def retrain_with_global_centroids(self, global_centroids):
    #     # Initialize clustering with global centroids
    #     model = KMeans(n_clusters=len(global_centroids), init=global_centroids, n_init=1, random_state=42)
    #     model.fit(self.data)
    #     self.centroids = model.cluster_centers_
    #     self.metadata = {
    #         "weights": np.bincount(model.labels_),
    #         "variance": [np.var(self.data[model.labels_ == i], axis=0) for i in range(self.n_clusters)],
    #     }

    #     if self.visualize:
    #         plt.scatter(self.data[:, 0], self.data[:, 1], c=model.labels_, cmap='viridis')
    #         plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', s=100, alpha=0.5)
    #         plt.title(f"Client {self.client_id} with {self.n_clusters} clusters using {self.clustering_method}")
    #         plt.show()

    #     return self.centroids
