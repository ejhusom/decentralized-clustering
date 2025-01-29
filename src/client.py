import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, MeanShift, MiniBatchKMeans
from scipy.spatial.distance import cdist
# from sklearn_extra.cluster import KMedoids
# import kmedoids
from sklearn.neighbors import NearestCentroid

import config

class LocalClient:
    def __init__(self, client_id, data, n_clusters=None, clustering_method="kmeans", visualize=False):
        self.client_id = client_id
        self.data = data
        self.n_clusters = n_clusters
        self.clustering_method = clustering_method
        self.centroids = None
        self.global_centroids = None
        self.metadata = None
        self.model = None
        self.visualize = visualize

        # if self.clustering_method not in ["kmeans", "mini_batch_kmeans", "meanshift", "kmedoids"]:
        if self.clustering_method not in ["kmeans", "mini_batch_kmeans", "meanshift"]:
            # raise ValueError("Invalid clustering method. Supported methods: ['kmeans', 'mini_batch_kmeans', 'meanshift', 'kmedoids']")
            raise ValueError("Invalid clustering method. Supported methods: ['kmeans', 'mini_batch_kmeans', 'meanshift']")

        if self.clustering_method in ["meanshift"] and self.n_clusters is not None:
            raise ValueError("Meanshift does not require n_clusters. Please set n_clusters=None")

    def train(self):
        if self.clustering_method == "kmeans":
            model = self._train_kmeans()
        elif self.clustering_method == "mini_batch_kmeans":
            model = self._train_mini_batch_kmeans()
        elif self.clustering_method == "meanshift":
            model = self._train_meanshift()
        # elif self.clustering_method == "kmedoids":
        #     model = self._train_kmedoids()

        model.fit(self.data)
        self.labels = model.labels_
        self.n_clusters = len(np.unique(model.labels_))

        # If cluster centroids are not defined by the model, calculate the centroids using NearestCentroid and the labels
        try:
            self.centroids = model.cluster_centers_
            # manual_centroids = np.array([np.mean(self.data[model.labels_ == i], axis=0) for i in range(self.n_clusters)])
            # clf = NearestCentroid()
            # clf.fit(self.data, model.labels_)
            # print(self.centroids)
            # print(manual_centroids)
            # print(clf.centroids_)
        except AttributeError:
            self.centroids = np.array([np.mean(self.data[model.labels_ == i], axis=0) for i in range(self.n_clusters)])

        self.metadata = {
            "weights": np.bincount(model.labels_),
            "variance": [np.mean(np.var(self.data[model.labels_ == i], axis=0))  # Scalar variance
                        for i in range(self.n_clusters)],
        }

        if self.visualize:
            plt.scatter(self.data[:, 0], self.data[:, 1], c=model.labels_, cmap='viridis')
            plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', s=100, alpha=0.5)
            plt.title(f"Client {self.client_id} with {self.n_clusters} clusters using {self.clustering_method}")
            plt.show()

        self.model = model
        
    def _train_kmeans(self):
        return KMeans(n_clusters=self.n_clusters, random_state=42, max_iter=config.max_iterations_clustering)

    def _train_mini_batch_kmeans(self):
        return MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42, max_iter=config.max_iterations_clustering)

    def _train_meanshift(self):
        return MeanShift()

    # def _train_kmedoids(self):
        # return KMedoids(n_clusters=self.n_clusters, random_state=42, max_iter=config.max_iterations_clustering)
        # return kmedoids.KMedoids(n_clusters=self.n_clusters, random_state=42, max_iter=config.max_iterations_clustering)

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

    def retrain(self, global_centroids):
        # Initialize clustering with global centroids
        if self.clustering_method == "kmeans":
            model = KMeans(n_clusters=len(global_centroids), init=global_centroids, n_init=1, random_state=42, max_iter=config.max_iterations_clustering)
        elif self.clustering_method == "mini_batch_kmeans":
            model = MiniBatchKMeans(n_clusters=len(global_centroids), init=global_centroids, n_init=1, random_state=42, max_iter=config.max_iterations_clustering)
        elif self.clustering_method == "meanshift":
            model = MeanShift(seeds=global_centroids)
        # elif self.clustering_method == "kmedoids":
            # model = KMedoids(n_clusters=len(global_centroids), init=global_centroids, n_init=1, random_state=42, max_iter=config.max_iterations_clustering)
            # model = kmedoids.KMedoids(n_clusters=len(global_centroids), init=global_centroids, n_init=1, random_state=42, max_iter=config.max_iterations_clustering)
        else:
            print("Only KMeans and MeanShift are supported for retraining, falling back to KMeans")
            model = KMeans(n_clusters=len(global_centroids), init=global_centroids, n_init=1, random_state=42, max_iter=config.max_iterations_clustering)

        model.fit(self.data)
        self.centroids = model.cluster_centers_

        if self.visualize:
            plt.scatter(self.data[:, 0], self.data[:, 1], c=model.labels_, cmap='viridis')
            plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', s=100, alpha=0.5)
            plt.title(f"Client {self.client_id} with {self.n_clusters} clusters using {self.clustering_method}")
            plt.show()

        return self.centroids
