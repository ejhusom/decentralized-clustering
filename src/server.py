import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

class ServerAggregator:
    def __init__(self, merging_threshold=1.0, visualize=False):
        self.global_centroids = None
        self.merging_threshold = merging_threshold
        self.visualize = visualize

    def aggregate(self, local_models):
        # Collect all centroids and metadata
        all_centroids = []
        for centroids, metadata, _ in local_models:
            all_centroids.append(centroids)
        all_centroids = np.vstack(all_centroids)

        self.unmerged_centroids = all_centroids

        # Merge clusters based on distance threshold
        self.global_centroids = self._merge_centroids(all_centroids)

    def _merge_centroids(self, centroids):
        # Hierarchical clustering or simple pairwise distance merging
        distances = cdist(centroids, centroids, metric='euclidean')
        merged_centroids = []
        used = set()
        for i in range(len(centroids)):
            if i in used:
                continue
            merge_group = [centroids[i]]
            for j in range(i+1, len(centroids)):
                if distances[i, j] < self.merging_threshold:
                    merge_group.append(centroids[j])
                    used.add(j)
            merged_centroids.append(np.mean(merge_group, axis=0))

        if self.visualize:
            plt.scatter(centroids[:, 0], centroids[:, 1], c='blue', s=50, alpha=0.5)
            plt.scatter(np.array(merged_centroids)[:, 0], np.array(merged_centroids)[:, 1], c='red', s=100, alpha=0.5)
            plt.title("Global Centroids")
            plt.show()

        return np.array(merged_centroids)
