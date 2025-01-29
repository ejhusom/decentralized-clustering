import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import MeanShift

class ServerAggregator:
    def __init__(self, merging_threshold=1.0, visualize=False, density_aware=False):
        self.global_centroids = None
        self.merging_threshold = merging_threshold
        self.visualize = visualize
        self.density_aware = density_aware  # New flag

    def aggregate(self, local_models, method='pairwise'):
        all_centroids = []
        all_variances = []  # Collect variances from metadata
        for centroids, metadata, _ in local_models:
            all_centroids.append(centroids)
            all_variances.extend(metadata["variance"])
        all_centroids = np.vstack(all_centroids)
        all_variances = np.array(all_variances)
        
        self.unmerged_centroids = all_centroids

        if method == 'pairwise':
            self.global_centroids = self._merge_centroids(all_centroids, all_variances)
        elif method == 'meanshift':
            self.global_centroids = self._merge_centroids_clustering(all_centroids)
        
        return self.global_centroids

    def _merge_centroids(self, centroids, variances):
        distances = cdist(centroids, centroids, metric='euclidean')
        np.fill_diagonal(distances, np.inf)  # Ignore self-comparisons
        
        merged_centroids = []
        used = set()
        
        for i in range(len(centroids)):
            if i in used:
                continue
            merge_group = [centroids[i]]
            for j in range(i+1, len(centroids)):
                if j in used:
                    continue
                
                # Dynamic threshold calculation
                if self.density_aware:
                    avg_variance = (variances[i] + variances[j]) / 2
                    dynamic_threshold = self.merging_threshold * (1 + avg_variance)
                else:
                    dynamic_threshold = self.merging_threshold
                
                if distances[i, j] < dynamic_threshold:
                    merge_group.append(centroids[j])
                    used.add(j)
            
            merged_centroids.append(np.mean(merge_group, axis=0))
        
        if self.visualize:
            plt.scatter(centroids[:, 0], centroids[:, 1], c='blue', s=50, alpha=0.5)
            plt.scatter(np.array(merged_centroids)[:, 0], np.array(merged_centroids)[:, 1], c='red', s=100, alpha=0.5)
            plt.title("Global Centroids")
            plt.show()

        return np.array(merged_centroids)

    def _merge_centroids_clustering(self, centroids):
        # Use MeanShift clustering to merge centroids
        ms = MeanShift()
        ms.fit(centroids)
        merged_centroids = ms.cluster_centers_

        if self.visualize:
            plt.scatter(centroids[:, 0], centroids[:, 1], c='blue', s=50, alpha=0.5)
            plt.scatter(merged_centroids[:, 0], merged_centroids[:, 1], c='red', s=100, alpha=0.5)
            plt.title("Global Centroids")
            plt.show()

        return merged_centroids