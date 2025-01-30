# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.spatial.distance import cdist
# from sklearn.cluster import MeanShift

# class ServerAggregator:
#     def __init__(self, merging_threshold=1.0, visualize=False):
#         self.global_centroids = None
#         self.merging_threshold = merging_threshold
#         self.visualize = visualize

#     def aggregate(self, local_models, method='pairwise'):
#         # Collect all centroids and metadata
#         all_centroids = []
#         for centroids, metadata, _ in local_models:
#             all_centroids.append(centroids)
#         all_centroids = np.vstack(all_centroids)

#         self.unmerged_centroids = all_centroids

#         if method == 'pairwise':
#             # Merge clusters based on distance threshold
#             self.global_centroids = self._merge_centroids(all_centroids)
#         elif method == 'meanshift':
#             # Merge clusters using MeanShift clustering
#             self.global_centroids = self._merge_centroids_clustering(all_centroids)

#     def _merge_centroids(self, centroids, variances, weights):
#         distances = cdist(centroids, centroids, metric='euclidean')
#         np.fill_diagonal(distances, np.inf)  # Ignore self-comparisons
        
#         merged_centroids = []
#         used = set()
        
#         for i in range(len(centroids)):
#             if i in used:
#                 continue
                
#             merge_group_indices = [i]
#             total_weight = weights[i] if self.use_weighted_merging else 0
            
#             # Initialize tracking for weighted merging
#             if self.use_weighted_merging:
#                 total_weight = weights[i]
            
#             for j in range(i+1, len(centroids)):
#                 if j in used:
#                     continue
                
#                 # Calculate dynamic threshold if enabled
#                 # if self.use_dynamic_threshold:
#                 #     avg_variance = (variances[i] + variances[j]) / 2
#                 #     threshold = self.merging_threshold * (1 + avg_variance)
#                 if self.use_dynamic_threshold:
#                     # Use geometric mean of variances
#                     joint_variance = np.sqrt(variances[i] * variances[j])
#                     threshold = self.merging_threshold * (1 + joint_variance)
#                 else:
#                     threshold = self.merging_threshold
                
#                 if distances[i, j] < threshold:
#                     merge_group_indices.append(j)
#                     used.add(j)
#                     if self.use_weighted_merging:
#                         total_weight += weights[j]
            
#             # Calculate merged centroid
#             # if self.use_weighted_merging:
#             #     weighted_sum = np.sum(weights[merge_group_indices][:, None] * 
#             #                         centroids[merge_group_indices], axis=0)
#             #     merged_centroid = weighted_sum / total_weight
#             # Weighted merging with minimum weight threshold
#             if self.use_weighted_merging:
#                 MIN_WEIGHT = 0.01  # Reject negligible clusters
#                 valid = weights[merge_group] > MIN_WEIGHT
#                 if np.any(valid):
#                     weighted_sum = np.sum(weights[merge_group][valid] * centroids[merge_group][valid])
#             else:
#                 merged_centroid = np.mean(centroids[merge_group_indices], axis=0)
            
#             merged_centroids.append(merged_centroid)
        
#         if self.visualize:
#             self._plot_merging(centroids, merged_centroids)
            
#         return np.array(merged_centroids)


#     def _merge_centroids_clustering(self, centroids):
#         # Use MeanShift clustering to merge centroids
#         ms = MeanShift()
#         ms.fit(centroids)
#         merged_centroids = ms.cluster_centers_

#         if self.visualize:
#             plt.scatter(centroids[:, 0], centroids[:, 1], c='blue', s=50, alpha=0.5)
#             plt.scatter(merged_centroids[:, 0], merged_centroids[:, 1], c='red', s=100, alpha=0.5)
#             plt.title("Global Centroids")
#             plt.show()

#         return merged_centroids


import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import MeanShift

class ServerAggregator:
    def __init__(self, merging_threshold=1.0, visualize=False,
                 use_dynamic_threshold=False, use_weighted_merging=False):
        self.global_centroids = None
        self.merging_threshold = merging_threshold
        self.visualize = visualize
        self.use_dynamic_threshold = use_dynamic_threshold  # Variance-based threshold
        self.use_weighted_merging = use_weighted_merging    # Weight-based averaging

    def aggregate(self, local_models, method='pairwise'):
        all_centroids = []
        all_variances = []
        all_weights = []
        
        # Collect metadata from all clients
        for centroids, metadata, _ in local_models:
            all_centroids.append(centroids)
            all_variances.extend(metadata["variance"])
            all_weights.extend(metadata["weights"])
            
        all_centroids = np.vstack(all_centroids)
        all_variances = np.array(all_variances)
        all_weights = np.array(all_weights)

        self.unmerged_centroids = all_centroids

        if method == 'pairwise':
            self.global_centroids = self._merge_centroids(
                all_centroids, all_variances, all_weights
            )
        elif method == 'meanshift':
            self.global_centroids = self._merge_centroids_clustering(all_centroids)
        
        return self.global_centroids

    def _merge_centroids(self, centroids, variances, weights):
        distances = cdist(centroids, centroids, metric='euclidean')
        np.fill_diagonal(distances, np.inf)  # Ignore self-comparisons
        
        merged_centroids = []
        used = set()
        
        for i in range(len(centroids)):
            if i in used:
                continue
                
            merge_group_indices = [i]
            total_weight = weights[i] if self.use_weighted_merging else 0
            
            # Initialize tracking for weighted merging
            if self.use_weighted_merging:
                total_weight = weights[i]
            
            for j in range(i+1, len(centroids)):
                if j in used:
                    continue
                
                # Calculate dynamic threshold if enabled
                # if self.use_dynamic_threshold:
                #     avg_variance = (variances[i] + variances[j]) / 2
                #     threshold = self.merging_threshold * (1 + avg_variance)
                if self.use_dynamic_threshold:
                    # Use geometric mean of variances
                    joint_variance = np.sqrt(variances[i] * variances[j])
                    threshold = self.merging_threshold * (1 + joint_variance)
                else:
                    threshold = self.merging_threshold
                
                if distances[i, j] < threshold:
                    merge_group_indices.append(j)
                    used.add(j)
                    if self.use_weighted_merging:
                        total_weight += weights[j]
            
            # Calculate merged centroid
            # if self.use_weighted_merging:
            #     weighted_sum = np.sum(weights[merge_group_indices][:, None] * 
            #                         centroids[merge_group_indices], axis=0)
            #     merged_centroid = weighted_sum / total_weight
            # Weighted merging with minimum weight threshold
            if self.use_weighted_merging:
                MIN_WEIGHT = 0.01  # Reject negligible clusters
                valid = weights[merge_group] > MIN_WEIGHT
                if np.any(valid):
                    weighted_sum = np.sum(weights[merge_group][valid] * centroids[merge_group][valid])
            else:
                merged_centroid = np.mean(centroids[merge_group_indices], axis=0)
            
            merged_centroids.append(merged_centroid)
        
        if self.visualize:
            self._plot_merging(centroids, merged_centroids)
            
        return np.array(merged_centroids)

    def _merge_centroids_clustering(self, centroids):
        ms = MeanShift()
        ms.fit(centroids)
        merged_centroids = ms.cluster_centers_
        
        if self.visualize:
            self._plot_merging(centroids, merged_centroids)
            
        return merged_centroids

    def _plot_merging(self, original, merged):
        plt.scatter(original[:, 0], original[:, 1], c='blue', s=50, alpha=0.5)
        plt.scatter(merged[:, 0], merged[:, 1], c='red', s=100, alpha=0.5)
        plt.title("Global Centroids")
        plt.show()