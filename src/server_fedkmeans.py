import numpy as np

class FedKMeansAggregator:
    def aggregate(self, local_models):
        all_centroids = [centroids for centroids, _, _ in local_models]
        return np.mean(np.vstack(all_centroids), axis=0)