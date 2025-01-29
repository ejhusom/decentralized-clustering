from sklearn.cluster import MeanShift
import numpy as np

class FedMeanShiftAggregator:
    def aggregate(self, local_centroids):
        ms = MeanShift()
        ms.fit(np.vstack(local_centroids))
        return ms.cluster_centers_