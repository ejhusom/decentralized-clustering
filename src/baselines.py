from sklearn.cluster import KMeans
import numpy as np

class CentralizedBaseline:
    def __init__(self, n_true_clusters):
        self.model = KMeans(n_clusters=n_true_clusters, random_state=42)
        
    def run(self, all_data):
        self.model.fit(np.vstack(all_data))
        return self.model.cluster_centers_