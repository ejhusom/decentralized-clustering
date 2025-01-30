n_centers_generated = 10
n_samples = 5000
n_samples_per_client = 10
n_features = 2
random_state = 42
merging_threshold = 4.0
visualize = False
verbose = False
n_iterations = 200
max_iterations_clustering = 10
n_clients = 10
clustering_methods = ["meanshift"] * (n_clients // 2) + ["kmeans"] * (n_clients // 2)
n_clusters = [None] * (n_clients // 2) + [8] * (n_clients // 2)
data_increase_factor = 0.1
distribution_shift_type = "significant"
# Merging parameters
use_dynamic_threshold = True
use_weighted_merging = True