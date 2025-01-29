n_centers_generated = 40
n_samples = 2000
n_samples_per_client = 10
n_features = 2
random_state = 42
merging_threshold = 4.0
visualize = False
verbose = False
n_iterations = 100
max_iterations_clustering = 1
n_clients = 12
clustering_methods = ["kmeans"] * (n_clients // 3) + ["mini_batch_kmeans"] * (n_clients // 3) + ["meanshift"] * (n_clients // 3)
n_clusters = [8] * (n_clients // 3) + [8] * (n_clients // 3) + [None] * (n_clients // 3)
density_aware_merging = False