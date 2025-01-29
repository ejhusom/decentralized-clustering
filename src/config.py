# n_centers_generated = 10
# n_samples = 5000
# n_samples_per_client = 100
# n_features = 2
# random_state = 42
# merging_threshold = 4.0
# visualize = False
# verbose = False
# n_iterations = 50
# max_iterations_clustering = 50
# n_clients = 12
# clustering_methods = ["kmeans"] * (n_clients // 3) + ["mini_batch_kmeans"] * (n_clients // 3) + ["meanshift"] * (n_clients // 3)
# n_clusters = [8] * (n_clients // 3) + [8] * (n_clients // 3) + [None] * (n_clients // 3)
# # n_clusters = [5, 6, 7, 8, 5, 6, 7, 8, None, None, None, None]
# use_dynamic_threshold = True  # Use variances for threshold adjustment
# use_weighted_merging = True   # Use weights for centroid averaging

n_centers_generated = 10  # Reduced for clearer patterns
n_samples = 3000
n_samples_per_client = 250  # More data per client
n_features = 2
random_state = 42
merging_threshold = 1.5  # Tightened merging
visualize = False  # Enable temporarily for debugging
verbose = False  # See per-iteration metrics
n_iterations = 30
max_iterations_clustering = 1  # Ensure convergence
n_clients = 9  # Reduced complexity

# Client configuration
clustering_methods = ["kmeans"]*3 + ["mini_batch_kmeans"]*3 + ["meanshift"]*3
# n_clusters = [None]*9  # All clients auto-detect clusters
n_clusters = [7, 8, 9, 8, 9, 10, None, None, None]  # Some clients auto-detect clusters

# Merging parameters
use_dynamic_threshold = True
use_weighted_merging = True

# Data evolution
distribution_shift_type = "moderate"  # From "significant"
new_data_percentage = 0.05  # From 0.1