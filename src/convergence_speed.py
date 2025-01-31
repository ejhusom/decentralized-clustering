import json
import numpy as np
import sys

def calculate_acceleration_factor(data):
    """Calculate how much faster global model reaches peak performance"""
    acceleration_factors = []
    
    for param_value in data.values():
        # Get score trajectories
        global_scores = param_value["global_avg"]
        local_scores = param_value["local_only_avg"]
        
        # Find peak performance thresholds (95% of max)
        global_threshold = 0.95 * max(global_scores)
        local_threshold = 0.95 * max(local_scores)
        
        # Find first iteration crossing threshold
        global_iter = next(i for i, score in enumerate(global_scores) 
                          if score >= global_threshold)
        try:
            local_iter = next(i for i, score in enumerate(local_scores) 
                             if score >= local_threshold)
        except StopIteration:
            local_iter = len(local_scores)  # If never reaches threshold
            
        # Calculate acceleration factor
        if global_iter == 0:  # Handle instant convergence
            acceleration = np.nan
        else:
            acceleration = local_iter / global_iter
            
        acceleration_factors.append(acceleration)
    
    # Filter invalid values and return statistics
    valid_factors = [x for x in acceleration_factors if not np.isnan(x)]
    return {
        "mean_acceleration": np.mean(valid_factors),
        "median_acceleration": np.median(valid_factors),
        "std_acceleration": np.std(valid_factors),
        "num_valid_samples": len(valid_factors)
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python acceleration.py <result_file1.json> [<result_file2.json> ...]")
        sys.exit(1)

    all_factors = []
    
    for json_file in sys.argv[1:]:
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        results = calculate_acceleration_factor(data)
        all_factors.extend([results["mean_acceleration"]] * results["num_valid_samples"])
        
        print(f"\nFile: {json_file}")
        print(f"Mean acceleration factor: {results['mean_acceleration']:.1f}x")
        print(f"Median acceleration factor: {results['median_acceleration']:.1f}x")
        print(f"Based on {results['num_valid_samples']} valid parameter configurations")

    # Combined results across all files
    print("\nOverall Acceleration Analysis:")
    print(f"Grand mean acceleration: {np.nanmean(all_factors):.1f}x")
    print(f"Grand median acceleration: {np.nanmedian(all_factors):.1f}x")
    print(f"Total configurations analyzed: {len(all_factors)}")
    print(f"Standard deviation: {np.nanstd(all_factors):.2f}")