# import json
# import numpy as np
# from scipy import stats
# import sys

# def analyze_performance(data_json):
#     # Parse input data
#     # data = json.loads(data_json)
#     data = data_json
    
#     # Client count analysis
#     client_counts = sorted([float(k) for k in data.keys()], key=float)
#     breakpoint()
#     global_final = [data[str(k)]["global_avg"][-1] for k in client_counts]
#     local_final = [data[str(k)]["local_only_avg"][-1] for k in client_counts]

#     # Calculate correlations
#     global_r = np.corrcoef(client_counts, global_final)[0,1]
#     local_r = np.corrcoef(client_counts, local_final)[0,1]
    
#     # Calculate recovery percentage (using max global score as proxy for theoretical max)
#     theoretical_max = max([max(v["global_avg"]) for v in data.values()])
#     recovery_percent = (max(global_final) / theoretical_max) * 100
    
#     # Adaptation speed (using slope of improvement)
#     global_slopes = []
#     local_slopes = []
#     for k in client_counts:
#         global_scores = data[str(k)]["global_avg"]
#         local_scores = data[str(k)]["local_only_avg"]
        
#         # Calculate slope of last 50 iterations
#         x = np.arange(150, 200)
#         global_slope, _ = np.polyfit(x, global_scores[-50:], 1)
#         local_slope, _ = np.polyfit(x, local_scores[-50:], 1)
        
#         global_slopes.append(global_slope)
#         local_slopes.append(local_slope)
    
#     adaptation_gain = ((np.mean(global_slopes) - np.mean(local_slopes)) / 
#                       np.mean(local_slopes)) * 100

#     return {
#         "global_correlation_r2": global_r**2,
#         "local_correlation_r2": local_r**2,
#         "recovery_percentage": recovery_percent,
#         "adaptation_gain_percent": adaptation_gain
#     }

# if __name__ == "__main__":
#     if len(sys.argv) > 1:
#         json_files = sys.argv[1:]

#     for json_file in json_files:
#         print(f"Reading data from {json_file}")
#         with open(json_file, "r") as f:
#             data = json.load(f)

#         import re
#         match = re.search(r"results_silhouette_data_(.+?)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.json$", json_file)
#         param_name = match.group(1).replace("_", " ") if match else None
#         param_name = param_name.replace(" ", "_")
#         print(f"Analyzing results for {param_name}")

#         # # Example usage with n_clients data:
#         # n_clients_data = """
#         # {
#         # "4": {"global_avg": [...], "local_only_avg": [...]},
#         # "8": {"global_avg": [...], "local_only_avg": [...]},
#         # "16": {"global_avg": [...], "local_only_avg": [...]}
#         # }
#         # """

#         results = analyze_performance(data)
#         print(f"Global model-param R²: {results['global_correlation_r2']:.2f}")
#         print(f"Local model-param count R²: {results['local_correlation_r2']:.2f}")
#         print(f"Recovery percentage: {results['recovery_percentage']:.1f}%")
#         print(f"Adaptation gain: {results['adaptation_gain_percent']:.1f}%")
#         print("===============================")
import json
import numpy as np
import sys
import re
from scipy import stats

def analyze_performance(data, param_name):
    """Analyze performance metrics for different parameter types"""
    # Parse parameter values based on type
    try:
        if param_name == "n_clients":
            param_values = [int(k) for k in data.keys()]
        elif param_name == "merging_threshold":
            param_values = [float(k) for k in data.keys()]
        elif param_name == "max_iterations_clustering":
            param_values = [int(k) for k in data.keys()]
        else:  # Fallback for unknown parameters
            param_values = [float(k) if '.' in k else int(k) for k in data.keys()]
    except ValueError:
        raise ValueError(f"Unable to parse parameter values for {param_name}")

    # Sort values while maintaining data association
    sorted_pairs = sorted(zip(param_values, data.values()), key=lambda x: x[0])
    sorted_params = [p for p, _ in sorted_pairs]
    sorted_data = [d for _, d in sorted_pairs]

    # Extract final scores
    global_final = [d["global_avg"][-1] for d in sorted_data]
    local_final = [d["local_only_avg"][-1] for d in sorted_data]
    print(sorted_params)
    print(global_final)
    print(local_final)
    # Calculate how much increase from smallest to largest parameter value
    global_increase = (global_final[-1] - global_final[0]) / global_final[0] * 100
    local_increase = (local_final[-1] - local_final[0]) / local_final[0] * 100
    # Print increase
    print(f"Global increase: {global_increase:.2f}%")
    print(f"Local increase: {local_increase:.2f}%")

    # Calculate correlations
    global_r = stats.pearsonr(sorted_params, global_final)[0]
    local_r = stats.pearsonr(sorted_params, local_final)[0]
    
    # Calculate recovery percentage
    all_global_scores = [score for d in data.values() for score in d["global_avg"]]
    theoretical_max = max(all_global_scores) if all_global_scores else 0
    recovery_percent = (max(global_final) / theoretical_max * 100) if theoretical_max else 0

    # Calculate adaptation rates
    global_slopes, local_slopes = [], []
    for d in sorted_data:
        # Use last 50 iterations or all available if fewer
        window = min(50, len(d["global_avg"]))
        x = np.arange(window)
        
        g_scores = d["global_avg"][-window:]
        l_scores = d["local_only_avg"][-window:]
        
        global_slopes.append(np.polyfit(x, g_scores, 1)[0])
        local_slopes.append(np.polyfit(x, l_scores, 1)[0])
    
    adaptation_gain = ((np.nanmean(global_slopes) - np.nanmean(local_slopes)) / 
                      np.abs(np.nanmean(local_slopes))) * 100

    return {
        "parameter": param_name,
        "global_correlation_r2": global_r**2,
        "local_correlation_r2": local_r**2,
        "recovery_percentage": recovery_percent,
        "adaptation_gain_percent": adaptation_gain,
        "optimal_value": sorted_params[np.argmax(global_final)]
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze.py <result_file1.json> [<result_file2.json> ...]")
        sys.exit(1)

    for json_file in sys.argv[1:]:
        print(f"\nAnalyzing {json_file}")
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Extract parameter name from filename
        match = re.search(r"results_silhouette_data_(.+?)_\d{4}", json_file)
        param_name = match.group(1).replace('_', ' ') if match else "unknown_parameter"
        param_name = param_name.replace(' ', '_')  # Standardize naming

        try:
            results = analyze_performance(data, param_name)
            print(f"Parameter: {results['parameter'].replace('_', ' ').title()}")
            print(f"Optimal Value: {results['optimal_value']}")
            print(f"Global Model Correlation (R²): {results['global_correlation_r2']:.2f}")
            print(f"Local Model Correlation (R²): {results['local_correlation_r2']:.2f}")
            print(f"Recovery Percentage: {results['recovery_percentage']:.1f}%")
            print(f"Adaptation Gain: {results['adaptation_gain_percent']:.1f}%")
        except Exception as e:
            print(f"Error analyzing {json_file}: {str(e)}")