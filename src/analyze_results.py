import json
import os
import sys
import numpy as np
import config_parameters
import matplotlib.pyplot as plt
import datetime
from plotting import plot_metrics_global_and_local_only

def calculate_average_gain(metrics_dict, param_name, param_values, metric_name="silhouette"):
    gains = {}
    for param_value in param_values:
        metrics = metrics_dict[param_value]
        if param_name == "n_clients":
            local_avg = np.mean([metrics[f"client_{client_id}"]["local"][metric_name] for client_id in range(param_value)], axis=0)
            global_avg = np.mean([metrics[f"client_{client_id}"]["global"][metric_name] for client_id in range(param_value)], axis=0)
            local_only_avg = np.mean([metrics[f"client_{client_id}"]["local_only"][metric_name] for client_id in range(param_value)], axis=0)
        else:
            local_avg = np.mean([metrics[f"client_{client_id}"]["local"][metric_name] for client_id in range(config_parameters.n_clients)], axis=0)
            global_avg = np.mean([metrics[f"client_{client_id}"]["global"][metric_name] for client_id in range(config_parameters.n_clients)], axis=0)
            local_only_avg = np.mean([metrics[f"client_{client_id}"]["local_only"][metric_name] for client_id in range(config_parameters.n_clients)], axis=0)
        
        absolute_gain_global = np.mean(global_avg - local_avg)
        relative_gain_global = np.mean((global_avg - local_avg) / local_avg) * 100  # in percentage

        absolute_gain_local_only = np.mean(local_only_avg - local_avg)
        relative_gain_local_only = np.mean((local_only_avg - local_avg) / local_avg) * 100  # in percentage

        gains[param_value] = {
            "global": {"absolute_gain": absolute_gain_global, "relative_gain": relative_gain_global},
            "local_only": {"absolute_gain": absolute_gain_local_only, "relative_gain": relative_gain_local_only}
        }
    
    return gains

def print_gain_summary(gains, param_name):
    print(f"\nSummary of gains for {param_name}:")
    for param_value, gain in gains.items():
        print(f"{param_name}={param_value}:")
        print(f"  Global - Absolute gain: {gain['global']['absolute_gain']:.4f}")
        print(f"  Global - Relative gain: {gain['global']['relative_gain']:.2f}%")
        print(f"  Local Only - Absolute gain: {gain['local_only']['absolute_gain']:.4f}")
        print(f"  Local Only - Relative gain: {gain['local_only']['relative_gain']:.2f}%")

def analyze_gains(output_dir):
    gain_files = [f for f in os.listdir(output_dir) if f.endswith("_gains.json")]
    overall_absolute_gains_global = []
    overall_relative_gains_global = []
    overall_absolute_gains_local_only = []
    overall_relative_gains_local_only = []

    report_lines = []

    for gain_file in gain_files:
        param_name = gain_file.replace("_gains.json", "")
        with open(os.path.join(output_dir, gain_file), "r") as f:
            gains = json.load(f)
            report_lines.append(f"\nSummary of gains for {param_name}:")
            for param_value, gain in gains.items():
                report_lines.append(f"{param_name}={param_value}:")
                report_lines.append(f"  Global - Absolute gain: {gain['global']['absolute_gain']:.4f}")
                report_lines.append(f"  Global - Relative gain: {gain['global']['relative_gain']:.2f}%")
                report_lines.append(f"  Local Only - Absolute gain: {gain['local_only']['absolute_gain']:.4f}")
                report_lines.append(f"  Local Only - Relative gain: {gain['local_only']['relative_gain']:.2f}%")
                overall_absolute_gains_global.append(gain["global"]["absolute_gain"])
                overall_relative_gains_global.append(gain["global"]["relative_gain"])
                overall_absolute_gains_local_only.append(gain["local_only"]["absolute_gain"])
                overall_relative_gains_local_only.append(gain["local_only"]["relative_gain"])

    overall_absolute_gain_avg_global = sum(overall_absolute_gains_global) / len(overall_absolute_gains_global)
    overall_relative_gain_avg_global = sum(overall_relative_gains_global) / len(overall_relative_gains_global)
    overall_absolute_gain_avg_local_only = sum(overall_absolute_gains_local_only) / len(overall_absolute_gains_local_only)
    overall_relative_gain_avg_local_only = sum(overall_relative_gains_local_only) / len(overall_relative_gains_local_only)

    report_lines.append("\nOverall average gains:")
    report_lines.append(f"  Global - Absolute gain: {overall_absolute_gain_avg_global:.4f}")
    report_lines.append(f"  Global - Relative gain: {overall_relative_gain_avg_global:.2f}%")
    report_lines.append(f"  Local Only - Absolute gain: {overall_absolute_gain_avg_local_only:.4f}")
    report_lines.append(f"  Local Only - Relative gain: {overall_relative_gain_avg_local_only:.2f}%")

    report = "\n".join(report_lines)
    print(report)

    with open(os.path.join(output_dir, "gain_summary.txt"), "w") as f:
        f.write(report)

def reproduce_plots(output_dir, metric_name="silhouette"):
    metric_files = [f for f in os.listdir(output_dir) if f.endswith("_metrics.json")]
    for metric_file in metric_files:
        param_name = metric_file.replace("_metrics.json", "")
        with open(os.path.join(output_dir, metric_file), "r") as f:
            metrics_dict = json.load(f)
        param_values = list(metrics_dict.keys())
        plot_metrics_global_and_local_only(metrics_dict, param_name, param_values, metric_name, output_dir)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <output_dir>")
        sys.exit(1)

    output_dir = sys.argv[1]
    analyze_gains(output_dir)
    reproduce_plots(output_dir)

