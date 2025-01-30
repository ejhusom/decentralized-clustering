import json
import os
import sys
from main import plot_metrics, print_gain_summary

def analyze_gains(output_dir):
    gain_files = [f for f in os.listdir(output_dir) if f.endswith("_gains.json")]
    overall_absolute_gains = []
    overall_relative_gains = []

    report_lines = []

    for gain_file in gain_files:
        param_name = gain_file.replace("_gains.json", "")
        with open(os.path.join(output_dir, gain_file), "r") as f:
            gains = json.load(f)
            report_lines.append(f"\nSummary of gains for {param_name}:")
            for param_value, gain in gains.items():
                report_lines.append(f"{param_name}={param_value}:")
                report_lines.append(f"  Absolute gain: {gain['absolute_gain']:.4f}")
                report_lines.append(f"  Relative gain: {gain['relative_gain']:.2f}%")
                overall_absolute_gains.append(gain["absolute_gain"])
                overall_relative_gains.append(gain["relative_gain"])

    overall_absolute_gain_avg = sum(overall_absolute_gains) / len(overall_absolute_gains)
    overall_relative_gain_avg = sum(overall_relative_gains) / len(overall_relative_gains)

    report_lines.append("\nOverall average gains:")
    report_lines.append(f"  Absolute gain: {overall_absolute_gain_avg:.4f}")
    report_lines.append(f"  Relative gain: {overall_relative_gain_avg:.2f}%")

    report = "\n".join(report_lines)
    print(report)

    with open(os.path.join(output_dir, "gain_summary.txt"), "w") as f:
        f.write(report)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_results.py <output_dir>")
        sys.exit(1)

    output_dir = sys.argv[1]
    analyze_gains(output_dir)

