import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
import json
import csv
import sys
import config_parameters

def plot_metrics(metrics_dict, param_name, param_values, metric_name="silhouette", output_dir="output"):
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for param_value in param_values:
        metrics = metrics_dict[param_value]
        if param_name == "n_clients":
            local_avg = np.mean([metrics[f"client_{client_id}"]["local"][metric_name] for client_id in range(param_value)], axis=0)
            global_avg = np.mean([metrics[f"client_{client_id}"]["global"][metric_name] for client_id in range(param_value)], axis=0)
        else:
            local_avg = np.mean([metrics[f"client_{client_id}"]["local"][metric_name] for client_id in range(config_parameters.n_clients)], axis=0)
            global_avg = np.mean([metrics[f"client_{client_id}"]["global"][metric_name] for client_id in range(config_parameters.n_clients)], axis=0)
        ax[0].plot(local_avg, label=f"{param_name}={param_value}")
        ax[1].plot(global_avg, label=f"{param_name}={param_value}")

    ax[0].set_title(f"Local average {metric_name} scores")
    ax[1].set_title(f"Global average {metric_name} scores")
    ax[0].legend(loc='lower right')
    ax[1].legend(loc='lower right')
    ax[0].set_ylim([0, 1])
    ax[1].set_ylim([0, 1])

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(output_dir + f"/{metric_name}_scores_{param_name}_{timestamp}.pdf")
    # plt.show()

    # Make another plot, only plotting the global scores
    fig, ax = plt.subplots(1, 1, figsize=(4,3))
    for param_value in param_values:
        metrics = metrics_dict[param_value]
        if param_name == "n_clients":
            global_avg = np.mean([metrics[f"client_{client_id}"]["global"][metric_name] for client_id in range(param_value)], axis=0)
        else:
            global_avg = np.mean([metrics[f"client_{client_id}"]["global"][metric_name] for client_id in range(config_parameters.n_clients)], axis=0)
        ax.plot(global_avg, label=f"{param_name}={param_value}")
    ax.set_title(f"Global average {metric_name} scores")
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1])

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(output_dir + f"/global_{metric_name}_scores_{param_name}_{timestamp}.pdf")
    # plt.show()

def plot_metrics_with_local_only(metrics_dict, param_name, param_values, metric_name="silhouette", output_dir="output"):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
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
        
        ax[0].plot(local_avg, label=f"{param_name}={param_value}")
        ax[1].plot(global_avg, label=f"{param_name}={param_value}")
        ax[2].plot(local_only_avg, label=f"{param_name}={param_value}")

    ax[0].set_title(f"Local average {metric_name} scores")
    ax[1].set_title(f"Global average {metric_name} scores")
    ax[2].set_title(f"Local only average {metric_name} scores")
    ax[0].legend(loc='lower right')
    ax[1].legend(loc='lower right')
    ax[2].legend(loc='lower right')
    ax[0].set_ylim([0, 1])
    ax[1].set_ylim([0, 1])
    ax[2].set_ylim([0, 1])

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(output_dir + f"/{metric_name}_scores_with_local_only_{param_name}_{timestamp}.pdf")
    # plt.show()

    # Make another plot, only plotting the global scores
    fig, ax = plt.subplots(1, 1, figsize=(4,3))
    for param_value in param_values:
        metrics = metrics_dict[param_value]
        if param_name == "n_clients":
            global_avg = np.mean([metrics[f"client_{client_id}"]["global"][metric_name] for client_id in range(param_value)], axis=0)
        else:
            global_avg = np.mean([metrics[f"client_{client_id}"]["global"][metric_name] for client_id in range(config_parameters.n_clients)], axis=0)
        ax.plot(global_avg, label=f"{param_name}={param_value}")
    ax.set_title(f"Global average {metric_name} scores")
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1])

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(output_dir + f"/global_{metric_name}_scores_with_local_only_{param_name}_{timestamp}.pdf")
    # plt.show()

def plot_metrics_global_and_local_only(metrics_dict, param_name, param_values, metric_name="silhouette", output_dir="output", one_plot=True):
    data_to_save = {}
    if one_plot:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        for param_value in param_values:
            metrics = metrics_dict[param_value]
            if param_name == "n_clients":
                global_avg = np.mean([metrics[f"client_{client_id}"]["global"][metric_name] for client_id in range(param_value)], axis=0)
                local_only_avg = np.mean([metrics[f"client_{client_id}"]["local_only"][metric_name] for client_id in range(param_value)], axis=0)
            else:
                global_avg = np.mean([metrics[f"client_{client_id}"]["global"][metric_name] for client_id in range(config_parameters.n_clients)], axis=0)
                local_only_avg = np.mean([metrics[f"client_{client_id}"]["local_only"][metric_name] for client_id in range(config_parameters.n_clients)], axis=0)
            
            color = next(ax._get_lines.prop_cycler)['color']
            ax.plot(global_avg, label=f"Global {param_name}={param_value}", color=color)
            ax.plot(local_only_avg, label=f"Local Only {param_name}={param_value}", linestyle='dotted', color=color)

            data_to_save[param_value] = {
                "global_avg": global_avg.tolist(),
                "local_only_avg": local_only_avg.tolist()
            }

        ax.set_title(f"Global and Local Only average {metric_name} scores")
        ax.legend(loc='lower right')
        ax.set_ylim([0, 1])

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(output_dir + f"/{metric_name}_scores_global_and_local_only_{param_name}_{timestamp}.pdf")
        # plt.show()
    else:
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        for param_value in param_values:
            metrics = metrics = metrics_dict[param_value]
            if param_name == "n_clients":
                global_avg = np.mean([metrics[f"client_{client_id}"]["global"][metric_name] for client_id in range(param_value)], axis=0)
                local_only_avg = np.mean([metrics[f"client_{client_id}"]["local_only"][metric_name] for client_id in range(param_value)], axis=0)
            else:
                global_avg = np.mean([metrics[f"client_{client_id}"]["global"][metric_name] for client_id in range(config_parameters.n_clients)], axis=0)
                local_only_avg = np.mean([metrics[f"client_{client_id}"]["local_only"][metric_name] for client_id in range(config_parameters.n_clients)], axis=0)
            
            color = next(ax[0]._get_lines.prop_cycler)['color']
            ax[0].plot(global_avg, label=f"{param_name}={param_value}", color=color)
            ax[1].plot(local_only_avg, label=f"{param_name}={param_value}", linestyle='dotted', color=color)

            data_to_save[param_value] = {
                "global_avg": global_avg.tolist(),
                "local_only_avg": local_only_avg.tolist()
            }

        ax[0].set_title(f"Global average {metric_name} scores")
        ax[1].set_title(f"Local only average {metric_name} scores")
        ax[0].legend(loc='lower right')
        ax[1].legend(loc='lower right')
        ax[0].set_ylim([0, 1])
        ax[1].set_ylim([0, 1])

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(output_dir + f"/{metric_name}_scores_global_and_local_only_{param_name}_{timestamp}.pdf")
        # plt.show()

    # Save the plotted data to JSON
    with open(output_dir + f"/results_{metric_name}_data_{param_name}_{timestamp}.json", "w") as f:
        json.dump(data_to_save, f)

    # Save the plotted data to CSV
    csv_file = output_dir + f"/results_{metric_name}_data_{param_name}_{timestamp}.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["param_value", "global_avg", "local_only_avg"])
        for param_value, data in data_to_save.items():
            writer.writerow([param_value, data["global_avg"], data["local_only_avg"]])

def plot_from_json_files(json_files, output_dir="output"):
    for json_file in json_files:
        print(f"Plotting data from {json_file}")
        with open(json_file, "r") as f:
            data = json.load(f)
        
        # Paran name must be inferred from [directory/]results_silhouette_data_[param_name_with_underlines]_[timestamp].json. The param_name can have two or more words separated by underscores.
        import re
        # Extract param_name using regex
        match = re.search(r"results_silhouette_data_(.+?)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.json$", json_file)
        param_name = match.group(1).replace("_", " ") if match else None
        param_name = param_name.replace(" ", "_")

        # Make a list of ten colors to cycle through for the lines of the plot
        colors = plt.cm.get_cmap('tab10', 10).colors

        print(param_name)  # Output: "my param name"
        print(f"Parameter name: {param_name}")
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        # If param is merging_threshold, the local_only_avg should be averaged into one line.
        if param_name == "merging_threshold":
            for i, (param_value, metrics) in enumerate(data.items()):
                global_avg = metrics["global_avg"]
                local_only_avg = metrics["local_only_avg"]
                color = colors[i % len(colors)]
                ax.plot(global_avg, label=f"Global {param_name}={param_value}", color=color)
            
            # Average the local_only_avg into one line
            local_only_avg_param_avg = np.mean([metrics["local_only_avg"] for metrics in data.values()], axis=0)
            ax.plot(local_only_avg_param_avg, label=f"Local Only", linestyle='dotted', color='black')
        else:
            for i, (param_value, metrics) in enumerate(data.items()):
                global_avg = metrics["global_avg"]
                local_only_avg = metrics["local_only_avg"]
                color = colors[i % len(colors)]
                ax.plot(global_avg, label=f"Global {param_name}={param_value}", color=color)
                ax.plot(local_only_avg, label=f"Local Only {param_name}={param_value}", linestyle='dotted', color=color)

        # Set axes labels
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Silhouette score")
        ax.set_title(f"Global and Local Only average silhouette scores")
        ax.legend(loc='lower right')
        ax.set_ylim([0, 1])

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(output_dir + f"/reproduced_{param_name}_scores_{timestamp}.pdf")
        # plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        json_files = sys.argv[1:]
        plot_from_json_files(json_files)