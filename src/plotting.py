import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
import json
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
            
            ax.plot(global_avg, label=f"Global {param_name}={param_value}")
            ax.plot(local_only_avg, label=f"Local Only {param_name}={param_value}", linestyle='dotted')

        ax.set_title(f"Global and Local Only average {metric_name} scores")
        ax.legend(loc='lower right')
        ax.set_ylim([0, 1])

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(output_dir + f"/{metric_name}_scores_global_and_local_only_{param_name}_{timestamp}.pdf")
        # plt.show()
    else:
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        for param_value in param_values:
            metrics = metrics_dict[param_value]
            if param_name == "n_clients":
                global_avg = np.mean([metrics[f"client_{client_id}"]["global"][metric_name] for client_id in range(param_value)], axis=0)
                local_only_avg = np.mean([metrics[f"client_{client_id}"]["local_only"][metric_name] for client_id in range(param_value)], axis=0)
            else:
                global_avg = np.mean([metrics[f"client_{client_id}"]["global"][metric_name] for client_id in range(config_parameters.n_clients)], axis=0)
                local_only_avg = np.mean([metrics[f"client_{client_id}"]["local_only"][metric_name] for client_id in range(config_parameters.n_clients)], axis=0)
            
            ax[0].plot(global_avg, label=f"{param_name}={param_value}")
            ax[1].plot(local_only_avg, label=f"{param_name}={param_value}")

        ax[0].set_title(f"Global average {metric_name} scores")
        ax[1].set_title(f"Local only average {metric_name} scores")
        ax[0].legend(loc='lower right')
        ax[1].legend(loc='lower right')
        ax[0].set_ylim([0, 1])
        ax[1].set_ylim([0, 1])

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(output_dir + f"/{metric_name}_scores_global_and_local_only_{param_name}_{timestamp}.pdf")
        # plt.show()