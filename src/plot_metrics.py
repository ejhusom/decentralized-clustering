import matplotlib.pyplot as plt

def plot_metrics(baselines, falcon_results):
    plt.figure(figsize=(15, 5))
    
    # Silhouette Score
    plt.subplot(1, 3, 1)
    for name, res in baselines.items():
        plt.axhline(y=res['silhouette'], label=name, linestyle='--')
    plt.plot(falcon_results['silhouette'], label='FALC')
    plt.title("Silhouette Score")
    plt.legend()
    
    # Cluster Recall
    plt.subplot(1, 3, 2)
    for name, res in baselines.items():
        plt.axhline(y=res['recall'], label=name, linestyle='--')
    plt.plot(falcon_results['recall'], label='FALC')
    plt.title("Cluster Recall")
    
    # Purity
    plt.subplot(1, 3, 3)
    for name, res in baselines.items():
        plt.axhline(y=res['purity'], label=name, linestyle='--')
    plt.plot(falcon_results['purity'], label='FALC')
    plt.title("Cluster Purity")
    
    plt.tight_layout()
    plt.savefig("metrics_comparison.pdf")