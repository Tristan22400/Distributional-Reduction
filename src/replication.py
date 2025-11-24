import torch
import numpy as np
import matplotlib.pyplot as plt
from src.clust_dr import DR_then_Clust, Clust_then_DR, DistR
from src.affinities import NormalizedGaussianAndStudentAffinity, SymmetricEntropicAffinity
from src.scores import compare_scores_sklearn

def get_majority_vote_labels(T, Y):
    """
    Assigns a class label to each prototype based on the majority class 
    of the samples mapped to it (weighted by the transport plan T).
    Operations are fully vectorized on GPU.
    """
    # T: (N_samples, n_prototypes) [GPU]
    # Y: (N_samples,) [CPU/Numpy]
    
    n_prototypes = T.shape[1]
    # Convert Y to GPU tensor once
    Y_tensor = torch.as_tensor(Y, device=T.device, dtype=torch.long)
    n_classes = Y_tensor.max().item() + 1
    
    # One-hot encode Y to aggregate mass: (N_samples, n_classes)
    # We use scatter on GPU
    Y_onehot = torch.zeros((len(Y), n_classes), device=T.device, dtype=T.dtype)
    Y_onehot.scatter_(1, Y_tensor.unsqueeze(1), 1.0)
    
    # Aggregate mass: Project Y onto Prototypes via T^T
    # vote_matrix: (n_prototypes, n_classes)
    # This matrix multiplication is the heavy lifting, kept on GPU
    vote_matrix = torch.mm(T.T, Y_onehot)
    
    # Get majority class for each prototype
    proto_labels = torch.argmax(vote_matrix, dim=1).cpu().numpy()
    return proto_labels

def evaluate_prototypes(Z, T, Y_true, X):
    """
    Computes metrics using the centralized scores implementation.
    """
    # Ensure inputs are tensors on the correct device
    if not isinstance(Z, torch.Tensor):
        Z = torch.tensor(Z)
    if not isinstance(T, torch.Tensor):
        T = torch.tensor(T)
    if not isinstance(Y_true, torch.Tensor):
        Y_true = torch.tensor(Y_true, device=T.device)
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, device=T.device)
        
    # Call the comparison function
    our_scores, sklearn_scores = compare_scores_sklearn(T, Z, Y_true, X)
    
    return our_scores

def run_experiment(data_dict, dataset_name, n_prototypes=50, device='cuda', 
                   affinity_data=None, affinity_embeddings=None, output_dim=2, loss_function="kl_loss", seed=0):
    """
    Runs DistR and Baselines on a single dataset with a specific seed.
    """
    print(f"=== Running Experiment on {dataset_name} (Seed={seed}) ===")
    print(f"Configuration: N={data_dict['X'].shape[0]}, Prototypes={n_prototypes}, Device={device}")
    
    # OPTIMIZATION: Use float32 for faster GPU computation (Tensor Cores)
    # Create tensor directly on device to avoid CPU->GPU copy
    X = torch.as_tensor(data_dict['X'], dtype=torch.float32, device=device)
    Y = data_dict['Y'] 
    
    # Initialize Affinities if not provided
    if affinity_data is None:
        # [cite: 329] Symmetric Entropic Affinity for X
        affinity_data = SymmetricEntropicAffinity(perp=30, verbose=False) 
    
    if affinity_embeddings is None:
        # [cite: 329] Student t-distribution for Z
        affinity_embeddings = NormalizedGaussianAndStudentAffinity(student=True) 
    
    results = {}
    
    # 1. DistR (Ours)
    print(f"\n[1/3] Training DistR (Device: {device})...")
    model_distr = DistR(
        affinity_data=affinity_data,
        affinity_embedding=affinity_embeddings,
        output_dim=output_dim,
        output_sam=n_prototypes,
        loss_fun=loss_function,       # [cite: 120]
        optimizer="Adam",         # [cite: 233]
        lr=0.1,
        max_iter=200,             # BCD Iterations [cite: 232]
        device=device,
        dtype=torch.float32,
        init="normal",
        init_T="kmeans",
        seed=seed
    )
    Z_distr = model_distr.fit_transform(X)
    metrics_distr = evaluate_prototypes(Z_distr, model_distr.T, Y, X)
    results['DistR'] = metrics_distr
    print(f"DistR Results: {metrics_distr}")

    # 2. DR -> Clustering
    print(f"\n[2/3] Training DR -> Clustering (Device: {device})...")
    model_drc = DR_then_Clust(
        affinity_data=affinity_data,
        affinity_embedding=affinity_embeddings,
        output_sam=n_prototypes,
        output_dim=2,
        loss_fun=loss_function,
        device=device,
        dtype=torch.float32,
        init="normal",
        init_T="kmeans",
        seed=seed
    )
    Z_drc = model_drc.fit_transform(X)
    metrics_drc = evaluate_prototypes(Z_drc, model_drc.T, Y, X)
    results['DR_then_Clust'] = metrics_drc
    print(f"DR->C Results: {metrics_drc}")

    # 3. Clustering -> DR
    print(f"\n[3/3] Training Clustering -> DR (Device: {device})...")
    model_cdr = Clust_then_DR(
        affinity_data=affinity_data,
        affinity_embedding=affinity_embeddings,
        output_sam=n_prototypes,
        output_dim=2,
        loss_fun=loss_function,
        device=device,
        dtype=torch.float32,
        init="normal",
        init_T="kmeans",
        seed=seed
    )
    Z_cdr = model_cdr.fit_transform(X)
    metrics_cdr = evaluate_prototypes(Z_cdr, model_cdr.T, Y, X)
    results['Clust_then_DR'] = metrics_cdr
    print(f"C->DR Results: {metrics_cdr}")
    
    return results

def plot_prototype_evolution(results_dict, prototype_counts, methods=['DistR', 'DR_then_Clust', 'Clust_then_DR'], 
                             metrics=["hom", "ami", "ari", "nmi", "sil"], filename="prototype_evolution.png"):
    """
    Plots the evolution of scores for multiple datasets in a grid.
    
    Args:
        results_dict: Dictionary where keys are dataset names and values are history dictionaries.
                      history structure: method -> metric -> list of lists (scores across seeds for each proto count)
        prototype_counts: List of prototype counts used.
        methods: List of method names to plot.
        metrics: List of metrics to plot.
        filename: Output filename for the plot.
    """
    n_datasets = len(results_dict)
    n_metrics = len(metrics)
    
    # Create a grid of subplots: rows = datasets, cols = metrics
    fig, axes = plt.subplots(n_datasets, n_metrics, figsize=(5 * n_metrics, 5 * n_datasets), squeeze=False)
    
    dataset_names = list(results_dict.keys())
    
    for row_idx, dataset_name in enumerate(dataset_names):
        history = results_dict[dataset_name]
        
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            
            for method in methods:
                # data: (n_prototypes, n_seeds)
                # We need to ensure we have data for this method/metric
                if method in history and metric in history[method]:
                    data = np.array(history[method][metric])
                    
                    if data.size > 0:
                        means = np.mean(data, axis=1)
                        stds = np.std(data, axis=1)
                        
                        ax.plot(prototype_counts, means, marker='o', label=method)
                        ax.fill_between(prototype_counts, means - stds, means + stds, alpha=0.2)
            
            # Set titles and labels
            if row_idx == 0:
                ax.set_title(metric.upper(), fontsize=14, fontweight='bold')
            
            if col_idx == 0:
                ax.set_ylabel(f"{dataset_name}\nScore", fontsize=12, fontweight='bold')
            else:
                ax.set_ylabel("Score")
                
            ax.set_xlabel("Number of Prototypes")
            
            # Add legend only to the first subplot to avoid clutter
            if row_idx == 0 and col_idx == 0:
                ax.legend()
                
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    # plt.show() # Commented out for headless execution
