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

def plot_prototype_evolution(prototype_counts, data_dict, dataset_name, device='cuda', 
                             affinity_data=None, affinity_embeddings=None, output_dim=2, loss_function="kl_loss", n_seeds=3):
    """
    Iterates over a range of prototype numbers, runs the experiment for each with multiple seeds, 
    and plots the evolution of scores for each method with variance envelopes.
    """
    
    # Initialize storage for results
    # Structure: methods -> metrics -> list of lists (seeds)
    methods = ['DistR', 'DR_then_Clust', 'Clust_then_DR']
    metrics = ["hom", "ami", "ari", "nmi", "sil"]
    
    # history[method][metric] will be a list of length len(prototype_counts)
    # each element will be a list of scores for that prototype count across seeds
    history = {method: {metric: [] for metric in metrics} for method in methods}
    
    for n in prototype_counts:
        print(f"\n\n>>>>> Running for {n} prototypes <<<<<")
        
        # Temporary storage for this prototype count
        current_proto_scores = {method: {metric: [] for metric in metrics} for method in methods}
        
        for seed in range(n_seeds):
            print(f"   --- Seed {seed+1}/{n_seeds} ---")
            results = run_experiment(
                data_dict, 
                dataset_name, 
                n_prototypes=n, 
                device=device,
                affinity_data=affinity_data,
                affinity_embeddings=affinity_embeddings,
                output_dim=output_dim,
                loss_function=loss_function,
                seed=seed
            )
            
            for method in methods:
                for metric in metrics:
                    val = results[method].get(metric, 0)
                    current_proto_scores[method][metric].append(val)
        
        # Append the list of scores for this prototype count to history
        for method in methods:
            for metric in metrics:
                history[method][metric].append(current_proto_scores[method][metric])
                
    # Plotting
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
        
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for method in methods:
            # data: (n_prototypes, n_seeds)
            data = np.array(history[method][metric])
            
            means = np.mean(data, axis=1)
            stds = np.std(data, axis=1)
            
            ax.plot(prototype_counts, means, marker='o', label=method)
            ax.fill_between(prototype_counts, means - stds, means + stds, alpha=0.2)
            
        ax.set_title(metric.upper())
        ax.set_xlabel("Number of Prototypes")
        ax.set_ylabel("Score")
        if i == 0:
            ax.legend()
            
    plt.tight_layout()
    filename = f"{dataset_name}_prototype_evolution.png"
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    # plt.show() # Cannot show in headless environment, but saving is good.
