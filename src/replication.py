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

def plot_tradeoff_analysis(results_dict, methods=['DistR', 'DR_then_Clust', 'Clust_then_DR'], filename="tradeoff_analysis.png"):
    """
    Replicates Figure 3: Trade-off analysis between NMI and Homogeneity.
    
    Args:
        results_dict: Dictionary where keys are dataset names and values are history dictionaries.
                      history structure: method -> metric -> list of lists (scores across seeds for each proto count)
        methods: List of method names to plot.
        filename: Output filename for the plot.
    """
    import matplotlib.patches as patches

    # Setup figure - Single plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Define colors for methods
    method_colors = {
        'DistR': 'blue',
        'DR_then_Clust': 'green',
        'Clust_then_DR': 'orange'
    }
    # Fallback for other methods
    cmap_methods = plt.get_cmap("Set1")
    for i, m in enumerate(methods):
        if m not in method_colors:
            method_colors[m] = cmap_methods(i)

    # Define markers for datasets
    datasets = list(results_dict.keys())
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    dataset_markers = {d: markers[i % len(markers)] for i, d in enumerate(datasets)}

    # 1. Calculate Normalization Constants per Dataset
    # We need min/max for NMI and Hom for each dataset across ALL methods
    norm_stats = {} # dataset -> {nmi_min, nmi_max, hom_min, hom_max}
    
    for dataset in datasets:
        all_nmi = []
        all_hom = []
        
        history = results_dict[dataset]
        for method in methods:
            if method in history:
                # history[method]['nmi'] is list of lists (protos -> seeds)
                # Flatten it
                nmis = np.array(history[method].get('nmi', [])).flatten()
                homs = np.array(history[method].get('hom', [])).flatten()
                
                all_nmi.extend(nmis)
                all_hom.extend(homs)
        
        if not all_nmi or not all_hom:
            print(f"Warning: No NMI/Hom data for dataset {dataset}")
            continue
            
        norm_stats[dataset] = {
            'nmi_min': np.min(all_nmi),
            'nmi_max': np.max(all_nmi),
            'hom_min': np.min(all_hom),
            'hom_max': np.max(all_hom)
        }

    # Data structure to hold aggregated normalized scores for quantile calculation
    # method -> {'nmi': [], 'hom': []}
    aggregated_scores = {m: {'nmi': [], 'hom': []} for m in methods}

    # 2. Plotting Points
    for dataset in datasets:
        if dataset not in norm_stats:
            continue
            
        stats = norm_stats[dataset]
        history = results_dict[dataset]
        
        for method in methods:
            if method in history:
                raw_nmi = np.array(history[method].get('nmi', [])).flatten()
                raw_hom = np.array(history[method].get('hom', [])).flatten()
                
                if len(raw_nmi) == 0:
                    continue
                
                # Normalize
                nmi_denom = stats['nmi_max'] - stats['nmi_min']
                hom_denom = stats['hom_max'] - stats['hom_min']
                
                if nmi_denom > 1e-9:
                    norm_nmi = (raw_nmi - stats['nmi_min']) / nmi_denom
                else:
                    norm_nmi = np.zeros_like(raw_nmi)

                if hom_denom > 1e-9:
                    norm_hom = (raw_hom - stats['hom_min']) / hom_denom
                else:
                    norm_hom = np.zeros_like(raw_hom)
                
                # Store for quantiles
                aggregated_scores[method]['nmi'].extend(norm_nmi)
                aggregated_scores[method]['hom'].extend(norm_hom)

                # Plot
                ax.scatter(norm_nmi, norm_hom, 
                           c=method_colors[method],
                           marker=dataset_markers[dataset],
                           alpha=0.6, s=40)

    # 3. Quantile Rectangles
    for method in methods:
        nmis = np.array(aggregated_scores[method]['nmi'])
        homs = np.array(aggregated_scores[method]['hom'])
        
        if len(nmis) > 0:
            nmi_20, nmi_80 = np.percentile(nmis, [20, 80])
            hom_20, hom_80 = np.percentile(homs, [20, 80])
            
            # Draw Rectangle
            # (x, y), width, height
            rect = patches.Rectangle((nmi_20, hom_20), nmi_80 - nmi_20, hom_80 - hom_20,
                                     linewidth=1, edgecolor='none', facecolor=method_colors[method], alpha=0.2)
            ax.add_patch(rect)

    # 4. Final Touches
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Normalized k-means NMI", fontsize=12)
    ax.set_ylabel("Normalized Homogeneity", fontsize=12)
    ax.set_title("Trade-off Analysis", fontsize=14, fontweight='bold')
    
    # Custom Legends
    # Method Legend
    method_handles = [patches.Patch(color=method_colors[m], label=m) for m in methods]
    legend1 = ax.legend(handles=method_handles, title="Methods", loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.add_artist(legend1)
    
    # Dataset Legend
    dataset_handles = [plt.Line2D([0], [0], marker=dataset_markers[d], color='w', label=d, 
                                  markerfacecolor='k', markersize=8) for d in datasets]
    ax.legend(handles=dataset_handles, title="Datasets", loc='upper left', bbox_to_anchor=(1.05, 0.6))

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    print(f"Trade-off plot saved to {filename}")

