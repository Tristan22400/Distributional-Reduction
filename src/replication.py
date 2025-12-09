import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from src.clust_dr import DR_then_Clust, Clust_then_DR, DistR
from src.affinities import NormalizedGaussianAndStudentAffinity, SymmetricEntropicAffinity
from src.scores import compare_scores_sklearn
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


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
    legend2 = ax.legend(handles=dataset_handles, title="Datasets", loc='upper left', bbox_to_anchor=(1.05, 0.6))

    plt.tight_layout()
    plt.savefig(filename, bbox_extra_artists=(legend1, legend2), bbox_inches='tight')
    print(f"Trade-off plot saved to {filename}")


# -------------------------------------------------------------------------
# Helpers: affinity extraction, medoids, shapes, plotting utilities
# -------------------------------------------------------------------------

def _to_torch(X, device="cpu", dtype=torch.float32):
    if isinstance(X, torch.Tensor):
        return X.to(device=device, dtype=dtype)
    return torch.as_tensor(X, device=device, dtype=dtype)

def _safe_get_attr(obj, names, default=None):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default

def get_affinity_matrix(affinity_obj, X):
    """
    Compatible with src/affinities.py API.
    """
    # Ensure torch tensor
    if not isinstance(X, torch.Tensor):
        X = torch.as_tensor(X, dtype=torch.float32)

    # Prefer the public API of your affinities
    if hasattr(affinity_obj, "compute_affinity"):
        C = affinity_obj.compute_affinity(X)
        return C

    if hasattr(affinity_obj, "compute_log_affinity"):
        logC = affinity_obj.compute_log_affinity(X)
        return torch.exp(logC)

    raise RuntimeError(
        "Affinity object has no compute_affinity/compute_log_affinity."
    )


def compute_cluster_medoids_from_Cx_T(Cx, T):
    """
    Implements the caption logic:
    medoid for cluster k is argmax_i [ (C_X @ T)_{i,k} ].
    Cx: (N, N)
    T:  (N, K)
    Returns: numpy array of shape (K,)
    """
    if not isinstance(Cx, torch.Tensor):
        Cx = torch.as_tensor(Cx)
    if not isinstance(T, torch.Tensor):
        T = torch.as_tensor(T)

    # Ensure same device/dtype
    Cx = Cx.to(device=T.device, dtype=T.dtype)

    scores = Cx @ T  # (N, K)
    medoid_idx = torch.argmax(scores, dim=0).detach().cpu().numpy()
    return medoid_idx

def infer_image_shape(X_row, fallback_square=True):
    """
    Infer H, W (and possibly C) from a flattened vector.
    """
    d = int(X_row.shape[0])
    if fallback_square:
        s = int(round(math.sqrt(d)))
        if s * s == d:
            return (s, s, 1)
    return (d, 1, 1)

def extract_images_from_data(data_dict):
    """
    Tries to retrieve raw images from data_dict.
    Priority:
      1) data_dict['images'] if present
      2) reshape from X using 'image_shape' if present
      3) infer square shape from X dimension
    Returns: numpy array (N, H, W) or (N, H, W, C)
    """
    if "images" in data_dict and data_dict["images"] is not None:
        imgs = data_dict["images"]
        return imgs

    X = data_dict["X"]
    if isinstance(X, torch.Tensor):
        Xn = X.detach().cpu().numpy()
    else:
        Xn = np.asarray(X)

    if "image_shape" in data_dict and data_dict["image_shape"] is not None:
        H, W = data_dict["image_shape"][:2]
        imgs = Xn.reshape(len(Xn), H, W)
        return imgs

    # Fallback: infer square
    H, W, _ = infer_image_shape(Xn[0])
    imgs = Xn.reshape(len(Xn), H, W)
    return imgs

def get_prototype_weights(model, n_prototypes):
    """
    Best-effort extraction of h_Z (prototype masses).
    """
    h = _safe_get_attr(model, ["hZ", "h_z", "hz", "h"], default=None)
    if h is None:
        # Fallback: uniform
        return np.ones(n_prototypes, dtype=np.float32) / float(n_prototypes)

    if isinstance(h, torch.Tensor):
        h = h.detach().cpu().numpy()
    else:
        h = np.asarray(h)

    # Normalize for safety
    s = h.sum()
    if s > 0:
        h = h / s
    return h

def plot_images_at_positions(ax, Z, images, medoid_idx, weights,
                             zoom_base=0.35, zoom_range=0.9):
    """
    Places medoid images at prototype positions.
    Image area ~ weight => zoom ~ sqrt(weight) scaling.
    """
    Z = np.asarray(Z)
    K = Z.shape[0]

    w = np.asarray(weights).clip(min=0)
    if w.sum() <= 0:
        w = np.ones_like(w) / len(w)

    # Normalize weights to [0,1] for stable scaling
    w_norm = w / (w.max() + 1e-12)

    for k in range(K):
        i = int(medoid_idx[k])
        img = images[i]

        # Ensure 2D for grayscale
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]

        # zoom scaling
        zoom = zoom_base + zoom_range * math.sqrt(float(w_norm[k]))

        oi = OffsetImage(img, zoom=zoom, cmap="gray" if img.ndim == 2 else None)
        ab = AnnotationBbox(oi, (Z[k, 0], Z[k, 1]), frameon=True, pad=0.15)
        ax.add_artist(ab)

    ax.set_xticks([])
    ax.set_yticks([])

def plot_poincare_disk(ax, Z, labels=None, title=None):
    """
    Simple Poincaré disk backdrop + scatter of points in the unit ball.
    We assume Z is already in the ball (||z||<1). If not, we rescale.
    """
    Z = np.asarray(Z)
    r = np.linalg.norm(Z, axis=1, keepdims=True)
    max_r = r.max() if len(r) else 1.0
    if max_r >= 0.999:
        Z = Z / (max_r + 1e-6) * 0.95

    # Draw unit circle
    circle = plt.Circle((0, 0), 1.0, fill=False, linewidth=2)
    ax.add_patch(circle)

    if labels is None:
        ax.scatter(Z[:, 0], Z[:, 1], s=35, alpha=0.9)
    else:
        labels = np.asarray(labels)
        ax.scatter(Z[:, 0], Z[:, 1], c=labels, s=35, alpha=0.9)

    ax.set_aspect("equal", "box")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)

# -------------------------------------------------------------------------
# Main: Figure 4 reproduction
# -------------------------------------------------------------------------

def fit_distr_for_figure4(data_dict, n_prototypes=50, output_dim=2,
                          device="cuda", seed=0, name="MNIST",
                          affinity_data=None, affinity_embeddings=None,
                          loss_function="kl_loss"):
    """
    Train DistR with the configuration indicated in the Figure 4 caption:
    - SEA similarity for C_X
    - Student kernel for C_Z
    """
    X = _to_torch(data_dict["X"], device=device, dtype=torch.float32)
    Y = data_dict.get("Y", None)

    if affinity_data is None:
        affinity_data = SymmetricEntropicAffinity(perp=30, verbose=False)

    if affinity_embeddings is None:
        affinity_embeddings = NormalizedGaussianAndStudentAffinity(student=True)

    distr_kwargs = dict(
        affinity_data=affinity_data,
        affinity_embedding=affinity_embeddings,
        output_dim=output_dim,
        output_sam=n_prototypes,
        loss_fun=loss_function,
        optimizer="Adam",
        lr=0.1,
        max_iter=200,
        device=device,
        dtype=torch.float32,
        init="normal",
        init_T="kmeans",
        seed=seed
    )

    model = DistR(
        affinity_data=affinity_data,
        affinity_embedding=affinity_embeddings,
        output_dim=output_dim,
        output_sam=n_prototypes,
        loss_fun=loss_function,
        optimizer="Adam",
        lr=0.1,
        max_iter=200,
        device=device,
        dtype=torch.float32,
        init="normal",
        init_T="kmeans",
        seed=seed
    )
    if name == "PBMC":
        for key, val in [
            ("output_space", "poincare"),
            ("space", "poincare"),
            ("manifold", "poincare"),
            ("hyperbolic", True),
        ]:
            try:
                model = DistR(**{**distr_kwargs, key: val})
                break
            except TypeError:
                model = None
        if model is None:
            model = DistR(**distr_kwargs)
    else:
        model = DistR(**distr_kwargs)

    Z = model.fit_transform(X)  # expected (K, output_dim)
    T = model.T                 # expected (N, K)

    # Convert Z to numpy
    if isinstance(Z, torch.Tensor):
        Z_np = Z.detach().cpu().numpy()
    else:
        Z_np = np.asarray(Z)

    return model, Z_np, T, Y, X

def reproduce_figure4(datasets,
                      n_prototypes_map=None,
                      device="cuda",
                      seed=0,
                      filename="figure4_replication.png",
                      save_path=save_path
                      ):
    """
    Reproduces Figure 4-style qualitative plots.

    Args:
        datasets: dict
            keys: dataset name among {"MNIST", "Fashion-MNIST", "COIL", "PBMC"}
            values: data_dict with at least "X" and ideally "Y"
                    For image datasets, you may also provide:
                        - "images" : (N,H,W) or (N,H,W,C)
                        - or "image_shape": (H,W)
        n_prototypes_map: dict or None
            optional per-dataset number of prototypes.
        device: "cuda" or "cpu"
        seed: int
        filename: output image
    """
    if n_prototypes_map is None:
        n_prototypes_map = {}

    order = ["MNIST", "Fashion-MNIST", "COIL", "PBMC"]
    present = [d for d in order if d in datasets]

    fig, axes = plt.subplots(1, len(present), figsize=(5.2 * len(present), 5))

    if len(present) == 1:
        axes = [axes]

    for ax, name in zip(axes, present):
        data_dict = datasets[name]
        n_prototypes = int(n_prototypes_map.get(name, 50))

        # Configure affinities
        affinity_data = SymmetricEntropicAffinity(perp=30, verbose=False)

        # For PBMC we attempt a hyperbolic-friendly Student affinity if your class supports it.
        if name == "PBMC":
            # best-effort: try to pass a flag understood by your implementation
            try:
                affinity_embeddings = NormalizedGaussianAndStudentAffinity(student=True, space="poincare")
            except Exception:
                try:
                    affinity_embeddings = NormalizedGaussianAndStudentAffinity(student=True, poincare=True)
                except Exception:
                    affinity_embeddings = NormalizedGaussianAndStudentAffinity(student=True)
        else:
            affinity_embeddings = NormalizedGaussianAndStudentAffinity(student=True)

        model, Z_np, T, Y, X_torch = fit_distr_for_figure4(
            data_dict,
            n_prototypes=n_prototypes,
            output_dim=2,
            device=device,
            seed=seed,
            affinity_data=affinity_data,
            affinity_embeddings=affinity_embeddings
        )

        # Prototype weights h_Z
        weights = get_prototype_weights(model, n_prototypes)

        if name in ["MNIST", "Fashion-MNIST", "COIL"]:
            # Compute Cx and medoids
            Cx = get_affinity_matrix(affinity_data, X_torch)  # (N,N)
            medoid_idx = compute_cluster_medoids_from_Cx_T(Cx, T)

            images = extract_images_from_data(data_dict)

            xmin, ymin = Z_np.min(axis=0)
            xmax, ymax = Z_np.max(axis=0)
            pad_x = 0.15 * (xmax - xmin + 1e-9)
            pad_y = 0.15 * (ymax - ymin + 1e-9)

            ax.set_xlim(xmin - pad_x, xmax + pad_x)
            ax.set_ylim(ymin - pad_y, ymax + pad_y)

            # Plot images at prototype positions
            plot_images_at_positions(
                ax, Z_np, images, medoid_idx, weights,
                zoom_base=0.25 if name != "COIL" else 0.22,
                zoom_range=0.95
            )
            ax.set_title(name)

        elif name == "PBMC":
            # Majority vote labels over prototypes (reuse your function if imported)
            try:
                from replication import get_majority_vote_labels  # in case this file is imported elsewhere
                proto_labels = get_majority_vote_labels(T, Y) if Y is not None else None
            except Exception:
                # Local lightweight version
                if Y is None:
                    proto_labels = None
                else:
                    Y_tensor = torch.as_tensor(Y, device=T.device, dtype=torch.long)
                    n_classes = int(Y_tensor.max().item()) + 1
                    Y_onehot = torch.zeros((len(Y), n_classes), device=T.device, dtype=T.dtype)
                    Y_onehot.scatter_(1, Y_tensor.unsqueeze(1), 1.0)
                    vote_matrix = torch.mm(T.T, Y_onehot)
                    proto_labels = torch.argmax(vote_matrix, dim=1).detach().cpu().numpy()

            plot_poincare_disk(ax, Z_np, labels=proto_labels, title=name)

        else:
            ax.text(0.5, 0.5, f"Unsupported dataset: {name}",
                    ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print(f"Figure 4 replication saved to {filename}")
