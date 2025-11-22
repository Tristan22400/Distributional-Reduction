# %%
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from src.dataset_pipeline import load_dataset
import numpy as np

# %%
# List of datasets to load
# Excluded 'coil20' due to current server downtime (404/522)
# 'snareseq' requires manual download of multiple files usually, but we can try if implemented.
# We focus on the ones known to work or standard in scanpy/torchvision.
datasets_to_load = ['mnist', 'fmnist', 'pbmc', 'zeisel']

loaded_data = {}

for name in datasets_to_load:
    try:
        print(f"Loading {name}...")
        # Load with default PCA dim 50
        data = load_dataset(name, pca_dim=50)
        loaded_data[name] = data
        
        print(f"Successfully loaded {name}")
        print(f"  X shape: {data['X'].shape}")
        print(f"  Y shape: {data['Y'].shape}")
        print(f"  Original dim: {data['original_dim']}")
        print("-" * 30)
    except Exception as e:
        print(f"Failed to load {name}: {e}")
        print("-" * 30)

# %%
# Example access
if 'mnist' in loaded_data:
    X_mnist = loaded_data['mnist']['X']
    Y_mnist = loaded_data['mnist']['Y']
    print("MNIST data ready for DistR.")

# %%
import torch
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, homogeneity_score, silhouette_score
from sklearn.cluster import KMeans

from build.lib.src.clust_dr import DR_then_Clust
from src.affinities import NormalizedGaussianAndStudentAffinity, SymmetricEntropicAffinity
from src.clust_dr import Clust_then_DR, DistR

# Assuming previous classes (DistR, DR_then_Clust, etc.) are in a module named 'methods'
# from methods import DistR, DR_then_Clust, Clust_then_DR, COOTClust
# from src.affinities import SymmetricEntropicAffinity, NormalizedGaussianAndStudentAffinity

def get_majority_vote_labels(T, Y):
    """
    Assigns a class label to each prototype based on the majority class 
    of the samples mapped to it (weighted by the transport plan T).
    """
    # T is (N_samples, n_prototypes)
    # Y is (N_samples,)
    n_prototypes = T.shape[1]
    n_classes = len(np.unique(Y))
    
    # Soft vote matrix: (n_prototypes, n_classes)
    vote_matrix = torch.zeros((n_prototypes, n_classes), device=T.device)
    
    # One-hot encode Y to aggregate mass
    Y_onehot = torch.zeros((len(Y), n_classes), device=T.device)
    Y_onehot.scatter_(1, torch.tensor(Y, device=T.device).unsqueeze(1).long(), 1)
    
    # Aggregate mass: Project Y onto Prototypes via T^T
    # vote_matrix[j, c] = sum of T_ij for all i where y_i = c
    vote_matrix = T.T @ Y_onehot
    
    # Get majority class
    proto_labels = torch.argmax(vote_matrix, dim=1).cpu().numpy()
    return proto_labels

def evaluate_prototypes(Z, T, Y_true):
    """
    Computes the three metrics defined in the paper [cite: 332-339]:
    1. Homogeneity: Do prototypes represent pure classes?
    2. K-Means NMI: Does clustering the prototypes recover global structure?
    3. Silhouette: Are prototypes well-separated?
    """
    Z_np = Z.detach().cpu().numpy()
    T_tensor = T.detach()
    
    # 1. Homogeneity Score
    # Treat prototype assignment as a clustering of inputs
    # Hard assignment: x_i assigned to prototype j with max T_ij
    sample_assignments = torch.argmax(T_tensor, dim=1).cpu().numpy()
    homogeneity = homogeneity_score(Y_true, sample_assignments)
    
    # 2. K-Means NMI Score
    # Cluster the prototypes themselves into K=n_classes clusters
    n_classes = len(np.unique(Y_true))
    kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
    proto_clusters = kmeans.fit_predict(Z_np)
    
    # Propagate prototype clusters back to samples
    predicted_labels = proto_clusters[sample_assignments]
    nmi = normalized_mutual_info_score(Y_true, predicted_labels)
    
    # 3. Silhouette Score
    # Computed on prototypes using majority vote labels
    proto_labels = get_majority_vote_labels(T_tensor, Y_true)
    # We weight samples by the mass of the prototype (sum of column T)
    weights = T_tensor.sum(dim=0).cpu().numpy()
    
    try:
        sil = silhouette_score(Z_np, proto_labels, sample_weight=weights)
    except ValueError:
        # Fails if < 2 labels present
        sil = -1.0
        
    return {
        "Homogeneity": homogeneity,
        "NMI": nmi,
        "Silhouette": sil
    }

def run_experiment(data_dict, dataset_name, n_prototypes=50, device='cuda'):
    """
    Runs DistR and Baselines on a single dataset.
    """
    print(f"=== Running Experiment on {dataset_name} ===")
    print(f"Configuration: N={data_dict['X'].shape[0]}, Prototypes={n_prototypes}, Device={device}")
    
    X = torch.tensor(data_dict['X'], dtype=torch.float64).to(device)
    Y = data_dict['Y'] # Keep as numpy/list for sklearn metrics
    
    # --- Configuration matching Paper Section 5  ---
    # Input Affinity: Symmetric Entropic Affinity (SEA)
    # Embedding Affinity: Student t-distribution
    # Loss: KL Divergence
    
    # Note: Perplexity typically N/100 or 30. Fixed to 30 for consistency.
    affinity_X = SymmetricEntropicAffinity(perp=30, verbose=False) 
    affinity_Z = NormalizedGaussianAndStudentAffinity(student=True) # Student kernel
    
    results = {}
    
    # 1. DistR (Ours)
    print("\n[1/3] Training DistR...")
    model_distr = DistR(
        affinity_data=affinity_X,
        affinity_embedding=affinity_Z,
        output_sam=n_prototypes,
        loss_fun="kl_loss",       # Matching NE objective [cite: 120]
        optimizer="Adam",
        lr=0.1,                   # Standard LR for Adam
        max_iter=200,             # Sufficient for demo
        device=device,
        init="normal",            # Must be "normal" or "WrappedNormal"
        init_T="kmeans"           # Good initialization
    )
    Z_distr = model_distr.fit_transform(X)
    metrics_distr = evaluate_prototypes(Z_distr, model_distr.T, Y)
    results['DistR'] = metrics_distr
    print(f"DistR Results: {metrics_distr}")

    # 2. DR -> Clustering
    print("\n[2/3] Training DR -> Clustering...")
    model_drc = DR_then_Clust(
        affinity_data=affinity_X,
        affinity_embedding=affinity_Z,
        output_sam=n_prototypes,
        output_dim=2,
        loss_fun="kl_loss",
        device=device,
        init="normal",            # Must be "normal" or "WrappedNormal"
        init_T="kmeans"           # Uses kmeans to cluster the embeddings
    )
    Z_drc = model_drc.fit_transform(X)
    metrics_drc = evaluate_prototypes(Z_drc, model_drc.T, Y)
    results['DR_then_Clust'] = metrics_drc
    print(f"DR->C Results: {metrics_drc}")

    # 3. Clustering -> DR
    print("\n[3/3] Training Clustering -> DR...")
    model_cdr = Clust_then_DR(
        affinity_data=affinity_X,
        affinity_embedding=affinity_Z,
        output_sam=n_prototypes,
        output_dim=2,
        loss_fun="kl_loss",
        device=device,
        init="normal",            # Must be "normal" or "WrappedNormal"
        init_T="kmeans"           # Uses kmeans to cluster input X
    )
    Z_cdr = model_cdr.fit_transform(X)
    metrics_cdr = evaluate_prototypes(Z_cdr, model_cdr.T, Y)
    results['Clust_then_DR'] = metrics_cdr
    print(f"C->DR Results: {metrics_cdr}")
    
    return results

# === Execution Block ===

if __name__ == "__main__":
    # Check for available device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Example Access as requested
    target_dataset = 'mnist'
    
    if target_dataset in loaded_data:
        # Prepare data dictionary from loaded context
        data_subset = {
            'X': loaded_data[target_dataset]['X'], # Ensure shape (N, Features)
            'Y': loaded_data[target_dataset]['Y']
        }
        
        # Set prototypes: typically n_classes + 20 or fixed number [cite: 1114]
        n_classes = len(np.unique(data_subset['Y']))
        n_prototypes = n_classes + 20 # Paper strategy
        
        final_scores = run_experiment(
            data_subset, 
            target_dataset, 
            n_prototypes=n_prototypes, 
            device=device
        )
        
        print("\n=== Final Benchmark Report ===")
        for method, scores in final_scores.items():
            print(f"{method}:")
            for metric, val in scores.items():
                print(f"  {metric}: {val:.4f}")
    else:
        print(f"Dataset {target_dataset} not found in loaded_data.")


