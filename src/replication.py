import torch
import numpy as np
from sklearn.metrics import homogeneity_score, normalized_mutual_info_score, silhouette_score
from sklearn.cluster import KMeans

from src.dataset_pipeline import load_dataset
from src.affinities import SymmetricEntropicAffinity, NormalizedGaussianAndStudentAffinity
from src.clust_dr import DistR



def get_distr_config(model_type='tsne', perplexity=30):
    """
    Get the affinity and loss configuration for DistR.
    
    Args:
        model_type: Type of model configuration (currently only 'tsne' supported).
        perplexity: Perplexity for the entropic affinity.
        
    Returns:
        config: Dictionary containing affinity_data, affinity_embedding, and loss_fun.
    """
    if model_type == 'tsne':
        # CX: Symmetric Entropic Affinity
        affinity_data = SymmetricEntropicAffinity(perp=perplexity, verbose=False)
        
        # CZ: Student-t kernel (nu=1 is standard t-SNE, which corresponds to Cauchy)
        # NormalizedGaussianAndStudentAffinity with student=True gives t-distribution
        affinity_embedding = NormalizedGaussianAndStudentAffinity(student=True, sigma=1.0, p=2)
        
        # Loss: KL divergence
        loss_fun = 'kl_loss'
        
        return {
            'affinity_data': affinity_data,
            'affinity_embedding': affinity_embedding,
            'loss_fun': loss_fun
        }
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def run_distr_replication(X, n_prototypes, embedding_dim=2, max_iter=100, device='cpu'):
    """
    Run the DistR training loop.
    
    Args:
        X: Input data (torch.Tensor).
        n_prototypes: Number of prototypes (clusters).
        embedding_dim: Dimension of the embedding space.
        max_iter: Maximum number of iterations.
        device: Device to run on ('cpu' or 'cuda').
        
    Returns:
        Z: Learned prototypes (torch.Tensor).
        T: Transport plan (torch.Tensor).
        model: The fitted DistR model.
    """
    config = get_distr_config()
    
    model = DistR(
        affinity_data=config['affinity_data'],
        affinity_embedding=config['affinity_embedding'],
        loss_fun=config['loss_fun'],
        output_sam=n_prototypes,
        output_dim=embedding_dim,
        optimizer="Adam",
        lr=1.0, # Standard learning rate for DistR
        init="normal",
        max_iter=max_iter,
        device=device,
        verbose=True
    )
    
    Z = model.fit_transform(X)
    T = model.T
    
    return Z, T, model

def calculate_metrics(X, Z, T, labels_true):
    """
    Calculate evaluation metrics: Homogeneity, Silhouette, NMI.
    
    Args:
        X: Input data (original or PCA-reduced).
        Z: Prototypes.
        T: Transport plan (n_samples, n_prototypes).
        labels_true: Ground truth labels.
        
    Returns:
        metrics: Dictionary of metric names and values.
    """
    # Convert to numpy for sklearn
    if isinstance(T, torch.Tensor):
        T_np = T.detach().cpu().numpy()
    else:
        T_np = T
        
    if isinstance(Z, torch.Tensor):
        Z_np = Z.detach().cpu().numpy()
    else:
        Z_np = Z
        
    if isinstance(labels_true, torch.Tensor):
        labels_true = labels_true.detach().cpu().numpy()
        
    # 1. Homogeneity
    # Assign each sample to the prototype with max transport mass
    labels_pred = np.argmax(T_np, axis=1)
    homogeneity = homogeneity_score(labels_true, labels_pred)
    
    # 2. Silhouette (Weighted)
    # The paper mentions "weighted silhouette score on the prototypes Z (Eq. 81/82)".
    # Standard silhouette_score computes mean silhouette coefficient of all samples.
    # If we want silhouette on Z, we treat Z as the dataset.
    # But what are the labels for Z?
    # The paper says: "We compute the silhouette score of the prototypes Z, weighted by their mass hZ."
    # And "The labels of the prototypes are assigned by majority voting of the points assigned to them."
    
    # Assign labels to prototypes
    # For each prototype j, find samples i where argmax(T_i) == j
    # Then take majority vote of labels_true[i]
    proto_labels = []
    weights = T_np.sum(axis=0) # hZ
    
    for j in range(Z_np.shape[0]):
        assigned_samples = (labels_pred == j)
        if np.sum(assigned_samples) > 0:
            majority_label = np.bincount(labels_true[assigned_samples]).argmax()
            proto_labels.append(majority_label)
        else:
            # Empty cluster, assign -1 or random?
            proto_labels.append(-1)
    
    proto_labels = np.array(proto_labels)
    
    # Filter out empty clusters for silhouette calculation if any
    valid_protos = (proto_labels != -1)
    if np.sum(valid_protos) > 1: # Need at least 2 clusters
        # sklearn silhouette_score doesn't directly support sample weights for the *score calculation* itself in the way we might want for "weighted silhouette",
        # but it does support `sample_weight` parameter which weights the contribution of each sample to the mean.
        # So we can pass hZ as sample_weight.
        from sklearn.metrics import silhouette_samples
        n_samples_z = np.sum(valid_protos)
        n_labels_z = len(np.unique(proto_labels[valid_protos]))
        
        if 1 < n_labels_z < n_samples_z:
            sample_silhouettes = silhouette_samples(Z_np[valid_protos], proto_labels[valid_protos])
            silhouette = np.average(sample_silhouettes, weights=weights[valid_protos])
        else:
            silhouette = 0.0
    else:
        silhouette = 0.0
        
    # 3. NMI
    # "Perform K-Means on prototypes Z, assign labels to samples via T, and compute Normalized Mutual Information against ground truth."
    # Wait, if we do K-Means on Z, we get cluster labels for Z.
    # Then we need to propagate these to samples?
    # Or does it mean: Cluster Z into K clusters (where K = number of ground truth classes),
    # then assign sample i to the cluster of its assigned prototype.
    
    n_classes = len(np.unique(labels_true))
    kmeans_Z = KMeans(n_clusters=n_classes, random_state=42, n_init=10).fit(Z_np)
    z_cluster_labels = kmeans_Z.labels_
    
    # Assign sample i to cluster of prototype j where j = argmax(T_i)
    sample_cluster_labels = z_cluster_labels[labels_pred]
    
    nmi = normalized_mutual_info_score(labels_true, sample_cluster_labels)
    
    return {
        'Homogeneity': homogeneity,
        'Silhouette': silhouette,
        'NMI': nmi
    }
