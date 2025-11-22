import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from src.dataset_pipeline import load_dataset
import numpy as np
import torch
from src.replication import run_experiment, plot_prototype_evolution
from src.affinities import SymmetricEntropicAffinity, NormalizedGaussianAndStudentAffinity

# ==========================================
# CONFIGURATION PARAMETERS
# ==========================================

# 1. Affinity Data (Similarity for CX)
# Options:
# - SymmetricEntropicAffinity(perp=30): SEA similarity (Van Assel et al., 2023). 
#   Computes entropic affinities with a fixed perplexity.
# - NormalizedGaussianAndStudentAffinity(sigma=1.0, student=False): Gaussian kernel.
#   Standard Gaussian similarity.
# - NormalizedGaussianAndStudentAffinity(student=True): Student's t-distribution.
#   Heavy-tailed distribution.
AFFINITY_DATA = SymmetricEntropicAffinity(perp=30, verbose=False)

# 2. Affinity Embeddings (Similarity for CZ)
# Options:
# - NormalizedGaussianAndStudentAffinity(student=True): Student's kernel (t-SNE like).
#   Standard for t-SNE and similar methods to avoid crowding problem.
# - NormalizedGaussianAndStudentAffinity(student=False): Gaussian kernel.
# - SymmetricEntropicAffinity(perp=30): SEA similarity.
AFFINITY_EMBEDDINGS = NormalizedGaussianAndStudentAffinity(student=True)

# 3. Loss Function
# Options:
# - "kl_loss": Kullback-Leibler divergence (LKL).
#   Minimizes divergence between P and Q. Standard for t-SNE/DistR.
# - "cross_entropy": Cross Entropy loss.
# - "l2": L2 / Mean Squared Error loss.
LOSS_FUNCTION = "kl_loss"

# 4. Output Dimension
OUTPUT_DIM = 2

# ==========================================
# MAIN SCRIPT
# ==========================================

# List of datasets to load
datasets_to_load = ['mnist']

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

if __name__ == "__main__":
    # Auto-detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Master Process Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    target_dataset = 'mnist'
    subset_size = 1000  # Defined subset size here
    
    # Run Experiment on Subset
    if target_dataset in loaded_data:
        print(f"\n--- PREPARING SUBSET (N={subset_size}) ---")
        
        # SLICING LOGIC: Take only the first subset_size elements
        X_full = loaded_data[target_dataset]['X']
        Y_full = loaded_data[target_dataset]['Y']
        
        data_subset = {
            'X': X_full[:subset_size],
            'Y': Y_full[:subset_size]
        }
        
        # Verify classes in subset (crucial for prototype count)
        unique_classes = np.unique(data_subset['Y'])
        n_classes = len(unique_classes)
        print(f"Classes in subset: {unique_classes}")
        
        # Define prototype counts to iterate over
        # For MNIST (10 classes), we can try a range.
        # Ensure they are <= subset_size
        prototype_counts = [10, 20, 30, 50, 100]
        prototype_counts = [p for p in prototype_counts if p <= subset_size]
        
        print(f"Iterating over prototype counts: {prototype_counts}")
        
        plot_prototype_evolution(
            prototype_counts,
            data_subset,
            target_dataset,
            device=device,
            affinity_data=AFFINITY_DATA,
            affinity_embeddings=AFFINITY_EMBEDDINGS,
            output_dim=OUTPUT_DIM,
            loss_function=LOSS_FUNCTION
        )
    else:
        print(f"Error: Dataset '{target_dataset}' could not be loaded.")