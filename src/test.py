import sys
import os
import json
sys.path.insert(0, os.path.abspath('..'))
from src.dataset_pipeline import load_dataset
import numpy as np
import torch
from src.replication import run_experiment, plot_prototype_evolution, plot_tradeoff_analysis
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

# 5. Dataset Loading and Preparation
# Available datasets: 'coil20', 'mnist', 'fmnist', 'pbmc', 'zeisel'
DATASETS = ['mnist']

# 6. Subset Size
subset_size = None # Set to None to use the full dataset

# 7. Number of seeds per experiment
n_seeds = 1

# ==========================================
# MAIN SCRIPT
# ==========================================

# List of datasets to load
datasets_to_load = DATASETS

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

# Auto-detect device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Master Process Device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Store results for all datasets
# Structure: dataset_name -> history (method -> metric -> list of lists)
all_results = {}

# Define prototype counts to iterate over
# We need a common set of prototype counts for the plot, 
# or we can handle different counts per dataset if we really wanted to,
# but for a grid plot, sharing the x-axis is better.
# Let's define a standard list, and filter if needed (though usually fixed).
standard_prototype_counts = [5, 10, 20, 30, 50, 100]

# We will use the same prototype counts for all datasets for consistency in plotting
# If subset_size is small, we filter.
if subset_size is not None:
    prototype_counts = [p for p in standard_prototype_counts if p <= subset_size]
else:
    prototype_counts = standard_prototype_counts
    
print(f"Iterating over prototype counts: {prototype_counts}")

methods = ['DistR', 'DR_then_Clust', 'Clust_then_DR']
metrics = ["hom", "ami", "ari", "nmi", "sil"]
n_seeds = 1 # Number of seeds per experiment

for target_dataset in datasets_to_load:
    if target_dataset in loaded_data:
        print(f"\n\n{'='*40}")
        print(f"PROCESSING DATASET: {target_dataset}")
        print(f"{'='*40}")
        
        X_full = loaded_data[target_dataset]['X']
        Y_full = loaded_data[target_dataset]['Y']
        
        if subset_size is not None:
            print(f"--- Using SUBSET (N={subset_size}) ---")
            data_subset = {
                'X': X_full[:subset_size],
                'Y': Y_full[:subset_size]
            }
        else:
            print(f"--- Using FULL DATASET (N={len(X_full)}) ---")
            data_subset = {
                'X': X_full,
                'Y': Y_full
            }
        
        # Verify classes
        unique_classes = np.unique(data_subset['Y'])
        print(f"Classes: {unique_classes}")
        
        # Initialize history for this dataset
        history = {method: {metric: [] for metric in metrics} for method in methods}
        
        for n in prototype_counts:
            print(f"\n  >> Prototypes: {n} <<")
            
            # Temporary storage for this prototype count
            current_proto_scores = {method: {metric: [] for metric in metrics} for method in methods}
            
            for seed in range(n_seeds):
                print(f"    -- Seed {seed+1}/{n_seeds} --")
                results = run_experiment(
                    data_subset, 
                    target_dataset, 
                    n_prototypes=n, 
                    device=device,
                    affinity_data=AFFINITY_DATA,
                    affinity_embeddings=AFFINITY_EMBEDDINGS,
                    output_dim=OUTPUT_DIM,
                    loss_function=LOSS_FUNCTION,
                    seed=seed
                )
                
                for method in methods:
                    for metric in metrics:
                        val = results[method].get(metric, 0)
                        current_proto_scores[method][metric].append(val)
            
            # Append to history
            for method in methods:
                for metric in metrics:
                    history[method][metric].append(current_proto_scores[method][metric])
        
        all_results[target_dataset] = history
        
    else:
        print(f"Error: Dataset '{target_dataset}' could not be loaded. Skipping.")

# Save results to disk
print("\nSaving results to 'results.json'...")
output_data = {
    "prototype_counts": prototype_counts,
    "datasets": datasets_to_load,
    "methods": methods,
    "metrics": metrics,
    "results": all_results
}
with open("results.json", "w") as f:
    json.dump(output_data, f, indent=4)
print("Results saved.")

# Plotting results for all datasets
if all_results:
    print(f"\nGenerating plots for {list(all_results.keys())}...")
    plot_prototype_evolution(
        all_results,
        prototype_counts,
        methods=methods,
        metrics=metrics,
        filename="multi_dataset_evolution.png"
    )
    
    print("\nGenerating trade-off analysis plot...")
    plot_tradeoff_analysis(
        all_results,
        methods=methods,
        filename="tradeoff_analysis.png"
    )
else:
    print("No results to plot.")