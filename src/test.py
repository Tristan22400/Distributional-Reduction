import sys
import os
import json
sys.path.insert(0, os.path.abspath('..'))
from src.dataset_pipeline import load_dataset
import numpy as np
import torch
from src.replication import run_experiment, plot_prototype_evolution, plot_tradeoff_analysis
from src.affinities import SymmetricEntropicAffinity, NormalizedGaussianAndStudentAffinity, LearnableNormalizedGaussianAndStudentAffinity
import matplotlib.pyplot as plt

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
DATASETS = ['coil20']

# 6. Subset Size
subset_size = None # Set to None to use the full dataset

# 7. Number of experiment
n_experiments = 1

# 8. Prototype counts
standard_prototype_counts = [5]

# 9. Methods
methods = ['DistR', 'DR_then_Clust', 'Clust_then_DR']
metrics = ["hom", "ami", "ari", "nmi", "sil"]

# 10. Alpha Analysis Configuration
RUN_ALPHA_ANALYSIS = True
ONLY_ALPHA_ANALYSIS = True # If True, skips the main experiment loop and runs only alpha analysis on full dataset
ALPHA_ANALYSIS_DATASET = 'coil20' # Dataset to use for alpha analysis
ALPHA_RANGE = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0] # Fixed alpha values to test
ALPHA_PROTOTYPES = 20 # Number of prototypes for alpha analysis
ALPHA_ANALYSIS_ITER = 10000 # Number of iterations for alpha analysis

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

# We will use the same prototype counts for all datasets for consistency in plotting
# If subset_size is small, we filter.
if subset_size is not None:
    prototype_counts = [p for p in standard_prototype_counts if p <= subset_size]
else:
    prototype_counts = standard_prototype_counts
    
print(f"Iterating over prototype counts: {prototype_counts}")




if ONLY_ALPHA_ANALYSIS:
    print("ONLY_ALPHA_ANALYSIS is True. Skipping main experiment loop.")
    datasets_to_load = [] # Skip main loop
    subset_size = None # Use full dataset for alpha analysis as requested

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
            
            for seed in range(n_experiments):
                print(f"    -- Seed {seed+1}/{n_experiments} --")
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

# ==========================================
# ALPHA ANALYSIS
# ==========================================
if RUN_ALPHA_ANALYSIS:
    print(f"\n\n{'='*40}")
    print(f"ALPHA ANALYSIS on {ALPHA_ANALYSIS_DATASET}")
    print(f"{'='*40}")
    
    if ALPHA_ANALYSIS_DATASET in loaded_data:
        X_full = loaded_data[ALPHA_ANALYSIS_DATASET]['X']
        Y_full = loaded_data[ALPHA_ANALYSIS_DATASET]['Y']
        
        if subset_size is not None:
             data_subset = {'X': X_full[:subset_size], 'Y': Y_full[:subset_size]}
        else:
             data_subset = {'X': X_full, 'Y': Y_full}
             
        # 1. Train with Learnable Alpha
        print("\n--- Training with LEARNABLE Alpha ---")
        learnable_affinity = LearnableNormalizedGaussianAndStudentAffinity(alpha_init=1.0)
        
        # We need to access the model to get the learned alpha, so we can't use run_experiment directly 
        # or we need to modify run_experiment to return the model or alpha.
        # Let's manually run DistR here to have full control.
        from src.clust_dr import DistR
        
        model_distr = DistR(
            affinity_data=AFFINITY_DATA,
            affinity_embedding=learnable_affinity,
            output_dim=OUTPUT_DIM,
            output_sam=ALPHA_PROTOTYPES,
            loss_fun=LOSS_FUNCTION,
            optimizer="Adam",
            lr=0.1,
            max_iter=ALPHA_ANALYSIS_ITER,
            device=device,
            dtype=torch.float32,
            init="normal",
            init_T="kmeans",
            tol=1e-9, # Lower tolerance to prevent early stopping
            max_iter_outer=100, # Increase outer loop limit
            seed=0
        )
        
        # Move data to device
        X_tensor = torch.as_tensor(data_subset['X'], dtype=torch.float32, device=device)
        Z_distr = model_distr.fit_transform(X_tensor)
        
        # Evaluate
        from src.replication import evaluate_prototypes
        metrics_learnable = evaluate_prototypes(Z_distr, model_distr.T, data_subset['Y'], X_tensor)
        learned_alpha = torch.nn.functional.softplus(learnable_affinity.alpha).item()
        learned_loss = model_distr.losses[-1] if model_distr.losses else None
        
        print(f"Learned Alpha: {learned_alpha:.4f}")
        print(f"Final Loss: {learned_loss}")
        print(f"Scores: {metrics_learnable}")
        
        # 2. Train with Fixed Alphas
        print("\n--- Training with FIXED Alphas ---")
        fixed_alpha_scores = {metric: [] for metric in metrics}
        fixed_alpha_losses = []
        
        for alpha in ALPHA_RANGE:
            print(f"  Testing Alpha = {alpha}...")
            # We use NormalizedGaussianAndStudentAffinity but we need to control alpha.
            # The existing class assumes Student t (alpha=1) or Gaussian.
            # We should use our new class but with requires_grad=False for fixed alpha.
            
            fixed_affinity = LearnableNormalizedGaussianAndStudentAffinity(alpha_init=alpha)
            fixed_affinity.alpha.requires_grad = False # Fix alpha
            
            # Manually run DistR to get loss
            model_fixed = DistR(
                affinity_data=AFFINITY_DATA,
                affinity_embedding=fixed_affinity,
                output_dim=OUTPUT_DIM,
                output_sam=ALPHA_PROTOTYPES,
                loss_fun=LOSS_FUNCTION,
                optimizer="Adam",
                lr=0.1,
                max_iter=int(ALPHA_ANALYSIS_ITER/10.0),
                device=device,
                dtype=torch.float32,
                init="normal",
                init_T="kmeans",
                tol=1e-9, # Lower tolerance to prevent early stopping
                max_iter_outer=100, # Increase outer loop limit
                seed=0
            )
            
            Z_fixed = model_fixed.fit_transform(X_tensor)
            metrics_fixed = evaluate_prototypes(Z_fixed, model_fixed.T, data_subset['Y'], X_tensor)
            loss_fixed = model_fixed.losses[-1] if model_fixed.losses else None
            
            fixed_alpha_losses.append(loss_fixed)
            for metric in metrics:
                fixed_alpha_scores[metric].append(metrics_fixed[metric])
                
        # 3. Plotting
        print("\nGenerating Alpha Analysis Plot...")
        
        # Combine data for plotting (Fixed + Learned)
        all_alphas = ALPHA_RANGE + [learned_alpha]
        all_nmi = fixed_alpha_scores["nmi"] + [metrics_learnable["nmi"]]
        all_ari = fixed_alpha_scores["ari"] + [metrics_learnable["ari"]]
        all_losses = fixed_alpha_losses + [learned_loss]
        
        # Sort by alpha
        combined_data = sorted(zip(all_alphas, all_nmi, all_ari, all_losses))
        sorted_alphas, sorted_nmi, sorted_ari, sorted_losses = zip(*combined_data)
        
        # Create a figure with 3 subplots (Absolute Scores, Difference, and Loss)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))
        
        # --- Subplot 1: Absolute Scores ---
        # Plot NMI and ARI for all alphas (connected line)
        if "nmi" in metrics:
            ax1.plot(sorted_alphas, sorted_nmi, 'o-', label="NMI (All Alphas)", color='blue', alpha=0.7)
        if "ari" in metrics:
            ax1.plot(sorted_alphas, sorted_ari, 's-', label="ARI (All Alphas)", color='green', alpha=0.7)
            
        # Highlight Learned Alpha
        if "nmi" in metrics:
            learned_nmi = metrics_learnable["nmi"]
            ax1.axhline(y=learned_nmi, color='red', linestyle='--', alpha=0.3, label=f"Learned Alpha NMI ({learned_nmi:.3f})")
            ax1.axvline(x=learned_alpha, color='red', linestyle='--', alpha=0.3, label=f"Learned Alpha Value ({learned_alpha:.2f})")
            ax1.scatter([learned_alpha], [learned_nmi], color='red', s=150, zorder=10, marker='*', label="Learned Result")
            
        ax1.set_xlabel("Alpha (Degrees of Freedom)")
        ax1.set_ylabel("Score")
        ax1.set_title(f"DistR Performance: Alpha Analysis")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # --- Subplot 2: Difference (Learned - Fixed) ---
        if "nmi" in metrics:
            # Calculate differences
            diffs = [learned_nmi - score for score in fixed_alpha_scores["nmi"]]
            
            # Bar chart for differences
            colors = ['green' if d >= 0 else 'red' for d in diffs]
            ax2.bar([str(a) for a in ALPHA_RANGE], diffs, color=colors, alpha=0.7)
            
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax2.set_xlabel("Fixed Alpha Value")
            ax2.set_ylabel("Score Difference (Learned - Fixed)")
            ax2.set_title("NMI Gain using Learned Alpha vs Fixed Alphas")
            ax2.grid(True, axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(diffs):
                ax2.text(i, v, f"{v:+.3f}", ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)

        # --- Subplot 3: Loss vs Alpha ---
        # Plot Loss for all alphas (connected line)
        ax3.plot(sorted_alphas, sorted_losses, 'o-', label="Loss (All Alphas)", color='purple', alpha=0.7)
        
        # Highlight Learned Alpha Loss
        if learned_loss is not None:
            ax3.axhline(y=learned_loss, color='red', linestyle='--', alpha=0.3, label=f"Learned Alpha Loss ({learned_loss:.3f})")
            ax3.axvline(x=learned_alpha, color='red', linestyle='--', alpha=0.3)
            ax3.scatter([learned_alpha], [learned_loss], color='red', s=150, zorder=10, marker='*', label="Learned Result")
            
        ax3.set_xlabel("Alpha (Degrees of Freedom)")
        ax3.set_ylabel("Loss")
        ax3.set_title("DistR Loss: Alpha Analysis")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("alpha_analysis.png")
        print("Plot saved to 'alpha_analysis.png'")
        
    else:
        print(f"Dataset {ALPHA_ANALYSIS_DATASET} could not be loaded. Skipping alpha analysis.")