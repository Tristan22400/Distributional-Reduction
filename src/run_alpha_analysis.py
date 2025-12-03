import sys
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.abspath('..'))

from src.dataset_pipeline import load_dataset
from src.replication import evaluate_prototypes
from src.affinities import SymmetricEntropicAffinity, NormalizedGaussianAndStudentAffinity, LearnableNormalizedGaussianAndStudentAffinity
from src.clust_dr import DistR
from src.affinities import NanError
from tqdm import tqdm
import torch.nn.functional as F

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
# AFFINITY_EMBEDDINGS = NormalizedGaussianAndStudentAffinity(student=True)

# 3. Loss Function
# Options:
# - "kl_loss": Kullback-Leibler divergence (LKL).
#   Minimizes divergence between P and Q. Standard for t-SNE/DistR.
# - "cross_entropy": Cross Entropy loss.
# - "l2": L2 / Mean Squared Error loss.
LOSS_FUNCTION = "kl_loss"


# 2. Output Dimension
OUTPUT_DIM = 2

# 3. Loss Function
LOSS_FUNCTION = "kl_loss"

# 4. Alpha Analysis Configuration
ALPHA_ANALYSIS_DATASET = 'coil20' # Dataset to use for alpha analysis
ALPHA_RANGE = [0.5, 1.0, 1.5, 5.0, 10.0, 50.0, 100.0] # Fixed alpha values to test
ALPHA_PROTOTYPES = 20 # Number of prototypes for alpha analysis
ALPHA_ANALYSIS_ITER = 10000 # Number of iterations for alpha analysis
subset_size = None # Set to None to use the full dataset

# Auto-detect device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print(f"\n\n{'='*40}")
print(f"ALPHA ANALYSIS on {ALPHA_ANALYSIS_DATASET}")
print(f"{'='*40}")

# Load Data
try:
    print(f"Loading {ALPHA_ANALYSIS_DATASET}...")
    data = load_dataset(ALPHA_ANALYSIS_DATASET, pca_dim=50)
    X_full = data['X']
    Y_full = data['Y']
    
    if subset_size is not None:
         data_subset = {'X': X_full[:subset_size], 'Y': Y_full[:subset_size]}
    else:
         data_subset = {'X': X_full, 'Y': Y_full}
         
    print(f"Successfully loaded {ALPHA_ANALYSIS_DATASET}")
    print(f"  X shape: {data_subset['X'].shape}")
    print(f"  Y shape: {data_subset['Y'].shape}")
    
except Exception as e:
    print(f"Failed to load {ALPHA_ANALYSIS_DATASET}: {e}")
    raise e


# ==========================================
# LOGGING IMPLEMENTATION
# ==========================================
class LoggedDistR(DistR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inner_logs = {
            "epoch": [],
            "physical_alpha": [],
            "raw_grad_norm": [],
            "z_grad_norm": [],
            "gw_loss": [],
            "alpha_delta": []
        }
        self.outer_logs = {
            "outer_step": [],
            "nmi": [],
            "ari": [],
            "hom": []
        }
        self.initial_alpha = None
        self.global_step = 0
        self.outer_step = 0

    def log_training_metrics(self, loss_val):
        # 1. Physical Alpha
        raw_alpha = self.affinity_embedding.alpha
        physical_alpha = F.softplus(raw_alpha).item()
        
        if self.initial_alpha is None:
            self.initial_alpha = physical_alpha

        # 2. Gradients
        if raw_alpha.grad is not None:
            raw_grad_norm = raw_alpha.grad.norm().item()
        else:
            raw_grad_norm = 0.0
            
        if self.Z.grad is not None:
            z_grad_norm = self.Z.grad.norm().item()
        else:
            z_grad_norm = 0.0

        # Update per-epoch logs
        self.inner_logs["epoch"].append(self.global_step)
        self.inner_logs["physical_alpha"].append(physical_alpha)
        self.inner_logs["raw_grad_norm"].append(raw_grad_norm)
        self.inner_logs["z_grad_norm"].append(z_grad_norm)
        self.inner_logs["gw_loss"].append(loss_val)
        self.inner_logs["alpha_delta"].append(physical_alpha - self.initial_alpha)

        self.global_step += 1

    def log_outer_metrics(self):
        # Evaluate Clustering
        if hasattr(self, 'Y_train') and hasattr(self, 'X_train'):
            with torch.no_grad():
                scores = evaluate_prototypes(self.Z, self.T, self.Y_train, self.X_train)
                self.outer_logs["nmi"].append(scores.get("nmi", 0))
                self.outer_logs["ari"].append(scores.get("ari", 0))
                self.outer_logs["hom"].append(scores.get("hom", 0))
        else:
            self.outer_logs["nmi"].append(None)
            self.outer_logs["ari"].append(None)
            self.outer_logs["hom"].append(None)
            
        self.outer_logs["outer_step"].append(self.outer_step)
        self.outer_step += 1

    def _update_T(self):
        super()._update_T()
        self.log_outer_metrics()

    def fit_transform(self, X, y=None):
        # Store data for evaluation
        self.X_train = X
        self.Y_train = y
        return super().fit_transform(X, y)

    def _update_embedding(self, max_iter=None):
        """
        Override to inject logging.
        """
        if max_iter is None:
            max_iter = self.max_iter
            
        self.Z.requires_grad = True
        pbar = tqdm(range(max_iter), disable=not self.verbose)
        
        for i in pbar:
            self.optimizer.zero_grad()
            Loss = self._embed_loss()
            if torch.isnan(Loss):
                raise NanError("NaN in embedding loss")
            Loss.backward()
            
            # --- LOGGING HOOK ---
            self.log_training_metrics(Loss.item())
            # --------------------
            
            self.optimizer.step()

            self.losses.append(Loss.item())
            if i > 1:
                delta = abs(self.losses[-1] - self.losses[-2]) / abs(self.losses[-2])
                if delta < self.tol:
                    if self.verbose:
                        print("---------- delta loss convergence ----------")
                    break
                if self.verbose:
                    pbar.set_description(
                        f"Loss : {float(self.losses[-1]): .3e}, "
                        f"delta : {float(delta): .3e} "
                    )

# 1. Train with Learnable Alpha
print("\n--- Training with LEARNABLE Alpha ---")
learnable_affinity = LearnableNormalizedGaussianAndStudentAffinity(alpha_init=1.0)

model_distr = LoggedDistR(
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
    lr_affinity=1,
    warmup_iter=50,
    seed=0
)

# Move data to device
X_tensor = torch.as_tensor(data_subset['X'], dtype=torch.float32, device=device)
# Pass Y to fit_transform for logging
Z_distr = model_distr.fit_transform(X_tensor, y=data_subset['Y'])

# Evaluate
metrics_learnable = evaluate_prototypes(Z_distr, model_distr.T, data_subset['Y'], X_tensor)
learned_alpha = torch.nn.functional.softplus(learnable_affinity.alpha).item()
learned_loss = model_distr.losses[-1] if model_distr.losses else None

print(f"Learned Alpha: {learned_alpha:.4f}")
print(f"Final Loss: {learned_loss}")
print(f"Scores: {metrics_learnable}")

# Save logs
# Save logs
import pandas as pd
inner_logs_df = pd.DataFrame(model_distr.inner_logs)
outer_logs_df = pd.DataFrame(model_distr.outer_logs)
inner_logs_df.to_csv("alpha_inner_logs.csv", index=False)
outer_logs_df.to_csv("alpha_outer_logs.csv", index=False)
print("Training logs saved to 'alpha_inner_logs.csv' and 'alpha_outer_logs.csv'")

# Plot Training Dynamics
plt.figure(figsize=(20, 5))
plt.subplot(1, 4, 1)
plt.plot(inner_logs_df['epoch'], inner_logs_df['physical_alpha'])
plt.title('Physical Alpha Trajectory')
plt.xlabel('Epoch')
plt.ylabel('Alpha')

plt.subplot(1, 4, 2)
plt.plot(inner_logs_df['epoch'], inner_logs_df['raw_grad_norm'], label='Alpha Grad')
plt.plot(inner_logs_df['epoch'], inner_logs_df['z_grad_norm'], label='Z Grad', alpha=0.7)
plt.title('Gradient Norms')
plt.xlabel('Epoch')
plt.ylabel('Norm')
plt.legend()
plt.yscale('log')

plt.subplot(1, 4, 3)
plt.plot(outer_logs_df['outer_step'], outer_logs_df['nmi'], marker='o')
plt.title('NMI Evolution (Outer Loop)')
plt.xlabel('Outer Step')
plt.ylabel('NMI')

plt.subplot(1, 4, 4)
# Scatter plot of final Z and projected data
Z_np = Z_distr.cpu().numpy()
T_np = model_distr.T.cpu().numpy()
Y_np = data_subset['Y']

# Normalize T row-wise to get barycentric projection of data
T_row_sum = T_np.sum(axis=1, keepdims=True)
T_row_sum[T_row_sum == 0] = 1.0 # Avoid division by zero
T_norm = T_np / T_row_sum
X_emb = T_norm @ Z_np

plt.scatter(X_emb[:, 0], X_emb[:, 1], c=Y_np, cmap='tab10', s=5, alpha=0.5, label='Data (Projected)')
plt.scatter(Z_np[:, 0], Z_np[:, 1], c='black', marker='x', s=50, label='Prototypes')
plt.title('Projected Data & Prototypes')
plt.xlabel('Dim 1')
plt.ylabel('Dim 2')
plt.legend()
# plt.colorbar(label='Class') # Colorbar might be cluttered with legend

plt.tight_layout()
plt.savefig("alpha_dynamics_and_embedding.png")
print("Dynamics and embedding plot saved to 'alpha_dynamics_and_embedding.png'")

# 2. Train with Fixed Alphas
print("\n--- Training with FIXED Alphas ---")
metrics = ["hom", "ami", "ari", "nmi", "sil"]
fixed_alpha_scores = {metric: [] for metric in metrics}
fixed_alpha_losses = []

for alpha in ALPHA_RANGE:
    print(f"  Testing Alpha = {alpha}...")
    
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
