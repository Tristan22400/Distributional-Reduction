import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
import ot
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_swiss_roll, make_s_curve
from sklearn.manifold import MDS, trustworthiness
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.preprocessing import StandardScaler
import random
import argparse
import sys
import os

# Add project root to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import real DistR and affinities
from src.clust_dr import DistR
from src.affinities import LearnableNormalizedGaussianAndStudentAffinity, NormalizedGaussianAndStudentAffinity, EntropicAffinity, LorentzHyperbolicAffinity

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

class SyntheticManifolds:
    def __init__(self, n_samples=1000, input_dim=50, seed=42):
        self.n_samples = n_samples
        self.input_dim = input_dim
        self.seed = seed

    def generate_blobs(self):
        X, y = make_blobs(n_samples=self.n_samples, n_features=self.input_dim, 
                          centers=5, cluster_std=1.0, random_state=self.seed)
        return torch.tensor(X, dtype=torch.float32), y, "Gaussian Blobs"

    def generate_swiss_roll(self):
        X_low, t = make_swiss_roll(n_samples=self.n_samples, noise=0.1, random_state=self.seed)
        # Project to high dim
        Q, _ = np.linalg.qr(np.random.randn(self.input_dim, 3))
        X = X_low @ Q.T
        return torch.tensor(X, dtype=torch.float32), t, "Swiss Roll"

    def generate_s_curve(self):
        X_low, t = make_s_curve(n_samples=self.n_samples, noise=0.1, random_state=self.seed)
        # Project to high dim
        Q, _ = np.linalg.qr(np.random.randn(self.input_dim, 3))
        X = X_low @ Q.T
        return torch.tensor(X, dtype=torch.float32), t, "S-Curve"

    def generate_tree(self):
        # Generate a balanced binary tree
        G = nx.balanced_tree(r=2, h=9)
        nodes = list(G.nodes())[:self.n_samples]
        G = G.subgraph(nodes)
        
        # Compute Shortest Path Distance matrix
        dist_matrix = dict(nx.all_pairs_shortest_path_length(G))
        n = len(G)
        D = np.zeros((n, n))
        for i, node_i in enumerate(G.nodes()):
            for j, node_j in enumerate(G.nodes()):
                D[i, j] = dist_matrix[node_i].get(node_j, 0)
        
        # MDS projection to R^50
        mds = MDS(n_components=self.input_dim, dissimilarity='precomputed', 
                  random_state=self.seed, normalized_stress='auto')
        X = mds.fit_transform(D)
        
        # Labels: distance from root (node 0)
        root = 0
        labels = [dist_matrix[root].get(node, 0) for node in G.nodes()]
        
        return torch.tensor(X, dtype=torch.float32), np.array(labels), "Synthetic Tree"

class DistRWithLogging(DistR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha_logs = []

    def _update_embedding(self, max_iter=None):
        if max_iter is None:
            max_iter = self.max_iter
            
        self.Z.requires_grad = True
        
        from tqdm import tqdm
        pbar = tqdm(range(max_iter), disable=not self.verbose)
        for i in pbar:
            self.optimizer.zero_grad()
            Loss = self._embed_loss()
            if torch.isnan(Loss):
                raise Exception("NaN in embedding loss")
            Loss.backward()
            self.optimizer.step()

            self.losses.append(Loss.item())
            
            # Log alpha
            if hasattr(self.affinity_embedding, 'alpha'):
                alpha_val = torch.nn.functional.softplus(self.affinity_embedding.alpha).item()
                self.alpha_logs.append(alpha_val)
            
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

def lorentz_to_poincare(x):
    """
    Convert Lorentz model coordinates (n, d+1) to Poincaré ball coordinates (n, d).
    x = (x0, x1, ..., xd)
    y_i = x_i / (1 + x0)
    """
    return x[:, 1:] / (1 + x[:, 0:1])

def hyperbolic_distance(Z):
    """
    Compute pairwise hyperbolic distances for Lorentz points Z.
    d(x, y) = arccosh(-<x, y>_L)
    """
    # <x, y>_L = -x0y0 + x1y1 + ...
    # Z: (N, D+1)
    # We can use src.utils_hyperbolic.minkowski_ip2 if available, or implement here.
    # Let's implement simple version.
    
    # Z is (N, dim)
    # Inner product
    # x0*y0
    xy0 = Z[:, 0:1] @ Z[:, 0:1].T
    # x_rest * y_rest
    xy_rest = Z[:, 1:] @ Z[:, 1:].T
    inner = -xy0 + xy_rest
    
    # Clamp for numerical stability
    inner = torch.clamp(inner, max=-1.0 - 1e-15)
    dist = torch.arccosh(-inner)
    return dist

def calculate_trustworthiness(X, Z, k=5, is_hyperbolic=False):
    """
    Calculate Continuity as Trustworthiness(Z, X).
    If is_hyperbolic is True, Z is in Lorentz model.
    """
    if is_hyperbolic:
        # Compute distance matrices
        D_X = pairwise_distances(X)
        Z_torch = torch.tensor(Z, dtype=torch.float64)
        D_Z = hyperbolic_distance(Z_torch).numpy()
        # Fill diagonal with 0
        np.fill_diagonal(D_Z, 0)
        
        return trustworthiness(D_Z, D_X, n_neighbors=k, metric='precomputed')
    else:
        return trustworthiness(Z, X, n_neighbors=k)

def calculate_continuity(X, Z, k=5, is_hyperbolic=False):
    """
    Calculate Continuity as Trustworthiness(Z, X).
    """
    if is_hyperbolic:
        D_X = pairwise_distances(X)
        Z_torch = torch.tensor(Z, dtype=torch.float64)
        D_Z = hyperbolic_distance(Z_torch).numpy()
        np.fill_diagonal(D_Z, 0)
        return trustworthiness(D_X, D_Z, n_neighbors=k, metric='precomputed')
    else:
        return trustworthiness(X, Z, n_neighbors=k)

def plot_shepard_diagram(X, Z, ax, title="Shepard Diagram", is_hyperbolic=False):
    """
    Plot Shepard Diagram: Input Distances vs Embedding Distances.
    """
    # Subsample if too large
    n = X.shape[0]
    if n > 500:
        indices = np.random.choice(n, 500, replace=False)
        X_sub = X[indices]
        Z_sub = Z[indices]
    else:
        X_sub = X
        Z_sub = Z
        
    D_X = pairwise_distances(X_sub)
    
    if is_hyperbolic:
        Z_torch = torch.tensor(Z_sub, dtype=torch.float64)
        D_Z = hyperbolic_distance(Z_torch).numpy()
    else:
        D_Z = pairwise_distances(Z_sub)
    
    # Flatten upper triangle
    mask = np.triu(np.ones_like(D_X, dtype=bool), k=1)
    dx = D_X[mask]
    dz = D_Z[mask]
    
    ax.scatter(dx, dz, s=1, alpha=0.5)
    ax.set_xlabel("Input Distance")
    ax.set_ylabel("Embedding Distance")
    ax.set_title(title)
    
    # Add correlation
    if len(dx) > 0:
        corr = np.corrcoef(dx, dz)[0, 1]
        ax.text(0.05, 0.95, f"Corr: {corr:.3f}", transform=ax.transAxes, verticalalignment='top')

def run_experiment(n_samples=1000, n_iter=500, perplexity=30):
    generator = SyntheticManifolds(n_samples=n_samples)
    datasets = [
        generator.generate_blobs(),
        generator.generate_swiss_roll(),
        generator.generate_s_curve(),
        generator.generate_tree()
    ]
    
    # Auto-detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Configurations to test
    configs = [
        {"name": "Learned Alpha", "alpha_init": 1.0, "learn_alpha": True, "reg": 0.0},
        {"name": "Fixed Alpha=1", "alpha_init": 1.0, "learn_alpha": False, "reg": 0.0},
        {"name": "Learned Alpha + Reg", "alpha_init": 1.0, "learn_alpha": True, "reg": 1.0},
    ]

    for i, (X, labels, name) in enumerate(datasets):
        print(f"\nProcessing {name}...")
        
        # Normalize input data
        X_norm = torch.tensor(StandardScaler().fit_transform(X), dtype=torch.float32)
        
        # Determine specific settings for this dataset
        dataset_configs = configs
        if name == "Synthetic Tree":
            # Add Hyperbolic config for Tree
            dataset_configs = configs + [{"name": "Hyperbolic", "alpha_init": 1.0, "learn_alpha": True, "reg": 0.0, "hyperbolic": True}]
        
        # Prepare figure for this dataset
        n_configs = len(dataset_configs)
        fig, axes = plt.subplots(n_configs, 3, figsize=(15, 5 * n_configs))
        if n_configs == 1: axes = axes[None, :]
        
        for j, config in enumerate(dataset_configs):
            print(f"  Config: {config['name']}")
            
            # Input Affinity
            affinity_data = EntropicAffinity(perp=perplexity)
            
            is_hyperbolic = config.get("hyperbolic", False)
            
            # Embedding Affinity
            if is_hyperbolic:
                affinity_embedding = LorentzHyperbolicAffinity() 
            else:
                affinity_embedding = LearnableNormalizedGaussianAndStudentAffinity(alpha_init=config["alpha_init"])
                if not config["learn_alpha"]:
                    affinity_embedding.alpha.requires_grad = False
            
            if isinstance(affinity_embedding, nn.Module):
                affinity_embedding.to(device)
            
            # Initialize T
            T_init = torch.eye(n_samples)
            
            # Initialize Z
            if is_hyperbolic:
                # Use WrappedNormal for Hyperbolic
                init_method = "WrappedNormal"
                optimizer_name = "RAdam"
                Z_init = None # Let DistR handle initialization
            else:
                init_method = "random" # Or PCA
                optimizer_name = "Adam"
                from sklearn.decomposition import PCA
                Z_pca = PCA(n_components=2).fit_transform(X_norm.numpy())
                Z_init = torch.tensor(Z_pca, dtype=torch.float32)
            
            # Initialize DistR model
            model = DistRWithLogging(
                affinity_data=affinity_data,
                affinity_embedding=affinity_embedding,
                output_sam=n_samples,
                output_dim=2,
                loss_fun="kl_loss",
                optimizer=optimizer_name,
                lr=0.01 if is_hyperbolic else 0.1,
                lr_affinity=0.1,
                max_iter=n_iter,
                max_iter_outer=1,
                init_T=T_init,
                init=Z_init if Z_init is not None else init_method,
                verbose=True,
                tol=0,
                early_stopping=False,
                dtype=torch.float64,
                device=device,
                alpha_reg=config.get("reg", 0.0)
            )
            
            Z = model.fit_transform(X_norm.to(dtype=torch.float64))
            Z_np = Z.detach().cpu().numpy()
            
            # Metrics
            trust = calculate_trustworthiness(X.numpy(), Z_np, k=5, is_hyperbolic=is_hyperbolic)
            cont = calculate_continuity(X.numpy(), Z_np, k=5, is_hyperbolic=is_hyperbolic)
            
            final_alpha = "N/A"
            if hasattr(model.affinity_embedding, 'alpha'):
                final_alpha = f"{torch.nn.functional.softplus(model.affinity_embedding.alpha).item():.2f}"
            
            print(f"    Trustworthiness: {trust:.4f}, Continuity: {cont:.4f}, Alpha: {final_alpha}")
            
            # Plot 1: Scatter
            ax_sc = axes[j, 0]
            
            if is_hyperbolic:
                # Project to Poincaré for visualization
                Z_vis = lorentz_to_poincare(Z.detach().cpu()).numpy()
                # Draw boundary circle
                circle = plt.Circle((0, 0), 1, color='r', fill=False, linestyle='--')
                ax_sc.add_artist(circle)
                ax_sc.set_xlim(-1.1, 1.1)
                ax_sc.set_ylim(-1.1, 1.1)
            else:
                Z_vis = Z_np
                
            sc = ax_sc.scatter(Z_vis[:, 0], Z_vis[:, 1], c=labels, cmap='viridis', s=10, alpha=0.7)
            ax_sc.set_title(f"{config['name']}\nTrust: {trust:.2f}, Cont: {cont:.2f}, Alpha: {final_alpha}")
            plt.colorbar(sc, ax=ax_sc)
            
            # Plot 2: Shepard
            ax_sh = axes[j, 1]
            plot_shepard_diagram(X.numpy(), Z_np, ax_sh, is_hyperbolic=is_hyperbolic)
            
            # Plot 3: Alpha Trajectory
            ax_al = axes[j, 2]
            if model.alpha_logs:
                ax_al.plot(model.alpha_logs)
                ax_al.set_title("Alpha Trajectory")
                ax_al.set_xlabel("Iter")
            else:
                ax_al.text(0.5, 0.5, "Fixed Alpha", ha='center')
        
        plt.tight_layout()
        plt.savefig(f'benchmark_{name.replace(" ", "_")}.png')
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=500) # Reduced for speed
    parser.add_argument('--n_iter', type=int, default=200)
    parser.add_argument('--perplexity', type=float, default=150.0)
    args = parser.parse_args()
    run_experiment(n_samples=args.n_samples, n_iter=args.n_iter, perplexity=args.perplexity)
