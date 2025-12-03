import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
import ot
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_swiss_roll, make_s_curve
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import random
import argparse
import sys
import os

# Add project root to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import real DistR and affinities
from src.clust_dr import DistR
from src.affinities import LearnableNormalizedGaussianAndStudentAffinity, NormalizedGaussianAndStudentAffinity, EntropicAffinity

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
        # Depth 9 gives 2^10 - 1 = 1023 nodes, close to 1000
        G = nx.balanced_tree(r=2, h=9)
        # Subsample to exactly n_samples if needed, but 1023 is close enough. 
        # Let's just take the first n_samples nodes if > n_samples, or regenerate.
        # Actually, let's just use the graph as is and slice.
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
        
        # FIX: Re-initialize optimizer because affinity_embedding parameters 
        # were missed due to initialization order in parent classes.
        # DataSummarizer.__init__ calls _init_embedding BEFORE AffinityBasedDataSummarizer sets affinity_embedding.
        
        # Re-collect parameters
        params = [{'params': [self.Z], 'lr': self.lr}]
        if hasattr(self, 'affinity_embedding') and hasattr(self.affinity_embedding, 'parameters'):
            params.append({'params': self.affinity_embedding.parameters(), 'lr': self.lr_affinity})
            
        # Re-create optimizer
        # We need to access the optimizer class. self.optimizer is an instance now.
        # But we stored the optimizer name in self.optimizer_name? No.
        # But we passed 'optimizer' string to __init__.
        # We can use OPTIMIZERS dict if we import it, or just use optim.Adam since we hardcoded it in run_experiment.
        # Or better, check type of self.optimizer?
        # Let's just use optim.Adam as we know we use it.
        self.optimizer = optim.Adam(params)

    def _update_embedding(self, max_iter=None):
        """
        Optimize the embeddings coordinates using a gradient-based optimization method.
        """
        if max_iter is None:
            max_iter = self.max_iter
            
        self.Z.requires_grad = True
        # Ensure alpha is optimized if it's in parameters
        # The parent class handles optimizer creation in _init_embedding
        
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
                
                # Debug: Check gradient
                if self.affinity_embedding.alpha.grad is not None:
                    # print(f"Alpha grad: {self.affinity_embedding.alpha.grad.item()}")
                    pass
                else:
                    if i == 0:
                        print("Alpha grad is None!")
            
            if i == 0:
                # Debug: Check optimizer params
                print(f"Optimizer param groups: {len(self.optimizer.param_groups)}")
                for idx, group in enumerate(self.optimizer.param_groups):
                    print(f"Group {idx} lr: {group['lr']}, params: {len(group['params'])}")
                    for p in group['params']:
                        print(f"  Param shape: {p.shape}, requires_grad: {p.requires_grad}")
            
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

def run_experiment(n_samples=1000, n_iter=500, perplexity=30):
    generator = SyntheticManifolds(n_samples=n_samples)
    datasets = [
        generator.generate_blobs(),
        generator.generate_swiss_roll(),
        generator.generate_s_curve(),
        generator.generate_tree()
    ]
    
    results = []
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    alpha_trajectories = {}
    
    for i, (X, labels, name) in enumerate(datasets):
        print(f"Processing {name}...")
        
        # Normalize input data
        X = torch.tensor(StandardScaler().fit_transform(X), dtype=torch.float32)
        
        # Define Affinities
        # Input Affinity: Entropic (adaptive sigma)
        affinity_data = EntropicAffinity(perp=perplexity)
        
        # Embedding Affinity: Learnable Student-t
        affinity_embedding = LearnableNormalizedGaussianAndStudentAffinity(alpha_init=1.0)
        
        # Initialize T to identity to enforce point-to-point correspondence
        # Row sums must be 1 (since h0 is ones)
        T_init = torch.eye(n_samples)
        
        # Initialize Z with PCA for better starting point
        from sklearn.decomposition import PCA
        Z_pca = PCA(n_components=2).fit_transform(X.numpy())
        Z_init = torch.tensor(Z_pca, dtype=torch.float32)
        
        # Initialize DistR model
        model = DistRWithLogging(
            affinity_data=affinity_data,
            affinity_embedding=affinity_embedding,
            output_sam=n_samples, # Full batch
            output_dim=2,
            loss_fun="kl_loss", # Use KL loss for affinities
            optimizer="Adam",
            lr=0.1, # Learning rate for Z
            lr_affinity=0.1, # Learning rate for alpha
            max_iter=n_iter,
            max_iter_outer=1, # We only do one outer loop (Z update) for this benchmark
            init_T=T_init, # Initialize T to identity
            init=Z_init, # Initialize Z with PCA
            verbose=True,
            tol=0, # Disable delta check
            early_stopping=False, # Disable early stopping to see full trajectory
            dtype=torch.float32 # Match input dtype
        )
        
        # Fit
        # DistR expects X.
        Z = model.fit_transform(X)
        
        alphas = model.alpha_logs
        alpha_trajectories[name] = alphas
        final_alpha = alphas[-1] if alphas else 1.0
        
        # Compute Silhouette Score
        # We use the final Z embeddings
        Z_np = Z.detach().cpu().numpy()
        # Silhouette requires discrete labels.
        # For Blobs, we have cluster labels.
        # For Manifolds, we have continuous t. We can discretize t for silhouette or just skip/use a different metric.
        # Prompt says: "Log the final Silhouette Score of the embeddings."
        # This implies we should use the provided labels.
        # For continuous labels, silhouette is not well defined.
        # Let's discretize continuous labels into bins for silhouette calculation.
        if name == "Gaussian Blobs":
            sil_score = silhouette_score(Z_np, labels)
        else:
            # Discretize into 10 bins
            labels_disc = np.digitize(labels, np.linspace(labels.min(), labels.max(), 10))
            sil_score = silhouette_score(Z_np, labels_disc)
            
        results.append({"Dataset": name, "Converged Alpha": final_alpha, "Silhouette": sil_score})
        
        # Plot Embeddings
        ax = axes[i]
        sc = ax.scatter(Z_np[:, 0], Z_np[:, 1], c=labels, cmap='viridis', s=10, alpha=0.7)
        ax.set_title(f"{name}\nAlpha: {final_alpha:.2f}")
        plt.colorbar(sc, ax=ax)
    
    plt.tight_layout()
    plt.savefig('benchmark_embeddings.png')
    plt.close()
    
    # Plot Alpha Dynamics
    plt.figure(figsize=(10, 6))
    for name, alphas in alpha_trajectories.items():
        plt.plot(alphas, label=name)
    plt.xlabel("Iterations")
    plt.ylabel("Alpha")
    plt.title("Alpha Dynamics during Optimization")
    plt.legend()
    plt.grid(True)
    plt.savefig('benchmark_alpha_dynamics.png')
    plt.close()
    
    # Console Report
    print("\nBenchmark Results:")
    print(f"{'Dataset':<20} | {'Converged Alpha':<15} | {'Silhouette':<10}")
    print("-" * 50)
    for res in results:
        print(f"{res['Dataset']:<20} | {res['Converged Alpha']:<15.4f} | {res['Silhouette']:<10.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--n_iter', type=int, default=500)
    parser.add_argument('--perplexity', type=float, default=150.0)
    args = parser.parse_args()
    run_experiment(n_samples=args.n_samples, n_iter=args.n_iter, perplexity=args.perplexity)
