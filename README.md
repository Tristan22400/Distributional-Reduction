# Distributional Reduction (DistR)

## 📌 Overview

**DistR** (Distributional Reduction) is a unified framework for simultaneous dimensionality reduction and clustering. It formulates the problem as the minimization of the **Gromov-Wasserstein (GW)** divergence between the empirical input distribution $\mu_X$ and a reduced prototype distribution $\mu_Z$.

This repository contains the implementation of the DistR framework modified to replicate the results of the original paper. This was done in the case of an academic course project in the MVA Master's program at ENS Paris-Saclay for the Geometric Data Analysis course.

## 🚀 Quick Start

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd Distributional-Reduction
    ```

2.  **Install dependencies:**
    ```bash
    vis_python_source="virtualenv" # or conda
    pip install -r requirements.txt
    ```
    *Note: GPU support requires a CUDA-enabled version of PyTorch.*

### Running the Replication Benchmark

To run the full replication benchmark as defined in the plan:

```bash
python src/test.py
```

This script is pre-configured to run a **fast test** on a subset of the data (100 samples) to verify functionality.

To run the **full experiments** as described in the paper, you need to modify the **Configuration Parameters** section at the top of `src/test.py`:

1.  **Disable Subsampling:**
    ```python
    # src/test.py
    subset_size = None  # Set to None to use the full dataset (instead of 100)
    ```

2.  **Increase Repetitions:**
    ```python
    n_experiments = 5  # Recommended for robust statistics (default is 1)
    ```

3.  **Adjust Datasets:**
    Ensure the `DATASETS` list includes all targets:
    ```python
    DATASETS = ['coil20', 'fmnist', 'pbmc', 'zeisel', 'mnist']
    ```

The script will generate:
*   `results.json`: Raw metrics for all runs.
*   `multi_dataset_evolution.png`: Evolution of metrics (NMI, ARI, etc.) vs number of prototypes.
*   `tradeoff_analysis.png`: Trade-off between local (Homogeneity) and global (NMI) structure.

## ⚙️ Supported Configurations

The codebase supports all four configurations required for proper benchmarking:

| Configuration | Input Similarity ($C_X$) | Output Similarity ($C_Z$) | Inner Loss | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Spectral** | Linear ($XX^T$) | Linear ($ZZ^T$) | Square Loss | Kernel PCA equivalent |
| **NE (t-SNE)** | Symmetric Entropic Affinity | Student's t-kernel | KL Divergence | Standard t-SNE / SNEkhorn behavior |
| **NE (UMAP)** | UMAP Affinity (Fuzzy Simplicial Set) | Parameterized t-kernel | Binary Cross Entropy | UMAP equivalent |
| **Hyperbolic** | Hyperbolic Entropic Affinity | Hyperbolic Student's t | KL Divergence | Geometric generalization |

## 📊 Datasets

The repository includes a unified data loading pipeline (`src/dataset_pipeline.py`) supporting:

*   **Image:** MNIST, Fashion-MNIST, COIL-20
*   **Genomics:** PBMC 3k, ZEISEL

## 📝 Compliance

This implementation strictly follows the original paper's implementation details:
*   **Solvers:** Implements **Conditional Gradient** (for unregularized GW) and **Mirror Descent** (for entropic GW/Sinkhorn).
*   **Optimization:** Uses **Block Coordinate Descent (BCD)** for alternating updates of Transport ($T$) and Embedding ($Z$).
*   **Complexity:** Leverages low-rank factorization and optimized matrix operations for efficiency.
