# Distributional-Reduction

Implementation of the paper [Distributional Reduction: Unifying Dimensionality Reduction and Clustering with Gromov-Wasserstein](https://arxiv.org/abs/2402.02239).

*Distributional Reduction* is a framework based on the Gromov-Wasserstein optimal transport problem to jointly address clustering and dimensionality reduction.

## Installation

### Prerequisites
- Python >= 3.7
- CUDA-compatible GPU (recommended for faster computation)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Tristan22400/Distributional-Reduction.git
cd Distributional-Reduction
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Alternatively, you can install the package in development mode:
```bash
pip install -e .
```

## Usage

### Quick Start: Generate Prototype Evolution Curves

To replicate the experiments and generate prototype evolution curves with variance envelopes:

```bash
cd src
python test.py
```

This will:
1. Load the MNIST dataset (1000 samples subset by default)
2. Run experiments with different numbers of prototypes (10, 20, 30, 50, 100)
3. Execute multiple runs with different random seeds to compute variance
4. Generate a plot showing the evolution of clustering metrics (Homogeneity, AMI, ARI, NMI, Silhouette) with variance envelopes
5. Save the plot as `mnist_prototype_evolution.png`

### Configuration

You can configure the experiment parameters by editing the top of `src/test.py`:

```python
# 1. Affinity Data (Similarity for input space)
# Options:
# - SymmetricEntropicAffinity(perp=30): SEA similarity
# - NormalizedGaussianAndStudentAffinity(sigma=1.0, student=False): Gaussian kernel
# - NormalizedGaussianAndStudentAffinity(student=True): Student's t-distribution
AFFINITY_DATA = SymmetricEntropicAffinity(perp=30, verbose=False)

# 2. Affinity Embeddings (Similarity for embedding space)
# Options:
# - NormalizedGaussianAndStudentAffinity(student=True): Student's kernel (t-SNE like)
# - NormalizedGaussianAndStudentAffinity(student=False): Gaussian kernel
# - SymmetricEntropicAffinity(perp=30): SEA similarity
AFFINITY_EMBEDDINGS = NormalizedGaussianAndStudentAffinity(student=True)

# 3. Loss Function
# Options:
# - "kl_loss": Kullback-Leibler divergence (standard for t-SNE/DistR)
# - "cross_entropy": Cross Entropy loss
# - "l2": L2 / Mean Squared Error loss
LOSS_FUNCTION = "kl_loss"

# 4. Output Dimension
OUTPUT_DIM = 2
```

You can also modify:
- `datasets_to_load`: List of datasets to load (e.g., `['mnist', 'fmnist', 'coil20']`)
- `subset_size`: Number of samples to use (line 76)
- `prototype_counts`: List of prototype numbers to test (line 99)
- `n_seeds`: Number of random seeds for variance computation (in `plot_prototype_evolution` call)

## Modifications Made to the Codebase

### 1. Dataset Loading Pipeline (`src/dataset_pipeline.py`)

**Purpose**: Provides a unified, reproducible pipeline for loading and preprocessing datasets.

**Features**:
- **Image datasets**: MNIST, Fashion-MNIST, COIL-20
- **Genomics datasets**: PBMC, Zeisel, SNAREseq (gene and chromatin)
- **Preprocessing**: 
  - Images: Normalization, flattening, StandardScaler, PCA
  - Genomics: Scanpy-based filtering, normalization, log1p, highly variable genes (HVG), PCA
- **Reproducibility**: Fixed random seeds (seed=42) for deterministic results
- **Unified interface**: `load_dataset(name, pca_dim=50)` returns a dictionary with:
  - `X`: PCA-reduced features (N × pca_dim)
  - `Y`: Labels (N,)
  - `original_dim`: Original feature dimension
  - `reduced_dim`: Dimension after PCA
  - `n_samples`: Number of samples
  - `name`: Dataset name

**Usage**:
```python
from src.dataset_pipeline import load_dataset

data = load_dataset('mnist', pca_dim=50)
X = data['X']  # Shape: (N, 50)
Y = data['Y']  # Shape: (N,)
```

### 2. Replication Protocol (`src/replication.py`)

**Purpose**: Implements the experimental protocol from the paper to compare DistR with baseline methods.

**Key Functions**:

#### `run_experiment(data_dict, dataset_name, n_prototypes, device, affinity_data, affinity_embeddings, output_dim, loss_function, seed)`
Runs a single experiment comparing three methods:
1. **DistR**: Joint dimensionality reduction and clustering
2. **DR → Clustering**: Sequential approach (dimensionality reduction first)
3. **Clustering → DR**: Sequential approach (clustering first)

**Parameters**:
- `data_dict`: Dictionary with 'X' (features) and 'Y' (labels)
- `n_prototypes`: Number of prototypes/clusters
- `device`: 'cuda' or 'cpu'
- `affinity_data`: Affinity function for input space
- `affinity_embeddings`: Affinity function for embedding space
- `output_dim`: Embedding dimension (typically 2)
- `loss_function`: Loss function ("kl_loss", "cross_entropy", or "l2")
- `seed`: Random seed for reproducibility

**Returns**: Dictionary with metrics for each method (homogeneity, AMI, ARI, NMI, silhouette)

#### `plot_prototype_evolution(prototype_counts, data_dict, dataset_name, device, affinity_data, affinity_embeddings, output_dim, loss_function, n_seeds)`
Generates evolution curves showing how metrics change with the number of prototypes.

**Features**:
- Runs experiments for each prototype count with multiple random seeds
- Computes mean and standard deviation across seeds
- Plots metrics with variance envelopes (mean ± std)
- Saves plot as `{dataset_name}_prototype_evolution.png`

**Parameters**:
- `prototype_counts`: List of prototype numbers to test (e.g., [10, 20, 50, 100])
- `n_seeds`: Number of random seeds for variance estimation (default: 3)
- Other parameters same as `run_experiment`

#### `get_majority_vote_labels(T, Y)`
Assigns class labels to prototypes based on majority voting (weighted by transport plan).

**GPU Optimization**: Fully vectorized operations on GPU for efficiency.

#### `evaluate_prototypes(Z, T, Y_true, X)`
Computes clustering metrics using the centralized scores implementation.

**Metrics**:
- **hom**: Homogeneity score
- **ami**: Adjusted Mutual Information
- **ari**: Adjusted Rand Index
- **nmi**: Normalized Mutual Information
- **sil**: Silhouette score

### 3. Test Script (`src/test.py`)

**Purpose**: Main entry point for running experiments and generating curves.

**Workflow**:
1. Load dataset(s) using `dataset_pipeline.load_dataset()`
2. Create a subset for faster experimentation
3. Configure experiment parameters (affinities, loss function, output dimension)
4. Call `plot_prototype_evolution()` to generate curves
5. Save results as PNG

**Key Features**:
- **Configurable parameters** at the top of the file (see Configuration section above)
- **Auto-detection** of GPU availability
- **Subset sampling** for quick testing (configurable `subset_size`)
- **Multiple prototype counts** tested in a single run

## Project Structure

```
Distributional-Reduction/
├── src/
│   ├── dataset_pipeline.py    # Dataset loading and preprocessing
│   ├── replication.py          # Experimental protocol and plotting
│   ├── test.py                 # Main script to run experiments
│   ├── clust_dr.py            # DistR and baseline implementations
│   ├── affinities.py          # Affinity functions
│   ├── scores.py              # Evaluation metrics
│   ├── dimension_reduction.py # Dimensionality reduction methods
│   ├── utils.py               # Utility functions
│   └── utils_hyperbolic.py    # Hyperbolic geometry utilities
├── local_ot/                  # Local optimal transport implementations
├── data/                      # Downloaded datasets (auto-created)
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Package configuration
└── README.md                 # This file
```

## How to Generate Curves

The prototype evolution curves show how clustering quality metrics evolve as the number of prototypes increases. Here's how they are generated:

1. **Multiple Seeds**: For each prototype count, the experiment runs `n_seeds` times with different random initializations
2. **Metric Computation**: For each run, clustering metrics (homogeneity, AMI, ARI, NMI, silhouette) are computed
3. **Statistical Aggregation**: Mean and standard deviation are computed across seeds
4. **Visualization**: 
   - Mean values are plotted as lines with markers
   - Variance envelopes (mean ± std) are shown as shaded regions
   - Each metric gets its own subplot

**Example**:
```python
from src.replication import plot_prototype_evolution
from src.dataset_pipeline import load_dataset

# Load dataset
data = load_dataset('mnist', pca_dim=50)

# Take subset
data_subset = {
    'X': data['X'][:1000],
    'Y': data['Y'][:1000]
}

# Generate curves
plot_prototype_evolution(
    prototype_counts=[10, 20, 30, 50, 100],
    data_subset=data_subset,
    dataset_name='mnist',
    device='cuda',
    n_seeds=5  # More seeds = better variance estimation
)
```

This will create `mnist_prototype_evolution.png` with 5 subplots (one per metric), each showing 3 curves (DistR, DR→Clust, Clust→DR) with variance envelopes.

## Citation

```bibtex
@article{van2024distributional,
  title={Distributional Reduction: Unifying Dimensionality Reduction and Clustering with Gromov-Wasserstein},
  author={Van Assel, Hugues and Vincent-Cuaz, C{\'e}dric and Courty, Nicolas and Flamary, R{\'e}mi and Frossard, Pascal and Vayer, Titouan},
  journal={arXiv preprint arXiv:2402.02239},
  year={2024},
  url={https://arxiv.org/abs/2402.02239}
}
```

## Authors

* [Hugues Van Assel](https://huguesva.github.io/)
* [Cédric Vincent-Cuaz](https://cedricvincentcuaz.github.io/)
* [Nicolas Courty](https://ncourty.github.io/)
* [Rémi Flamary](https://remi.flamary.com/)
* [Pascal Frossard](https://people.epfl.ch/pascal.frossard)
* [Titouan Vayer](https://tvayer.github.io/)

## License

MIT License
