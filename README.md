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

#### `evaluate_prototypes(Z, T, Y_true, X)`
Computes clustering metrics using the centralized scores implementation.

**Metrics**:
- **hom**: Homogeneity score
- **ami**: Adjusted Mutual Information
- **ari**: Adjusted Rand Index
- **nmi**: Normalized Mutual Information
- **sil**: Silhouette score

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
