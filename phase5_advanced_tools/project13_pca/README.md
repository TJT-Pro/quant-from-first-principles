# Project 13 — PCA from Scratch

Principal Component Analysis implemented from its mathematical definition.
No numpy, no scipy, no sklearn.  Builds on the matrix engine from Project 12.

---

## What is implemented

### Preprocessing
`column_means`, `centre` (subtract column means), `column_stds`,
`standardise` (zero mean, unit variance per column).

### Covariance matrix
`covariance_matrix(X)` — computes XᵀX / (n−1) for a centred n×p matrix.

### Jacobi eigenvalue algorithm
`jacobi_eigen(C)` — exact eigenvalues and eigenvectors of any real symmetric
matrix via iterated Givens rotations.  Each rotation zeros one off-diagonal
pair; sweeps continue until the off-diagonal Frobenius norm falls below
tolerance.  Returns sorted (descending) eigenvalues and the orthonormal
eigenvector matrix V such that CV = VΛ.

### Full PCA pipeline
`pca(X, n_components, standardise_data)` — one-call entry point:
1. Centre (or standardise) the data
2. Compute the covariance (or correlation) matrix
3. Run Jacobi to get eigenpairs
4. Sort by descending eigenvalue
5. Project: scores = X_centred @ V[:, :k]

Returns components, eigenvalues, explained variance ratios, cumulative
ratios, scores, means, stds.

### Reconstruction
`reconstruct(result, n_components)` — maps scores back to the original
feature space: X_hat = scores[:, :k] @ V[:, :k]ᵀ + means.

`reconstruction_error(X, result, n_components)` — mean squared error
between the original and reconstructed data.

### Whitening
`whitened_scores(result)` — divides each score column by √eigenvalue,
giving identity covariance in component space.

### Variance threshold
`n_components_for_variance(result, threshold)` — minimum k such that
cumulative explained variance ≥ threshold.

### Lag matrix (time series feature engineering)
`lag_matrix(series, n_lags)` — builds an n×p matrix from a 1D time
series where each row contains the current value and n_lags−1 previous
values.  Used to apply PCA to the EUR/USD return series.

---

## Key formulas

### Covariance matrix
    C = XᵀX / (n−1)   where X is mean-centred

### Jacobi Givens rotation
    θ = ½ · (C_qq − C_pp) / C_pq
    t = sign(θ) / (|θ| + √(1 + θ²))
    c = 1/√(1+t²),   s = tc

### Reconstruction
    X̂ = Z @ Vᵀ + μ   where Z = X_c @ V[:, :k]

### Whitened scores
    z̃_ij = z_ij / √λ_j

---

## Limitations

The Jacobi algorithm is O(p³ · sweeps) and is practical for p up to
a few hundred.  For p in the thousands, iterative Lanczos or randomised
SVD methods are more appropriate.

The lag matrix approach treats a time series as i.i.d. rows, ignoring
the autocorrelation structure.  It is used here to demonstrate the
mechanics of PCA on real financial data, not as a forecasting model.

---

## Running the tests

```
python3 -m pytest test_pca.py -v
```

42 tests covering preprocessing, covariance matrix, Jacobi eigenvalue
algorithm (eigenvalues, eigenvectors, orthonormality, Cv=λv), full PCA
pipeline (shapes, explained variance, cumulative ratio, score means,
component orthogonality), reconstruction (full and partial), whitening,
variance threshold, lag matrix, and EUR/USD integration tests.

## Oracle verification

```
python3 cli.py verify
```

18 checks: column means, centring, covariance of perfectly correlated
features, covariance symmetry, eigenvalue sum = trace, Jacobi eigenvalues
of a known 2×2, eigenvector orthonormality, Cv=λv for both eigenpairs,
PCA explained ratio sum, eigenvalue ordering, cumulative ratio monotonicity,
score zero-mean, component orthogonality, full reconstruction exactness,
whitened unit variance, and lag matrix shape and values.

## CLI examples

```bash
# PCA on lag-5 feature matrix of EUR/USD returns
python3 cli.py pca  --data data/DEXUSEU_returns.csv --lags 5 --k 5

# Scree plot showing variance per component
python3 cli.py scree --data data/DEXUSEU_returns.csv --lags 5

# Reconstruction error as function of k
python3 cli.py reconstruct --data data/DEXUSEU_returns.csv --lags 5 --k 5

# Whitened scores (unit variance per component)
python3 cli.py whiten --data data/DEXUSEU_returns.csv --lags 5 --k 3
```
