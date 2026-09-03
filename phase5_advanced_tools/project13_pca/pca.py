import csv
import math
import os
import sys

# Import the matrix engine built in project 12
_HERE = os.path.dirname(os.path.abspath(__file__))
_P12  = os.path.join(_HERE, "..", "project12_matrix_engine")
sys.path.insert(0, _P12)
import matrix_engine as mx


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path):
    """Load a two-column CSV (date, value) and return the value column."""
    values = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)           # skip header
        for row in reader:
            if row and row[1].strip():
                values.append(float(row[1]))
    return values


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def column_means(X):
    """Mean of each column of an n×p data matrix X.

    Parameters
    ----------
    X : list of n rows, each a list of p floats

    Returns a list of p means.
    """
    n, p = len(X), len(X[0])
    return [sum(X[i][j] for i in range(n)) / n for j in range(p)]


def centre(X):
    """Subtract the column mean from each column.

    Returns (X_centred, means) where means can be used to invert later.
    """
    means  = column_means(X)
    p      = len(means)
    Xc     = [[X[i][j] - means[j] for j in range(p)] for i in range(len(X))]
    return Xc, means


def column_stds(X):
    """Sample standard deviation of each column."""
    n, p = len(X), len(X[0])
    means = column_means(X)
    stds  = []
    for j in range(p):
        var = sum((X[i][j] - means[j])**2 for i in range(n)) / (n - 1)
        stds.append(math.sqrt(var))
    return stds


def standardise(X):
    """Centre and scale each column to unit variance.

    Returns (X_standardised, means, stds).
    """
    n, p  = len(X), len(X[0])
    means = column_means(X)
    stds  = column_stds(X)
    Xs    = [[(X[i][j] - means[j]) / (stds[j] if stds[j] > 0 else 1.0)
               for j in range(p)]
              for i in range(n)]
    return Xs, means, stds


# ---------------------------------------------------------------------------
# Covariance matrix
# ---------------------------------------------------------------------------

def covariance_matrix(X):
    """Sample covariance matrix of an n×p centred data matrix.

    C[i][j] = (1/(n-1)) * sum_k X[k][i] * X[k][j]

    Equivalent to XᵀX / (n-1).

    Parameters
    ----------
    X : n×p list-of-lists, should be mean-centred already
    """
    n, p = len(X), len(X[0])
    if n <= 1:
        raise ValueError("need at least 2 observations for covariance")
    C = mx.zeros(p, p)
    for i in range(p):
        for j in range(i, p):
            s = sum(X[k][i] * X[k][j] for k in range(n)) / (n - 1)
            C[i][j] = s
            C[j][i] = s
    return C


# ---------------------------------------------------------------------------
# Jacobi eigenvalue algorithm for symmetric matrices
# ---------------------------------------------------------------------------

def _jacobi_rotate(A, V, p, q):
    """Apply one Jacobi Givens rotation to zero A[p][q].

    Updates A and V in-place.
    A is the symmetric matrix being diagonalised.
    V accumulates the product of all rotation matrices (eigenvectors).
    """
    n = len(A)
    if abs(A[p][q]) < 1e-15:
        return
    theta = 0.5 * (A[q][q] - A[p][p]) / A[p][q]
    t     = math.copysign(1.0 / (abs(theta) + math.sqrt(1.0 + theta*theta)),
                          theta)
    c     = 1.0 / math.sqrt(1.0 + t*t)
    s     = t * c

    # Update the full matrix
    Apq   = A[p][q]
    A[p][p] -= t * Apq
    A[q][q] += t * Apq
    A[p][q]  = 0.0
    A[q][p]  = 0.0

    for r in range(n):
        if r != p and r != q:
            Apr = A[p][r]
            Aqr = A[q][r]
            A[p][r] = c * Apr - s * Aqr
            A[r][p] = A[p][r]
            A[q][r] = s * Apr + c * Aqr
            A[r][q] = A[q][r]

    for r in range(n):
        Vrp = V[r][p]
        Vrq = V[r][q]
        V[r][p] = c * Vrp - s * Vrq
        V[r][q] = s * Vrp + c * Vrq


def jacobi_eigen(C, max_sweeps=100, tolerance=1e-12):
    """Eigenvalues and eigenvectors of a real symmetric matrix via Jacobi.

    Repeatedly applies Givens rotations (Jacobi sweeps) to zero the
    off-diagonal elements until the off-diagonal Frobenius norm is below
    tolerance.

    Returns (eigenvalues, eigenvectors) where:
        eigenvalues  : list of p floats (diagonal of the diagonalised matrix)
        eigenvectors : p×p matrix whose columns are the eigenvectors

    Parameters
    ----------
    C          : p×p real symmetric matrix
    max_sweeps : maximum number of full sweeps
    tolerance  : stop when off-diagonal Frobenius norm < tolerance
    """
    p = len(C)
    A = mx.copy_matrix(C)      # working copy, will be diagonalised
    V = mx.identity(p)         # accumulates rotation product

    for _ in range(max_sweeps):
        # Off-diagonal Frobenius norm
        off = math.sqrt(sum(A[i][j]**2
                            for i in range(p) for j in range(p) if i != j))
        if off < tolerance:
            break
        # One sweep: zero every off-diagonal element
        for i in range(p - 1):
            for j in range(i + 1, p):
                if abs(A[i][j]) > tolerance * 0.01:
                    _jacobi_rotate(A, V, i, j)

    eigenvalues = [A[i][i] for i in range(p)]
    return eigenvalues, V


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def pca(X, n_components=None, standardise_data=False):
    """Full PCA pipeline on an n×p data matrix X.

    Steps:
        1. Optionally standardise each feature to zero mean, unit variance
        2. Otherwise just centre (subtract column means)
        3. Compute the p×p covariance (or correlation) matrix
        4. Compute eigenpairs via Jacobi
        5. Sort by descending eigenvalue
        6. Project data onto the top n_components principal components

    Returns a dict with:
        components      : p×k matrix (columns = principal components,
                          sorted by descending explained variance)
        eigenvalues     : list of k eigenvalues (variances per component)
        explained_ratio : list of k floats (fraction of total variance)
        cumulative_ratio: list of k floats (cumulative explained variance)
        scores          : n×k projection of the centred data
        means           : column means (for reconstruction)
        stds            : column stds if standardised (else None)
        n_components    : k actually used

    Parameters
    ----------
    X              : n×p data matrix (list of lists)
    n_components   : number of components to keep (default: all p)
    standardise_data: if True, scale each feature to unit variance first
    """
    n, p = len(X), len(X[0])
    if n_components is None or n_components > p:
        n_components = p

    if standardise_data:
        Xp, means, stds = standardise(X)
    else:
        Xp, means = centre(X)
        stds = None

    C = covariance_matrix(Xp)
    eigenvalues, V = jacobi_eigen(C)

    # Sort by descending eigenvalue
    order = sorted(range(p), key=lambda i: -eigenvalues[i])
    eigenvalues = [eigenvalues[order[i]] for i in range(p)]
    # V columns (eigenvectors): extract in sorted order
    components = [[V[r][order[i]] for i in range(n_components)]
                  for r in range(p)]

    total_var      = sum(max(ev, 0) for ev in eigenvalues)
    explained      = [max(eigenvalues[i], 0) / total_var
                      if total_var > 0 else 0.0
                      for i in range(n_components)]
    cumulative     = []
    running        = 0.0
    for e in explained:
        running += e
        cumulative.append(running)

    # Project: scores[i][k] = dot(Xp[i], components[:, k])
    scores = [[sum(Xp[i][r] * components[r][k]
                   for r in range(p))
               for k in range(n_components)]
              for i in range(n)]

    return {
        "components":       components,
        "eigenvalues":      eigenvalues[:n_components],
        "explained_ratio":  explained,
        "cumulative_ratio": cumulative,
        "scores":           scores,
        "means":            means,
        "stds":             stds,
        "n_components":     n_components,
    }


def reconstruct(result, n_components=None):
    """Reconstruct the original data from a PCA result.

    Uses the first n_components principal components (default: all
    retained in the result).  Adds back the column means (and scales
    by stds if the data was standardised).

    Returns the reconstructed n×p matrix in the original feature space.

    Parameters
    ----------
    result       : dict returned by pca()
    n_components : number of components to use for reconstruction
                   (must be <= result['n_components'])
    """
    k_max  = result["n_components"]
    k      = n_components if n_components is not None else k_max
    if k > k_max:
        raise ValueError(
            "n_components {} > retained components {}".format(k, k_max)
        )

    scores     = result["scores"]
    components = result["components"]
    means      = result["means"]
    stds       = result["stds"]
    n          = len(scores)
    p          = len(means)

    # X_hat = scores[:, :k] @ components[:, :k]^T + means
    Xhat = []
    for i in range(n):
        row = []
        for j in range(p):
            val = sum(scores[i][c] * components[j][c] for c in range(k))
            if stds is not None:
                val = val * stds[j] + means[j]
            else:
                val = val + means[j]
            row.append(val)
        Xhat.append(row)
    return Xhat


def reconstruction_error(X, result, n_components=None):
    """Mean squared reconstruction error using n_components components.

    MSE = (1/np) * sum_{i,j} (X[i][j] - X_hat[i][j])²

    Parameters
    ----------
    X            : original n×p data matrix
    result       : dict returned by pca()
    n_components : components to use (default: all retained)
    """
    Xhat = reconstruct(result, n_components)
    n, p = len(X), len(X[0])
    return sum((X[i][j] - Xhat[i][j])**2
               for i in range(n) for j in range(p)) / (n * p)


def whitened_scores(result):
    """Return whitened scores: each component scaled to unit variance.

    Whitening divides each score column by sqrt(eigenvalue), giving
    data with identity covariance in the principal component space.
    Used as a preprocessing step for some machine learning algorithms.

    Returns a list of lists with the same shape as result['scores'].
    """
    scores = result["scores"]
    evs    = result["eigenvalues"]
    n      = len(scores)
    k      = result["n_components"]
    return [[scores[i][c] / math.sqrt(max(evs[c], 1e-15))
             for c in range(k)]
            for i in range(n)]


def n_components_for_variance(result, threshold=0.95):
    """Minimum number of components to explain at least threshold of variance.

    Parameters
    ----------
    result    : dict returned by pca()
    threshold : target cumulative explained variance (default 0.95)
    """
    for k, cum in enumerate(result["cumulative_ratio"], start=1):
        if cum >= threshold:
            return k
    return result["n_components"]


# ---------------------------------------------------------------------------
# Build a lagged feature matrix from a time series
# ---------------------------------------------------------------------------

def lag_matrix(series, n_lags):
    """Build an n×p matrix of lagged values from a 1D time series.

    Row i contains [series[i], series[i-1], ..., series[i-n_lags+1]].
    The first n_lags-1 rows are dropped (incomplete lags).

    Parameters
    ----------
    series : list of floats
    n_lags : number of lag columns (including lag-0)
    """
    if n_lags < 1:
        raise ValueError("n_lags must be at least 1")
    result = []
    for i in range(n_lags - 1, len(series)):
        row = [series[i - lag] for lag in range(n_lags)]
        result.append(row)
    return result
