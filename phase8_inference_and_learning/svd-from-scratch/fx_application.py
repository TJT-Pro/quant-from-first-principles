import csv
import statistics
from typing import List, Dict

import linear_algebra as la
import jacobi_eigen as je
import svd


def load_log_returns(csv_path: str, column: str = "log_return") -> List[float]:
    values = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row[column]))
    return values


def build_volatility_matrix(returns: List[float], windows: List[int]) -> la.Matrix:
    """A data matrix with one row per date and one column per rolling-
    window realised volatility (the sample standard deviation of
    returns over the trailing `w` days, for each `w` in `windows`).
    Different window lengths are all measuring the same underlying
    quantity (how volatile the market currently is) at different
    timescales, so they are expected to be highly, but not perfectly,
    correlated: exactly the kind of cross-sectional structure PCA and
    SVD are built to summarise with a small number of components,
    unlike a single univariate time series, which has no cross-
    sectional structure for either method to find.
    """
    max_window = max(windows)
    rows = []
    for t in range(max_window, len(returns)):
        row = [statistics.pstdev(returns[t - w:t]) for w in windows]
        rows.append(row)
    return rows


def center_columns(x: la.Matrix) -> la.Matrix:
    m = len(x)
    n = len(x[0])
    means = [statistics.mean([x[i][j] for i in range(m)]) for j in range(n)]
    return [[x[i][j] - means[j] for j in range(n)] for i in range(m)]


def pca_via_covariance(x_centered: la.Matrix, auto_rescale: bool = True) -> Dict:
    """Principal Component Analysis via explicit eigendecomposition of
    the sample covariance matrix, cov = X^T*X / (m-1).

    `auto_rescale` defaults to True for a specific, tested reason: the
    Jacobi eigenvalue algorithm implemented in this project loses
    real, usable precision when applied directly to a covariance
    matrix built from realistically small-magnitude data (financial
    daily volatilities are typically O(1e-3) to O(1e-2), making
    covariance entries O(1e-5) to O(1e-4)). Found directly by testing:
    on synthetic data with a known true dominant loading vector at
    this realistic scale, eigendecomposing the raw covariance matrix
    returned a nonsense leading eigenvector, [1,0,0,0], instead of
    anything resembling the true loadings, while eigendecomposing the
    *same* covariance matrix after dividing every entry by its own
    largest diagonal value (a matrix similarity that does not change
    the eigenvectors and only rescales the eigenvalues by a known
    constant) recovered the true loadings to within about 0.1%. This
    is not a fundamental limitation of PCA or of eigendecomposition;
    it is a fixable numerical-conditioning issue specific to running
    an iterative eigenvalue algorithm on very-small-magnitude matrix
    entries, and rescaling before decomposing (then rescaling the
    returned eigenvalues back) is the fix applied here by default.
    """
    m = len(x_centered)
    n = len(x_centered[0])
    xt = la.transpose(x_centered)
    cov = la.scalar_mul(la.matmul(xt, x_centered), 1.0 / (m - 1))

    if auto_rescale:
        scale = max(abs(cov[i][i]) for i in range(n))
        if scale <= 0:
            raise ValueError("covariance matrix has zero variance in every dimension")
        cov_scaled = la.scalar_mul(cov, 1.0 / scale)
        eigenvalues_scaled, eigenvectors = je.sorted_eigen(cov_scaled)
        eigenvalues = [ev * scale for ev in eigenvalues_scaled]
    else:
        eigenvalues, eigenvectors = je.sorted_eigen(cov)

    total_variance = sum(eigenvalues)
    variance_explained = [ev / total_variance for ev in eigenvalues] if total_variance > 0 else eigenvalues
    return {"eigenvalues": eigenvalues, "eigenvectors": eigenvectors,
            "variance_explained": variance_explained}


def pca_via_svd(x_centered: la.Matrix) -> Dict:
    """Principal Component Analysis via SVD of the centered data matrix
    directly, never forming the covariance matrix explicitly. The
    equivalence: for X_centered = U*Sigma*V^T,

        cov = X_centered^T * X_centered / (m-1) = V*Sigma^2*V^T / (m-1),

    which is exactly an eigendecomposition of cov with eigenvalues
    sigma_i^2/(m-1) and eigenvectors given by V's columns: the
    principal directions (loadings) are identical to PCA's, and each
    principal component's variance is sigma_i^2/(m-1). Because this
    route only ever forms X_centered (whose entries are at the data's
    own natural scale, not squared into covariance-matrix scale), it
    does not suffer the small-magnitude precision loss
    `pca_via_covariance` needs an explicit rescaling step to avoid.
    """
    m = len(x_centered)
    u, s, v = svd.singular_value_decompose(x_centered)
    eigenvalues = [si ** 2 / (m - 1) for si in s]
    total_variance = sum(eigenvalues)
    variance_explained = [ev / total_variance for ev in eigenvalues] if total_variance > 0 else eigenvalues
    return {"eigenvalues": eigenvalues, "eigenvectors": v, "variance_explained": variance_explained,
            "singular_values": s}


def compare_pca_methods(x: la.Matrix) -> Dict:
    """Run PCA three ways on the same (raw, not yet centered) data
    matrix: via covariance eigendecomposition with the rescaling fix,
    via covariance eigendecomposition without it, and via direct SVD,
    and report how far the naive (unrescaled) covariance route drifts
    from the other two, which should agree closely with each other.
    """
    x_centered = center_columns(x)
    rescaled = pca_via_covariance(x_centered, auto_rescale=True)
    naive = pca_via_covariance(x_centered, auto_rescale=False)
    via_svd = pca_via_svd(x_centered)

    def leading_vector_agreement(a, b):
        va = la.column(a["eigenvectors"], 0)
        vb = la.column(b["eigenvectors"], 0)
        if la.dot(va, vb) < 0:
            vb = [-x for x in vb]
        return la.vector_norm([x - y for x, y in zip(va, vb)])

    return {
        "rescaled": rescaled,
        "naive": naive,
        "svd": via_svd,
        "naive_vs_svd_leading_vector_gap": leading_vector_agreement(naive, via_svd),
        "rescaled_vs_svd_leading_vector_gap": leading_vector_agreement(rescaled, via_svd),
    }
