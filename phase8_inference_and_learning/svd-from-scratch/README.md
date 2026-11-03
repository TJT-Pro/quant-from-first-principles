# svd-from-scratch

Singular Value Decomposition built from scratch on top of a from-
scratch symmetric eigenvalue solver (the Jacobi eigenvalue algorithm):
low-rank approximation, the Moore-Penrose pseudoinverse, rank
estimation, condition numbers, and a direct investigation of why PCA
and SVD are mathematically the same computation but not always the
same numerically safe one, applied to real EUR/USD volatility data.

This repository is independent of the sibling repositories and reuses
none of their code, though the application connects to the
volatility-clustering theme those repositories established.

## Project structure

```
svd-from-scratch/
    README.md
    data/DEXUSEU_returns.csv
    linear_algebra.py    basic matrix operations
    jacobi_eigen.py       Jacobi eigenvalue algorithm for symmetric matrices
    svd.py                 SVD, low-rank approximation, pseudoinverse, rank, condition number
    fx_application.py     PCA vs SVD on real EUR/USD volatility term structure
    test_*.py
    cli.py
```

## Status

60 tests, 10/10 verify checks, all passing.

## Setup

Python 3.11+. All decomposition code uses only the standard library.
`cli.py verify` uses numpy as an independent oracle.

```bash
pip install --break-system-packages numpy
```

## Running tests

```bash
python3 -m pytest test_linear_algebra.py test_jacobi_eigen.py test_svd.py test_fx_application.py -v
```

## Running the oracle

```bash
python3 cli.py verify
```

## The data

`data/DEXUSEU_returns.csv`: daily EUR/USD exchange rates from the
Federal Reserve (FRED series DEXUSEU), 1999-01-04 to 2026-08-21
(6,931 price levels, 6,930 daily returns), the same file used in the
sibling repositories.

## Mathematical background

### Why singular values are the square roots of A^T A's eigenvalues

Substituting A = U*Sigma*V^T (with U, V orthogonal) into A^T*A gives

    A^T*A = V*Sigma^T*U^T*U*Sigma*V^T = V*Sigma^2*V^T,

using U^T*U=I. This is exactly an eigendecomposition of the symmetric
matrix A^T*A, with eigenvalues sigma_i^2 and eigenvectors given by V's
columns. A^T*A is always positive semi-definite
(x^T*A^T*A*x = ||Ax||^2 >= 0 for every x), so every eigenvalue is
non-negative and every singular value sigma_i = sqrt(lambda_i) is a
well-defined real number. Given V from this eigendecomposition, each
column of U for a nonzero singular value follows directly from
A*v_i = sigma_i*u_i; for a numerically zero singular value, no such
formula determines u_i, so those columns are instead completed via
Gram-Schmidt against the columns already found, finishing an
orthonormal basis of R^m without pretending a specific direction was
recovered from data that does not determine one.

### The Jacobi eigenvalue algorithm, and a real sign-convention bug

A symmetric matrix can be diagonalised by a sequence of plane
rotations, each zeroing one off-diagonal pair A_pq. A genuine bug was
found and fixed during development here: the rotation angle formula
and the row/column update formulas must agree on which sign
convention the rotation matrix R uses (which of R_pq, R_qp carries the
minus sign). An earlier version paired the textbook
theta = 0.5*atan2(2*A_pq, A_pp-A_qq) with row/column updates that
actually implemented the other convention. The bug was completely
invisible from two of the three properties a Jacobi eigensolver should
have: the eigenvalues on the diagonal converged correctly (matching
the trace exactly), and the accumulated V was still perfectly
orthogonal (a valid product of genuine rotations, regardless of their
angles' sign). It only showed up when checking A*v_i = lambda_i*v_i
directly for individual eigenpairs, or equivalently V^T*A*V against
the diagonal eigenvalue matrix, which is why the test suite checks
exactly that (`test_jacobi_eigen.py`), not just eigenvalue correctness
or V's orthogonality in isolation. The fix flips the angle's sign,
theta = -0.5*atan2(2*A_pq, A_pp-A_qq), verified directly against the
reference computation R^T*A*R via ordinary matrix multiplication
before being trusted again in the general algorithm.

### Low-rank approximation and the Eckart-Young-Mirsky theorem

Truncating the SVD to its k largest singular triples,
A_k = sum_{i=1}^k sigma_i*u_i*v_i^T, gives the best possible rank-k
approximation to A in both the Frobenius and spectral norms
simultaneously (Eckart-Young-Mirsky), with reconstruction error
exactly sqrt(sum_{i>k} sigma_i^2). This is checked directly, not
merely cited: at every rank k from 0 to full rank, the measured
Frobenius error matches this formula to six decimal places, and a
deliberately different (non-optimal) rank-2 approximation built from
singular triples 2 and 3 instead of 1 and 2 is confirmed to have
strictly larger error.

### The Moore-Penrose pseudoinverse

A^+ = V*Sigma^+*U^T, where Sigma^+ replaces every nonzero singular
value with its reciprocal and leaves numerically zero singular values
at zero rather than dividing by them. This is the unique matrix
satisfying A*A^+*A=A and A^+*A*A^+=A^+ (both checked directly), and
for a rank-deficient A it specifically gives the minimum-norm solution
to a least-squares problem, which is exactly why zero singular values
map to zero in Sigma^+ rather than to any other finite value.

### Rank and condition number, and the cost of forming A^T A explicitly

Numerical rank counts singular values above a relative threshold (a
fixed absolute cutoff would not respect that A and 1000*A should be
judged to have the same rank). The condition number,
sigma_max/sigma_min over the numerically nonzero singular values,
measures how much A can amplify relative error. A second real finding
surfaced testing this directly: a matrix with two columns differing by
only 1e-8 was expected to show a large but finite condition number
(around 1e8); instead, the smallest singular value came back as
exactly 0.0. The reason, confirmed by inspecting A^T*A directly: the
true eigenvalue gap this construction implies works out to about
1.5e-15 against diagonal entries of magnitude ~15, right at the edge
of double-precision resolution. Forming A^T*A explicitly squares the
original problem's condition number, and this is precisely why
professional numerical libraries compute the SVD via direct
bidiagonalisation of A (Golub-Kahan) rather than by eigendecomposing
A^T*A the way this project does for simplicity; the test suite
documents the actual observed behaviour (the matrix is detected as
rank-deficient, not merely ill-conditioned) rather than asserting the
naively-expected one.

## Application: PCA vs SVD, and a numerical-precision finding that turned out to matter more than the equivalence itself

**Setup.** A data matrix of daily realised volatility (trailing
standard deviation of returns) at four horizons, 5, 10, 20, and 60
days, giving a 6,870 x 4 matrix with strong but imperfect
cross-sectional correlation (0.63-0.89 between every pair of
horizons): exactly the kind of structure PCA and SVD summarise, unlike
a single univariate series with no cross-sectional structure for
either method to find.

**The headline result, and it holds up**: the first principal
component explains 84.05% of the joint variance across all four
volatility horizons, a single dominant "how volatile is the market
right now" factor, consistent with everything the sibling repositories
established about EUR/USD volatility clustering, now demonstrated as a
genuinely dominant cross-sectional factor rather than only a
time-domain persistence property.

**But getting a trustworthy number required catching a real numerical
trap first.** PCA computed via explicit eigendecomposition of the
4x4 sample covariance matrix, and PCA computed via SVD of the centered
data matrix directly, are mathematically identical
(cov = V*Sigma^2*V^T/(m-1)). On this real data, they agreed on
variance explained to four decimal places, but the naive covariance
route's leading eigenvector differed from SVD's by a Euclidean norm of
0.0375, a small but real and repeatable discrepancy, not noise (it
reproduces exactly with the same random seed every run). Investigating
why, on a controlled synthetic example with a known true loading
vector at the same realistic financial-data scale (~1e-5), made the
mechanism unambiguous: naive covariance eigendecomposition returned
the nonsense vector [1,0,0,0], nowhere near the true loadings, because
the Jacobi algorithm's rotation-angle formula loses real precision
operating on covariance entries at O(1e-10) scale. Rescaling the
covariance matrix by its own largest diagonal entry before decomposing
it (a similarity transform that changes no eigenvector and rescales
eigenvalues by a known, exactly-invertible constant) fixed this
completely, recovering the true loadings to within 0.1%; SVD of the
centered data matrix needed no such fix, since it never forms the
small-magnitude covariance matrix at all. This is not a hypothetical
concern: the real EUR/USD data showed the same failure mode, just less
severely (a 3.75% leading-vector gap rather than complete failure,
since 6,870 real observations partially average out the precision
loss that only 2,000 synthetic points could not). `pca_via_covariance`
now rescales by default for exactly this reason, and the un-rescaled
behaviour is kept available specifically so the failure mode stays
directly demonstrable rather than silently patched away.

## Limitations

- `singular_value_decompose` assumes m >= n (more rows than columns,
  the standard "tall" data-matrix shape); handling m < n requires
  transposing A first and swapping the roles of U and V in the
  result, not implemented as an automatic code path here.
- The Jacobi eigenvalue algorithm is O(n^3) per sweep and typically
  needs several sweeps to converge; it is well suited to the small
  matrices (n <= 10 or so) this project actually decomposes, but
  would not scale to the hundreds-or-thousands-of-dimensions problems
  professional PCA implementations routinely handle, where an
  iterative method exploiting sparsity or a randomised SVD algorithm
  would be used instead.
- The "form A^T*A and eigendecompose" approach to SVD used throughout
  this project, even with the rescaling fix applied to PCA, remains
  fundamentally more sensitive to ill-conditioning than a genuine
  direct SVD algorithm (Golub-Kahan bidiagonalisation followed by an
  implicit-shift QR-type iteration) would be, since squaring A's
  condition number by forming A^T*A is a property of the algorithm's
  structure, not just of the eigenvalue solver's numerical care; the
  rescaling fix addresses the specific failure mode found by testing
  here, not the underlying sensitivity in general.

## CLI examples

```bash
# Compute the SVD of a matrix
python3 cli.py decompose --matrix "1,2;3,4;5,6"

# Best rank-k approximation
python3 cli.py lowrank --matrix "1,2,3;4,5,6;7,8,10;2,1,3" --k 1

# Moore-Penrose pseudoinverse
python3 cli.py pseudoinverse --matrix "1,2;3,4;5,6"

# PCA via covariance vs PCA via SVD, on real EUR/USD volatility data
python3 cli.py pcavssvd --windows "5,10,20,60"

# Cross-check every decomposition against numpy oracles and known theorems
python3 cli.py verify
```
