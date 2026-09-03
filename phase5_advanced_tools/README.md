# Phase 5: Advanced Tools

The final phase builds the core machinery of linear algebra, dimensionality
reduction, and numerical optimisation from scratch.  The three projects form
a chain: the matrix engine (12) is imported by PCA (13), and the LP solver
and optimizers (14) use linear algebra concepts from project 12.

| Project | Directory | Tests | Status |
|---------|-----------|-------|--------|
| 12 - Matrix Engine | `project12_matrix_engine/` | 71 | done |
| 13 - PCA from Scratch | `project13_pca/` | 42 | done |
| 14 - Gradient Descent & LP Solver | `project14_optimizer/` | 36 | done |

---

**Project 12** implements the complete linear algebra toolkit: matrix
construction and arithmetic, matrix multiply and matvec, vector norms,
LU decomposition with partial pivoting, forward and backward substitution,
linear solve (Ax=b), matrix inverse via LU, eigenvalues of 2×2 matrices
by the quadratic formula, power iteration for the dominant eigenpair
of symmetric matrices, QR decomposition via modified Gram-Schmidt,
least squares solve, and condition number approximation.

**Project 13** implements PCA from its mathematical definition: centring
and standardisation, the sample covariance matrix (XᵀX/(n-1)), the
Jacobi eigenvalue algorithm for exact eigenpairs of real symmetric
matrices via iterated Givens rotations, full PCA pipeline with explained
variance ratios, reconstruction from k components, whitening (unit
variance in component space), minimum-components-for-variance threshold,
and a lag matrix builder for time series analysis.  Applied to the
EUR/USD daily return series.

**Project 14** implements unconstrained optimisation (vanilla gradient
descent, heavy-ball momentum, RMSProp, Adam with bias correction,
backtracking line search, and derivative-free Nelder-Mead), the revised
simplex method for linear programming (min cᵀx s.t. Ax≤b, x≥0), and
applications: binary logistic regression fitted by Adam, a minimum-risk
portfolio LP, and L1 regression reformulated as an LP.
