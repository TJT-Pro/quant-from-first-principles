# Project 12 — Matrix Engine from Scratch

A complete linear algebra library built from Python lists and arithmetic.
No numpy, no scipy, no external linear algebra libraries.

---

## What is implemented

### Representation and constructors
Matrices are list-of-lists of floats.  `zeros(m,n)`, `identity(n)`,
`copy_matrix(A)`, `shape(A)`.

### Basic operations
`add`, `subtract`, `scalar_multiply`, `transpose`, `matmul` (O(n³)
schoolbook), `matvec`.

### Vector operations
`dot`, `vec_add`, `vec_sub`, `vec_scale`, `norm` (L1, L2, L∞),
`normalize`.

### Scalar properties
`trace` (sum of diagonal), `frobenius_norm` (√(sum of squares)).

### Determinant
`determinant` — cofactor expansion, exact, O(n!), practical for n ≤ 6.
`det_lu` — via LU decomposition, O(n³), numerically stable.

### LU decomposition  `PA = LU`
`lu_decompose` — Doolittle's algorithm with partial pivoting (largest
pivot in column).  Returns L (unit lower triangular), U (upper
triangular), P (permutation matrix), and n_swaps (row swap count used
for determinant sign).

`forward_substitution(L, b)` — solve Lx = b in O(n²).  
`back_substitution(U, b)` — solve Ux = b in O(n²).  
`solve(A, b)` — solves Ax = b via LU: PA = LU → Ly = Pb → Ux = y.

### Matrix inverse
`inverse(A)` — solves Ax = eᵢ for each basis vector, assembles columns.
O(n³).

### Eigenvalues
`eigenvalues_2x2` — closed-form via the quadratic formula.  Returns
real floats or complex (real, imag) tuples.

`power_iteration` — dominant eigenpair of a symmetric matrix.  Uses a
deterministic LCG for the initial vector.  Converges geometrically at
rate |λ₂/λ₁|.

### QR decomposition  `A = QR`
`qr_decompose` — modified Gram-Schmidt orthogonalisation.  More stable
than classical Gram-Schmidt because it re-orthogonalises each column
against already-computed basis vectors rather than against the
originals.  Returns Q (m×n, orthonormal columns) and R (n×n, upper
triangular).

### Least squares
`least_squares(A, b)` — minimises ||Ax − b||₂ via QR decomposition.
Solves the reduced system Rx = Qᵀb by back substitution.

`residual_norm(A, b, x)` — ||Ax − b||₂.

### Condition number
`condition_number(A)` — approximates κ₂(A) = σ_max/σ_min via power
iteration on AᵀA.  Uses deflation to estimate the second-largest
eigenvalue.

---

## Key formulas

### LU solve
    PA = LU,  Pb = c,  Ly = c,  Ux = y  →  x = A⁻¹b

### Least squares via QR
    A = QR  →  Qᵀb = Rx  →  x = R⁻¹Qᵀb  (back substitution)

### 2×2 characteristic polynomial
    λ² − tr(A)λ + det(A) = 0
    λ = [tr(A) ± √(tr(A)² − 4 det(A))] / 2

### Condition number
    κ(A) = σ_max(A) / σ_min(A) = √(λ_max(AᵀA) / λ_min(AᵀA))

---

## Limitations

The cofactor-expansion determinant is O(n!) and only practical for
n ≤ 6.  Use `det_lu` for anything larger.

Power iteration finds only the dominant eigenpair.  It converges slowly
when |λ₂/λ₁| is close to 1 (nearly repeated eigenvalues).

The condition number approximation uses deflation to estimate the
smallest eigenvalue of AᵀA; for nearly rank-deficient matrices the
deflation can lose accuracy.

---

## Running the tests

```
python3 -m pytest test_matrix_engine.py -v
```

71 tests covering constructors, arithmetic, matmul, vector ops, trace,
Frobenius norm, determinant (cofactor and LU), LU decomposition (PA=LU,
L unit lower, U upper), linear solve, matrix inverse, eigenvalues
(2×2 closed form and power iteration), QR decomposition, least squares,
and condition number.

## Oracle verification

```
python3 cli.py verify
```

21 checks: matrix operations against hand-computed values, determinant
by both methods, LU factorisation identity PA=LU, linear solve residual,
A·A⁻¹=I, eigenvalue sum/product relations, power iteration accuracy,
QR factorisation identity A=QR and orthonormality QᵀQ=I, least-squares
normal equations, and condition numbers of the identity and a diagonal
matrix.

## CLI examples

```bash
python3 cli.py matmul  --A "[[1,2],[3,4]]" --B "[[5,6],[7,8]]"
python3 cli.py solve   --A "[[2,1],[5,7]]" --b "[11,13]"
python3 cli.py inverse --A "[[4,7],[2,6]]"
python3 cli.py det     --A "[[1,2],[3,4]]"
python3 cli.py lu      --A "[[2,1],[1,3]]"
python3 cli.py qr      --A "[[1,2],[3,4],[5,6]]"
python3 cli.py lstsq   --A "[[1,0],[1,1],[1,2]]" --b "[1,2,4]"
python3 cli.py eigen   --A "[[2,1],[1,2]]"
python3 cli.py power   --A "[[4,1],[1,3]]"
python3 cli.py cond    --A "[[100,0],[0,1]]"
```
