# Project 14 — Gradient Descent Optimizer & LP Solver

Unconstrained optimisation algorithms and a linear programming simplex
solver, built from scratch.  No numpy, no scipy, no cvxpy.

---

## Optimizers implemented

### Vanilla gradient descent
`gradient_descent(f, x0, learning_rate, max_iter, tolerance, grad_fn)`

The canonical first-order method: x ← x − η·∇f(x).  Accepts an
optional analytical gradient; falls back to central-difference numerical
gradient.  Returns full history of f values.

### Momentum (heavy-ball)
`momentum_gd(f, x0, learning_rate, beta, ...)`

Accumulates a velocity vector:
    v ← β·v + ∇f(x)
    x ← x − η·v

The momentum term damps oscillations on ill-conditioned objectives and
accelerates convergence in consistent directions.

### RMSProp
`rmsprop(f, x0, learning_rate, rho, eps, ...)`

Adapts the per-coordinate learning rate by dividing by the root mean
square of recent gradients:
    cache ← ρ·cache + (1−ρ)·∇²
    x     ← x − η·∇ / (√cache + ε)

Useful when gradient magnitudes vary greatly across coordinates.

### Adam (Adaptive Moment Estimation)
`adam(f, x0, learning_rate, beta1, beta2, eps, ...)`

Combines momentum (first moment) with RMSProp (second moment), with
bias correction for the cold-start (t=0) period:
    m ← β₁·m + (1−β₁)·∇
    v ← β₂·v + (1−β₂)·∇²
    x ← x − η · m̂ / (√v̂ + ε)

The workhorse of modern deep learning.  Converges reliably on the
Rosenbrock banana function from a cold start.

### Backtracking line search
`backtracking_line_search(f, x, direction, f0, grad, alpha, rho, c)`

Armijo condition: shrinks the initial step size by factor ρ until:
    f(x + α·d) ≤ f(x) + c·α·∇f(x)ᵀd

Used in `gradient_descent_ls` (GD with adaptive step size) which is
more robust than a fixed learning rate on steep or ill-scaled problems.

### Nelder-Mead (derivative-free simplex)
`nelder_mead(f, x0, step, max_iter, tolerance)`

Maintains n+1 vertices in n-dimensional space and iteratively reflects,
expands, contracts, and shrinks the simplex toward the minimum.  No
gradient required — works on non-smooth or black-box functions.

---

## LP solver — revised simplex

`lp_simplex(c, A, b, max_iter)` — solves the standard form LP:

    min cᵀx   subject to  Ax ≤ b,  x ≥ 0

Converts to equality form by adding slack variables, starts from the
all-slack feasible basis (requires all b_i ≥ 0), and pivots using the
most-negative reduced cost entering rule and minimum ratio leaving rule.

Returns x (primal solution), obj (optimal value), status
('optimal', 'unbounded', or 'max_iter_reached'), and pivot count.

**Limitation:** requires all b_i ≥ 0.  For constraints with negative
right-hand sides, Phase I (two-phase) or Big-M methods are needed.

---

## Applications

### Logistic regression
`logistic_loss`, `logistic_gradient`, `fit_logistic`, `predict_logistic`

Binary cross-entropy loss with optional L2 regularisation, fitted by
Adam.  Demonstrated on EUR/USD daily returns (predict positive vs
negative return from prior return).

### Portfolio LP
`portfolio_min_risk_lp(expected_returns, budget, min_return)`

Minimum-risk portfolio subject to a return target, weight bounds, and
budget constraint, solved via the simplex method.

### L1 regression LP
`regression_l1_lp(X, y, lam)`

Least absolute deviations regression reformulated as an LP using the
w+ / w− variable split.  **Limitation:** the constraint matrix
includes rows with potentially negative right-hand side (b < 0) when
y values are positive.  The row-flip workaround is not always
numerically correct; Phase I simplex is the proper fix.

---

## Key formulas

### Adam bias correction
    m̂ = m / (1 − β₁ᵗ),   v̂ = v / (1 − β₂ᵗ)

### Armijo sufficient decrease condition
    f(x + αd) ≤ f(x) + c·α·∇fᵀd

### LP optimality (reduced costs)
    x is optimal ⟺ all reduced costs c̄_j = c_j − cᵀ_B B⁻¹ A_j ≥ 0

---

## Running the tests

```
python3 -m pytest test_optimizer.py -v
```

36 tests covering gradient descent convergence and accuracy on quadratic
and shifted objectives, momentum, RMSProp, Adam (including Rosenbrock),
backtracking line search, Nelder-Mead on quadratic and Rosenbrock,
LP simplex (standard form, constraint satisfaction, non-negativity,
unbounded detection, negative-b error), logistic regression loss,
analytical vs numerical gradient, fit accuracy on separable data, and
L1 regression LP mechanics.

## Oracle verification

```
python3 cli.py verify
```

23 checks: GD convergence and accuracy (origin and shifted minimum),
history monotonicity, momentum convergence, RMSProp convergence,
Adam convergence, line-search convergence and accuracy, Nelder-Mead
accuracy on both coordinates, LP optimality and objective value, LP
constraint satisfaction and non-negativity, logistic loss at zero
weights, logistic gradient vs numerical gradient, logistic classification
accuracy, and training loss decrease.

## CLI examples

```bash
# Gradient descent
python3 cli.py gd --fn "(x0-3)**2+(x1+2)**2" --x0 0,0 --lr 0.1

# Adam on Rosenbrock
python3 cli.py adam --fn "(1-x0)**2+100*(x1-x0**2)**2" --x0 0,0 --lr 0.01 --iters 20000

# Nelder-Mead (derivative-free)
python3 cli.py neldermead --fn "(x0-1)**2+(x1-2)**2" --x0 0,0

# LP: max 5x+4y s.t. 6x+4y<=24, x+2y<=6
python3 cli.py lp --c "-5,-4" --A "[[6,4],[1,2]]" --b "24,6"

# Logistic regression on EUR/USD
python3 cli.py logistic --data data/DEXUSEU_returns.csv
```
