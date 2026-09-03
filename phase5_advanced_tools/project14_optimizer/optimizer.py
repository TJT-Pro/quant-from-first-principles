import csv
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_P12  = os.path.join(_HERE, "..", "project12_matrix_engine")
sys.path.insert(0, _P12)
import matrix_engine as mx


# ---------------------------------------------------------------------------
# Numerical gradient (reused from project 11 logic, self-contained here)
# ---------------------------------------------------------------------------

def _grad(f, x, h=1e-5):
    """Numerical gradient of f at x by central differences."""
    g = []
    for i in range(len(x)):
        xp = list(x); xp[i] += h
        xm = list(x); xm[i] -= h
        g.append((f(xp) - f(xm)) / (2.0 * h))
    return g


# ---------------------------------------------------------------------------
# Vanilla gradient descent
# ---------------------------------------------------------------------------

def gradient_descent(f, x0, learning_rate=0.01, max_iter=10000,
                     tolerance=1e-8, grad_fn=None):
    """Vanilla gradient descent: x ← x − lr·∇f(x).

    Parameters
    ----------
    f             : callable, f(x) -> float (minimise this)
    x0            : list of floats, starting point
    learning_rate : step size
    max_iter      : hard cap on iterations
    tolerance     : stop when ||∇f|| < tolerance
    grad_fn       : optional analytical gradient; uses numerical if None

    Returns a result dict with keys:
        x, f_val, iterations, converged, history (list of f values)
    """
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    x       = list(x0)
    history = [f(x)]
    gfn     = grad_fn if grad_fn is not None else lambda v: _grad(f, v)
    converged = False

    for it in range(max_iter):
        g   = gfn(x)
        gnorm = math.sqrt(sum(gi*gi for gi in g))
        if gnorm < tolerance:
            converged = True
            break
        x = [xi - learning_rate * gi for xi, gi in zip(x, g)]
        history.append(f(x))

    return {"x": x, "f_val": f(x), "iterations": len(history)-1,
            "converged": converged, "history": history}


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

def momentum_gd(f, x0, learning_rate=0.01, beta=0.9, max_iter=10000,
                tolerance=1e-8, grad_fn=None):
    """Gradient descent with heavy-ball momentum.

    v ← β·v + ∇f(x)
    x ← x − lr·v
    """
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1)")
    x       = list(x0)
    v       = [0.0] * len(x)
    history = [f(x)]
    gfn     = grad_fn if grad_fn is not None else lambda w: _grad(f, w)
    converged = False

    for _ in range(max_iter):
        g     = gfn(x)
        gnorm = math.sqrt(sum(gi*gi for gi in g))
        if gnorm < tolerance:
            converged = True
            break
        v = [beta * vi + gi for vi, gi in zip(v, g)]
        x = [xi - learning_rate * vi for xi, vi in zip(x, v)]
        history.append(f(x))

    return {"x": x, "f_val": f(x), "iterations": len(history)-1,
            "converged": converged, "history": history}


# ---------------------------------------------------------------------------
# RMSProp
# ---------------------------------------------------------------------------

def rmsprop(f, x0, learning_rate=0.01, rho=0.9, eps=1e-8,
            max_iter=10000, tolerance=1e-8, grad_fn=None):
    """RMSProp: adapts learning rate per coordinate using gradient variance.

    cache ← ρ·cache + (1−ρ)·∇²
    x     ← x − lr·∇ / (√cache + ε)
    """
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    x       = list(x0)
    cache   = [0.0] * len(x)
    history = [f(x)]
    gfn     = grad_fn if grad_fn is not None else lambda w: _grad(f, w)
    converged = False

    for _ in range(max_iter):
        g     = gfn(x)
        gnorm = math.sqrt(sum(gi*gi for gi in g))
        if gnorm < tolerance:
            converged = True
            break
        cache = [rho * ci + (1.0 - rho) * gi*gi for ci, gi in zip(cache, g)]
        x     = [xi - learning_rate * gi / (math.sqrt(ci) + eps)
                 for xi, gi, ci in zip(x, g, cache)]
        history.append(f(x))

    return {"x": x, "f_val": f(x), "iterations": len(history)-1,
            "converged": converged, "history": history}


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------

def adam(f, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8,
         max_iter=10000, tolerance=1e-8, grad_fn=None):
    """Adam: adaptive moment estimation.

    m ← β₁·m + (1−β₁)·∇           (first moment, exponential avg of grads)
    v ← β₂·v + (1−β₂)·∇²          (second moment, exponential avg of sq grads)
    m̂ = m / (1 − β₁ᵗ)             (bias correction)
    v̂ = v / (1 − β₂ᵗ)
    x ← x − lr·m̂ / (√v̂ + ε)

    Adam combines momentum (m) and RMSProp (v) with bias correction for
    the early steps when the moment estimates are initialised at zero.
    """
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("beta1 and beta2 must be in [0, 1)")
    x       = list(x0)
    m       = [0.0] * len(x)
    v       = [0.0] * len(x)
    history = [f(x)]
    gfn     = grad_fn if grad_fn is not None else lambda w: _grad(f, w)
    converged = False

    for t in range(1, max_iter + 1):
        g     = gfn(x)
        gnorm = math.sqrt(sum(gi*gi for gi in g))
        if gnorm < tolerance:
            converged = True
            break
        m = [beta1 * mi + (1.0 - beta1) * gi for mi, gi in zip(m, g)]
        v = [beta2 * vi + (1.0 - beta2) * gi*gi for vi, gi in zip(v, g)]
        m_hat = [mi / (1.0 - beta1**t) for mi in m]
        v_hat = [vi / (1.0 - beta2**t) for vi in v]
        x = [xi - learning_rate * mh / (math.sqrt(vh) + eps)
             for xi, mh, vh in zip(x, m_hat, v_hat)]
        history.append(f(x))

    return {"x": x, "f_val": f(x), "iterations": len(history)-1,
            "converged": converged, "history": history}


# ---------------------------------------------------------------------------
# Backtracking line search
# ---------------------------------------------------------------------------

def backtracking_line_search(f, x, direction, f0, grad, alpha=1.0,
                              rho=0.5, c=1e-4):
    """Armijo backtracking line search.

    Shrinks the step size alpha by factor rho until the Armijo condition:

        f(x + alpha·d) ≤ f(x) + c·alpha·∇f(x)ᵀd

    is satisfied.  Ensures sufficient decrease.

    Parameters
    ----------
    f         : callable
    x         : current point
    direction : search direction (usually −∇f)
    f0        : f(x) (pre-computed)
    grad      : ∇f(x) (pre-computed)
    alpha     : initial step size
    rho       : shrinkage factor
    c         : Armijo constant

    Returns the accepted step size.
    """
    slope = sum(gi * di for gi, di in zip(grad, direction))
    for _ in range(100):
        x_new = [xi + alpha * di for xi, di in zip(x, direction)]
        if f(x_new) <= f0 + c * alpha * slope:
            return alpha
        alpha *= rho
    return alpha


def gradient_descent_ls(f, x0, max_iter=1000, tolerance=1e-8, grad_fn=None):
    """Gradient descent with backtracking line search.

    Uses the Armijo condition to choose an adaptive step size at each
    iteration.  More robust than a fixed learning rate on ill-conditioned
    problems.
    """
    x       = list(x0)
    history = [f(x)]
    gfn     = grad_fn if grad_fn is not None else lambda w: _grad(f, w)
    converged = False

    for _ in range(max_iter):
        g     = gfn(x)
        gnorm = math.sqrt(sum(gi*gi for gi in g))
        if gnorm < tolerance:
            converged = True
            break
        d    = [-gi for gi in g]
        step = backtracking_line_search(f, x, d, f(x), g)
        x    = [xi + step * di for xi, di in zip(x, d)]
        history.append(f(x))

    return {"x": x, "f_val": f(x), "iterations": len(history)-1,
            "converged": converged, "history": history}


# ---------------------------------------------------------------------------
# Nelder-Mead (derivative-free)
# ---------------------------------------------------------------------------

def nelder_mead(f, x0, step=0.1, max_iter=5000, tolerance=1e-8):
    """Nelder-Mead simplex method for derivative-free minimisation.

    Maintains a simplex of n+1 vertices in n dimensions.  At each step:
        - Reflect the worst vertex through the centroid
        - Expand or contract depending on the reflected value
        - Shrink the whole simplex toward the best if needed

    Parameters
    ----------
    f        : callable, f(x) -> float
    x0       : list of floats, starting point
    step     : initial simplex size
    max_iter : maximum iterations
    tolerance: stop when simplex diameter < tolerance

    Returns the standard result dict.
    """
    n = len(x0)
    # Build initial simplex
    simplex = [list(x0)]
    for i in range(n):
        v = list(x0)
        v[i] += step
        simplex.append(v)
    fvals   = [f(v) for v in simplex]
    history = [min(fvals)]

    for _ in range(max_iter):
        # Sort
        order   = sorted(range(n + 1), key=lambda i: fvals[i])
        simplex = [simplex[i] for i in order]
        fvals   = [fvals[i]   for i in order]

        # Convergence: simplex diameter
        diam = max(math.sqrt(sum((simplex[i][j] - simplex[0][j])**2
                                  for j in range(n)))
                   for i in range(1, n + 1))
        if diam < tolerance:
            break

        # Centroid of all but worst
        centroid = [sum(simplex[i][j] for i in range(n)) / n
                    for j in range(n)]

        # Reflect
        xr    = [centroid[j] + (centroid[j] - simplex[n][j]) for j in range(n)]
        fr    = f(xr)

        if fvals[0] <= fr < fvals[n - 1]:
            simplex[n] = xr
            fvals[n]   = fr
        elif fr < fvals[0]:
            # Try expansion
            xe  = [centroid[j] + 2.0 * (centroid[j] - simplex[n][j]) for j in range(n)]
            fe  = f(xe)
            if fe < fr:
                simplex[n] = xe
                fvals[n]   = fe
            else:
                simplex[n] = xr
                fvals[n]   = fr
        else:
            # Contraction
            xc  = [centroid[j] + 0.5 * (simplex[n][j] - centroid[j]) for j in range(n)]
            fc  = f(xc)
            if fc < fvals[n]:
                simplex[n] = xc
                fvals[n]   = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j])
                                   for j in range(n)]
                    fvals[i]   = f(simplex[i])

        history.append(fvals[0])

    best = simplex[order[0] if fvals else 0]
    return {"x": best, "f_val": f(best), "iterations": len(history)-1,
            "converged": diam < tolerance, "history": history}


# ---------------------------------------------------------------------------
# Logistic regression loss
# ---------------------------------------------------------------------------

def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def logistic_loss(w, X, y, lam=0.0):
    """Binary cross-entropy loss with optional L2 regularisation.

    L(w) = -sum_i [y_i log σ(xᵢᵀw) + (1-yᵢ) log(1 - σ(xᵢᵀw))] + λ/2 ||w||²

    Parameters
    ----------
    w   : list of floats, model weights (length p)
    X   : n×p data matrix (list of lists)
    y   : list of n floats in {0, 1}
    lam : L2 regularisation strength
    """
    n   = len(X)
    eps = 1e-12
    loss = 0.0
    for i in range(n):
        z = sum(X[i][j] * w[j] for j in range(len(w)))
        p = _sigmoid(z)
        loss -= y[i] * math.log(p + eps) + (1.0 - y[i]) * math.log(1.0 - p + eps)
    loss /= n
    if lam > 0:
        loss += 0.5 * lam * sum(wj*wj for wj in w)
    return loss


def logistic_gradient(w, X, y, lam=0.0):
    """Gradient of the logistic loss with respect to w."""
    n, p = len(X), len(w)
    grad = [0.0] * p
    for i in range(n):
        z    = sum(X[i][j] * w[j] for j in range(p))
        err  = _sigmoid(z) - y[i]
        for j in range(p):
            grad[j] += err * X[i][j]
    grad = [g / n + lam * w[j] for j, g in enumerate(grad)]
    return grad


def fit_logistic(X, y, learning_rate=0.1, max_iter=1000, lam=0.0):
    """Fit a logistic regression model using Adam.

    Returns (weights, history) where weights is the fitted w vector.
    """
    p  = len(X[0])
    w0 = [0.0] * p
    result = adam(
        lambda w: logistic_loss(w, X, y, lam),
        w0,
        learning_rate=learning_rate,
        max_iter=max_iter,
        tolerance=1e-7,
        grad_fn=lambda w: logistic_gradient(w, X, y, lam),
    )
    return result["x"], result["history"]


def predict_logistic(w, X, threshold=0.5):
    """Predict binary labels for X given weights w."""
    labels = []
    for row in X:
        z = sum(row[j] * w[j] for j in range(len(w)))
        labels.append(1 if _sigmoid(z) >= threshold else 0)
    return labels


# ---------------------------------------------------------------------------
# Simplex LP solver (revised simplex, standard form)
# ---------------------------------------------------------------------------

def lp_simplex(c, A, b, max_iter=1000):
    """Solve the linear program min cᵀx s.t. Ax ≤ b, x ≥ 0.

    Converts to standard form by adding slack variables:
        min cᵀx  s.t. Ax + s = b,  x ≥ 0,  s ≥ 0

    Uses the revised simplex method with the full-tableau representation
    for clarity.  Starts from the all-slack basis (feasible when b ≥ 0).

    Parameters
    ----------
    c        : list of n floats (cost vector)
    A        : m×n constraint matrix (list of lists)
    b        : list of m floats, right-hand side (must all be ≥ 0)
    max_iter : maximum pivot steps

    Returns a dict with:
        x          : optimal primal solution (length n)
        obj        : optimal objective value
        status     : 'optimal', 'unbounded', or 'max_iter_reached'
        iterations : number of pivots
    """
    if any(bi < -1e-10 for bi in b):
        raise ValueError("all b_i must be ≥ 0 for the all-slack initial basis")
    m = len(b)
    n = len(c)

    # Build the full tableau: [A | I | b], objective row [c | 0 | 0]
    # Rows 0..m-1 are constraints, row m is the objective (negated for tableau form)
    tableau = []
    for i in range(m):
        row = list(A[i]) + [1.0 if j == i else 0.0 for j in range(m)] + [b[i]]
        tableau.append(row)
    # Objective row: reduced costs of structural variables, zeros for slacks
    obj_row = list(c) + [0.0] * m + [0.0]
    tableau.append(obj_row)

    total_cols = n + m + 1  # structural + slack + rhs
    rhs_col    = total_cols - 1
    basis      = list(range(n, n + m))  # initial basis: slack variables

    status = "max_iter_reached"
    for it in range(max_iter):
        # Find entering variable: most negative reduced cost in objective row
        obj = tableau[m]
        enter = -1
        min_rc = -1e-9
        for j in range(n + m):
            if obj[j] < min_rc:
                min_rc = obj[j]
                enter  = j
        if enter == -1:
            status = "optimal"
            break

        # Ratio test: find leaving variable
        leave = -1
        min_ratio = math.inf
        for i in range(m):
            if tableau[i][enter] > 1e-10:
                ratio = tableau[i][rhs_col] / tableau[i][enter]
                if ratio < min_ratio:
                    min_ratio = ratio
                    leave     = i
        if leave == -1:
            status = "unbounded"
            break

        # Pivot: make tableau[leave][enter] = 1, zero out rest of column
        pivot = tableau[leave][enter]
        tableau[leave] = [v / pivot for v in tableau[leave]]
        for i in range(m + 1):
            if i == leave:
                continue
            factor = tableau[i][enter]
            tableau[i] = [tableau[i][j] - factor * tableau[leave][j]
                           for j in range(total_cols)]
        basis[leave] = enter

    # Extract solution
    x = [0.0] * n
    for i, b_var in enumerate(basis):
        if b_var < n:
            x[b_var] = tableau[i][rhs_col]

    return {
        "x":          x,
        "obj":        -tableau[m][rhs_col],   # tableau stores -obj
        "status":     status,
        "iterations": it + 1,
    }


# ---------------------------------------------------------------------------
# LP applications
# ---------------------------------------------------------------------------

def portfolio_min_risk_lp(expected_returns, max_drawdown_budget,
                           min_return_target):
    """Find minimum-drawdown-budget portfolio meeting a return target.

    Formulated as:
        min  sum(d_i)
        s.t. sum(r_i * w_i) >= R_min
             w_i >= 0
             sum(w_i) = 1
             d_i >= -r_i * w_i  (linearised drawdown proxy)

    This is a simplified LP formulation for demonstration purposes.
    Returns (weights, objective).

    Parameters
    ----------
    expected_returns    : list of n asset expected returns
    max_drawdown_budget : not used directly (kept for API consistency)
    min_return_target   : minimum portfolio expected return
    """
    n = len(expected_returns)
    # Variables: [w_0,...,w_{n-1}, d_0,...,d_{n-1}]
    # min sum(d_i) = c = [0,...,0, 1,...,1]
    c = [0.0] * n + [1.0] * n

    # Constraints (Ax <= b form, all b >= 0):
    # 1. -sum(r_i * w_i) <= -R_min  (return requirement, flipped)
    # 2. -w_i + d_i <= 0  (d_i >= -r_i * w_i proxy: d_i >= w_i * max(0,-r_i))
    # 3. -w_i <= 0  (non-negativity of weights, redundant but explicit)
    # 4.  w_i <= 1  (box constraint)
    # 5.  sum(w_i) <= 1 + eps, -sum(w_i) <= -(1-eps) (sum-to-1 approx)

    rows = []
    bs   = []

    # Return constraint: -sum(r_i * w_i) <= -R_min
    row = [-r for r in expected_returns] + [0.0] * n
    rows.append(row)
    bs.append(-min_return_target)

    # d_i >= 0: already enforced by LP non-negativity
    # d_i >= -loss proxy: d_i + r_i * w_i >= 0 => -d_i - r_i*w_i <= 0
    for i in range(n):
        row = [0.0] * (2 * n)
        row[i]     = -expected_returns[i] if expected_returns[i] < 0 else 0.0
        row[n + i] = -1.0
        rows.append(row)
        bs.append(0.0)

    # Box: w_i <= 1
    for i in range(n):
        row = [0.0] * (2 * n)
        row[i] = 1.0
        rows.append(row)
        bs.append(1.0)

    # sum(w) <= 1
    rows.append([1.0] * n + [0.0] * n)
    bs.append(1.0)

    # -sum(w) <= -1 (sum >= 1)
    rows.append([-1.0] * n + [0.0] * n)
    bs.append(-1.0)

    try:
        result = lp_simplex(c, rows, bs)
    except ValueError as e:
        return None, str(e)

    if result["status"] != "optimal":
        return None, result["status"]

    weights = result["x"][:n]
    return weights, result["obj"]


def regression_l1_lp(X, y, lam=0.0):
    """L1 (least absolute deviations) regression via LP.

    Minimises sum_i |y_i - xᵢᵀw| + λ sum_j |w_j|.

    Reformulated as LP by introducing auxiliary variables:
        t_i >= y_i - xᵢᵀw   and   t_i >= -(y_i - xᵢᵀw)
        u_j >= w_j            and   u_j >= -w_j

    min sum(t_i) + λ sum(u_j)
    s.t. t_i - xᵢᵀw >= -y_i     → xᵢᵀw - t_i <= y_i
         t_i + xᵢᵀw >=  y_i     → -xᵢᵀw - t_i <= -y_i
         u_j - w_j >= 0          → w_j - u_j <= 0
         u_j + w_j >= 0          → -w_j - u_j <= 0

    Variables: [w_0..w_{p-1}, t_0..t_{n-1}, u_0..u_{p-1}]

    Parameters
    ----------
    X   : n×p design matrix
    y   : list of n floats (targets)
    lam : L1 regularisation on weights
    """
    n, p = len(X), len(X[0])
    # variables: w (p), t (n), u (p); LP needs x >= 0
    # Split w = w_pos - w_neg; all LP variables non-negative
    # This adds p more variables but keeps the LP standard
    # Simpler: just solve without the w non-negativity (use t,u only)
    # Here we use the full split for correctness.

    # Variables: [w+ (p), w- (p), t (n)]   (u absorbed into w+ + w-)
    # cost: [lam,...,lam, lam,...,lam, 1,...,1]
    nvars = 2 * p + n
    c     = [lam] * (2*p) + [1.0] * n

    rows = []
    bs   = []

    for i in range(n):
        xi = X[i]
        # t_i - xᵢᵀ(w+ - w-) >= -y_i  →  xᵢᵀw+ - xᵢᵀw- - t_i <= y_i
        row = [xi[j] for j in range(p)] + [-xi[j] for j in range(p)] + \
              [-1.0 if k == i else 0.0 for k in range(n)]
        # Need rhs y_i >= 0; if y_i < 0 flip: -row <= -y_i
        if y[i] >= 0:
            rows.append(row)
            bs.append(y[i])
        else:
            rows.append([-v for v in row])
            bs.append(-y[i])

        # -t_i - xᵢᵀ(w+ - w-) >= y_i  →  -xᵢᵀw+ + xᵢᵀw- - t_i <= -y_i
        row = [-xi[j] for j in range(p)] + [xi[j] for j in range(p)] + \
              [-1.0 if k == i else 0.0 for k in range(n)]
        rhs = -y[i]
        if rhs >= 0:
            rows.append(row)
            bs.append(rhs)
        else:
            rows.append([-v for v in row])
            bs.append(-rhs)

    result = lp_simplex(c, rows, bs)
    if result["status"] == "optimal":
        w_pos = result["x"][:p]
        w_neg = result["x"][p:2*p]
        w     = [wp - wm for wp, wm in zip(w_pos, w_neg)]
        return {"w": w, "obj": result["obj"], "status": "optimal"}
    return {"w": None, "obj": None, "status": result["status"]}


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def load_returns(path):
    """Load a two-column CSV and return the return column."""
    values = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row and row[1].strip():
                values.append(float(row[1]))
    return values
