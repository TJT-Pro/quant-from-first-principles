import math


# ---------------------------------------------------------------------------
# Step sizes
# ---------------------------------------------------------------------------

_H_FIRST  = 1e-4    # step for first-order central differences
_H_SECOND = 1e-3    # step for second-order (Hessian) central differences


def _h_scale(x, h_base):
    """Scale h by the magnitude of x to avoid tiny relative steps."""
    return h_base * max(1.0, abs(x))


# ---------------------------------------------------------------------------
# Partial derivatives
# ---------------------------------------------------------------------------

def partial_derivative(f, point, i, h=None):
    """Partial derivative of f with respect to the i-th variable at point.

    Uses the central difference formula with Richardson extrapolation:

        df/dx_i ≈ [f(x + h*e_i) - f(x - h*e_i)] / (2h)

    Richardson extrapolation cancels the leading O(h²) error:

        df/dx_i ≈ [4·D(h/2) - D(h)] / 3

    giving an O(h⁴) estimate.

    Parameters
    ----------
    f     : callable, f(point) -> float, where point is a list of floats
    point : list of floats, the evaluation point
    i     : index of the variable to differentiate (0-based)
    h     : base step size; defaults to 1e-4 scaled by |point[i]|
    """
    if i < 0 or i >= len(point):
        raise ValueError(
            "index {} out of range for point of length {}".format(i, len(point))
        )
    h_use = h if h is not None else _h_scale(point[i], _H_FIRST)

    def _diff(step):
        p_plus  = list(point)
        p_minus = list(point)
        p_plus[i]  += step
        p_minus[i] -= step
        return (f(p_plus) - f(p_minus)) / (2.0 * step)

    d_h  = _diff(h_use)
    d_h2 = _diff(h_use / 2.0)
    return (4.0 * d_h2 - d_h) / 3.0


def gradient(f, point, h=None):
    """Gradient vector of f at point.

    Returns a list [df/dx_0, df/dx_1, ..., df/dx_{n-1}] computed by
    calling partial_derivative for each coordinate.

    The gradient points in the direction of steepest increase of f.
    Its magnitude is the maximum rate of change at that point.

    Parameters
    ----------
    f     : callable, f(point) -> float
    point : list of floats
    h     : base step size; if None each component uses its own scale
    """
    return [partial_derivative(f, point, i, h) for i in range(len(point))]


def gradient_magnitude(f, point, h=None):
    """Magnitude of the gradient of f at point: ||∇f(point)||."""
    g = gradient(f, point, h)
    return math.sqrt(sum(gi * gi for gi in g))


def directional_derivative(f, point, direction, h=None):
    """Derivative of f at point in the given direction.

    The directional derivative is the dot product of the gradient with
    the unit vector in the given direction:

        D_v f(x) = ∇f(x) · v̂

    where v̂ = direction / ||direction||.

    Parameters
    ----------
    f         : callable
    point     : list of floats
    direction : list of floats, the direction vector (need not be unit)
    """
    if len(direction) != len(point):
        raise ValueError("direction and point must have the same length")
    norm = math.sqrt(sum(d * d for d in direction))
    if norm == 0.0:
        raise ValueError("direction vector must be non-zero")
    unit = [d / norm for d in direction]
    g    = gradient(f, point, h)
    return sum(gi * ui for gi, ui in zip(g, unit))


# ---------------------------------------------------------------------------
# Second-order quantities
# ---------------------------------------------------------------------------

def second_partial(f, point, i, j, h=None):
    """Mixed second partial derivative d²f / (dx_i dx_j) at point.

    For i == j (pure second derivative):
        d²f/dx_i² ≈ [f(x+h·eᵢ) - 2f(x) + f(x-h·eᵢ)] / h²

    For i != j (mixed):
        d²f/(dx_i dx_j) ≈ [f(x+h·eᵢ+h·eⱼ) - f(x+h·eᵢ-h·eⱼ)
                           - f(x-h·eᵢ+h·eⱼ) + f(x-h·eᵢ-h·eⱼ)] / (4h²)

    Both use Richardson extrapolation for improved accuracy.

    Parameters
    ----------
    f     : callable
    point : list of floats
    i, j  : indices of the two variables
    h     : step size; defaults to 1e-3 scaled by point magnitude
    """
    n = len(point)
    if not (0 <= i < n and 0 <= j < n):
        raise ValueError("indices out of range")
    h_use = h if h is not None else _h_scale(
        max(abs(point[i]), abs(point[j]), 1.0), _H_SECOND
    )

    def _pure(step):
        p_plus  = list(point)
        p_minus = list(point)
        p_plus[i]  += step
        p_minus[i] -= step
        return (f(p_plus) - 2.0 * f(point) + f(p_minus)) / (step * step)

    def _mixed(step):
        def _p(di, dj):
            p = list(point)
            p[i] += di * step
            p[j] += dj * step
            return f(p)
        return (_p(1, 1) - _p(1, -1) - _p(-1, 1) + _p(-1, -1)) / (4.0 * step * step)

    if i == j:
        d_h  = _pure(h_use)
        d_h2 = _pure(h_use / 2.0)
    else:
        d_h  = _mixed(h_use)
        d_h2 = _mixed(h_use / 2.0)

    return (4.0 * d_h2 - d_h) / 3.0


def hessian(f, point, h=None):
    """Hessian matrix of f at point.

    Returns an n×n list-of-lists H where H[i][j] = d²f/(dx_i dx_j).

    The Hessian encodes the local curvature of f.  Its eigenvalues
    determine whether a critical point is a minimum (all positive),
    maximum (all negative), or saddle (mixed signs).

    Parameters
    ----------
    f     : callable
    point : list of floats
    h     : step size
    """
    n = len(point)
    H = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            val = second_partial(f, point, i, j, h)
            H[i][j] = val
            H[j][i] = val
    return H


def laplacian(f, point, h=None):
    """Laplacian of f at point: ∇²f = sum of d²f/dx_i².

    The Laplacian measures how the function value at a point compares
    to the average in a small neighbourhood.  Positive means the point
    is a local minimum of the function; negative means a local maximum.
    Zero is characteristic of harmonic functions.

    Parameters
    ----------
    f     : callable
    point : list of floats
    h     : step size
    """
    return sum(second_partial(f, point, i, i, h) for i in range(len(point)))


# ---------------------------------------------------------------------------
# Critical point classification
# ---------------------------------------------------------------------------

def classify_critical_2d(f, point, h=None):
    """Classify a critical point of a 2D function via the Hessian determinant.

    For f: R² → R at a critical point (where ∇f ≈ 0):

        D = f_xx * f_yy - f_xy²  (determinant of 2×2 Hessian)

        D > 0, f_xx > 0  →  local minimum
        D > 0, f_xx < 0  →  local maximum
        D < 0            →  saddle point
        D ≈ 0            →  inconclusive (degenerate)

    Returns a dict with keys: fxx, fyy, fxy, det, classification, grad_norm.

    Parameters
    ----------
    f     : callable, f([x, y]) -> float
    point : [x, y]
    h     : step size
    """
    if len(point) != 2:
        raise ValueError("classify_critical_2d requires a 2D point")
    H       = hessian(f, point, h)
    fxx     = H[0][0]
    fyy     = H[1][1]
    fxy     = H[0][1]
    det     = fxx * fyy - fxy * fxy
    g_norm  = gradient_magnitude(f, point)

    if abs(det) < 1e-8:
        classification = "inconclusive"
    elif det < 0:
        classification = "saddle"
    elif fxx > 0:
        classification = "minimum"
    else:
        classification = "maximum"

    return {
        "fxx":            fxx,
        "fyy":            fyy,
        "fxy":            fxy,
        "det":            det,
        "classification": classification,
        "grad_norm":      g_norm,
    }


# ---------------------------------------------------------------------------
# Gradient descent with trajectory
# ---------------------------------------------------------------------------

def gradient_descent_2d(f, x0, y0, learning_rate=0.01,
                         max_iterations=1000, tolerance=1e-8):
    """Gradient descent on f: R² → R starting from (x0, y0).

    At each step:
        x_{k+1} = x_k - lr * df/dx(x_k, y_k)
        y_{k+1} = y_k - lr * df/dy(x_k, y_k)

    Stops when ||∇f|| < tolerance or max_iterations is reached.

    Returns a dict with:
        x, y          : final position
        f_val         : f at the final position
        iterations    : number of steps taken
        converged     : True if tolerance was met
        trajectory    : list of (x, y, f_val) for every step
        grad_norms    : list of ||∇f|| at each step

    Parameters
    ----------
    f              : callable, f([x, y]) -> float
    x0, y0         : starting position
    learning_rate  : step size multiplier
    max_iterations : hard cap on iterations
    tolerance      : stop when ||∇f|| < tolerance
    """
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    x, y      = float(x0), float(y0)
    trajectory  = [(x, y, f([x, y]))]
    grad_norms  = []
    converged   = False

    for _ in range(max_iterations):
        g      = gradient(f, [x, y])
        gnorm  = math.sqrt(g[0]*g[0] + g[1]*g[1])
        grad_norms.append(gnorm)
        if gnorm < tolerance:
            converged = True
            break
        x -= learning_rate * g[0]
        y -= learning_rate * g[1]
        trajectory.append((x, y, f([x, y])))

    return {
        "x":          x,
        "y":          y,
        "f_val":      f([x, y]),
        "iterations": len(trajectory) - 1,
        "converged":  converged,
        "trajectory": trajectory,
        "grad_norms": grad_norms,
    }


def compare_learning_rates(f, x0, y0, rates, max_iterations=500, tolerance=1e-8):
    """Run gradient descent with several learning rates and compare results.

    Returns a list of dicts, one per rate, each containing the result
    of gradient_descent_2d plus 'learning_rate' and 'final_grad_norm'.

    Parameters
    ----------
    f              : callable
    x0, y0         : starting position
    rates          : list of learning rates to try
    max_iterations : passed to gradient_descent_2d
    tolerance      : passed to gradient_descent_2d
    """
    results = []
    for lr in rates:
        res = gradient_descent_2d(f, x0, y0,
                                   learning_rate=lr,
                                   max_iterations=max_iterations,
                                   tolerance=tolerance)
        res["learning_rate"]    = lr
        res["final_grad_norm"]  = gradient_magnitude(f, [res["x"], res["y"]])
        results.append(res)
    return results


# ---------------------------------------------------------------------------
# Gradient field on a grid
# ---------------------------------------------------------------------------

def gradient_field_grid(f, x_range, y_range, nx=20, ny=20):
    """Evaluate the gradient of f on a 2D grid.

    Returns a dict with:
        xs, ys        : 1D lists of grid coordinates
        gx, gy        : 2D lists (ny × nx) of x and y gradient components
        magnitudes    : 2D list (ny × nx) of ||∇f||

    Parameters
    ----------
    f       : callable, f([x, y]) -> float
    x_range : (x_min, x_max)
    y_range : (y_min, y_max)
    nx, ny  : number of grid points in each dimension
    """
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must be at least 2")
    x_min, x_max = x_range
    y_min, y_max = y_range
    xs = [x_min + i * (x_max - x_min) / (nx - 1) for i in range(nx)]
    ys = [y_min + j * (y_max - y_min) / (ny - 1) for j in range(ny)]

    gx   = [[0.0] * nx for _ in range(ny)]
    gy   = [[0.0] * nx for _ in range(ny)]
    mags = [[0.0] * nx for _ in range(ny)]

    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            g = gradient(f, [x, y])
            gx[j][i]   = g[0]
            gy[j][i]   = g[1]
            mags[j][i] = math.sqrt(g[0]*g[0] + g[1]*g[1])

    return {"xs": xs, "ys": ys, "gx": gx, "gy": gy, "magnitudes": mags}


def function_grid(f, x_range, y_range, nx=50, ny=50):
    """Evaluate f on a 2D grid.

    Returns a dict with xs, ys (1D coordinate lists) and
    values (2D ny×nx list of f(x, y)).
    """
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must be at least 2")
    x_min, x_max = x_range
    y_min, y_max = y_range
    xs = [x_min + i * (x_max - x_min) / (nx - 1) for i in range(nx)]
    ys = [y_min + j * (y_max - y_min) / (ny - 1) for j in range(ny)]
    values = [[f([x, y]) for x in xs] for y in ys]
    return {"xs": xs, "ys": ys, "values": values}


def level_curve_points(f, c, x_range, y_range, nx=200, ny=200, tol=None):
    """Find grid points near the level curve f(x, y) = c.

    Scans the grid and returns all points (x, y) where |f(x,y) - c|
    is within tol of zero.  tol defaults to one grid-cell diagonal.

    Returns a list of (x, y, f_val) triples.
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)
    if tol is None:
        tol = math.sqrt(dx*dx + dy*dy)

    pts = []
    for j in range(ny):
        y = y_min + j * dy
        for i in range(nx):
            x = x_min + i * dx
            val = f([x, y])
            if abs(val - c) <= tol:
                pts.append((x, y, val))
    return pts


# ---------------------------------------------------------------------------
# Vector field operations
# ---------------------------------------------------------------------------

def divergence(Fx, Fy, point, h=None):
    """Divergence of the 2D vector field F = (Fx, Fy) at point.

    ∇·F = dFx/dx + dFy/dy

    Parameters
    ----------
    Fx, Fy : callables, each takes [x, y] and returns a float
    point  : [x, y]
    h      : step size
    """
    dFx_dx = partial_derivative(Fx, point, 0, h)
    dFy_dy = partial_derivative(Fy, point, 1, h)
    return dFx_dx + dFy_dy


def curl_z(Fx, Fy, point, h=None):
    """Z-component of the curl of a 2D vector field (scalar curl).

    For F = (Fx, Fy, 0) in 3D, the curl has only a z-component:

        (∇×F)_z = dFy/dx - dFx/dy

    This is the relevant quantity for 2D flow fields: positive means
    counter-clockwise rotation, negative means clockwise.

    Parameters
    ----------
    Fx, Fy : callables, each takes [x, y] and returns a float
    point  : [x, y]
    h      : step size
    """
    dFy_dx = partial_derivative(Fy, point, 0, h)
    dFx_dy = partial_derivative(Fx, point, 1, h)
    return dFy_dx - dFx_dy
