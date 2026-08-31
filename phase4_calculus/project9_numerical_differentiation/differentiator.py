import math


# ---------------------------------------------------------------------------
# Step size selection
# ---------------------------------------------------------------------------

# The machine epsilon for double precision floating point
_EPSILON = 2.220446049250313e-16


def _optimal_h(x, method="central"):
    """A reasonable default step size for the given method and point.

    For central differences the approximation error has two competing
    terms: the truncation error O(h^2) from the Taylor expansion, and
    the cancellation error O(eps/h) from floating-point subtraction.
    The optimal h minimises their sum, giving h ~ eps^(1/3) ~ 6e-6 for
    central differences and h ~ eps^(1/2) ~ 1.5e-8 for forward/backward.

    The point x is used to scale h so that the step is relative to the
    magnitude of x. A fixed step of 1e-6 would be too small near x=0
    and too large near x=1e9.
    """
    scale = max(1.0, abs(x))
    if method in ("forward", "backward"):
        return scale * (_EPSILON ** 0.5)
    elif method == "central":
        return scale * (_EPSILON ** (1.0 / 3.0))
    elif method == "five-point":
        return scale * (_EPSILON ** (1.0 / 5.0))
    elif method == "second":
        return scale * (_EPSILON ** (1.0 / 4.0))
    else:
        raise ValueError("unknown method {!r}".format(method))


# ---------------------------------------------------------------------------
# First derivative approximations
# ---------------------------------------------------------------------------

def forward_difference(f, x, h=None):
    """Forward difference approximation of f'(x).

    f'(x) ≈ [f(x + h) - f(x)] / h

    First-order accurate: the error is O(h). Requires two function
    evaluations. Cheap but less accurate than central difference.
    Use this only when you cannot evaluate f to the left of x.
    """
    if h is None:
        h = _optimal_h(x, "forward")
    if h == 0.0:
        raise ValueError("step size h must not be zero")
    return (f(x + h) - f(x)) / h


def backward_difference(f, x, h=None):
    """Backward difference approximation of f'(x).

    f'(x) ≈ [f(x) - f(x - h)] / h

    First-order accurate: the error is O(h). Use this only when you
    cannot evaluate f to the right of x.
    """
    if h is None:
        h = _optimal_h(x, "backward")
    if h == 0.0:
        raise ValueError("step size h must not be zero")
    return (f(x) - f(x - h)) / h


def central_difference(f, x, h=None):
    """Central difference approximation of f'(x).

    f'(x) ≈ [f(x + h) - f(x - h)] / (2h)

    Second-order accurate: the error is O(h^2). The odd-order terms in
    the Taylor expansions of f(x+h) and f(x-h) cancel, leaving only
    the h^2 term in the error. Use this as the default.
    """
    if h is None:
        h = _optimal_h(x, "central")
    if h == 0.0:
        raise ValueError("step size h must not be zero")
    return (f(x + h) - f(x - h)) / (2.0 * h)


def five_point_stencil(f, x, h=None):
    """Five-point stencil approximation of f'(x).

    f'(x) ≈ [-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)] / (12h)

    Fourth-order accurate: the error is O(h^4). The coefficients are
    chosen so that the h, h^2, and h^3 error terms all cancel. Requires
    four function evaluations. The most accurate formula in this module
    for smooth functions.
    """
    if h is None:
        h = _optimal_h(x, "five-point")
    if h == 0.0:
        raise ValueError("step size h must not be zero")
    return (-f(x + 2*h) + 8*f(x + h) - 8*f(x - h) + f(x - 2*h)) / (12.0 * h)


def differentiate(f, x, method="central", h=None):
    """Differentiate f at x using the chosen method.

    method is one of: forward, backward, central, five-point.
    h is the step size; if None a sensible default is chosen.
    """
    if method == "forward":
        return forward_difference(f, x, h)
    elif method == "backward":
        return backward_difference(f, x, h)
    elif method == "central":
        return central_difference(f, x, h)
    elif method == "five-point":
        return five_point_stencil(f, x, h)
    else:
        raise ValueError("method must be forward, backward, central, or five-point")


# ---------------------------------------------------------------------------
# Second derivative
# ---------------------------------------------------------------------------

def second_derivative(f, x, h=None):
    """Central difference approximation of f''(x).

    f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h^2

    This formula comes from adding the Taylor expansions of f(x+h) and
    f(x-h). The first-derivative terms cancel and the h^2 terms double,
    giving a second-order accurate approximation of the second derivative.
    """
    if h is None:
        h = _optimal_h(x, "second")
    if h == 0.0:
        raise ValueError("step size h must not be zero")
    return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def approximation_error(f, x, f_prime_exact, h_values, method="central"):
    """Compute the approximation error for a range of step sizes.

    Returns a list of (h, error) pairs where error is the absolute
    difference between the numerical derivative and the exact value.
    Useful for plotting the error vs h curve and observing the optimal
    step size before floating-point cancellation dominates.

    Parameters
    ----------
    f             : function to differentiate
    x             : point at which to differentiate
    f_prime_exact : exact derivative value at x
    h_values      : list of step sizes to try
    method        : differentiation method to use
    """
    results = []
    for h in h_values:
        try:
            approx = differentiate(f, x, method=method, h=h)
            error  = abs(approx - f_prime_exact)
        except (ZeroDivisionError, ValueError, OverflowError):
            error = float("inf")
        results.append((h, error))
    return results


# ---------------------------------------------------------------------------
# Multivariate calculus
# ---------------------------------------------------------------------------

def partial_derivative(f, x, i, h=None, method="central"):
    """Partial derivative of f with respect to its i-th argument.

    f takes a list of floats and returns a float.  The partial
    derivative at x is approximated by perturbing x[i] by h while
    holding all other arguments fixed.

    Parameters
    ----------
    f      : callable, f(x) -> float, x is a list
    x      : list of floats, the point at which to differentiate
    i      : index of the variable to differentiate with respect to
    h      : step size; if None a sensible default is chosen
    method : forward, backward, central, or five-point
    """
    x = list(x)
    if i < 0 or i >= len(x):
        raise ValueError(
            "index {} out of range for x of length {}".format(i, len(x))
        )
    xi = x[i]
    if h is None:
        h = _optimal_h(xi, method)

    def f_1d(t):
        xp = list(x)
        xp[i] = t
        return f(xp)

    return differentiate(f_1d, xi, method=method, h=h)


def gradient(f, x, h=None, method="central"):
    """Gradient of a scalar field f at point x.

    Returns a list of partial derivatives [df/dx_0, df/dx_1, ...].
    Each partial derivative is computed by central difference with the
    other variables held fixed.

    The gradient points in the direction of steepest ascent of f at x.
    Its magnitude is the rate of change in that direction.

    Parameters
    ----------
    f      : callable, f(x) -> float
    x      : list of floats
    h      : step size (same h used for all partial derivatives)
    method : differentiation method
    """
    return [partial_derivative(f, x, i, h=h, method=method)
            for i in range(len(x))]


def directional_derivative(f, x, direction, h=None, method="central"):
    """Directional derivative of f at x along a unit vector.

    The directional derivative is the dot product of the gradient with
    the unit direction vector. It measures the rate of change of f as
    you move from x in the given direction.

    direction does not need to be a unit vector; it is normalised here.
    """
    norm = math.sqrt(sum(d * d for d in direction))
    if norm == 0.0:
        raise ValueError("direction vector must not be the zero vector")
    unit = [d / norm for d in direction]
    grad = gradient(f, x, h=h, method=method)
    return sum(g * u for g, u in zip(grad, unit))


def second_partial(f, x, i, j, h=None):
    """Mixed second partial derivative d^2f / (dx_i dx_j).

    Computed by applying central difference twice: first differentiate
    with respect to x_j to get g(x) = df/dx_j, then differentiate g
    with respect to x_i.

    For smooth functions Schwarz's theorem guarantees d^2f/dx_i dx_j =
    d^2f/dx_j dx_i, so the order does not matter.
    """
    x = list(x)
    if h is None:
        h = _optimal_h(max(abs(x[i]), abs(x[j]), 1.0), "central")

    def g(t):
        xp = list(x)
        xp[i] = t
        return partial_derivative(f, xp, j, h=h)

    xi = x[i]
    return central_difference(g, xi, h=h)


def jacobian(f, x, h=None, method="central"):
    """Jacobian matrix of a vector-valued function f at x.

    f maps a list of n floats to a list of m floats.
    The Jacobian is an m x n matrix where J[i][j] = df_i/dx_j.

    Parameters
    ----------
    f      : callable, f(x) -> list of floats
    x      : list of floats, the point
    h      : step size
    method : differentiation method
    """
    x    = list(x)
    n    = len(x)
    fx   = f(x)
    m    = len(fx)
    jac  = [[0.0] * n for _ in range(m)]
    for j in range(n):
        def fj(t, j=j):
            xp    = list(x)
            xp[j] = t
            return f(xp)
        xj = x[j]
        if h is None:
            hj = _optimal_h(xj, method)
        else:
            hj = h
        # differentiate each output component with respect to x[j]
        for i in range(m):
            def fi_j(t, i=i, j=j):
                xp    = list(x)
                xp[j] = t
                return f(xp)[i]
            jac[i][j] = differentiate(fi_j, xj, method=method, h=hj)
    return jac


# ---------------------------------------------------------------------------
# Derivative rules verified numerically
# ---------------------------------------------------------------------------

def verify_product_rule(f, g, x, h=None):
    """Check (fg)'(x) = f'(x)g(x) + f(x)g'(x) numerically.

    Returns a dict with the left-hand side, the right-hand side, and
    the absolute error between them.
    """
    lhs = central_difference(lambda t: f(t) * g(t), x, h=h)
    rhs = (central_difference(f, x, h=h) * g(x)
           + f(x) * central_difference(g, x, h=h))
    return {"lhs": lhs, "rhs": rhs, "error": abs(lhs - rhs)}


def verify_chain_rule(f, g, x, h=None):
    """Check (f o g)'(x) = f'(g(x)) * g'(x) numerically.

    Returns a dict with the left-hand side, the right-hand side, and
    the absolute error between them.
    """
    lhs = central_difference(lambda t: f(g(t)), x, h=h)
    gx  = g(x)
    rhs = central_difference(f, gx, h=h) * central_difference(g, x, h=h)
    return {"lhs": lhs, "rhs": rhs, "error": abs(lhs - rhs)}


# ---------------------------------------------------------------------------
# Gradient descent
# ---------------------------------------------------------------------------

def gradient_descent(f, x0, learning_rate=0.01, max_iterations=10000,
                     tolerance=1e-8, h=None):
    """Find a local minimum of f by following the negative gradient.

    Starting from x0, repeatedly moves in the direction of the negative
    gradient by the learning_rate until either the gradient is smaller
    than tolerance or max_iterations is reached.

    Parameters
    ----------
    f             : callable, f(x) -> float, x is a list
    x0            : starting point, a list of floats
    learning_rate : step size along the gradient direction
    max_iterations: safety cap on the number of steps
    tolerance     : stop when the gradient magnitude is below this
    h             : step size for numerical gradient computation

    Returns a dict with:
        x          : the final point
        f_x        : f at the final point
        iterations : number of iterations taken
        converged  : True if the gradient magnitude fell below tolerance
        path       : list of (x, f(x)) pairs, one per iteration
    """
    x    = list(x0)
    path = [(list(x), f(x))]
    for iteration in range(1, max_iterations + 1):
        grad = gradient(f, x, h=h)
        mag  = math.sqrt(sum(g * g for g in grad))
        if mag < tolerance:
            return {
                "x":          x,
                "f_x":        f(x),
                "iterations": iteration,
                "converged":  True,
                "path":       path,
            }
        x = [xi - learning_rate * gi for xi, gi in zip(x, grad)]
        path.append((list(x), f(x)))
    return {
        "x":          x,
        "f_x":        f(x),
        "iterations": max_iterations,
        "converged":  False,
        "path":       path,
    }


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def sensitivity(f, x, h=None, method="central", relative=False):
    """Sensitivity of f's output to each input.

    Returns a list of partial derivatives [df/dx_0, df/dx_1, ...].
    When relative=True returns [x_i/f(x) * df/dx_i], the relative (or
    percentage) sensitivity normalised so that a value of 1 means a 1%
    change in x_i produces a 1% change in f(x).

    Parameters
    ----------
    f        : callable, f(x) -> float
    x        : list of floats
    h        : step size
    method   : differentiation method
    relative : whether to return relative (elasticity) sensitivities
    """
    grad = gradient(f, x, h=h, method=method)
    if not relative:
        return grad
    fx = f(x)
    if fx == 0.0:
        raise ValueError(
            "relative sensitivity is undefined when f(x) = 0"
        )
    return [g * xi / fx for g, xi in zip(grad, x)]


# ---------------------------------------------------------------------------
# Finance applications
# ---------------------------------------------------------------------------

def bond_price(yield_rate, coupon, face, n_periods, periods_per_year=1):
    """Price of a fixed-coupon bond.

    P = sum_{t=1}^{n} C / (1+y)^t + F / (1+y)^n

    where C is the coupon per period, F is the face value, y is the
    yield per period, and n is the number of periods.

    Parameters
    ----------
    yield_rate      : yield per period (e.g. 0.05 for 5% annual)
    coupon          : coupon payment per period
    face            : face (par) value
    n_periods       : total number of coupon periods
    periods_per_year: used only for annualising duration (default 1)
    """
    if yield_rate <= -1.0:
        raise ValueError("yield must be greater than -1")
    price = 0.0
    for t in range(1, n_periods + 1):
        price += coupon / (1.0 + yield_rate) ** t
    price += face / (1.0 + yield_rate) ** n_periods
    return price


def bond_duration(yield_rate, coupon, face, n_periods, kind="modified"):
    """Modified or Macaulay duration of a bond.

    Macaulay duration is the weighted average time to each cash flow,
    weighted by the present value of that cash flow:

        D_mac = (1/P) * sum_{t=1}^{n} t * PV(CF_t)

    Modified duration is Macaulay duration divided by (1 + y):

        D_mod = D_mac / (1 + y) = -dP/dy * 1/P

    Modified duration is the negative logarithmic derivative of price
    with respect to yield: a bond with modified duration 7 loses
    approximately 7% in price for a 1% rise in yield.

    This function computes modified duration by numerical differentiation
    of the price function.  The analytical formula is also computed and
    returned for comparison.

    kind is 'modified' (default) or 'macaulay'.
    """
    price = bond_price(yield_rate, coupon, face, n_periods)

    # numerical duration: -dP/dy / P
    dp_dy    = central_difference(
        lambda y: bond_price(y, coupon, face, n_periods), yield_rate
    )
    num_mod  = -dp_dy / price

    # analytical Macaulay duration
    mac = 0.0
    for t in range(1, n_periods + 1):
        pv   = coupon / (1.0 + yield_rate) ** t
        mac += t * pv
    pv_face  = face / (1.0 + yield_rate) ** n_periods
    mac     += n_periods * pv_face
    mac     /= price
    ana_mod  = mac / (1.0 + yield_rate)

    if kind == "macaulay":
        return {"numerical": num_mod * (1.0 + yield_rate),
                "analytical": mac,
                "price": price}
    return {"numerical": num_mod, "analytical": ana_mod, "price": price}


def bond_convexity(yield_rate, coupon, face, n_periods):
    """Convexity of a bond.

    Convexity = (1/P) * d^2P/dy^2

    It is the second derivative of price with respect to yield, divided
    by price. A bond with higher convexity benefits more from falling
    yields and loses less from rising yields than one with lower
    convexity at the same duration.

    Computed here by numerical second derivative of the price function.
    """
    price  = bond_price(yield_rate, coupon, face, n_periods)
    d2p_dy = second_derivative(
        lambda y: bond_price(y, coupon, face, n_periods), yield_rate
    )
    return d2p_dy / price


def bond_price_change(yield_rate, coupon, face, n_periods, delta_y):
    """Approximate change in bond price for a small yield change.

    Uses the duration-convexity approximation:

        dP/P ≈ -D_mod * dy + 0.5 * C * dy^2

    where D_mod is modified duration and C is convexity.

    Returns a dict with the duration-only approximation, the
    duration-convexity approximation, and the exact price change.
    """
    dur    = bond_duration(yield_rate, coupon, face, n_periods)["numerical"]
    conv   = bond_convexity(yield_rate, coupon, face, n_periods)
    p0     = bond_price(yield_rate, coupon, face, n_periods)
    p1     = bond_price(yield_rate + delta_y, coupon, face, n_periods)
    exact  = (p1 - p0) / p0
    dur_approx  = -dur * delta_y
    durconv_approx = -dur * delta_y + 0.5 * conv * delta_y ** 2
    return {
        "exact_pct_change":        exact,
        "duration_approx":         dur_approx,
        "duration_convexity_approx": durconv_approx,
        "duration_error":          abs(exact - dur_approx),
        "duration_convexity_error": abs(exact - durconv_approx),
    }
