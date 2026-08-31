import math


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_interval(a, b):
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("use improper_integral for infinite limits")
    if a == b:
        return False   # degenerate interval, integral is zero
    return True


def _check_n(n, minimum=1):
    if not isinstance(n, int) or n < minimum:
        raise ValueError(
            "n must be an integer >= {}".format(minimum)
        )


# ---------------------------------------------------------------------------
# Riemann sums
# ---------------------------------------------------------------------------

def left_riemann(f, a, b, n=1000):
    """Left Riemann sum with n equal subintervals.

    Approximates the integral of f from a to b by evaluating f at the
    left endpoint of each subinterval and summing the rectangle areas:

        sum_{i=0}^{n-1} f(a + i*h) * h    where h = (b-a)/n

    First-order accurate: the error is O(h) = O((b-a)/n).  The sign of
    the error depends on whether f is increasing or decreasing.

    Parameters
    ----------
    f : callable, f(x) -> float
    a : left endpoint of the integration interval
    b : right endpoint
    n : number of subintervals
    """
    if not _check_interval(a, b):
        return 0.0
    _check_n(n)
    h = (b - a) / n
    return h * sum(f(a + i * h) for i in range(n))


def right_riemann(f, a, b, n=1000):
    """Right Riemann sum with n equal subintervals.

    Like the left sum but evaluates f at the right endpoint of each
    subinterval:

        sum_{i=1}^{n} f(a + i*h) * h    where h = (b-a)/n

    Also first-order accurate.  The left and right sums bracket the true
    integral for monotone functions: whichever is smaller is a lower
    bound and whichever is larger is an upper bound.
    """
    if not _check_interval(a, b):
        return 0.0
    _check_n(n)
    h = (b - a) / n
    return h * sum(f(a + i * h) for i in range(1, n + 1))


def midpoint_riemann(f, a, b, n=1000):
    """Midpoint Riemann sum with n equal subintervals.

    Evaluates f at the midpoint of each subinterval:

        sum_{i=0}^{n-1} f(a + (i + 0.5)*h) * h    where h = (b-a)/n

    Second-order accurate, O(h^2), despite using only one function
    evaluation per subinterval.  The midpoint cancels the first-order
    error term that plagues the left and right sums.  Usually more
    accurate than the trapezoidal rule by a factor of two for the same
    number of evaluations.
    """
    if not _check_interval(a, b):
        return 0.0
    _check_n(n)
    h = (b - a) / n
    return h * sum(f(a + (i + 0.5) * h) for i in range(n))


# ---------------------------------------------------------------------------
# Trapezoidal rule
# ---------------------------------------------------------------------------

def trapezoidal(f, a, b, n=1000):
    """Trapezoidal rule with n equal subintervals.

    Connects adjacent function values with straight lines and sums the
    trapezoid areas:

        h/2 * [f(a) + 2*f(x_1) + 2*f(x_2) + ... + 2*f(x_{n-1}) + f(b)]

    where h = (b-a)/n.  Second-order accurate, O(h^2).  The endpoint
    values are weighted by 1 and the interior values by 2.

    The trapezoidal rule is exact for linear functions.  For periodic
    functions on a full period it achieves spectral accuracy (faster
    than any power of h) because the endpoint error terms cancel.
    """
    if not _check_interval(a, b):
        return 0.0
    _check_n(n)
    h = (b - a) / n
    total = f(a) + f(b)
    total += 2.0 * sum(f(a + i * h) for i in range(1, n))
    return total * h / 2.0


def trapezoidal_error_estimate(f, a, b, n):
    """Richardson extrapolation estimate of the trapezoidal error.

    Computes the integral with n and 2n subintervals.  By Richardson
    extrapolation the error of the n-panel estimate is approximately:

        error ≈ (I(2n) - I(n)) / 3

    Returns (I(2n), estimated_error).  The 2n estimate is returned
    because it is more accurate than the n estimate.
    """
    _check_n(n)
    i_n  = trapezoidal(f, a, b, n)
    i_2n = trapezoidal(f, a, b, 2 * n)
    return i_2n, abs(i_2n - i_n) / 3.0


# ---------------------------------------------------------------------------
# Simpson's rule
# ---------------------------------------------------------------------------

def simpsons(f, a, b, n=1000):
    """Simpson's rule with n subintervals (n must be even).

    Fits a quadratic polynomial through each triple of adjacent points
    and integrates the polynomial exactly:

        h/3 * [f(x_0) + 4*f(x_1) + 2*f(x_2) + 4*f(x_3) + ... + f(x_n)]

    where h = (b-a)/n.  The pattern of weights is 1, 4, 2, 4, 2, ...,
    4, 1.  Fourth-order accurate, O(h^4): halving n reduces the error
    by a factor of 16.  The standard method for smooth functions.

    n must be even because each quadratic panel spans two subintervals.

    Parameters
    ----------
    f : callable
    a : left endpoint
    b : right endpoint
    n : number of subintervals, must be a positive even integer
    """
    if not _check_interval(a, b):
        return 0.0
    _check_n(n, minimum=2)
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson's rule")
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        weight = 4 if i % 2 == 1 else 2
        total += weight * f(a + i * h)
    return total * h / 3.0


def simpsons_error_estimate(f, a, b, n):
    """Richardson extrapolation estimate of the Simpson's rule error.

    Uses the 4th-order convergence rate: error ≈ (I(2n) - I(n)) / 15.

    Returns (I(2n), estimated_error).
    """
    _check_n(n, minimum=2)
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson's rule")
    i_n  = simpsons(f, a, b, n)
    i_2n = simpsons(f, a, b, 2 * n)
    return i_2n, abs(i_2n - i_n) / 15.0


# ---------------------------------------------------------------------------
# Adaptive quadrature
# ---------------------------------------------------------------------------

def adaptive_simpsons(f, a, b, tolerance=1e-8, max_depth=50):
    """Adaptive Simpson's rule to a target absolute tolerance.

    Recursively subdivides the interval, applying Simpson's rule on each
    subinterval and estimating the local error by comparing the coarse
    and fine estimates.  Only subintervals with error above their share
    of the tolerance are refined.

    This achieves the same accuracy as a uniform grid with far fewer
    function evaluations because it concentrates effort where the
    function is changing rapidly.

    Parameters
    ----------
    f         : callable
    a, b      : integration limits
    tolerance : target absolute error for the whole integral
    max_depth : maximum recursion depth per subinterval

    Returns (integral, error_estimate).
    """
    if not _check_interval(a, b):
        return 0.0, 0.0

    def _recursive(a, b, fa, fm, fb, whole, tolerance, depth):
        m  = 0.5 * (a + b)
        lm = 0.5 * (a + m)
        rm = 0.5 * (m + b)
        flm = f(lm)
        frm = f(rm)
        left  = (m - a) / 6.0 * (fa + 4.0 * flm + fm)
        right = (b - m) / 6.0 * (fm + 4.0 * frm + fb)
        delta = left + right - whole
        if depth <= 0 or abs(delta) <= 15.0 * tolerance:
            return left + right + delta / 15.0, abs(delta / 15.0)
        l_val, l_err = _recursive(a, m, fa, flm, fm, left,
                                   tolerance / 2.0, depth - 1)
        r_val, r_err = _recursive(m, b, fm, frm, fb, right,
                                   tolerance / 2.0, depth - 1)
        return l_val + r_val, l_err + r_err

    m   = 0.5 * (a + b)
    fa  = f(a)
    fm  = f(m)
    fb  = f(b)
    whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb)
    return _recursive(a, b, fa, fm, fb, whole, tolerance, max_depth)


# ---------------------------------------------------------------------------
# Gaussian quadrature
# ---------------------------------------------------------------------------

# Gauss-Legendre nodes and weights on [-1, 1] for n = 2, 3, 4, 5, 6, 7, 10.
# These are pre-computed from the exact eigenvalue problem for the Legendre
# polynomials. Storing them here avoids needing scipy for a 5-node rule.
_GL_NODES_WEIGHTS = {
    2: (
        [-0.5773502691896257, 0.5773502691896257],
        [1.0, 1.0],
    ),
    3: (
        [-0.7745966692414834, 0.0, 0.7745966692414834],
        [0.5555555555555556, 0.8888888888888888, 0.5555555555555556],
    ),
    4: (
        [-0.8611363115940526, -0.3399810435848563,
          0.3399810435848563,  0.8611363115940526],
        [0.3478548451374538, 0.6521451548625461,
         0.6521451548625461, 0.3478548451374538],
    ),
    5: (
        [-0.9061798459386640, -0.5384693101056831, 0.0,
          0.5384693101056831,  0.9061798459386640],
        [0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
         0.4786286704993665, 0.2369268850561891],
    ),
    6: (
        [-0.9324695142031521, -0.6612093864662645, -0.2386191860831969,
          0.2386191860831969,  0.6612093864662645,  0.9324695142031521],
        [0.1713244923791704, 0.3607615730481386, 0.4679139345726910,
         0.4679139345726910, 0.3607615730481386, 0.1713244923791704],
    ),
    7: (
        [-0.9491079123427585, -0.7415311855993945, -0.4058451513773972,
          0.0,
          0.4058451513773972,  0.7415311855993945,  0.9491079123427585],
        [0.1294849661688697, 0.2797053914892767, 0.3818300505051189,
         0.4179591836734694,
         0.3818300505051189, 0.2797053914892767, 0.1294849661688697],
    ),
    10: (
        [-0.9739065285171717, -0.8650633666889845, -0.6794095682990244,
         -0.4333953941292472, -0.1488743389816312,
          0.1488743389816312,  0.4333953941292472,  0.6794095682990244,
          0.8650633666889845,  0.9739065285171717],
        [0.0666713443086881, 0.1494513491505806, 0.2190863625159820,
         0.2692667193099963, 0.2955242247147529,
         0.2955242247147529, 0.2692667193099963, 0.2190863625159820,
         0.1494513491505806, 0.0666713443086881],
    ),
}


def gauss_legendre(f, a, b, n=5):
    """Gauss-Legendre quadrature with n points on [a, b].

    Integrates f exactly for polynomials of degree up to 2n-1.  For
    smooth non-polynomial functions it converges exponentially fast as n
    increases -- far faster than any equally-spaced method.

    The standard nodes and weights are defined on [-1, 1].  A linear
    change of variables maps [a, b] to [-1, 1]:

        x = (b+a)/2 + (b-a)/2 * t,   dx = (b-a)/2 * dt

    so the integral becomes ((b-a)/2) * sum_i w_i * f(x(t_i)).

    Parameters
    ----------
    f : callable
    a : left endpoint
    b : right endpoint
    n : number of quadrature points; must be one of 2, 3, 4, 5, 6, 7, 10
    """
    if n not in _GL_NODES_WEIGHTS:
        raise ValueError(
            "n must be one of {}".format(sorted(_GL_NODES_WEIGHTS))
        )
    if not _check_interval(a, b):
        return 0.0
    nodes, weights = _GL_NODES_WEIGHTS[n]
    mid   = 0.5 * (a + b)
    half  = 0.5 * (b - a)
    return half * sum(w * f(mid + half * t)
                      for t, w in zip(nodes, weights))


# ---------------------------------------------------------------------------
# Convergence analysis
# ---------------------------------------------------------------------------

def convergence_table(f, a, b, exact, methods=None, n_values=None):
    """Compute integration errors for increasing n, for each method.

    Returns a list of dicts, one per (method, n) combination, with keys:
        method, n, estimate, error, ratio

    ratio is the error at n divided by the error at n/2, which reveals
    the convergence rate: a ratio of ~4 means O(h^2), ~16 means O(h^4).

    Parameters
    ----------
    f       : callable
    a, b    : integration limits
    exact   : exact value of the integral
    methods : list of method names from ('left', 'right', 'midpoint',
              'trapezoidal', 'simpsons'); default all five
    n_values: list of n values; must be even for simpsons; default
              [10, 20, 50, 100, 200, 500, 1000]
    """
    if methods is None:
        methods = ['left', 'right', 'midpoint', 'trapezoidal', 'simpsons']
    if n_values is None:
        n_values = [10, 20, 50, 100, 200, 500, 1000]

    _dispatch = {
        'left':        left_riemann,
        'right':       right_riemann,
        'midpoint':    midpoint_riemann,
        'trapezoidal': trapezoidal,
        'simpsons':    simpsons,
    }
    for m in methods:
        if m not in _dispatch:
            raise ValueError("unknown method {!r}".format(m))

    rows  = []
    prev_errors = {m: None for m in methods}
    for n in n_values:
        for m in methods:
            if m == 'simpsons' and n % 2 != 0:
                continue
            est   = _dispatch[m](f, a, b, n)
            err   = abs(est - exact)
            prev  = prev_errors[m]
            ratio = (prev / err) if (prev is not None and err > 0) else None
            rows.append({
                'method':   m,
                'n':        n,
                'estimate': est,
                'error':    err,
                'ratio':    ratio,
            })
            prev_errors[m] = err
    return rows


# ---------------------------------------------------------------------------
# Improper integrals
# ---------------------------------------------------------------------------

def improper_integral(f, a, b, n=10000):
    """Integrate f over an interval that may include infinite limits.

    Handles four cases:
        finite [a, b]          : standard Simpson's rule
        [a, +inf)              : substitution t = 1/(1+x-a), maps to (0,1]
        (-inf, b]              : substitution t = 1/(1+b-x), maps to (0,1]
        (-inf, +inf)           : split at 0, handle each half separately

    The substitution for [a, +inf) is x = a + t/(1-t), dx = 1/(1-t)^2 dt,
    which maps t in (0,1) to x in (a, +inf).  The transformed integrand
    is f(a + t/(1-t)) / (1-t)^2.

    This is a simple but effective approach.  Integrands that decay
    slowly near infinity may need larger n.

    Parameters
    ----------
    f    : callable
    a    : lower limit, or -math.inf
    b    : upper limit, or math.inf
    n    : number of subintervals for the transformed integral
    """
    _check_n(n, minimum=2)
    if n % 2 != 0:
        n += 1  # Simpson's needs even n

    a_inf = math.isinf(a) and a < 0
    b_inf = math.isinf(b) and b > 0

    if not a_inf and not b_inf:
        return simpsons(f, a, b, n)

    if not a_inf and b_inf:
        # substitution: t = (x-a)/(1+(x-a)), x = a + t/(1-t), dx = 1/(1-t)^2
        def g(t):
            if t >= 1.0:
                return 0.0
            x = a + t / (1.0 - t)
            return f(x) / (1.0 - t) ** 2
        return simpsons(g, 0.0, 1.0, n)

    if a_inf and not b_inf:
        # substitution: t = (b-x)/(1+(b-x)), x = b - t/(1-t)
        def g(t):
            if t >= 1.0:
                return 0.0
            x = b - t / (1.0 - t)
            return f(x) / (1.0 - t) ** 2
        return simpsons(g, 0.0, 1.0, n)

    # both infinite: split at 0
    left  = improper_integral(f, a, 0.0, n)
    right = improper_integral(f, 0.0, b, n)
    return left + right


# ---------------------------------------------------------------------------
# Taylor series
# ---------------------------------------------------------------------------

def _central_diff(f, x0, k, h):
    """k-th derivative of f at x0 by the central difference formula.

    The formulas are derived from the Taylor expansion by choosing
    coefficients that cancel all terms below order k.  Each formula
    is second-order accurate in h (error O(h^2)).

    Symmetric (even k) formulas use f at x0 +/- jh.
    Antisymmetric (odd k) formulas use differences f(x0+jh) - f(x0-jh).
    """
    if k == 0:
        return f(x0)
    # Odd derivatives: use antisymmetric combinations
    # d_k = [a*(f(x+h)-f(x-h)) + b*(f(x+2h)-f(x-2h)) + c*(f(x+3h)-f(x-3h)) + ...] / (2h^k)
    # Coefficients derived so that all lower odd derivatives cancel.
    if k == 1:
        # (f(x+h) - f(x-h)) / (2h)
        return (f(x0 + h) - f(x0 - h)) / (2.0 * h)
    if k == 2:
        # (f(x+h) - 2f(x) + f(x-h)) / h^2
        return (f(x0 + h) - 2.0*f(x0) + f(x0 - h)) / (h * h)
    if k == 3:
        # Derived: [-2*(f(x+h)-f(x-h)) + (f(x+2h)-f(x-2h))] / (2h^3)
        return (-2.0*(f(x0+h) - f(x0-h))
                + (f(x0+2*h) - f(x0-2*h))) / (2.0 * h**3)
    if k == 4:
        # (f(x+2h) - 4f(x+h) + 6f(x) - 4f(x-h) + f(x-2h)) / h^4
        return (f(x0+2*h) - 4.0*f(x0+h) + 6.0*f(x0)
                - 4.0*f(x0-h) + f(x0-2*h)) / h**4
    if k == 5:
        # Derived: [5*(f(x+h)-f(x-h)) - 4*(f(x+2h)-f(x-2h)) + (f(x+3h)-f(x-3h))] / (2h^5)
        return (5.0*(f(x0+h) - f(x0-h))
                - 4.0*(f(x0+2*h) - f(x0-2*h))
                + (f(x0+3*h) - f(x0-3*h))) / (2.0 * h**5)
    if k == 6:
        # (f(x+3h) - 6f(x+2h) + 15f(x+h) - 20f(x) + 15f(x-h) - 6f(x-2h) + f(x-3h)) / h^6
        return (f(x0+3*h) - 6.0*f(x0+2*h) + 15.0*f(x0+h) - 20.0*f(x0)
                + 15.0*f(x0-h) - 6.0*f(x0-2*h) + f(x0-3*h)) / h**6
    if k == 7:
        # Derived antisymmetric: [-7*(f(x+h)-f(x-h)) + 7*(f(x+2h)-f(x-2h))
        #                        - (7/3)*(f(x+3h)-f(x-3h)) + (1/6)*(f(x+4h)-f(x-4h))] / h^7
        # Simplified with common denominator 6:
        return (-42.0*(f(x0+h)-f(x0-h))
                + 42.0*(f(x0+2*h)-f(x0-2*h))
                - 14.0*(f(x0+3*h)-f(x0-3*h))
                +  2.0*(f(x0+4*h)-f(x0-4*h))) / (12.0 * h**7)
    if k == 8:
        return (f(x0+4*h) - 8.0*f(x0+3*h) + 28.0*f(x0+2*h) - 56.0*f(x0+h)
                + 70.0*f(x0) - 56.0*f(x0-h) + 28.0*f(x0-2*h)
                - 8.0*f(x0-3*h) + f(x0-4*h)) / h**8
    # For k > 8: recurse: f^(k) = (f'')^(k-2)
    def f2(x):
        return (f(x + h) - 2.0*f(x) + f(x - h)) / (h * h)
    return _central_diff(f2, x0, k - 2, h)


def taylor_coefficients(f, x0, degree, h=None):
    """Compute Taylor series coefficients of f at x0 up to given degree.

    The k-th Taylor coefficient is f^(k)(x0) / k!, where f^(k) is the
    k-th derivative.  Each derivative is estimated by the standard
    central difference formula for that order, then improved with one
    step of Richardson extrapolation which cancels the leading O(h^2)
    error term.

    Richardson extrapolation for O(h^2) methods:

        f^(k)(x0) ≈ [4 * D(h/2) - D(h)] / 3

    where D(h) is the central difference estimate at step size h.
    The extrapolated estimate is O(h^4) accurate.

    Returns a list [a_0, a_1, ..., a_degree] where

        f(x) ≈ sum_{k=0}^{degree} a_k * (x - x0)^k

    Accurate for smooth functions for degree up to about 6.  Degrees 7
    and 9 require antisymmetric stencils with points at 4h and 5h from
    x0; the resulting near-cancellation of large numbers limits accuracy
    to at best 2-3 significant digits for those coefficients regardless
    of how h is chosen.  If you need reliable high-degree coefficients,
    use a computer algebra system.

    Parameters
    ----------
    f      : callable, f(x) -> float
    x0     : expansion point
    degree : highest power to compute (non-negative integer)
    h      : step size; if None an order-specific default is chosen
    """
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if h is None:
        h = None  # signal to choose per-order below

    scale = max(1.0, abs(x0))

    # Optimal step size depends on derivative order.  High-order stencils
    # use points far from x0 (up to 4h away), so h must be large enough
    # to avoid floating-point cancellation but small enough to keep
    # truncation error low.  Empirically tuned per order:
    _h_defaults = {
        0: 0.0,          # unused
        1: 0.005,
        2: 0.01,
        3: 0.01,
        4: 0.03,
        5: 0.03,
        6: 0.1,
        7: 0.07,
        8: 0.1,
    }

    factorial_k = 1
    coeffs = [f(x0)]
    for k in range(1, degree + 1):
        factorial_k *= k
        hk = (h if h is not None
              else scale * _h_defaults.get(k, 0.1))
        d_h   = _central_diff(f, x0, k, hk)
        d_h2  = _central_diff(f, x0, k, hk / 2.0)
        dk    = (4.0 * d_h2 - d_h) / 3.0
        coeffs.append(dk / factorial_k)
    return coeffs


def taylor_evaluate(coeffs, x0, x):
    """Evaluate a Taylor polynomial at x given coefficients and expansion point.

    coeffs is the list returned by taylor_coefficients.
    Uses Horner's method for numerical stability.

    Horner's method rewrites the polynomial:
        a_0 + a_1*(x-x0) + a_2*(x-x0)^2 + ...
    as:
        a_0 + (x-x0) * (a_1 + (x-x0) * (a_2 + ...))

    which requires only n multiplications and n additions for a degree-n
    polynomial, avoiding the repeated computation of (x-x0)^k.
    """
    dx  = x - x0
    val = 0.0
    for c in reversed(coeffs):
        val = val * dx + c
    return val


def taylor_error(f, coeffs, x0, x_values):
    """Absolute error of the Taylor polynomial at a list of test points.

    Returns a list of (x, exact, approx, error) tuples.
    """
    results = []
    for x in x_values:
        exact  = f(x)
        approx = taylor_evaluate(coeffs, x0, x)
        results.append((x, exact, approx, abs(exact - approx)))
    return results


def radius_of_convergence(coeffs):
    """Estimate the radius of convergence from successive Taylor coefficients.

    For a power series sum a_k * (x-x0)^k the radius of convergence is:

        R = lim_{k -> inf} |a_k / a_{k+1}|   (ratio test)

    The ratio fluctuates for small k, so this returns the ratio for the
    last pair of non-negligible coefficients as the best estimate.  A
    radius of infinity means the series converges everywhere.

    Returns the estimated radius, or math.inf if all ratios are very large.
    """
    ratios = []
    for k in range(len(coeffs) - 1):
        ak   = abs(coeffs[k])
        ak1  = abs(coeffs[k + 1])
        if ak1 < 1e-15:
            # coefficient is negligibly small; series has likely converged
            return math.inf
        ratios.append(ak / ak1)
    if not ratios:
        return math.inf
    # Return the last ratio as the best estimate (most terms included)
    return ratios[-1]


def convergent_at(coeffs, x0, x):
    """Check whether the Taylor series appears to converge at x.

    Returns True if |x - x0| < estimated radius of convergence.
    """
    r = radius_of_convergence(coeffs)
    return abs(x - x0) < r


# ---------------------------------------------------------------------------
# Probability applications
# ---------------------------------------------------------------------------

def normal_cdf(x, mu=0.0, sigma=1.0, n=200):
    """CDF of the normal distribution by numerical integration of the PDF.

    The standard normal PDF is phi(t) = exp(-t^2/2) / sqrt(2*pi).
    The CDF is the integral of phi from -inf to x.

    Computed as 0.5 + integral of phi from 0 to x (for x >= 0),
    using the identity Phi(-x) = 1 - Phi(x) for x < 0.

    The integral is computed with adaptive Simpson's rule.

    Parameters
    ----------
    x     : quantile
    mu    : mean of the distribution
    sigma : standard deviation
    n     : number of subintervals for numerical integration
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    z = (x - mu) / sigma

    def phi(t):
        return math.exp(-0.5 * t * t) / math.sqrt(2.0 * math.pi)

    if z == 0.0:
        return 0.5
    if z > 0:
        val, _ = adaptive_simpsons(phi, 0.0, z)
        return 0.5 + val
    else:
        val, _ = adaptive_simpsons(phi, z, 0.0)
        return 0.5 - val


def exponential_cdf(x, rate=1.0):
    """CDF of the exponential distribution by numerical integration.

    The exponential PDF is f(t) = rate * exp(-rate * t) for t >= 0.
    The CDF is 1 - exp(-rate * x), computed here by numerical integration
    to demonstrate the method.

    Parameters
    ----------
    x    : quantile (must be >= 0)
    rate : rate parameter lambda > 0
    """
    if rate <= 0.0:
        raise ValueError("rate must be positive")
    if x < 0.0:
        return 0.0
    result, _ = adaptive_simpsons(lambda t: rate * math.exp(-rate * t),
                                  0.0, x)
    return result


def beta_cdf(x, alpha, beta_param, n=500):
    """CDF of the beta distribution by numerical integration.

    The beta PDF is f(t) = t^(alpha-1) * (1-t)^(beta-1) / B(alpha, beta)
    where B is the beta function.  This is computed by numerical integration
    on [0, x] with special handling near 0 and 1 where the integrand may
    be singular for alpha < 1 or beta < 1.

    Parameters
    ----------
    x          : quantile in [0, 1]
    alpha      : first shape parameter > 0
    beta_param : second shape parameter > 0
    n          : number of subintervals
    """
    if not 0.0 <= x <= 1.0:
        raise ValueError("x must be in [0, 1] for the beta distribution")
    if alpha <= 0.0 or beta_param <= 0.0:
        raise ValueError("alpha and beta must be positive")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    log_beta = math.lgamma(alpha) + math.lgamma(beta_param) - math.lgamma(alpha + beta_param)

    def pdf(t):
        if t <= 0.0 or t >= 1.0:
            return 0.0
        return math.exp((alpha - 1.0) * math.log(t) +
                        (beta_param - 1.0) * math.log(1.0 - t) -
                        log_beta)

    # Use a small epsilon to avoid endpoints for non-integer shape params
    eps = 1e-10
    lo  = eps if alpha < 1.0 else 0.0
    hi  = min(x, 1.0 - eps) if beta_param < 1.0 else x
    if hi <= lo:
        return 0.0 if x < 0.5 else 1.0
    _check_n(n, minimum=2)
    if n % 2 != 0:
        n += 1
    return simpsons(pdf, lo, hi, n)


# ---------------------------------------------------------------------------
# Finance applications
# ---------------------------------------------------------------------------

def present_value_continuous(cash_flow, rate, T, n=1000):
    """Present value of a continuous cash flow stream by numerical integration.

    The present value of a stream c(t) discounted at continuous rate r is:

        PV = integral_0^T c(t) * exp(-r*t) dt

    Parameters
    ----------
    cash_flow : callable, c(t) -> float, the cash flow rate at time t
    rate      : continuous discount rate
    T         : horizon
    n         : number of subintervals
    """
    if T <= 0.0:
        raise ValueError("horizon T must be positive")
    if n % 2 != 0:
        n += 1
    return simpsons(lambda t: cash_flow(t) * math.exp(-rate * t), 0.0, T, n)


def black_scholes_call(S, K, r, sigma, T):
    """Black-Scholes price of a European call option.

    The formula is:

        C = S * N(d1) - K * exp(-r*T) * N(d2)

        d1 = (ln(S/K) + (r + sigma^2/2)*T) / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)

    where N is the standard normal CDF, computed here by numerical
    integration rather than a library function.

    Parameters
    ----------
    S     : current stock price
    K     : strike price
    r     : risk-free rate (continuous)
    sigma : volatility (annualised)
    T     : time to expiry in years
    """
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        raise ValueError("S, K, sigma, T must all be positive")
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return S * normal_cdf(d1) - K * math.exp(-r * T) * normal_cdf(d2)


def black_scholes_put(S, K, r, sigma, T):
    """Black-Scholes price of a European put option via put-call parity.

    P = C - S + K * exp(-r*T)

    where C is the call price from black_scholes_call.
    """
    call = black_scholes_call(S, K, r, sigma, T)
    return call - S + K * math.exp(-r * T)


def continuous_bond_duration(yield_rate, coupon_rate, T, n=1000):
    """Macaulay duration of a continuous-coupon bond by numerical integration.

    For a bond with continuous coupon payments at rate c and face value 1,
    the price and duration are:

        P = integral_0^T c * exp(-y*t) dt + exp(-y*T)

        D = (1/P) * [integral_0^T t * c * exp(-y*t) dt + T * exp(-y*T)]

    where y is the yield.  Both integrals are computed numerically.

    Parameters
    ----------
    yield_rate  : continuous yield y
    coupon_rate : continuous coupon rate c (as a fraction of face value)
    T           : maturity
    n           : number of subintervals
    """
    if T <= 0.0:
        raise ValueError("maturity T must be positive")
    if n % 2 != 0:
        n += 1

    pv_coupons = simpsons(
        lambda t: coupon_rate * math.exp(-yield_rate * t), 0.0, T, n
    )
    pv_face    = math.exp(-yield_rate * T)
    price      = pv_coupons + pv_face

    dur_coupons = simpsons(
        lambda t: t * coupon_rate * math.exp(-yield_rate * t), 0.0, T, n
    )
    dur_face    = T * pv_face
    duration    = (dur_coupons + dur_face) / price

    return {"price": price, "duration": duration}
