import cmath
import random
import statistics
from typing import List, Optional, Dict

import autocorrelation as ac


def simulate_ar(n: int, c: float, phis: List[float], sigma: float = 1.0,
                 seed: Optional[int] = None, burn_in: int = 200) -> List[float]:
    """Simulate an AR(p) process X_t = c + sum_i phi_i X_{t-i} + eps_t,
    eps_t ~ N(0, sigma^2). A burn-in period is discarded so the
    returned series is close to its stationary distribution rather
    than reflecting the arbitrary zero initial condition, provided the
    process is actually stationary (see `is_stationary`); simulating
    an explosive or unit-root process with this function will simply
    produce a diverging or wandering series, which is itself the
    correct behaviour, not a bug to guard against here.
    """
    rng = random.Random(seed)
    p = len(phis)
    x = [0.0] * p
    for _ in range(n + burn_in):
        value = c + sum(phis[i] * x[-(i + 1)] for i in range(p)) + rng.gauss(0, sigma)
        x.append(value)
    return x[p + burn_in:]


def durand_kerner_roots(coeffs: List[float], max_iter: int = 500, tol: float = 1e-12) -> List[complex]:
    """All roots of the polynomial coeffs[0] + coeffs[1]*z + ... +
    coeffs[n]*z^n = 0, via the Durand-Kerner (Weierstrass) method: n
    initial complex guesses are refined simultaneously by

        z_i <- z_i - p(z_i) / prod_{j != i} (z_i - z_j)

    which is Newton's method applied with the polynomial's leading
    coefficient and the *current* guesses for the other roots standing
    in for the true (unknown) factorisation p(z) = a_n * prod(z - z_j).
    Starting from n points spread around a circle in the complex plane
    (rather than, say, all starting at the same value) guarantees the
    initial guesses are distinct, which the update rule requires to
    avoid division by zero.
    """
    n = len(coeffs) - 1
    if n < 1:
        raise ValueError("polynomial must have degree at least 1")
    leading = coeffs[-1]
    if leading == 0:
        raise ValueError("leading coefficient must be nonzero")

    def poly(z: complex) -> complex:
        result = 0.0 + 0.0j
        for power, a in enumerate(coeffs):
            result += a * z ** power
        return result

    radius = 1.0 + max(abs(c) for c in coeffs) / abs(leading)
    roots = [radius * cmath.exp(2j * cmath.pi * k / n + 0.4j) for k in range(n)]

    for _ in range(max_iter):
        max_shift = 0.0
        new_roots = list(roots)
        for i in range(n):
            denom = leading
            for j in range(n):
                if j != i:
                    denom *= (roots[i] - roots[j])
            if abs(denom) < 1e-300:
                continue
            shift = poly(roots[i]) / denom
            new_roots[i] = roots[i] - shift
            max_shift = max(max_shift, abs(shift))
        roots = new_roots
        if max_shift < tol:
            break
    return roots


def is_stationary(phis: List[float]) -> bool:
    """An AR(p) process is (weakly) stationary iff every root of the
    characteristic polynomial 1 - phi_1*z - phi_2*z^2 - ... - phi_p*z^p
    lies strictly outside the unit circle in the complex plane. For
    p=1 this is just |phi_1| < 1; for general p, the roots are found
    numerically via `durand_kerner_roots` rather than relying on a
    closed-form condition, which only exists in simple, explicit terms
    for p <= 2.
    """
    if not phis:
        raise ValueError("phis must be non-empty")
    if len(phis) == 1:
        return abs(phis[0]) < 1.0
    coeffs = [1.0] + [-phi for phi in phis]
    roots = durand_kerner_roots(coeffs)
    return all(abs(r) > 1.0 for r in roots)


def fit_yule_walker(x: List[float], order: int) -> Dict[str, float]:
    """Estimate an AR(order) model by the Yule-Walker method: the
    autoregressive coefficients solve the Yule-Walker equations (see
    `autocorrelation.yule_walker_coefficients`), the intercept follows
    from the mean of a stationary AR process,
    E[X_t] = c / (1 - sum phi_i), rearranged to
    c = mean * (1 - sum phi_i), and the innovation variance follows
    from the same equations' implied residual variance,
    sigma^2 = gamma_0 * (1 - sum_i phi_i * rho_i).
    """
    phis = ac.yule_walker_coefficients(x, order)
    mean = statistics.mean(x)
    c = mean * (1.0 - sum(phis))
    gamma0 = ac.autocovariance(x, 0)
    rho = ac.acf(x, order)
    sigma2 = gamma0 * (1.0 - sum(phis[i] * rho[i + 1] for i in range(order)))
    return {"c": c, "phis": phis, "sigma2": max(sigma2, 0.0)}


def fit_ols(x: List[float], order: int) -> Dict[str, float]:
    """Estimate an AR(order) model by ordinary least squares: regress
    X_t on X_{t-1}, ..., X_{t-order} and a constant, using the
    observations actually available (starting from t=order). This is
    the conditional maximum-likelihood estimator under Gaussian
    innovations, conditioning on the first `order` observations rather
    than modelling their (stationary, but more complicated) marginal
    distribution the way an exact likelihood would; the two methods
    coincide as the sample size grows, and are compared directly in
    the test suite and in `verify` to check they roughly agree, not
    merely trusted to.
    """
    n = len(x)
    if n <= order:
        raise ValueError("series too short for the requested order")

    y = x[order:]
    design = [[1.0] + [x[t - 1 - i] for i in range(order)] for t in range(order, n)]

    m = len(design)
    k = order + 1
    xtx = [[sum(design[row][a] * design[row][b] for row in range(m)) for b in range(k)] for a in range(k)]
    xty = [sum(design[row][a] * y[row] for row in range(m)) for a in range(k)]

    beta = _solve_linear_system(xtx, xty)
    c = beta[0]
    phis = beta[1:]

    residuals = [y[t] - (c + sum(phis[i] * design[t][i + 1] for i in range(order))) for t in range(m)]
    sigma2 = sum(e ** 2 for e in residuals) / (m - k)
    return {"c": c, "phis": phis, "sigma2": sigma2, "residuals": residuals}


def _solve_linear_system(a: List[List[float]], b: List[float]) -> List[float]:
    n = len(a)
    aug = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-14:
            raise ValueError("design matrix is singular or nearly singular")
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot
        for row in range(n):
            if row != col and aug[row][col] != 0.0:
                factor = aug[row][col]
                for j in range(col, n + 1):
                    aug[row][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]


def forecast_ar(x: List[float], c: float, phis: List[float], steps: int) -> List[float]:
    """Recursive point forecasts: each future value's forecast plugs
    in the forecasts (not the unobserved true values) for any lagged
    term that falls beyond the end of the observed data, which is the
    standard definition of a multi-step-ahead point forecast for an AR
    process.
    """
    order = len(phis)
    history = list(x[-order:]) if order > 0 else []
    forecasts = []
    for _ in range(steps):
        value = c + sum(phis[i] * history[-(i + 1)] for i in range(order))
        forecasts.append(value)
        history.append(value)
    return forecasts
