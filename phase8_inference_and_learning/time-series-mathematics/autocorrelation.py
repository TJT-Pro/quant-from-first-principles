import csv
import math
import statistics
from typing import List, Dict


def autocovariance(x: List[float], lag: int) -> float:
    """Sample autocovariance at `lag`: (1/n) * sum_{t=1}^{n-lag} (x_t - xbar)(x_{t+lag} - xbar).

    Dividing by n (not n-lag) throughout, even though only n-lag terms
    are summed, is the standard convention (not a typo): it is what
    guarantees the resulting sample autocovariance *matrix* (built
    from these values at every lag) is positive semi-definite, which
    matters for every downstream computation here that treats the
    autocovariances as coming from a genuine covariance structure
    (the Yule-Walker equations and the Durbin-Levinson recursion both
    implicitly assume this).
    """
    n = len(x)
    if lag < 0 or lag >= n:
        raise ValueError("lag must be in [0, n)")
    mean = statistics.mean(x)
    total = sum((x[t] - mean) * (x[t + lag] - mean) for t in range(n - lag))
    return total / n


def autocorrelation(x: List[float], lag: int) -> float:
    """rho_k = gamma_k / gamma_0, the autocovariance at lag k
    normalised by the variance (autocovariance at lag 0).
    """
    gamma_0 = autocovariance(x, 0)
    if gamma_0 == 0:
        raise ValueError("series has zero variance")
    return autocovariance(x, lag) / gamma_0


def acf(x: List[float], max_lag: int) -> List[float]:
    """rho_0, rho_1, ..., rho_{max_lag}."""
    return [autocorrelation(x, k) for k in range(max_lag + 1)]


def bartlett_confidence_band(n: int, confidence: float = 0.95) -> float:
    """Under the null hypothesis that x is white noise, the sample
    autocorrelation at any fixed lag k > 0 is asymptotically Normal
    with mean 0 and variance 1/n (Bartlett's formula, in the special
    case relevant to testing individual ACF spikes for significance
    against a white-noise null, rather than the general formula for
    the covariance between two sample autocorrelations of an
    already-autocorrelated process). The two-sided critical value is
    z * sqrt(1/n) for the standard normal quantile z at the requested
    confidence level, computed here via a rational approximation to
    the inverse normal CDF (Acklam's algorithm) so that no scipy
    dependency is needed in the core module.
    """
    z = _inverse_normal_cdf(0.5 + confidence / 2)
    return z / math.sqrt(n)


def _inverse_normal_cdf(p: float) -> float:
    """Acklam's rational approximation to the standard normal inverse
    CDF, accurate to about 1.15e-9. Implemented locally (rather than
    calling into a statistics library) purely to keep the confidence
    band computation self-contained; a bisection search against the
    project's own `standard_normal_cdf`-style construction would work
    just as well but is far slower for the single evaluation needed
    here.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def pacf_durbin_levinson(x: List[float], max_lag: int) -> List[float]:
    """Partial autocorrelations at lags 1..max_lag, via the
    Durbin-Levinson recursion.

    The partial autocorrelation at lag k, phi_kk, is the coefficient
    on X_{t-k} in the best linear predictor of X_t from
    X_{t-1}, ..., X_{t-k}: it measures the correlation between X_t and
    X_{t-k} after removing the linear effect of everything in between.
    Computing it by solving a fresh k-th order Yule-Walker system for
    every k would cost O(k^3) per lag; the Durbin-Levinson recursion
    instead updates the order-(k-1) solution to the order-k solution
    in O(k) work using the identities

        phi_kk = (rho_k - sum_{j=1}^{k-1} phi_{k-1,j} * rho_{k-j}) / (1 - sum_{j=1}^{k-1} phi_{k-1,j} * rho_j)
        phi_kj = phi_{k-1,j} - phi_kk * phi_{k-1,k-j},  j = 1, ..., k-1

    which is also exactly the algorithm used to fit an AR(p) model by
    Yule-Walker: the final row of coefficients phi_{p,1}, ..., phi_{p,p}
    at k=p *is* the Yule-Walker AR(p) coefficient vector, which is why
    `ar_models.fit_yule_walker` calls into this same function.
    """
    if max_lag < 1:
        raise ValueError("max_lag must be at least 1")
    rho = acf(x, max_lag)

    phi = [[0.0] * (max_lag + 1) for _ in range(max_lag + 1)]
    pacf_values = [0.0] * (max_lag + 1)

    phi[1][1] = rho[1]
    pacf_values[1] = rho[1]

    for k in range(2, max_lag + 1):
        numerator = rho[k] - sum(phi[k - 1][j] * rho[k - j] for j in range(1, k))
        denominator = 1.0 - sum(phi[k - 1][j] * rho[j] for j in range(1, k))
        if abs(denominator) < 1e-12:
            raise ValueError(f"Durbin-Levinson recursion is numerically unstable at lag {k}")
        phi_kk = numerator / denominator
        for j in range(1, k):
            phi[k][j] = phi[k - 1][j] - phi_kk * phi[k - 1][k - j]
        phi[k][k] = phi_kk
        pacf_values[k] = phi_kk

    return pacf_values[1:]


def yule_walker_coefficients(x: List[float], order: int) -> List[float]:
    """The AR(order) coefficients phi_1, ..., phi_order solved from the
    Yule-Walker equations, read directly off the final row of the
    Durbin-Levinson recursion at k=order (see `pacf_durbin_levinson`
    for the derivation of why this recursion solves exactly this
    system).
    """
    if order < 1:
        raise ValueError("order must be at least 1")
    rho = acf(x, order)
    phi = [[0.0] * (order + 1) for _ in range(order + 1)]
    phi[1][1] = rho[1]
    for k in range(2, order + 1):
        numerator = rho[k] - sum(phi[k - 1][j] * rho[k - j] for j in range(1, k))
        denominator = 1.0 - sum(phi[k - 1][j] * rho[j] for j in range(1, k))
        if abs(denominator) < 1e-12:
            raise ValueError(f"Yule-Walker system is numerically unstable at order {k}")
        phi_kk = numerator / denominator
        for j in range(1, k):
            phi[k][j] = phi[k - 1][j] - phi_kk * phi[k - 1][k - j]
        phi[k][k] = phi_kk
    return phi[order][1:order + 1]


# Ljung-Box white-noise test: needs the chi-square survival
# function, built from the same regularised incomplete gamma
# construction used elsewhere in this project family, reimplemented
# locally so this project stands on its own.

def _gamma_series(a: float, x: float, itmax: int = 500, eps: float = 3e-14) -> float:
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(itmax):
        ap += 1
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * eps:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_continued_fraction(a: float, x: float, itmax: int = 500,
                               eps: float = 3e-14, fpmin: float = 1e-300) -> float:
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in range(1, itmax + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_sf(x: float, df: int) -> float:
    if x <= 0:
        return 1.0
    a, xx = df / 2.0, x / 2.0
    if xx < a + 1.0:
        return 1.0 - _gamma_series(a, xx)
    return _gamma_continued_fraction(a, xx)


def ljung_box_test(x: List[float], max_lag: int, n_params_fitted: int = 0) -> Dict[str, float]:
    """Tests H0: the series is white noise (no autocorrelation at any
    lag up to max_lag), using the statistic

        Q = n(n+2) * sum_{k=1}^{h} rho_k^2 / (n - k)

    which is asymptotically chi-square with (h - n_params_fitted)
    degrees of freedom under H0. The (n-k) denominator, rather than a
    flat n, is the standard small-sample correction (Ljung & Box,
    1978) to the simpler Box-Pierce statistic, improving the
    chi-square approximation's accuracy at finite n. `n_params_fitted`
    should be set to p+q when testing the *residuals* of a fitted
    ARMA(p,q) model, since fitting parameters uses up degrees of
    freedom that must be subtracted from the nominal lag count.
    """
    n = len(x)
    df = max_lag - n_params_fitted
    if df < 1:
        raise ValueError("degrees of freedom must be at least 1; reduce n_params_fitted or increase max_lag")
    rho = acf(x, max_lag)
    q_stat = n * (n + 2) * sum((rho[k] ** 2) / (n - k) for k in range(1, max_lag + 1))
    p_value = chi2_sf(q_stat, df)
    return {"statistic": q_stat, "df": df, "p_value": p_value}


def load_log_returns(csv_path: str, column: str = "log_return") -> List[float]:
    values = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row[column]))
    return values


def load_prices(csv_path: str, column: str = "price") -> List[float]:
    values = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row[column]))
    return values
