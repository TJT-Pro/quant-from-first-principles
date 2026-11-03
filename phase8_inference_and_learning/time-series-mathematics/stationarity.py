import random
import statistics
from typing import List, Optional, Dict

from ar_models import _solve_linear_system


def _difference(x: List[float]) -> List[float]:
    return [x[t] - x[t - 1] for t in range(1, len(x))]


def _adf_regression(x: List[float], n_lags: int) -> Dict[str, float]:
    """Run the augmented Dickey-Fuller test regression

        Delta(x_t) = alpha + gamma * x_{t-1} + sum_i delta_i * Delta(x_{t-i}) + e_t

    by ordinary least squares, and return gamma's coefficient and its
    standard error. The null hypothesis of a unit root is
    H0: gamma = 0 (x is I(1), a random walk); the alternative is
    H1: gamma < 0 (x is stationary, reverting toward its mean at rate
    |gamma|). `n_lags` augmenting difference terms are included to
    absorb any remaining short-run autocorrelation in the residuals,
    which is what distinguishes the "augmented" test from the original
    Dickey-Fuller test (n_lags=0).
    """
    dx = _difference(x)
    n = len(dx)
    if n <= n_lags + 2:
        raise ValueError("series too short for the requested number of lags")

    y = dx[n_lags:]
    rows = []
    for t in range(n_lags, n):
        row = [1.0, x[t]]
        for i in range(1, n_lags + 1):
            row.append(dx[t - i])
        rows.append(row)

    m = len(rows)
    k = 2 + n_lags
    xtx = [[sum(rows[r][a] * rows[r][b] for r in range(m)) for b in range(k)] for a in range(k)]
    xty = [sum(rows[r][a] * y[r] for r in range(m)) for a in range(k)]
    beta = _solve_linear_system(xtx, xty)

    residuals = [y[r] - sum(beta[j] * rows[r][j] for j in range(k)) for r in range(m)]
    sigma2 = sum(e ** 2 for e in residuals) / (m - k)

    xtx_inv_diag = _matrix_inverse_diagonal(xtx)
    gamma = beta[1]
    se_gamma = (sigma2 * xtx_inv_diag[1]) ** 0.5

    return {"gamma": gamma, "se_gamma": se_gamma, "t_stat": gamma / se_gamma}


def _matrix_inverse_diagonal(a: List[List[float]]) -> List[float]:
    """The diagonal of A^{-1}, obtained by solving A*e_i = x_i for each
    standard basis vector e_i and keeping only the i-th entry of each
    solution; used here only to get Var(beta_hat)'s diagonal, so
    computing the full inverse would do unnecessary extra work.
    """
    n = len(a)
    diag = []
    for i in range(n):
        e_i = [1.0 if j == i else 0.0 for j in range(n)]
        solution = _solve_linear_system([row[:] for row in a], e_i)
        diag.append(solution[i])
    return diag


def adf_statistic(x: List[float], n_lags: int = 0) -> float:
    return _adf_regression(x, n_lags)["t_stat"]


def simulate_adf_null_distribution(n: int, n_trials: int, n_lags: int = 0,
                                    seed: Optional[int] = None) -> List[float]:
    """Monte Carlo the null distribution of the ADF t-statistic: the
    statistic's exact finite-sample distribution under H0 (a pure
    random walk with no drift) is famously non-standard (not a
    t-distribution, not even asymptotically normal), which is exactly
    why the original Dickey-Fuller critical value tables were
    themselves built from simulation rather than derived in closed
    form. Simulating the null directly, at the *same sample size* as
    the data being tested (critical: the distribution is markedly
    different at n=50 than at n=5000), is the honest way to get a
    valid critical value or p-value here without importing a
    pre-computed table from an external statistics library.
    """
    rng = random.Random(seed)
    stats = []
    for _ in range(n_trials):
        walk = [0.0]
        for _ in range(n - 1):
            walk.append(walk[-1] + rng.gauss(0, 1))
        stats.append(adf_statistic(walk, n_lags))
    return stats


def adf_test(x: List[float], n_lags: int = 0, n_trials: int = 2000,
             seed: Optional[int] = None) -> Dict[str, float]:
    """The full ADF test: compute the observed t-statistic, simulate
    its null distribution at the same sample size, and report the
    one-sided p-value (the null is rejected, in favour of stationarity,
    for sufficiently *negative* t-statistics, so the p-value is the
    fraction of simulated null statistics at or below the observed
    one).
    """
    observed = adf_statistic(x, n_lags)
    null_stats = simulate_adf_null_distribution(len(x), n_trials, n_lags, seed)
    p_value = sum(1 for s in null_stats if s <= observed) / n_trials
    return {
        "statistic": observed,
        "p_value": p_value,
        "n_trials": n_trials,
        "critical_value_5pct": sorted(null_stats)[int(0.05 * n_trials)],
    }


def rolling_mean_variance(x: List[float], window: int) -> Dict[str, List[float]]:
    """Rolling mean and variance over non-overlapping windows, the
    simplest possible visual/numerical check of the two properties a
    stationary series must have that a non-stationary one typically
    does not: a roughly constant mean and a roughly constant variance
    across different stretches of the sample.
    """
    if window <= 0 or window > len(x):
        raise ValueError("window must be positive and no larger than the series")
    means, variances = [], []
    for start in range(0, len(x) - window + 1, window):
        segment = x[start:start + window]
        means.append(statistics.mean(segment))
        variances.append(statistics.variance(segment))
    return {"means": means, "variances": variances}
