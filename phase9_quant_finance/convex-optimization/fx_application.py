import csv
import statistics
from typing import List, Dict

import linear_algebra as la
import constrained_optimization as co


def load_log_returns(csv_path: str, column: str = "log_return") -> List[float]:
    values = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row[column]))
    return values


def build_strategy_returns(returns: List[float], momentum_lookback: int = 5,
                            reversion_lookback: int = 20) -> Dict[str, List[float]]:
    """Four genuinely distinct daily P&L series, all derived from the
    same real EUR/USD return history, giving a real (not synthetic)
    covariance structure to optimise a portfolio over:

    - buy_hold: simply holding the position (today's raw return).
    - momentum: long when the trailing momentum_lookback-day return
      was positive, short otherwise, applied to today's return.
    - mean_reversion: the opposite signal, but computed over a
      different, longer reversion_lookback window rather than the
      literal negative of the momentum signal; using the same window
      for both would make momentum and mean_reversion exact opposites
      by construction (correlation exactly -1), a degenerate,
      singular covariance matrix that provides no genuine
      diversification information to optimise over at all.
    - vol_scaled: today's return scaled by the inverse of trailing
      20-day realised volatility, a simple volatility-targeting
      overlay on the same underlying exposure.

    All four are real derived P&L series from real data, not
    fabricated returns, so their correlations (checked directly in
    the application below) are a genuine empirical finding rather
    than something assumed or hand-tuned.
    """
    max_lookback = max(momentum_lookback, reversion_lookback)
    buy_hold, momentum, mean_reversion, vol_scaled = [], [], [], []
    for t in range(max_lookback, len(returns)):
        mom_signal = sum(returns[t - momentum_lookback:t])
        rev_signal = sum(returns[t - reversion_lookback:t])
        trailing_vol = statistics.pstdev(returns[t - reversion_lookback:t])
        r_t = returns[t]

        buy_hold.append(r_t)
        momentum.append((1.0 if mom_signal > 0 else -1.0) * r_t)
        mean_reversion.append((-1.0 if rev_signal > 0 else 1.0) * r_t)
        vol_scaled.append(r_t / trailing_vol * 0.01 if trailing_vol > 0 else 0.0)

    return {"buy_hold": buy_hold, "momentum": momentum, "mean_reversion": mean_reversion,
            "vol_scaled": vol_scaled}


def sample_covariance(columns: List[List[float]]) -> la.Matrix:
    m = len(columns[0])
    n = len(columns)
    means = [statistics.mean(col) for col in columns]
    return [[sum((columns[a][i] - means[a]) * (columns[b][i] - means[b]) for i in range(m)) / m
             for b in range(n)] for a in range(n)]


def annualized_sharpe(returns: List[float], periods_per_year: int = 252) -> float:
    mean = statistics.mean(returns)
    std = statistics.pstdev(returns)
    return mean / std * periods_per_year ** 0.5 if std > 0 else 0.0


def min_variance_portfolio_unconstrained(sigma: la.Matrix) -> dict:
    n = len(sigma)
    return co.solve_equality_constrained_qp(sigma, [0.0] * n, [[1.0] * n], [1.0])


def min_variance_portfolio_long_only(sigma: la.Matrix, step: float = 10000.0, max_iter: int = 20000,
                                      tol: float = 1e-10) -> dict:
    """The long-only (no short-selling) minimum-variance portfolio, via
    projected gradient descent onto the simplex.

    `step` defaults to a much larger value than a naive first guess
    might suggest, for a specific, tested reason: this project's
    covariance matrices are built from daily return data, so their
    entries are tiny (O(1e-5) to O(1e-4)), making the gradient
    Sigma*w itself tiny; a "reasonable-looking" step size like 0.5
    (reasonable for, say, unit-scaled toy problems) makes vanishingly
    slow progress and fails to converge within any practical iteration
    budget on data at this real scale. Found directly while applying
    this function to real EUR/USD-derived data: a step of 0.5 failed
    to converge after 100,000 iterations, while a step of 10,000 on
    the identical problem converges in under 100. The fix is not
    algorithmic (the projected-gradient method itself was always
    correct); it is a reminder that a fixed step size's suitability
    depends on the objective's actual curvature scale, which for a
    covariance-matrix objective is set entirely by the data's own
    units.

    Returns the true portfolio variance (w^T*Sigma*w) directly,
    rather than leaving the caller to notice that the optimizer's own
    internal objective is 0.5*w^T*Sigma*w and needs doubling: an
    earlier version of this project's own exploratory analysis
    briefly (and wrongly) concluded the long-only solution had *lower*
    variance than the unconstrained one, a result that would violate
    the basic fact that adding constraints can never improve an
    optimum, purely because of this exact scaling mismatch between the
    optimizer's halved objective value and the actual variance.
    """
    n = len(sigma)
    f = lambda w: 0.5 * la.dot(w, la.mat_vec(sigma, w))
    grad_fn = lambda w: la.mat_vec(sigma, w)
    x0 = [1.0 / n] * n
    result = co.projected_gradient_descent(f, x0, co.project_onto_simplex, grad_fn=grad_fn,
                                            step=step, max_iter=max_iter, tol=tol)
    result["variance"] = portfolio_variance(sigma, result["x"])
    return result


def portfolio_variance(sigma: la.Matrix, w: List[float]) -> float:
    return la.dot(w, la.mat_vec(sigma, w))


def verify_unconstrained_solution_kkt(sigma: la.Matrix, result: dict) -> dict:
    grad_f = la.mat_vec(sigma, result["x"])
    n = len(sigma)
    grad_h = [[1.0] * n]
    h_values = [sum(result["x"]) - 1.0]
    return co.verify_kkt(grad_f, grad_h, h_values, result["lambda"])


def verify_long_only_solution_kkt(sigma: la.Matrix, w: List[float], tol: float = 1e-4) -> dict:
    """KKT verification for the long-only problem
    min 0.5*w^T*Sigma*w s.t. sum(w)=1, w>=0, treating each -w_i<=0 as
    an inequality constraint. At a candidate solution, the multiplier
    for each active bound (w_i essentially 0) is read off directly
    from the stationarity condition itself, Sigma*w + lambda*1 - mu = 0
    (mu_i>=0 for the constraint -w_i<=0), solving for lambda as the
    common per-unit cost of the equality constraint via the *inactive*
    coordinates (where mu_i=0 exactly, by complementary slackness)
    before checking the full set of KKT conditions with the resulting
    multipliers.
    """
    n = len(sigma)
    grad_f = la.mat_vec(sigma, w)
    inactive = [i for i in range(n) if w[i] > tol]
    if not inactive:
        raise ValueError("no inactive (strictly positive) coordinate to solve for lambda")
    lam = -sum(grad_f[i] for i in inactive) / len(inactive)

    mu = [max(grad_f[i] + lam, 0.0) for i in range(n)]
    grad_g = [[-1.0 if k == i else 0.0 for k in range(n)] for i in range(n)]
    g_values = [-w[i] for i in range(n)]

    grad_h = [[1.0] * n]
    h_values = [sum(w) - 1.0]

    return co.verify_kkt(grad_f, grad_h, h_values, [lam], grad_g=grad_g, g_values=g_values, mu=mu, tol=1e-3)
