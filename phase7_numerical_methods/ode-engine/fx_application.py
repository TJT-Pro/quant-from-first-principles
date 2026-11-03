import csv
import math
import statistics
from typing import List, Dict

import ode_solvers as ode


def load_log_returns(csv_path: str, column: str = "log_return") -> List[float]:
    values = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row[column]))
    return values


def autocorrelation(x: List[float], lag: int) -> float:
    mean = statistics.mean(x)
    n = len(x)
    numer = sum((x[t] - mean) * (x[t + lag] - mean) for t in range(n - lag))
    denom = sum((v - mean) ** 2 for v in x)
    return numer / denom


def calibrate_mean_reversion(squared_returns: List[float], max_lag: int) -> Dict:
    """Calibrate a deterministic mean-reversion ODE dV/dt = kappa*(theta-V)
    to the volatility persistence actually observed in the data: theta
    is the unconditional (long-run) variance, and kappa is fit from
    the autocorrelation function of squared returns. If squared
    returns followed a simple AR(1)-in-variance (equivalently, a
    GARCH(1,1)/Ornstein-Uhlenbeck-style single-timescale) process, the
    ACF at lag k would decay exactly exponentially, rho_k = rho_1^k,
    so ln(rho_k) would be exactly linear in k with slope -kappa (up to
    a units conversion between the discrete lag index and continuous
    time, treated here as one trading day per lag). Fitting that line
    by ordinary least squares over `max_lag` lags gives kappa directly,
    along with an R^2 diagnosing how well a *single* exponential
    actually describes the observed decay (see the README for why,
    empirically, it only partially does).
    """
    theta = statistics.mean(squared_returns)
    lags = list(range(1, max_lag + 1))
    acfs = [autocorrelation(squared_returns, k) for k in lags]
    points = [(k, a) for k, a in zip(lags, acfs) if a > 0]
    if len(points) < 2:
        raise ValueError("not enough positive-autocorrelation lags to fit a decay rate")

    xs = [p[0] for p in points]
    ys = [math.log(p[1]) for p in points]
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    slope = num / den
    intercept = mean_y - slope * mean_x
    kappa = -slope

    y_pred = [intercept + slope * x for x in xs]
    ss_res = sum((y - yp) ** 2 for y, yp in zip(ys, y_pred))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "theta": theta, "kappa": kappa, "r_squared": r_squared,
        "n_lags_used": len(points), "max_lag": max_lag,
        "half_life_days": math.log(2) / kappa if kappa > 0 else float("inf"),
    }


def simulate_volatility_shock(theta: float, kappa: float, v0: float, t_end: float,
                               h: float = 0.5) -> Dict:
    """Solve dV/dt = kappa*(theta - V), V(0) = v0 with every fixed-step
    method plus the adaptive solver, and compare each against the
    exact analytic solution V(t) = theta + (v0-theta)*exp(-kappa*t)
    (obtained by separating variables: dV/(theta-V) = kappa*dt
    integrates directly to -ln|theta-V| = kappa*t + C). This is a
    genuine cross-check with a known closed form, not merely a
    demonstration, and reports each method's final-time error.
    """
    f = lambda t, v: kappa * (theta - v)
    exact = lambda t: theta + (v0 - theta) * math.exp(-kappa * t)

    results = {}
    for method in ("euler", "improved_euler", "midpoint", "rk4"):
        sol = ode.solve_ivp(f, 0.0, v0, t_end, h, method)
        results[method] = {"final_value": sol["y"][-1], "error": abs(sol["y"][-1] - exact(t_end))}

    adaptive = ode.solve_ivp_adaptive(f, 0.0, v0, t_end, h_init=h, tol=1e-10, method="rk4")
    results["adaptive_rk4"] = {
        "final_value": adaptive["y"][-1],
        "error": abs(adaptive["y"][-1] - exact(t_end)),
        "n_steps": len(adaptive["t"]) - 1,
    }
    results["exact_final_value"] = exact(t_end)

    if v0 != theta:
        results["days_to_within_5pct"] = -math.log(0.05 * abs(theta) / abs(v0 - theta)) / kappa \
            if abs(v0 - theta) > 0 else 0.0
    return results


def two_state_generator_from_persistence(p_stay_low: float, p_stay_high: float) -> List[List[float]]:
    """Build a continuous-time generator matrix Q for a 2-state chain
    from daily discrete self-transition probabilities, via the
    small-interval approximation P(dt) ~ I + Q*dt for dt=1 (one
    trading day): matching P_ii ~ 1 + Q_ii gives Q_ii = -(1-P_ii), and
    each row of a generator matrix must sum to zero (probability
    leaving a state must equal probability arriving elsewhere), which
    fixes the single off-diagonal entry in each row of a 2-state chain
    directly. This is an approximation (exact only in the dt->0 limit;
    here used at dt=1 as a reasonable, simple way to get a plausible,
    real-world-motivated generator without solving the full matrix
    logarithm problem), used purely to build a demonstration system
    for the Kolmogorov forward equations below.
    """
    q11 = -(1 - p_stay_low)
    q22 = -(1 - p_stay_high)
    return [[q11, -q11], [-q22, q22]]


def kolmogorov_forward_rhs(q: List[List[float]]):
    """The Kolmogorov forward equation dp/dt = p*Q for a continuous-
    time Markov chain with generator Q: the rate of change of the
    probability of being in state j is the net flow of probability
    into j from every state (including the flow "out of j itself",
    captured by Q_jj being negative). Returned as a function usable
    directly as the right-hand side of `ode_solvers.solve_ivp`.
    """
    n = len(q)

    def rhs(t, p):
        return [sum(p[i] * q[i][j] for i in range(n)) for j in range(n)]

    return rhs


def stationary_distribution_generator(q: List[List[float]]) -> List[float]:
    """The stationary distribution of a continuous-time chain with
    generator Q solves pi*Q = 0, sum(pi) = 1: replace one (redundant,
    since Q's rows sum to zero) balance equation with the
    normalisation constraint and solve the resulting linear system
    directly, the continuous-time analogue of the discrete-time
    stationary-distribution calculation used elsewhere in this
    project family.
    """
    n = len(q)
    qt = [[q[j][i] for j in range(n)] for i in range(n)]
    a = [row[:] for row in qt]
    a[-1] = [1.0] * n
    b = [0.0] * (n - 1) + [1.0]
    return _solve_linear_system(a, b)


def _solve_linear_system(a: List[List[float]], b: List[float]) -> List[float]:
    n = len(a)
    aug = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-14:
            raise ValueError("singular system")
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
