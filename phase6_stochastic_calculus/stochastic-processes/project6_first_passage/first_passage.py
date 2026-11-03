import csv
import math
import random
import statistics
from typing import List, Optional, Dict, Tuple


def standard_normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# Method 1: the discrete two-boundary problem solved as an absorbing
# Markov chain. States -b+1, ..., a-1 are transient; a and -b are
# absorbing. This is the same gambler's-ruin problem Project 1 solved
# by a combinatorial difference equation and Project 5 solved by the
# optional stopping theorem; solving it a third way, by linear algebra
# on the transition matrix, is a genuine cross-check that all three
# very different arguments describe the same object.

def _build_transient_generator(a: int, b: int, p: float) -> Tuple[List[List[float]], List[float], int]:
    """Build the sub-stochastic transition matrix Q over the transient
    states {-b+1, ..., a-1} and the absorption column R giving the
    probability of moving directly to the upper boundary a from each
    transient state (needed to get absorption-side probabilities).
    Returns (Q, r_to_upper, start_index).
    """
    states = list(range(-b + 1, a))
    n = len(states)
    index = {s: i for i, s in enumerate(states)}
    q = [[0.0] * n for _ in range(n)]
    r_upper = [0.0] * n
    for i, s in enumerate(states):
        up = s + 1
        down = s - 1
        if up == a:
            r_upper[i] += p
        else:
            q[i][index[up]] += p
        if down == -b:
            pass  # absorbed at the lower boundary; no need to track that column separately
        else:
            q[i][index[down]] += (1 - p)
    start_index = index[0]
    return q, r_upper, start_index


def solve_linear_system(a: List[List[float]], b: List[float]) -> List[float]:
    """Gaussian elimination with partial pivoting."""
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
            if row == col:
                continue
            factor = aug[row][col]
            if factor != 0.0:
                for j in range(col, n + 1):
                    aug[row][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]


def matrix_vector_multiply(m: List[List[float]], v: List[float]) -> List[float]:
    return [sum(row[j] * v[j] for j in range(len(v))) for row in m]


def absorbing_chain_ruin_probability(a: int, b: int, p: float) -> float:
    """P(hit a before -b), solved as an absorbing Markov chain: h_i,
    the probability of eventual absorption at a starting from
    transient state i, satisfies h = Q h + r_upper, i.e.
    (I - Q) h = r_upper, solved directly rather than iteratively.
    """
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive")
    q, r_upper, start = _build_transient_generator(a, b, p)
    n = len(q)
    lhs = [[(1.0 if i == j else 0.0) - q[i][j] for j in range(n)] for i in range(n)]
    h = solve_linear_system(lhs, r_upper)
    return h[start]


def absorbing_chain_expected_duration(a: int, b: int, p: float) -> float:
    """E[T] starting from 0, via the fundamental matrix identity
    (I - Q) t = 1 (a vector of ones): each transient state's expected
    remaining time is 1 (this step) plus the expectation from wherever
    it moves next, which rearranges to exactly this linear system.
    """
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive")
    q, _, start = _build_transient_generator(a, b, p)
    n = len(q)
    lhs = [[(1.0 if i == j else 0.0) - q[i][j] for j in range(n)] for i in range(n)]
    ones = [1.0] * n
    t = solve_linear_system(lhs, ones)
    return t[start]


def absorbing_chain_survival_curve(a: int, b: int, p: float, n_steps: int) -> List[float]:
    """P(T > n) for n = 0, 1, ..., n_steps, i.e. the probability the
    chain has not yet been absorbed after n steps, computed by
    repeated multiplication of the current state distribution by Q
    (rather than by forming Q^n directly, which is unnecessary here
    since only one starting state is of interest).
    """
    q, _, start = _build_transient_generator(a, b, p)
    n = len(q)
    dist = [0.0] * n
    dist[start] = 1.0
    survival = [sum(dist)]
    for _ in range(n_steps):
        dist = matrix_vector_multiply(q, dist)
        survival.append(sum(dist))
    return survival


# Local simple random walk, self-contained, used for the Monte Carlo
# cross-checks in this project.

def simulate_ruin(a: int, b: int, p: float, seed: Optional[int] = None,
                   max_steps: int = 10 ** 7) -> Tuple[int, int]:
    rng = random.Random(seed)
    position = 0
    for step in range(1, max_steps + 1):
        position += 1 if rng.random() < p else -1
        if position >= a:
            return 1, step
        if position <= -b:
            return 0, step
    raise RuntimeError("did not absorb within max_steps")


# Method 2: the continuous-time double-barrier problem for Brownian
# motion with drift, dX_t = mu dt + sigma dW_t, started at 0, absorbed
# at a > 0 or -b < 0. This is the direct continuous-time analogue of
# Project 5's discrete exponential-martingale argument.

def bm_ruin_probability(a: float, b: float, mu: float, sigma: float) -> float:
    """P(hit a before -b) for drifted Brownian motion.

    M_t = e^{theta X_t} is a martingale exactly when theta*mu +
    0.5*theta^2*sigma^2 = 0 (matching the drift term to zero via Ito's
    lemma), giving theta = -2*mu/sigma^2 as the nontrivial root. The
    optional stopping theorem gives E[M_T] = 1, i.e.
    alpha*e^{theta a} + (1-alpha)*e^{-theta b} = 1, solved for alpha.
    At mu = 0 this degenerates (theta = 0) and X_t itself is used as
    the martingale instead, giving the driftless closed form b/(a+b).
    """
    if a <= 0 or b <= 0 or sigma <= 0:
        raise ValueError("a, b, sigma must be positive")
    if mu == 0:
        return b / (a + b)
    theta = -2.0 * mu / (sigma ** 2)
    return (1.0 - math.exp(-theta * b)) / (math.exp(theta * a) - math.exp(-theta * b))


def bm_expected_duration(a: float, b: float, mu: float, sigma: float) -> float:
    """E[T] for the same problem.

    mu != 0: X_t - mu*t is a martingale, so E[X_T] = mu*E[T].
    mu == 0: X_t^2 - sigma^2*t is a martingale, so E[X_T^2] = sigma^2*E[T].
    """
    if a <= 0 or b <= 0 or sigma <= 0:
        raise ValueError("a, b, sigma must be positive")
    alpha = bm_ruin_probability(a, b, mu, sigma)
    if mu == 0:
        return (alpha * a ** 2 + (1 - alpha) * b ** 2) / (sigma ** 2)
    expected_xt = alpha * a - (1 - alpha) * b
    return expected_xt / mu


def simulate_bm_ruin(a: float, b: float, mu: float, sigma: float, dt: float,
                      seed: Optional[int] = None, max_steps: int = 10 ** 7) -> Tuple[int, float]:
    """Simulate drifted Brownian motion in fixed increments dt until it
    is absorbed at a or -b. Returns (outcome, time), outcome=1 if a was
    hit first.
    """
    rng = random.Random(seed)
    x = 0.0
    scale = sigma * math.sqrt(dt)
    for step in range(1, max_steps + 1):
        x += mu * dt + rng.gauss(0.0, 1.0) * scale
        if x >= a:
            return 1, step * dt
        if x <= -b:
            return 0, step * dt
    raise RuntimeError("did not absorb within max_steps")


# Single-barrier, drift-adjusted reflection principle: the probability
# that drifted Brownian motion ever exceeds a single level a > 0 within
# a finite horizon t, needed for the knock-out barrier application.
# Project 4 covered the mu = 0 case; this extends it.

def bm_max_exceeds(a: float, t: float, mu: float, sigma: float) -> float:
    """P(max_{0<=s<=t} X_s >= a) for X_t = mu*t + sigma*W_t, a > 0.

    Girsanov's theorem (a change of measure that removes the drift at
    the cost of reweighting probabilities by exp(theta*W_t - ...))
    gives the closed form

        P(M_t >= a) = Phi((mu*t - a)/(sigma*sqrt(t)))
                    + exp(2*mu*a/sigma^2) * Phi((-mu*t - a)/(sigma*sqrt(t))).

    Setting mu = 0 collapses this to
    Phi(-a/(sigma*sqrt(t))) + Phi(-a/(sigma*sqrt(t))) = 2*(1-Phi(a/(sigma*sqrt(t)))),
    exactly Project 4's driftless reflection-principle formula, which
    is checked directly as a consistency test.
    """
    if a <= 0 or t <= 0 or sigma <= 0:
        raise ValueError("a, t, sigma must be positive")
    term1 = standard_normal_cdf((mu * t - a) / (sigma * math.sqrt(t)))
    term2 = math.exp(2.0 * mu * a / (sigma ** 2)) * standard_normal_cdf((-mu * t - a) / (sigma * math.sqrt(t)))
    return term1 + term2


# Application: calibrate a driftless/drifted Brownian model to real
# EUR/USD log returns, and compare its stop-loss / take-profit
# predictions against genuine historical simulation.

def load_log_returns(csv_path: str, column: str = "log_return") -> List[float]:
    values = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row[column]))
    return values


def calibrate_gbm(log_returns: List[float]) -> Dict[str, float]:
    """Daily drift and volatility estimated from log returns: with
    X_t = mu*t + sigma*W_t modelling the log-price, the sample mean of
    daily log returns estimates mu and the sample standard deviation
    estimates sigma, both in per-day units.
    """
    mu = statistics.mean(log_returns)
    sigma = statistics.pstdev(log_returns)
    return {"mu": mu, "sigma": sigma}


def historical_stop_target_backtest(log_returns: List[float], target: float, stop: float,
                                     max_horizon: int) -> Dict[str, float]:
    """For every possible entry day in the historical log-return series,
    walk forward along the actual subsequent path (not a simulation)
    and record whether the cumulative log return hits +target before
    -stop within max_horizon days, and how long that took. This is a
    genuine historical (overlapping-window) backtest, not a model.
    """
    n = len(log_returns)
    hits_target = 0
    hits_stop = 0
    unresolved = 0
    durations = []
    for start in range(n - max_horizon):
        cumulative = 0.0
        resolved = False
        for h in range(1, max_horizon + 1):
            cumulative += log_returns[start + h - 1]
            if cumulative >= target:
                hits_target += 1
                durations.append(h)
                resolved = True
                break
            if cumulative <= -stop:
                hits_stop += 1
                durations.append(h)
                resolved = True
                break
        if not resolved:
            unresolved += 1
    n_trials = n - max_horizon
    return {
        "n_trials": n_trials,
        "p_hit_target": hits_target / n_trials,
        "p_hit_stop": hits_stop / n_trials,
        "p_unresolved": unresolved / n_trials,
        "mean_duration_when_resolved": statistics.mean(durations) if durations else float("nan"),
    }


def model_stop_target_prediction(target: float, stop: float, mu: float, sigma: float) -> Dict[str, float]:
    """The Brownian-motion-with-drift model's prediction for the same
    stop-loss/take-profit setup, using the exact two-boundary formulas
    above with a = target, b = stop.
    """
    p_target = bm_ruin_probability(target, stop, mu, sigma)
    expected_duration = bm_expected_duration(target, stop, mu, sigma)
    return {
        "p_hit_target": p_target,
        "p_hit_stop": 1.0 - p_target,
        "expected_duration": expected_duration,
    }
