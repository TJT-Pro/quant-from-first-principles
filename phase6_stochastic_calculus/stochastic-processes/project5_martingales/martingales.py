import csv
import math
import random
import statistics
from typing import List, Optional, Dict, Tuple, Callable


# Standard normal CDF, self-contained (same construction as Project 4,
# reimplemented here so this project does not depend on another one).

def standard_normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# A local simple random walk, the building block for every martingale
# construction below.

def simulate_walk(n_steps: int, p: float = 0.5, seed: Optional[int] = None) -> List[int]:
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    rng = random.Random(seed)
    path = [0]
    position = 0
    for _ in range(n_steps):
        position += 1 if rng.random() < p else -1
        path.append(position)
    return path


# Empirically testing the martingale property E[X_{n+1} - X_n | X_n] = 0
# by bucketing many simulated (or observed) transitions by the current
# value and averaging the next increment within each bucket.

def empirical_conditional_drift(xs: List[float], dxs: List[float],
                                 n_buckets: int = 10) -> List[Dict[str, float]]:
    """Split (xs, dxs) pairs into n_buckets equal-frequency groups by
    the value of x, and report the mean of dx within each group. For a
    martingale, dx = X_{n+1} - X_n given X_n = x should average to zero
    regardless of the bucket; systematic departure from zero in one
    direction is direct evidence against the martingale property.
    """
    if len(xs) != len(dxs):
        raise ValueError("xs and dxs must have the same length")
    if n_buckets < 1:
        raise ValueError("n_buckets must be at least 1")

    order = sorted(range(len(xs)), key=lambda i: xs[i])
    n = len(xs)
    bucket_size = max(1, n // n_buckets)
    results = []
    for b in range(n_buckets):
        start = b * bucket_size
        end = n if b == n_buckets - 1 else min(n, start + bucket_size)
        if start >= end:
            continue
        idx = order[start:end]
        bucket_xs = [xs[i] for i in idx]
        bucket_dxs = [dxs[i] for i in idx]
        results.append({
            "mean_x": statistics.mean(bucket_xs),
            "mean_dx": statistics.mean(bucket_dxs),
            "count": len(idx),
        })
    return results


# Martingale constructions from a (possibly biased) simple random walk.

def compensated_walk(n_steps: int, p: float, seed: Optional[int] = None) -> List[float]:
    """M_n = S_n - n*(2p-1). Subtracting the linear drift restores the
    martingale property for a biased walk: E[M_{n+1} - M_n | F_n]
    = E[S_{n+1} - S_n] - (2p-1) = (2p-1) - (2p-1) = 0.
    """
    path = simulate_walk(n_steps, p=p, seed=seed)
    mu = 2 * p - 1
    return [path[n] - n * mu for n in range(len(path))]


def exponential_martingale_path(n_steps: int, p: float, seed: Optional[int] = None) -> List[float]:
    """M_n = ((1-p)/p)^{S_n}, Wald's exponential martingale.

    For a step X taking value +1 w.p. p and -1 w.p. 1-p, and theta such
    that E[theta^X] = 1, the process theta^{S_n} is a martingale. Here
    theta = (1-p)/p satisfies p*theta + (1-p)/theta = p*((1-p)/p) +
    (1-p)*(p/(1-p)) = (1-p) + p = 1 exactly, so E[theta^{S_{n+1}} | F_n]
    = theta^{S_n} * E[theta^{X_{n+1}}] = theta^{S_n} * 1 = theta^{S_n}.
    Degenerates to the constant sequence 1 when p = 0.5 (theta = 1),
    which is why the symmetric case needs `compensated_walk` instead.
    """
    if p == 0.5:
        raise ValueError("the exponential martingale is degenerate at p=0.5; use compensated_walk")
    theta = (1 - p) / p
    path = simulate_walk(n_steps, p=p, seed=seed)
    return [theta ** s for s in path]


# Optional stopping applied to two-boundary gambler's ruin: the walk
# starts at 0 and stops the first time it hits a > 0 or -b < 0.

def ruin_probability(a: int, b: int, p: float) -> float:
    """P(hit a before -b), derived from the optional stopping theorem.

    Symmetric case (p = 0.5): S_n itself is a martingale, so OST gives
    E[S_T] = 0 = alpha*a - (1-alpha)*b, hence alpha = b / (a + b).

    Biased case: the exponential martingale M_n = theta^{S_n} with
    theta = (1-p)/p gives E[M_T] = M_0 = 1, so
    alpha*theta^a + (1-alpha)*theta^{-b} = 1, solved directly for alpha.
    """
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive")
    if p == 0.5:
        return b / (a + b)
    theta = (1 - p) / p
    return (1.0 - theta ** (-b)) / (theta ** a - theta ** (-b))


def expected_duration(a: int, b: int, p: float) -> float:
    """E[T], the expected number of steps until the walk hits a or -b.

    Symmetric case: S_n^2 - n is a martingale (since E[S_{n+1}^2 | F_n]
    = S_n^2 + 1 for a +-1 step of variance 1), so OST gives
    E[S_T^2] = E[T]. With alpha = P(hit a first) = b/(a+b),
    E[S_T^2] = alpha*a^2 + (1-alpha)*b^2, which simplifies to a*b.

    Biased case: S_n - n*mu is a martingale (mu = 2p-1), so
    E[S_T] = mu*E[T]. With alpha as returned by `ruin_probability`,
    E[S_T] = alpha*a - (1-alpha)*b, giving E[T] = E[S_T] / mu.
    """
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive")
    alpha = ruin_probability(a, b, p)
    if p == 0.5:
        return float(a * b)
    mu = 2 * p - 1
    expected_st = alpha * a - (1 - alpha) * b
    return expected_st / mu


def simulate_ruin(a: int, b: int, p: float, seed: Optional[int] = None,
                   max_steps: int = 10 ** 7) -> Tuple[int, int]:
    """Simulate a single run of the two-boundary gambler's ruin problem.
    Returns (outcome, duration) where outcome is 1 if a was hit first,
    0 if -b was hit first.
    """
    rng = random.Random(seed)
    position = 0
    for step in range(1, max_steps + 1):
        position += 1 if rng.random() < p else -1
        if position >= a:
            return 1, step
        if position <= -b:
            return 0, step
    raise RuntimeError("gambler's ruin did not terminate within max_steps")


# Doob's maximal inequality for a non-negative submartingale.

def doob_maximal_bound(n_steps: int, level: float) -> float:
    """Doob's maximal inequality for the submartingale M_n = S_n^2 of a
    symmetric simple random walk: P(max_{k<=n} M_k >= level) <=
    E[M_n] / level = n / level, since E[S_n^2] = n exactly and S_n^2 is
    a submartingale (a convex function, x -> x^2, applied to the
    martingale S_n, by Jensen's inequality applied conditionally).
    """
    if level <= 0:
        raise ValueError("level must be positive")
    return n_steps / level


def doob_maximal_inequality_check(n_steps: int, level: float, trials: int,
                                   seed: Optional[int] = None) -> Dict[str, float]:
    """Empirical P(max_{k<=n} S_k^2 >= level) against Doob's bound."""
    rng = random.Random(seed)
    hits = 0
    for i in range(trials):
        path = simulate_walk(n_steps, p=0.5, seed=rng.randrange(2 ** 31))
        if max(s ** 2 for s in path) >= level:
            hits += 1
    empirical = hits / trials
    bound = doob_maximal_bound(n_steps, level)
    return {"empirical_probability": empirical, "doob_bound": bound, "bound_holds": empirical <= bound}


# The Polya urn: a bounded martingale with a known, non-degenerate
# almost-sure limit, used to demonstrate Doob's martingale convergence
# theorem concretely.

def polya_urn_path(r0: int, b0: int, n_draws: int, seed: Optional[int] = None) -> List[float]:
    """Fraction of red balls after each draw, starting from r0 red and
    b0 blue balls, where each draw returns the drawn ball plus one more
    of the same colour. X_n = R_n / (R_n + B_n) is a martingale:
    E[X_{n+1} | F_n] = X_n * (R_n+1)/(R_n+B_n+1) + (1-X_n) * R_n/(R_n+B_n+1)
    simplifies to X_n exactly, since drawing red raises both the count
    and the total by amounts that exactly preserve the expected next
    fraction.
    """
    if r0 <= 0 or b0 <= 0 or n_draws < 0:
        raise ValueError("r0, b0 must be positive and n_draws non-negative")
    rng = random.Random(seed)
    r, b = r0, b0
    fractions = [r / (r + b)]
    for _ in range(n_draws):
        if rng.random() < r / (r + b):
            r += 1
        else:
            b += 1
        fractions.append(r / (r + b))
    return fractions


def polya_urn_terminal_moments(r0: int, b0: int) -> Dict[str, float]:
    """The classical result: the a.s. limit of the Polya urn fraction
    is Beta(r0, b0) distributed, with mean r0/(r0+b0) and variance
    r0*b0 / [(r0+b0)^2 * (r0+b0+1)].
    """
    total = r0 + b0
    mean = r0 / total
    variance = (r0 * b0) / (total ** 2 * (total + 1))
    return {"mean": mean, "variance": variance}


# Ordinary least squares from scratch, used to test the martingale
# difference hypothesis (does yesterday's return predict today's?).
# Both a homoskedastic and a heteroskedasticity-robust (White/HC0)
# standard error are computed, since Projects 2-4 already established
# that EUR/USD returns are conditionally heteroskedastic.

def ols_fit(x: List[float], y: List[float]) -> Dict[str, float]:
    n = len(x)
    if n != len(y) or n < 3:
        raise ValueError("x and y must have equal length >= 3")
    x_bar = statistics.mean(x)
    y_bar = statistics.mean(y)
    sxx = sum((xi - x_bar) ** 2 for xi in x)
    if sxx == 0:
        raise ValueError("x has zero variance")
    sxy = sum((xi - x_bar) * (yi - y_bar) for xi, yi in zip(x, y))

    beta1 = sxy / sxx
    beta0 = y_bar - beta1 * x_bar
    residuals = [yi - (beta0 + beta1 * xi) for xi, yi in zip(x, y)]
    sse = sum(e ** 2 for e in residuals)
    syy = sum((yi - y_bar) ** 2 for yi in y)
    r_squared = 1.0 - sse / syy if syy > 0 else float("nan")

    sigma2_hat = sse / (n - 2)
    se_homoskedastic = math.sqrt(sigma2_hat / sxx) if sigma2_hat > 0 else 0.0

    robust_var = sum(((xi - x_bar) ** 2) * (e ** 2) for xi, e in zip(x, residuals)) / (sxx ** 2)
    se_robust = math.sqrt(robust_var) if robust_var > 0 else 0.0

    if se_homoskedastic > 0:
        z_homo = beta1 / se_homoskedastic
        p_homo = 2.0 * (1.0 - standard_normal_cdf(abs(z_homo)))
    else:
        # a perfect (zero-residual) fit: the slope is exactly determined
        # by the data, so treat any nonzero beta1 as infinitely significant
        z_homo = math.inf if beta1 != 0 else 0.0
        p_homo = 0.0 if beta1 != 0 else 1.0

    if se_robust > 0:
        z_robust = beta1 / se_robust
        p_robust = 2.0 * (1.0 - standard_normal_cdf(abs(z_robust)))
    else:
        z_robust = math.inf if beta1 != 0 else 0.0
        p_robust = 0.0 if beta1 != 0 else 1.0

    return {
        "n": n,
        "beta0": beta0,
        "beta1": beta1,
        "r_squared": r_squared,
        "se_homoskedastic": se_homoskedastic,
        "se_robust": se_robust,
        "z_homoskedastic": z_homo,
        "z_robust": z_robust,
        "p_homoskedastic": p_homo,
        "p_robust": p_robust,
    }


def martingale_difference_test(returns: List[float]) -> Dict[str, float]:
    """Regress r_t on r_{t-1}. The martingale difference hypothesis
    E[r_t | F_{t-1}] = 0 implies the slope should be statistically
    indistinguishable from zero: if lagged returns carried information
    about the conditional mean of the next return, the martingale
    property (for the return-generating process, not the price level)
    would fail.
    """
    if len(returns) < 4:
        raise ValueError("need at least 4 observations")
    x = returns[:-1]
    y = returns[1:]
    return ols_fit(x, y)


# The classic doubling ("martingale") betting system, used to
# demonstrate what the optional stopping theorem actually requires:
# a bounded stopping time or bounded stakes. Remove either bound and
# the "guaranteed win" intuition breaks down exactly where the theorem
# stops applying.

def simulate_doubling_strategy(outcomes: List[int], initial_stake: float,
                                max_doublings: Optional[int]) -> Dict[str, float]:
    """Play the doubling system against a fixed sequence of +1/-1
    outcomes (a win doubles nothing and resets the stake; a loss
    doubles the next stake). If max_doublings is given, betting stops
    (forced loss realised, no further recovery attempted) once that
    many consecutive losses have occurred without a win; if None, the
    stake is allowed to double without limit for as long as the
    outcome sequence runs, which is the classical, capital-unbounded
    version of the system.
    """
    stake = initial_stake
    pnl = 0.0
    consecutive_losses = 0
    forced_stops = 0
    for outcome in outcomes:
        if outcome == 1:
            pnl += stake
            stake = initial_stake
            consecutive_losses = 0
        else:
            pnl -= stake
            consecutive_losses += 1
            if max_doublings is not None and consecutive_losses >= max_doublings:
                forced_stops += 1
                stake = initial_stake
                consecutive_losses = 0
            else:
                stake *= 2
    return {"final_pnl": pnl, "forced_stops": forced_stops}


def block_bootstrap_resample(series: List[int], block_length: int,
                              n_out: int, seed: Optional[int] = None) -> List[int]:
    """Resample a dependent series by concatenating randomly chosen
    contiguous blocks of length `block_length` (Kunsch's moving block
    bootstrap), preserving local dependence structure that an i.i.d.
    resample would destroy, up to producing a series of length n_out.
    """
    if block_length < 1 or block_length > len(series):
        raise ValueError("invalid block_length")
    rng = random.Random(seed)
    n = len(series)
    out = []
    while len(out) < n_out:
        start = rng.randrange(0, n - block_length + 1)
        out.extend(series[start:start + block_length])
    return out[:n_out]


def exact_expected_pnl(n_steps: int, p: float, initial_stake: float,
                        max_doublings: Optional[int]) -> float:
    """Exact expected final P&L of the doubling strategy after n_steps
    fixed rounds, computed by forward propagation of the exact
    probability distribution over the current losing-streak length (a
    Markov chain on non-negative integers, or on {0,...,max_doublings-1}
    when capped) rather than by Monte Carlo or path enumeration.

    At every step the stake is a deterministic function of the current
    streak length k (stake = initial_stake * 2^k), and that stake is
    predictable (fixed before the step's outcome is drawn), so the
    exact contribution to expected P&L at that step is
    E[stake_before_step] * (2p - 1), summed over all n_steps. Capping
    only changes which state a loss transitions to (forced back to 0
    instead of to k+1); it does not change this per-step accounting,
    since the pnl -= stake line fires identically whether or not the
    loss also triggers a forced stop.
    """
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    dist = {0: 1.0}
    expected_pnl = 0.0
    for _ in range(n_steps):
        new_dist: Dict[int, float] = {}
        for k, mass in dist.items():
            win_mass = mass * p
            loss_mass = mass * (1 - p)
            # mass * initial_stake * 2**k, computed via ldexp to avoid
            # ever forming the (potentially astronomically large)
            # integer 2**k directly, which overflows a float long
            # before its product with the correspondingly tiny mass
            # would.
            expected_pnl += math.ldexp(win_mass, k) * initial_stake
            expected_pnl -= math.ldexp(loss_mass, k) * initial_stake
            new_dist[0] = new_dist.get(0, 0.0) + win_mass
            if max_doublings is not None and k + 1 >= max_doublings:
                new_dist[0] = new_dist.get(0, 0.0) + loss_mass
            else:
                new_dist[k + 1] = new_dist.get(k + 1, 0.0) + loss_mass
        dist = new_dist
    return expected_pnl


def probability_max_streak_at_least(n_steps: int, p: float, k: int) -> float:
    """Exact probability that a run of at least k consecutive losses
    occurs at some point within n_steps rounds, via forward propagation
    of a Markov chain on the current streak length with an absorbing
    'reached k' state. Quantifies how rare the loss event has to be for
    the doubling strategy's books to balance.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    dist = {0: 1.0}
    absorbed = 0.0
    for _ in range(n_steps):
        new_dist: Dict[int, float] = {}
        for streak, mass in dist.items():
            new_dist[0] = new_dist.get(0, 0.0) + mass * p
            next_streak = streak + 1
            if next_streak >= k:
                absorbed += mass * (1 - p)
            else:
                new_dist[next_streak] = new_dist.get(next_streak, 0.0) + mass * (1 - p)
        dist = new_dist
    return absorbed


# Application: EUR/USD returns.

def load_returns(csv_path: str, column: str = "return") -> List[float]:
    values = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row[column]))
    return values


def sign_outcomes(returns: List[float], seed: Optional[int] = None) -> List[int]:
    """Map each return to +1 (up) or -1 (down) for the doubling-strategy
    demonstration; a return of exactly zero (not observed in this
    dataset) is broken by a coin flip.
    """
    rng = random.Random(seed)
    outcomes = []
    for r in returns:
        if r > 0:
            outcomes.append(1)
        elif r < 0:
            outcomes.append(-1)
        else:
            outcomes.append(1 if rng.random() < 0.5 else -1)
    return outcomes


def down_streak_lengths(outcomes: List[int]) -> List[int]:
    """Lengths of every maximal run of consecutive -1 outcomes."""
    streaks = []
    current = 0
    for o in outcomes:
        if o == -1:
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)
    return streaks


def streak_tail_probabilities(outcomes: List[int], thresholds: List[int]) -> Dict[str, List[float]]:
    """For each threshold k, compare the empirical fraction of observed
    down-streaks with length >= k against the fraction implied by an
    i.i.d. Bernoulli model with the same marginal down-probability
    (streak lengths under i.i.d. Bernoulli(p) trials are geometric,
    so P(streak >= k) = q^k with q = 1 - p being the down-probability).
    This is the diagnostic used to explain why a path-dependent
    strategy like doubling can show a return not predicted by a simple
    lag-1 independence test: the two tests are sensitive to different
    features of the same dependence structure.
    """
    streaks = down_streak_lengths(outcomes)
    q_hat = sum(1 for o in outcomes if o == -1) / len(outcomes)
    empirical = []
    theoretical = []
    for k in thresholds:
        empirical.append(sum(1 for s in streaks if s >= k) / len(streaks))
        theoretical.append(q_hat ** k)
    return {"thresholds": thresholds, "empirical": empirical, "theoretical": theoretical}
