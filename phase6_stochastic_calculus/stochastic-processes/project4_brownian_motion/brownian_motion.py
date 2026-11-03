import csv
import math
import random
import statistics
from typing import List, Optional, Dict, Tuple


# Standard normal sampling and CDF, built entirely from the standard
# library. Everything downstream (path simulation, the reflection
# principle, the variance ratio test) reduces to these two primitives.

def generate_standard_normals(n: int, rng: random.Random) -> List[float]:
    """n draws from N(0,1) via the Box-Muller transform: two independent
    uniforms U1, U2 on (0,1) produce two independent standard normals

        Z0 = sqrt(-2 ln U1) cos(2*pi*U2)
        Z1 = sqrt(-2 ln U1) sin(2*pi*U2)

    because (Z0, Z1) is then a point sampled uniformly in angle with a
    Rayleigh-distributed radius, which is exactly the joint density of
    two independent standard normals in polar form.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    values = []
    while len(values) < n:
        u1 = rng.random()
        u2 = rng.random()
        # avoid log(0)
        u1 = u1 if u1 > 1e-300 else 1e-300
        radius = math.sqrt(-2.0 * math.log(u1))
        angle = 2.0 * math.pi * u2
        values.append(radius * math.cos(angle))
        if len(values) < n:
            values.append(radius * math.sin(angle))
    return values[:n]


def standard_normal_cdf(x: float) -> float:
    """Phi(x), via the identity Phi(x) = (1 + erf(x/sqrt(2))) / 2. math.erf
    is a standard-library special function, not a probability routine,
    so this is a legitimate from-scratch construction of the normal
    CDF rather than a call into a statistics library.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# Simulating standard Brownian motion.

def simulate_brownian_motion(t_max: float, n_steps: int,
                              seed: Optional[int] = None) -> Tuple[List[float], List[float]]:
    """Simulate a standard Brownian motion path on [0, t_max] using
    n_steps equal increments. Each increment W(t+dt) - W(t) is drawn
    from N(0, dt), since Brownian motion has independent, stationary,
    Gaussian increments with variance equal to the elapsed time.
    Returns (times, values) with W(0) = 0 included.
    """
    if t_max <= 0 or n_steps <= 0:
        raise ValueError("t_max and n_steps must be positive")
    rng = random.Random(seed)
    dt = t_max / n_steps
    normals = generate_standard_normals(n_steps, rng)
    scale = math.sqrt(dt)

    times = [0.0]
    values = [0.0]
    w = 0.0
    for i in range(n_steps):
        w += normals[i] * scale
        times.append(times[-1] + dt)
        values.append(w)
    return times, values


def simulate_brownian_motion_with_drift(t_max: float, n_steps: int, mu: float, sigma: float,
                                         seed: Optional[int] = None) -> Tuple[List[float], List[float]]:
    """X(t) = mu*t + sigma*W(t), the generalised Brownian motion used to
    model a log-price path with drift mu and volatility sigma.
    """
    times, w = simulate_brownian_motion(t_max, n_steps, seed=seed)
    values = [mu * t + sigma * wt for t, wt in zip(times, w)]
    return times, values


# Path functionals: quadratic variation and total variation.

def quadratic_variation(values: List[float]) -> float:
    """Sum of squared increments along a sampled path. For Brownian
    motion, this converges (in probability, and in fact almost surely
    along nested partitions) to the elapsed time t as the partition is
    refined, because each squared increment has expectation dt and the
    increments are independent, so the sum's mean is exactly the total
    elapsed time regardless of how many steps it is split into, while
    its variance shrinks as the steps get finer.
    """
    return sum((values[i + 1] - values[i]) ** 2 for i in range(len(values) - 1))


def total_variation(values: List[float]) -> float:
    """Sum of absolute increments. Unlike quadratic variation, this
    grows without bound as the partition of a Brownian path is
    refined: it does not converge to a finite limit, which is the
    standard demonstration that Brownian paths have infinite total
    variation and are therefore not differentiable in the ordinary
    (Riemann-Stieltjes) sense.
    """
    return sum(abs(values[i + 1] - values[i]) for i in range(len(values) - 1))


# Reflection principle, first-passage times.

def reflection_principle_bm(a: float, t: float) -> float:
    """P(max_{0<=s<=t} W_s >= a) for standard Brownian motion, a > 0.

    Because a continuous Brownian path started at 0 must cross level a
    before it can be found above a, this event is identical to
    {T_a <= t}, so the same formula gives the first-passage-time CDF.
    The reflection principle argument mirrors Project 1's random-walk
    case exactly, but here the reflected path (post-hitting-time
    portion negated about y=a) is an exact, not merely combinatorial,
    symmetry of Brownian motion, giving

        P(M_t >= a) = P(W_t >= a) + P(W_t > a) = 2 P(W_t >= a),

    since W_t is continuous so P(W_t = a) = 0, and W_t / sqrt(t) is
    standard normal.
    """
    if a <= 0:
        raise ValueError("a must be positive")
    if t <= 0:
        raise ValueError("t must be positive")
    return 2.0 * (1.0 - standard_normal_cdf(a / math.sqrt(t)))


def first_passage_cdf_bm(a: float, t: float) -> float:
    """P(T_a <= t): identical to reflection_principle_bm, kept as a
    separate name because the two questions (running maximum vs.
    hitting time) are conceptually different even though a continuous
    path makes them numerically the same event.
    """
    return reflection_principle_bm(a, t)


def first_passage_density_bm(a: float, t: float) -> float:
    """Density of T_a, the first time standard Brownian motion hits
    level a > 0. Differentiating the CDF above with respect to t gives
    the inverse-Gaussian-type density

        f(t) = (a / sqrt(2*pi*t^3)) * exp(-a^2 / (2t)).
    """
    if a <= 0:
        raise ValueError("a must be positive")
    if t <= 0:
        return 0.0
    return (a / math.sqrt(2.0 * math.pi * t ** 3)) * math.exp(-(a ** 2) / (2.0 * t))


def simulate_first_passage_bm(a: float, dt: float, max_steps: int,
                               seed: Optional[int] = None) -> Optional[float]:
    """Simulate a Brownian path in fixed increments dt and return the
    first time it reaches or exceeds level a > 0, or None if it does
    not within max_steps.
    """
    if a <= 0:
        raise ValueError("a must be positive")
    rng = random.Random(seed)
    scale = math.sqrt(dt)
    position = 0.0
    for step in range(1, max_steps + 1):
        position += rng.gauss(0.0, 1.0) * scale
        if position >= a:
            return step * dt
    return None


# Donsker's invariance principle: a scaled random walk approximates BM.

def donsker_walk(n_steps: int, t_max: float, seed: Optional[int] = None) -> Tuple[List[float], List[float]]:
    """A simple symmetric random walk (+-1 steps) rescaled in both time
    and space so that it approximates standard Brownian motion. With
    n_steps over [0, t_max], each step has size sqrt(t_max / n_steps)
    so that the variance of the walk after k steps is
    k * (t_max/n_steps) = (k/n_steps) * t_max, matching Var(W_t) = t
    at time t = (k/n_steps) * t_max. This is Donsker's invariance
    principle in its simplest form: as n_steps grows, the rescaled
    random walk converges in distribution to Brownian motion.
    """
    if n_steps <= 0 or t_max <= 0:
        raise ValueError("n_steps and t_max must be positive")
    rng = random.Random(seed)
    step_size = math.sqrt(t_max / n_steps)
    dt = t_max / n_steps

    times = [0.0]
    values = [0.0]
    position = 0.0
    for _ in range(n_steps):
        step = step_size if rng.random() < 0.5 else -step_size
        position += step
        times.append(times[-1] + dt)
        values.append(position)
    return times, values


# Brownian bridge.

def brownian_bridge(t_max: float, n_steps: int, seed: Optional[int] = None) -> Tuple[List[float], List[float]]:
    """A Brownian motion path conditioned to return to 0 at time t_max,
    constructed from a free path W by

        B(t) = W(t) - (t / t_max) * W(t_max).

    This is the standard construction of the Brownian bridge: it is
    linear in the free path's terminal value, which is exactly what is
    needed to force B(t_max) = 0 while preserving the Gaussian
    structure. Var(B(t)) = t(t_max - t)/t_max, which is 0 at both
    endpoints and maximal at the midpoint.
    """
    times, w = simulate_brownian_motion(t_max, n_steps, seed=seed)
    w_end = w[-1]
    bridge = [w[i] - (times[i] / t_max) * w_end for i in range(len(times))]
    return times, bridge


# Application: EUR/USD log-price as a Brownian motion with drift.

def load_log_returns(csv_path: str, column: str = "log_return") -> List[float]:
    values = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row[column]))
    return values


def realized_variance(log_returns: List[float]) -> float:
    """Sum of squared log returns: the discrete-time analogue of
    quadratic variation, and the standard realized-variance estimator
    of cumulative variance used in volatility research.
    """
    return sum(r ** 2 for r in log_returns)


def variance_ratio_test(log_returns: List[float], q: int) -> Dict[str, float]:
    """Non-overlapping variance ratio test (Lo-MacKinlay style null):
    if log-price increments were uncorrelated with constant variance
    (consistent with log-price following a driftless Brownian motion
    plus drift), the variance of a q-period sum of increments must
    equal exactly q times the variance of a single increment, which is
    just Brownian scaling applied to the discrete-time analogue.

    The q-period returns are built from non-overlapping blocks (not a
    rolling window), so each block's variance-additivity is an exact
    algebraic identity under the null of uncorrelated increments, not
    an asymptotic approximation. The test statistic VR(q) is

        VR(q) = Var(sum of q consecutive returns) / (q * Var(single return))

    which should be 1 under the null. Both the numerator and
    denominator are estimated from overlapping information (the same
    n = m*q observations feed both the block sums and the one-period
    variance), so the exact asymptotic variance is derived here rather
    than assumed. Writing S1 for the one-period variance estimator and
    Sq for the q-period block-variance estimator, expanding Y_j^2 into
    within-block squared terms plus within-block cross terms gives
    Sq = q*S1 + C where C is uncorrelated with S1 for Gaussian
    increments (odd-moment cross terms vanish), so Cov(S1, Sq) = q*Var(S1).
    Applying the delta method to VR = Sq / (q*S1) with
    Var(S1) = 2/n, Var(Sq) = 2q^2/m, Cov(S1, Sq) = 2/m gives

        Var(VR(q)) = Var(S1) + Var(Sq)/q^2 - (2/q)*Cov(S1, Sq)
                   = 2/(qm) + 2/m - 4/(qm)
                   = 2(q-1) / (m*q).

    so the z-statistic (VR(q) - 1) / sqrt(2(q-1)/(m*q)) is compared
    against the standard normal distribution. This is checked directly
    against Monte Carlo replication in `cli.py verify`, both for
    rejection-rate calibration and for the exact variance formula.
    """
    if q < 2:
        raise ValueError("q must be at least 2")
    n = len(log_returns)
    m = n // q
    if m < 30:
        raise ValueError("not enough data for the requested q to have a stable asymptotic variance")

    mu = statistics.mean(log_returns)
    one_period_var = sum((r - mu) ** 2 for r in log_returns) / n

    block_sums = []
    for j in range(m):
        block = log_returns[j * q:(j + 1) * q]
        block_sums.append(sum(block))
    q_period_var = sum((s - q * mu) ** 2 for s in block_sums) / m

    vr = q_period_var / (q * one_period_var)
    se = math.sqrt(2.0 * (q - 1) / (m * q))
    z = (vr - 1.0) / se
    p_value = 2.0 * (1.0 - standard_normal_cdf(abs(z)))

    return {"q": q, "m": m, "vr": vr, "z": z, "se": se, "p_value": p_value}


def compare_scaling_to_theory(log_returns: List[float], horizons: List[int]) -> Dict[str, List[float]]:
    """For each horizon q in `horizons`, compute the empirical variance
    of non-overlapping q-period sums and the theoretical value implied
    by Brownian scaling (q times the 1-period variance), so the two
    can be plotted or printed side by side.
    """
    mu = statistics.mean(log_returns)
    n = len(log_returns)
    one_period_var = sum((r - mu) ** 2 for r in log_returns) / n

    empirical = []
    theoretical = []
    for q in horizons:
        m = n // q
        block_sums = [sum(log_returns[j * q:(j + 1) * q]) for j in range(m)]
        empirical.append(sum((s - q * mu) ** 2 for s in block_sums) / m)
        theoretical.append(q * one_period_var)

    return {"horizons": horizons, "empirical_variance": empirical, "theoretical_variance": theoretical}
