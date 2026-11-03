import csv
import math
import random
import statistics
from typing import List, Optional, Callable, Dict, Tuple


# Exact distribution of a homogeneous Poisson process.

def pmf(rate: float, t: float, k: int) -> float:
    """P(N(t) = k) for a homogeneous Poisson process with intensity
    `rate`, observed over an interval of length `t`.
    """
    if rate < 0 or t < 0:
        raise ValueError("rate and t must be non-negative")
    if k < 0:
        return 0.0
    mean = rate * t
    if mean == 0:
        return 1.0 if k == 0 else 0.0
    log_pmf = k * math.log(mean) - mean - math.lgamma(k + 1)
    return math.exp(log_pmf)


def cdf(rate: float, t: float, k: int) -> float:
    """P(N(t) <= k)."""
    if k < 0:
        return 0.0
    return sum(pmf(rate, t, j) for j in range(k + 1))


def mean_count(rate: float, t: float) -> float:
    return rate * t


def variance_count(rate: float, t: float) -> float:
    """For a Poisson distribution, variance equals the mean. This is
    the defining equidispersion property tested empirically below.
    """
    return rate * t


# Simulating a homogeneous process two independent ways.

def simulate_arrivals(rate: float, t_max: float, seed: Optional[int] = None) -> List[float]:
    """Generate arrival times on [0, t_max] via i.i.d. Exponential(rate)
    inter-arrival gaps, drawn by inverse-transform sampling:
    if U ~ Uniform(0,1) then -ln(U)/rate ~ Exponential(rate).
    """
    if rate <= 0:
        raise ValueError("rate must be positive")
    rng = random.Random(seed)
    arrivals = []
    t = 0.0
    while True:
        u = rng.random()
        gap = -math.log(u) / rate
        t += gap
        if t > t_max:
            break
        arrivals.append(t)
    return arrivals


def simulate_count_direct(rate: float, t: float, seed: Optional[int] = None) -> int:
    """Generate a single Poisson(rate * t) count by Knuth's algorithm:
    multiply uniforms until the running product drops below exp(-mean).
    This is an independent construction from `simulate_arrivals`, used
    to cross-check the arrival-based counting method.
    """
    if rate < 0 or t < 0:
        raise ValueError("rate and t must be non-negative")
    mean = rate * t
    rng = random.Random(seed)
    threshold = math.exp(-mean)
    product = 1.0
    count = -1
    while product > threshold:
        product *= rng.random()
        count += 1
    return count


def interarrival_times(arrival_times: List[float]) -> List[float]:
    """Gaps between successive arrivals, including the gap from time 0
    to the first arrival.
    """
    if not arrival_times:
        return []
    gaps = [arrival_times[0]]
    for i in range(1, len(arrival_times)):
        gaps.append(arrival_times[i] - arrival_times[i - 1])
    return gaps


def count_in_window(arrival_times: List[float], start: float, end: float) -> int:
    return sum(1 for a in arrival_times if start <= a < end)


def counts_in_bins(arrival_times: List[float], t_max: float, n_bins: int) -> List[int]:
    width = t_max / n_bins
    counts = [0] * n_bins
    for a in arrival_times:
        if a >= t_max:
            continue
        idx = min(int(a / width), n_bins - 1)
        counts[idx] += 1
    return counts


# Superposition and thinning.

def superpose(*arrival_lists: List[float]) -> List[float]:
    """Merge independent Poisson arrival streams into a single sorted
    stream. If the inputs are independent Poisson processes with rates
    lambda_1, ..., lambda_k, the superposition theorem says the merged
    process is Poisson with rate sum(lambda_i).
    """
    merged = []
    for lst in arrival_lists:
        merged.extend(lst)
    merged.sort()
    return merged


def thin(arrival_times: List[float], keep_prob: float,
         seed: Optional[int] = None) -> Tuple[List[float], List[float]]:
    """Independently mark each arrival as kept (probability keep_prob)
    or discarded. The thinning theorem says that if the input is
    Poisson(rate), the kept and discarded streams are themselves
    independent Poisson processes with rates rate*keep_prob and
    rate*(1-keep_prob).
    """
    if not 0.0 <= keep_prob <= 1.0:
        raise ValueError("keep_prob must be in [0, 1]")
    rng = random.Random(seed)
    kept, discarded = [], []
    for a in arrival_times:
        if rng.random() < keep_prob:
            kept.append(a)
        else:
            discarded.append(a)
    return kept, discarded


# Inhomogeneous Poisson process via Lewis-Shedler thinning.

def simulate_inhomogeneous(rate_func: Callable[[float], float], t_max: float,
                            rate_max: float, seed: Optional[int] = None) -> List[float]:
    """Simulate a Poisson process with time-varying intensity rate_func
    by generating a homogeneous Poisson(rate_max) process and keeping
    each candidate point u with probability rate_func(u) / rate_max.
    Requires rate_func(t) <= rate_max for all t in [0, t_max].
    """
    if rate_max <= 0:
        raise ValueError("rate_max must be positive")
    rng = random.Random(seed)
    candidates = simulate_arrivals(rate_max, t_max, seed=rng.randrange(2 ** 31))
    kept = []
    for t in candidates:
        r = rate_func(t)
        if r > rate_max + 1e-9:
            raise ValueError("rate_func exceeded rate_max at t=%.6f" % t)
        if rng.random() < r / rate_max:
            kept.append(t)
    return kept


def expected_inhomogeneous_count(rate_func: Callable[[float], float], t_max: float,
                                  n_grid: int = 10000) -> float:
    """Trapezoidal-rule estimate of the mean count, integral of
    rate_func over [0, t_max]. This is the mean of an inhomogeneous
    Poisson process by definition.
    """
    if n_grid < 2:
        raise ValueError("n_grid must be at least 2")
    h = t_max / n_grid
    total = 0.5 * (rate_func(0.0) + rate_func(t_max))
    for i in range(1, n_grid):
        total += rate_func(i * h)
    return total * h


# Regularised incomplete gamma function, used for the dispersion test
# below. Same construction as the chi-square work in Project 2 (series
# and continued-fraction split), reimplemented here so this project
# stays self-contained.

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


def regularized_gamma_q(a: float, x: float) -> float:
    if x < 0 or a <= 0:
        raise ValueError("require a > 0 and x >= 0")
    if x == 0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _gamma_series(a, x)
    return _gamma_continued_fraction(a, x)


def chi2_sf(x: float, df: int) -> float:
    if x <= 0:
        return 1.0
    return regularized_gamma_q(df / 2.0, x / 2.0)


# Fisher's dispersion test: is a set of bin counts consistent with
# equidispersed (Poisson) counts, or over/under-dispersed?

def fisher_dispersion_test(counts: List[int]) -> Dict[str, float]:
    """Under the null hypothesis that counts are i.i.d. Poisson(mu),
    D = sum((n_i - mean)^2) / mean is approximately chi-square with
    len(counts) - 1 degrees of freedom. D much larger than its
    degrees of freedom indicates overdispersion (clustering); much
    smaller indicates underdispersion (regularity).
    """
    n = len(counts)
    if n < 2:
        raise ValueError("need at least two bins")
    mean = statistics.mean(counts)
    if mean == 0:
        raise ValueError("mean count is zero; dispersion test is undefined")
    statistic = sum((c - mean) ** 2 for c in counts) / mean
    df = n - 1
    p_value = chi2_sf(statistic, df)
    index_of_dispersion = statistic / df
    return {
        "statistic": statistic,
        "df": df,
        "p_value": p_value,
        "index_of_dispersion": index_of_dispersion,
        "mean_count": mean,
    }


def exponential_moments_check(gaps: List[float]) -> Dict[str, float]:
    """Sample mean, variance, and coefficient of variation of a set of
    inter-arrival gaps. For a true Exponential(rate) distribution the
    coefficient of variation (std / mean) is exactly 1; departures
    above 1 indicate more clustering/burstiness than memorylessness
    would produce, departures below 1 indicate more regularity.
    """
    if len(gaps) < 2:
        raise ValueError("need at least two gaps")
    mean = statistics.mean(gaps)
    var = statistics.variance(gaps)
    cv = math.sqrt(var) / mean if mean > 0 else math.inf
    return {"mean": mean, "variance": var, "coefficient_of_variation": cv}


# Application: are large EUR/USD moves a homogeneous Poisson process?

def load_returns(csv_path: str, column: str = "return") -> List[float]:
    values = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row[column]))
    return values


def percentile(data: List[float], q: float) -> float:
    """Linear-interpolation percentile, q in [0, 1]."""
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    ordered = sorted(data)
    n = len(ordered)
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return ordered[lo] + frac * (ordered[hi] - ordered[lo])


def large_move_arrival_times(returns: List[float], quantile: float = 0.95) -> List[float]:
    """Treat each trading day as a unit time step and record the
    (integer) day index of every day whose absolute return exceeds the
    given quantile of the absolute-return distribution, itself
    computed from the data rather than fixed in advance.
    """
    threshold = percentile([abs(r) for r in returns], quantile)
    return [float(i) for i, r in enumerate(returns) if abs(r) > threshold]


def analyze_fx_poisson_process(returns: List[float], quantile: float = 0.95,
                                n_bins: int = 20) -> Dict:
    """Test whether large EUR/USD moves arrive as a homogeneous Poisson
    process: estimate the rate, compare inter-arrival gaps to the
    implied exponential distribution, and run Fisher's dispersion test
    on counts across equal-length windows.
    """
    t_max = float(len(returns))
    arrivals = large_move_arrival_times(returns, quantile=quantile)
    gaps = interarrival_times(arrivals)

    rate_hat = len(arrivals) / t_max
    theoretical_mean_gap = 1.0 / rate_hat
    moments = exponential_moments_check(gaps)

    counts = counts_in_bins(arrivals, t_max, n_bins)
    dispersion = fisher_dispersion_test(counts)

    return {
        "n_observations": len(returns),
        "quantile": quantile,
        "n_arrivals": len(arrivals),
        "rate_hat": rate_hat,
        "theoretical_mean_gap": theoretical_mean_gap,
        "empirical_mean_gap": moments["mean"],
        "coefficient_of_variation": moments["coefficient_of_variation"],
        "n_bins": n_bins,
        "counts": counts,
        "dispersion_test": dispersion,
    }
