"""A resampling library built entirely by hand.

The bootstrap, the jackknife and the permutation test all answer the
same question in slightly different ways: what would this statistic do
if we could resample the world?  No numpy, scipy or pandas here.  The
resampling is done with the standard library's random module, seeded
so any run can be reproduced exactly.

The bootstrap draws many samples with replacement and recalculates the
statistic on each one, giving an empirical distribution of the
statistic.  Confidence intervals come from that distribution in four
ways:

    percentile   the alpha/2 and 1-alpha/2 quantiles directly
    basic        the same quantiles reflected around the estimate
    normal       estimate +- z * bootstrap standard error
    BCa          bias-corrected and accelerated, the most careful

The jackknife leaves out one observation at a time.  It gives a
de-biased estimate, a bias term and a standard error, and for the mean
it reproduces the classic formula exactly: the jackknife standard
error of the mean is the sample standard deviation over sqrt(n).

The permutation test shuffles the pooled data many times, recomputes
the test statistic on each shuffle, and reports how often the shuffled
statistic reaches or passes the observed one.  That is the p-value: an
answer computed from the data alone, with no parametric assumptions.

The CLT functions draw many samples from a population and inspect the
distribution of their means.  As the sample size grows that distribution
should approach a normal one centered on the population mean with
standard deviation population_sd / sqrt(sample_size), which is exactly
what the central limit theorem promises.
"""

import math
import random


# ---------------------------------------------------------------------------
# Small statistic helpers, all plain definitions
# ---------------------------------------------------------------------------

def mean(data):
    """Arithmetic mean."""
    return sum(data) / len(data)


def median(data):
    """Middle value of the sorted data; averages the two middles."""
    n = len(data)
    ordered = sorted(data)
    if n % 2 == 1:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def variance(data):
    """Sample variance with the n-1 correction."""
    if len(data) < 2:
        raise ValueError("need at least two values")
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / (len(data) - 1)


def std(data):
    """Sample standard deviation."""
    return math.sqrt(variance(data))


def population_std(data):
    """Standard deviation treating the data as the whole population."""
    if not data:
        raise ValueError("need at least one value")
    m = mean(data)
    return math.sqrt(sum((x - m) ** 2 for x in data) / len(data))


def mean_diff(a, b):
    """Difference of two group means, mean(a) - mean(b)."""
    return mean(a) - mean(b)


def _skewness(data):
    """Population third standardized moment."""
    m = mean(data)
    m2 = sum((x - m) ** 2 for x in data) / len(data)
    m3 = sum((x - m) ** 3 for x in data) / len(data)
    return m3 / m2 ** 1.5


def _kurtosis(data):
    """Population excess fourth standardized moment."""
    m = mean(data)
    m2 = sum((x - m) ** 2 for x in data) / len(data)
    m4 = sum((x - m) ** 4 for x in data) / len(data)
    return m4 / m2 ** 2 - 3.0


def _quantile(sorted_data, q):
    """Linear-interpolation quantile, matching the numpy default."""
    if not sorted_data:
        raise ValueError("need at least one value")
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    pos = (n - 1) * q
    lo = int(pos)
    frac = pos - lo
    if lo + 1 < n:
        return sorted_data[lo] * (1 - frac) + sorted_data[lo + 1] * frac
    return sorted_data[lo]


def _normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_quantile(p):
    """Acklam's approximation refined with Newton steps."""
    if not 0.0 < p < 1.0:
        raise ValueError("probability must lie strictly between 0 and 1")
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) \
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p <= 1.0 - p_low:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q \
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) \
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    for _ in range(3):
        x -= (_normal_cdf(x) - p) / (math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi))
    return x


def _sample_with_replacement(population, count, rng):
    """Draw count values from population, each with equal probability."""
    n = len(population)
    return [population[rng.randrange(n)] for _ in range(count)]


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap(data, statistic=mean, n_boot=1000, seed=None):
    """Draw n_boot resamples with replacement and evaluate statistic.

    Returns the list of bootstrap replicates in the order they were
    drawn, so the same seed always produces the same list.
    """
    if not data:
        raise ValueError("need at least one value")
    if n_boot < 2:
        raise ValueError("need at least two bootstrap resamples")
    rng = random.Random(seed)
    return [statistic(_sample_with_replacement(data, len(data), rng))
            for _ in range(n_boot)]


def _bca_bounds(data, statistic, estimate, alpha):
    """Bias-correction and acceleration for the BCa interval."""
    n = len(data)
    loo = [statistic(data[:i] + data[i + 1:]) for i in range(n)]
    mean_loo = mean(loo)
    num = sum((mean_loo - v) ** 3 for v in loo)
    den = sum((mean_loo - v) ** 2 for v in loo)
    acceleration = num / (6.0 * den ** 1.5) if den else 0.0
    return acceleration


def bootstrap_ci(data, statistic=mean, n_boot=1000, alpha=0.05,
                 method="percentile", seed=None):
    """Bootstrap confidence interval for a statistic of the data.

    alpha is the total error rate, so a 95% interval has alpha 0.05.
    The methods are described in the module docstring.  Returns a dict
    with the point estimate, the bootstrap standard error and the lower
    and upper bounds of the interval.
    """
    if not data:
        raise ValueError("need at least one value")
    if n_boot < 2:
        raise ValueError("need at least two bootstrap resamples")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if method not in ("percentile", "basic", "normal", "BCa"):
        raise ValueError("method must be percentile, basic, normal or BCa")
    estimate = statistic(data)
    reps = bootstrap(data, statistic, n_boot, seed)
    ordered = sorted(reps)
    se = std(reps)
    if method == "percentile":
        lower = _quantile(ordered, alpha / 2.0)
        upper = _quantile(ordered, 1.0 - alpha / 2.0)
    elif method == "basic":
        lower = 2.0 * estimate - _quantile(ordered, 1.0 - alpha / 2.0)
        upper = 2.0 * estimate - _quantile(ordered, alpha / 2.0)
    elif method == "normal":
        z = _normal_quantile(1.0 - alpha / 2.0)
        lower = estimate - z * se
        upper = estimate + z * se
    else:
        less = sum(1 for r in reps if r < estimate) / len(reps)
        z0 = _normal_quantile(min(max(less, 1e-10), 1 - 1e-10))
        acc = _bca_bounds(data, statistic, estimate, alpha)

        def adjusted(z):
            return _normal_cdf(z0 + (z0 + z) / (1.0 - acc * (z0 + z)))

        p_low = adjusted(_normal_quantile(alpha / 2.0))
        p_high = adjusted(_normal_quantile(1.0 - alpha / 2.0))
        lower = _quantile(ordered, min(max(p_low, 0.0), 1.0))
        upper = _quantile(ordered, min(max(p_high, 0.0), 1.0))
    return {"estimate": estimate, "se": se, "lower": lower, "upper": upper,
            "method": method, "alpha": alpha, "n_boot": n_boot}


# ---------------------------------------------------------------------------
# Jackknife
# ---------------------------------------------------------------------------

def jackknife(data, statistic=mean):
    """Leave-one-out estimates, bias and standard error.

    For the mean the standard error is exactly the sample standard
    deviation over sqrt(n), which is a useful sanity check on the
    implementation.  The bias term corrects the estimate by how much
    leaving each observation out moves it, on average.
    """
    if not data:
        raise ValueError("need at least one value")
    n = len(data)
    loo = [statistic(data[:i] + data[i + 1:]) for i in range(n)]
    estimate = statistic(data)
    mean_loo = mean(loo)
    bias = (n - 1) * (mean_loo - estimate)
    var = (n - 1) / n * sum((v - mean_loo) ** 2 for v in loo)
    return {"estimate": estimate, "bias": bias, "se": math.sqrt(var),
            "leave_one_out": loo}


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def permutation_test(a, b, statistic=mean_diff, n_perm=10000, seed=None):
    """Two-sample permutation test of the statistic against zero.

    The null hypothesis is that both groups come from the same
    population, in which case shuffling the pooled data changes nothing
    on average.  The p-value is the fraction of shuffles whose statistic
    reaches or passes the observed one in magnitude, with the 1/(N+1)
    convention so the smallest reportable p-value is never zero.

    The shuffle only randomizes the first len(a) positions, which is a
    partial Fisher-Yates; those positions are a uniform random subset
    either way, and the rest is their complement, so the split is a
    uniform random one at about half the cost of a full shuffle.
    """
    if not a or not b:
        raise ValueError("both groups need at least one value")
    if n_perm < 2:
        raise ValueError("need at least two permutations")
    na = len(a)
    n = na + len(b)
    pooled = list(a) + list(b)
    observed = statistic(a, b)
    rng = random.Random(seed)
    count = 0
    if statistic is mean_diff:
        nb = n - na
        total = sum(pooled)
        for _ in range(n_perm):
            for i in range(na):
                j = rng.randrange(i, n)
                pooled[i], pooled[j] = pooled[j], pooled[i]
            sa = sum(pooled[:na])
            stat = sa / na - (total - sa) / nb
            if abs(stat) >= abs(observed):
                count += 1
    else:
        for _ in range(n_perm):
            for i in range(na):
                j = rng.randrange(i, n)
                pooled[i], pooled[j] = pooled[j], pooled[i]
            if abs(statistic(pooled[:na], pooled[na:])) >= abs(observed):
                count += 1
    return {"statistic": observed, "p": (count + 1) / (n_perm + 1),
            "n_perm": n_perm, "count": count}


# ---------------------------------------------------------------------------
# Central limit theorem
# ---------------------------------------------------------------------------

def sample_means(population, sample_size, n_samples=2000, seed=None):
    """Draw n_samples samples of sample_size, return their means.

    Sampling is with replacement from the given population, which keeps
    the theoretical comparison exact: the variance of each mean is the
    population variance over the sample size.
    """
    if not population:
        raise ValueError("need at least one value in the population")
    if sample_size < 1:
        raise ValueError("sample size must be at least one")
    if n_samples < 2:
        raise ValueError("need at least two samples")
    rng = random.Random(seed)
    return [mean(_sample_with_replacement(population, sample_size, rng))
            for _ in range(n_samples)]


def clt_summary(population, sample_size, n_samples=2000, seed=None):
    """Shape of the sampling distribution of the mean for one sample size.

    Returns the empirical mean, standard deviation, skewness and excess
    kurtosis of the sample means, plus the theoretical standard
    deviation population_sd / sqrt(sample_size) that the CLT predicts.
    As the sample size grows, the empirical standard deviation should
    approach the theoretical one and the skewness should shrink toward
    zero.
    """
    means = sample_means(population, sample_size, n_samples, seed)
    pop_sd = population_std(population)
    return {"sample_size": sample_size, "n_samples": n_samples,
            "mean": mean(means), "sd": std(means),
            "skewness": _skewness(means), "kurtosis": _kurtosis(means),
            "theoretical_sd": pop_sd / math.sqrt(sample_size),
            "population_sd": pop_sd, "population_mean": mean(population)}


def clt_convergence(population, sample_sizes, n_samples=2000, seed=None):
    """Run clt_summary for each sample size in turn."""
    return [clt_summary(population, k, n_samples, seed)
            for k in sample_sizes]
