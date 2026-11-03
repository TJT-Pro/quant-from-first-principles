import math
import statistics
from typing import List, Dict


def inverse_normal_cdf(p: float) -> float:
    """Rational approximation to the standard normal quantile function
    (Beasley-Springer-Moro), needed here to convert a VaR confidence
    level into the corresponding z-score for the parametric Gaussian
    VaR formula below.
    """
    a = (2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637)
    b = (-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833)
    c = (0.3374754822726147, 0.9761690190917186, 0.1607979714918209, 0.0276438810333863,
         0.0038405729373609, 0.0003951896511919, 0.0000321767881768, 0.0000002888167364,
         0.0000003960315187)

    y = p - 0.5
    if abs(y) < 0.42:
        r = y * y
        num = y * (((a[3] * r + a[2]) * r + a[1]) * r + a[0])
        den = (((b[3] * r + b[2]) * r + b[1]) * r + b[0]) * r + 1
        return num / den

    r = p if y < 0 else 1 - p
    r = math.log(-math.log(r))
    x = c[0] + r * (c[1] + r * (c[2] + r * (c[3] + r * (c[4] + r * (c[5] + r * (c[6] + r * (c[7] + r * c[8])))))))
    return -x if y < 0 else x


def parametric_gaussian_var(returns: List[float], confidence: float) -> float:
    """The parametric (Gaussian) Value-at-Risk: assuming returns are
    normally distributed with the sample mean and standard deviation,
    VaR at confidence level `confidence` (e.g. 0.95, 0.99) is the loss
    threshold L such that P(return < -L) = 1-confidence, i.e.
    L = -(mu + z_{1-confidence}*sigma), reported here as a positive
    number representing a potential loss magnitude. This is the
    single most common VaR model in practice specifically because of
    its simplicity, and specifically the model this project's own
    application below shows fails a rigorous backtest on real data.
    """
    mu = statistics.mean(returns)
    sigma = statistics.pstdev(returns)
    z = inverse_normal_cdf(1 - confidence)
    return -(mu + z * sigma)


def historical_var(returns: List[float], confidence: float) -> float:
    """Historical simulation VaR: the empirical (1-confidence)
    quantile of the realised return distribution directly, with no
    distributional assumption at all. This automatically captures
    whatever tail heaviness the data actually has (fat tails, skew,
    anything), at the cost of relying entirely on the historical
    sample actually containing a representative range of outcomes,
    including rare ones.
    """
    sorted_returns = sorted(returns)
    n = len(sorted_returns)
    idx = int((1 - confidence) * n)
    idx = max(0, min(idx, n - 1))
    return -sorted_returns[idx]


def count_exceptions(returns: List[float], var_estimate: float) -> int:
    """The number of days a realised loss exceeded the VaR estimate
    (equivalently, return < -var_estimate): the raw ingredient every
    VaR backtest is built from.
    """
    return sum(1 for r in returns if r < -var_estimate)


def chi_squared_1_cdf(x: float) -> float:
    """The CDF of a chi-squared distribution with exactly 1 degree of
    freedom, via the exact closed-form relationship to the standard
    normal: chi-squared-1 is the distribution of Z^2 for Z~N(0,1), so
    P(Z^2<=x) = P(-sqrt(x)<=Z<=sqrt(x)) = 2*Phi(sqrt(x))-1, which
    simplifies directly to erf(sqrt(x/2)) using Phi's own definition
    in terms of erf. This closed form is exactly what is needed for
    Kupiec's POF test below, whose likelihood-ratio statistic is
    chi-squared distributed with 1 degree of freedom under the null
    hypothesis of a correctly specified VaR model, with no need for a
    general (any degrees of freedom) chi-squared CDF implementation.
    """
    if x < 0:
        return 0.0
    return math.erf(math.sqrt(x / 2))


def kupiec_pof_test(n_observations: int, n_exceptions: int, confidence: float) -> Dict:
    """Kupiec's (1995) Proportion-of-Failures test: under a correctly
    specified VaR model at confidence level `confidence`, exceptions
    should occur independently at exactly rate p=1-confidence. The
    likelihood-ratio statistic

        LR = -2*ln[ (1-p)^(n-x)*p^x / ((1-x/n)^(n-x)*(x/n)^x) ]

    compares the likelihood of the observed exception count x under
    the model's claimed rate p against the likelihood under the
    empirically observed rate x/n (the unconstrained maximum
    likelihood estimate), and is chi-squared distributed with 1
    degree of freedom under the null hypothesis that the model's
    claimed rate is correct. A large LR (equivalently, a small
    p-value) rejects the model: its actual exception rate differs
    significantly from what it claims to be delivering, in either
    direction (too many exceptions means the model understates risk;
    too few, less commonly flagged in practice but equally a
    miscalibration, means it overstates risk).
    """
    n, x = n_observations, n_exceptions
    p = 1 - confidence
    x_rate = x / n

    def safe_log(v):
        return math.log(v) if v > 0 else -1e10

    log_likelihood_null = (n - x) * safe_log(1 - p) + x * safe_log(p)
    log_likelihood_alt = (n - x) * safe_log(1 - x_rate) + x * safe_log(x_rate)
    lr_stat = -2 * (log_likelihood_null - log_likelihood_alt)

    p_value = 1 - chi_squared_1_cdf(lr_stat)
    expected_exceptions = n * p

    return {"n_observations": n, "n_exceptions": x, "expected_exceptions": expected_exceptions,
            "exception_rate": x_rate, "expected_rate": p, "lr_statistic": lr_stat,
            "p_value": p_value, "rejected_at_5_percent": p_value < 0.05}
