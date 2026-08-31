import math

_TWO_PI = 2.0 * math.pi
_FPMIN = 1e-300


def _raise_bad_probability(p):
    if not 0.0 < p < 1.0:
        raise ValueError("probability must lie strictly between 0 and 1")


# ---------------------------------------------------------------------------
# Normal distribution
# ---------------------------------------------------------------------------

def normal_pdf(x):
    """Probability density of the standard normal at x."""
    return math.exp(-0.5 * x * x) / math.sqrt(_TWO_PI)


def normal_cdf(x):
    """P(Z <= x) for Z standard normal, via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_cdf_inverse(p):
    """Quantile of the standard normal: the x with P(Z <= x) = p.

    Acklam's rational approximation provides a seed accurate to about
    nine decimal places; a few Newton steps against the exact cdf and
    pdf then bring it to machine precision.
    """
    _raise_bad_probability(p)
    if p <= 0.0:
        return -float("inf")
    if p >= 1.0:
        return float("inf")
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
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        x = num / den
    elif p <= 1.0 - p_low:
        q = p - 0.5
        r = q * q
        num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        x = num / den
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        x = -num / den
    for _ in range(3):
        x -= (normal_cdf(x) - p) / normal_pdf(x)
    return x


# ---------------------------------------------------------------------------
# Incomplete gamma, used for the chi-squared CDF
# ---------------------------------------------------------------------------

def _regularized_lower_gamma(a, x):
    """P(a, x), the lower regularized incomplete gamma function."""
    if x < a + 1.0:
        ap = a
        total = 1.0 / a
        delta = total
        for _ in range(10000):
            ap += 1.0
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * 1e-15:
                break
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b = x + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    upper = math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
    return 1.0 - upper


def chi2_cdf(x, df):
    """P(X <= x) for a chi-squared variable with df degrees of freedom."""
    if x < 0.0:
        return 0.0
    if x == 0.0:
        return 0.0
    return _regularized_lower_gamma(0.5 * df, 0.5 * x)


def chi2_sf(x, df):
    """P(X > x) for a chi-squared variable with df degrees of freedom."""
    return 1.0 - chi2_cdf(x, df)


def chi2_ppf(p, df):
    """Quantile of the chi-squared distribution with df degrees of freedom."""
    _raise_bad_probability(p)
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return float("inf")
    lo, hi = 0.0, 1.0
    while chi2_cdf(hi, df) < p:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if chi2_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Incomplete beta, used for the t distribution
# ---------------------------------------------------------------------------

def _betacf(a, b, x):
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, 10000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def _regularized_beta(a, b, x):
    """I_x(a, b), the regularized incomplete beta function."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_cdf(t, df):
    """P(T <= t) for a Student t variable with df degrees of freedom."""
    if df <= 0.0:
        raise ValueError("degrees of freedom must be positive")
    x = df / (df + t * t)
    ib = _regularized_beta(0.5 * df, 0.5, x)
    if t >= 0.0:
        return 1.0 - 0.5 * ib
    return 0.5 * ib


def t_sf(t, df):
    """P(T > t) for a Student t variable with df degrees of freedom."""
    return 1.0 - t_cdf(t, df)


def t_ppf(p, df):
    """Quantile of the Student t distribution with df degrees of freedom."""
    _raise_bad_probability(p)
    if p == 0.5:
        return 0.0
    if p < 0.5:
        lo, hi = -1e8, 0.0
    else:
        lo, hi = 0.0, 1e8
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Helpers shared by the tests
# ---------------------------------------------------------------------------

def _check_alternative(alternative):
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative must be two-sided, greater or less")


def _mean(values):
    return sum(values) / len(values)


def _sample_stddev(values):
    n = len(values)
    if n < 2:
        raise ValueError("need at least two values")
    m = _mean(values)
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)


def _z_to_p(z, alternative):
    _check_alternative(alternative)
    if alternative == "two-sided":
        return 2.0 * (1.0 - normal_cdf(abs(z)))
    if alternative == "greater":
        return 1.0 - normal_cdf(z)
    return normal_cdf(z)


def _t_to_p(t, df, alternative):
    _check_alternative(alternative)
    if alternative == "two-sided":
        return 2.0 * t_sf(abs(t), df)
    if alternative == "greater":
        return t_sf(t, df)
    return t_cdf(t, df)


# ---------------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------------

def z_test(x_bar, mu0, sigma, n, alternative="two-sided"):
    """One-sample z test with known population standard deviation.

    The null hypothesis is that the population mean equals mu0.  The z
    statistic measures how many standard errors the observed mean sits
    from mu0.  Returns a dict with the statistic and the p-value.
    """
    _check_alternative(alternative)
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if n < 1:
        raise ValueError("n must be at least one")
    z = (x_bar - mu0) / (sigma / math.sqrt(n))
    return {"z": z, "p": _z_to_p(z, alternative)}


def t_test_one_sample(data, mu0=0.0, alternative="two-sided"):
    """One-sample t test of the null mean == mu0, sigma unknown.

    Uses the sample standard deviation with the n-1 correction, so the
    degrees of freedom are n-1.  If every value in the sample is the
    same number, the test is degenerate: a constant equal to mu0 gives
    t = 0, anything else gives t = +-infinity and a p-value of 0.
    """
    _check_alternative(alternative)
    n = len(data)
    if n < 2:
        raise ValueError("need at least two observations")
    m = _mean(data)
    s = _sample_stddev(data)
    if s == 0.0:
        if m == mu0:
            return {"t": 0.0, "df": n - 1, "p": 1.0}
        if m > mu0:
            t = float("inf")
        else:
            t = float("-inf")
        p = 0.0 if alternative == "two-sided" else (1.0 if t > 0.0 else 0.0)
        return {"t": t, "df": n - 1, "p": p}
    t = (m - mu0) / (s / math.sqrt(n))
    return {"t": t, "df": n - 1, "p": _t_to_p(t, n - 1, alternative)}


def t_test_two_sample(a, b, equal_var=False, alternative="two-sided"):
    """Two-sample t test for independent groups.

    With equal_var False (the default) it is Welch's t test, which does
    not assume the two populations share a variance and uses the
    Satterthwaite degrees of freedom.  With equal_var True it pools the
    variances and uses n1 + n2 - 2 degrees of freedom.  The null is
    that the two population means are equal.
    """
    _check_alternative(alternative)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        raise ValueError("each group needs at least two observations")
    m1, m2 = _mean(a), _mean(b)
    s1, s2 = _sample_stddev(a), _sample_stddev(b)
    if s1 == 0.0 and s2 == 0.0:
        raise ValueError("both groups are constant, the test is undefined")
    if equal_var:
        sp = math.sqrt(((n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2)
                       / (n1 + n2 - 2))
        t = (m1 - m2) / (sp * math.sqrt(1.0 / n1 + 1.0 / n2))
        df = n1 + n2 - 2
    else:
        se = math.sqrt(s1 * s1 / n1 + s2 * s2 / n2)
        if se == 0.0:
            raise ValueError("both groups are constant, the test is undefined")
        t = (m1 - m2) / se
        num = (s1 * s1 / n1 + s2 * s2 / n2) ** 2
        den = ((s1 * s1 / n1) ** 2 / (n1 - 1) + (s2 * s2 / n2) ** 2 / (n2 - 1))
        df = num / den
    return {"t": t, "df": df, "p": _t_to_p(t, df, alternative)}


def t_test_paired(a, b, alternative="two-sided"):
    """Paired t test: each element of b is matched to one in a.

    Reduces to a one-sample t test on the differences a_i - b_i, so the
    pairing is used directly rather than thrown away.  The null is that
    the mean difference is zero.
    """
    if len(a) != len(b):
        raise ValueError("paired test needs equally sized groups")
    diffs = [x - y for x, y in zip(a, b)]
    return t_test_one_sample(diffs, mu0=0.0, alternative=alternative)


def chi2_goodness_of_fit(observed, expected):
    """Pearson chi-squared test of observed counts against expected.

    The null is that the observed counts were drawn from the
    distribution described by expected.  The degrees of freedom are
    len(observed) - 1.  Returns the statistic, the degrees of freedom
    and the p-value, plus the per-cell contribution to the statistic.
    """
    if len(observed) != len(expected):
        raise ValueError("observed and expected must have the same length")
    if len(observed) < 2:
        raise ValueError("need at least two categories")
    contributions = []
    total = 0.0
    for o, e in zip(observed, expected):
        if e <= 0.0:
            raise ValueError("expected counts must be positive")
        if o < 0:
            raise ValueError("observed counts cannot be negative")
        term = (o - e) * (o - e) / e
        contributions.append(term)
        total += term
    df = len(observed) - 1
    return {"chi2": total, "df": df, "p": chi2_sf(total, df),
            "contributions": contributions}


def chi2_independence(contingency):
    """Chi-squared test of independence for a table of counts.

    Rows are categories of one variable, columns of the other.  The
    expected count in each cell is row_total * col_total / grand_total,
    which is exactly what independence would predict.  The null is that
    the two variables are independent.
    """
    rows = len(contingency)
    if rows < 2:
        raise ValueError("need at least two rows")
    cols = len(contingency[0])
    if cols < 2:
        raise ValueError("need at least two columns")
    row_totals = [sum(r) for r in contingency]
    col_totals = [0] * cols
    grand = 0.0
    for r in contingency:
        if len(r) != cols:
            raise ValueError("all rows must have the same length")
        for j, v in enumerate(r):
            if v < 0:
                raise ValueError("counts cannot be negative")
            col_totals[j] += v
            grand += v
    if grand == 0:
        raise ValueError("the table is empty")
    total = 0.0
    for i in range(rows):
        for j in range(cols):
            e = row_totals[i] * col_totals[j] / grand
            total += (contingency[i][j] - e) * (contingency[i][j] - e) / e
    df = (rows - 1) * (cols - 1)
    return {"chi2": total, "df": df, "p": chi2_sf(total, df)}


def binomial_proportion_test(k, n, p0, alternative="two-sided",
                             continuity_correction=False):
    """z test for a single proportion.

    Tests whether the observed count k out of n trials is consistent
    with a true proportion p0.  The standard error of the difference is
    sqrt(p0(1-p0)/n) under the null.  The continuity correction nudges
    the observed proportion toward p0 by 1/(2n) before testing, which
    makes the normal approximation to the discrete binomial less
    generous.
    """
    _check_alternative(alternative)
    if n < 1:
        raise ValueError("n must be at least one")
    if not 0.0 <= p0 <= 1.0:
        raise ValueError("p0 must be between 0 and 1")
    if not 0 <= k <= n:
        raise ValueError("k must be between 0 and n")
    p_hat = k / n
    se = math.sqrt(p0 * (1.0 - p0) / n)
    if se == 0.0:
        return {"z": 0.0, "p": 1.0, "p_hat": p_hat}
    if continuity_correction:
        shift = 1.0 / (2.0 * n)
        p_hat = min(max(p_hat - shift, p0), p_hat) if p_hat > p0 else p_hat
        if p_hat < p0:
            p_hat = min(p_hat + shift, p0) if p_hat + shift < p0 else p0
    z = (p_hat - p0) / se
    return {"z": z, "p": _z_to_p(z, alternative), "p_hat": p_hat}


# ---------------------------------------------------------------------------
# Confidence intervals and effect size
# ---------------------------------------------------------------------------

def confidence_interval_mean(data, alpha=0.05):
    """t-based confidence interval for the population mean.

    mean +- t(1-alpha/2, n-1) * s / sqrt(n).  With a small sample this
    interval is honest about the extra uncertainty from estimating the
    standard deviation instead of knowing it.
    """
    n = len(data)
    if n < 2:
        raise ValueError("need at least two observations")
    m = _mean(data)
    s = _sample_stddev(data)
    tcrit = t_ppf(1.0 - alpha / 2.0, n - 1)
    half = tcrit * s / math.sqrt(n)
    return {"mean": m, "se": s / math.sqrt(n), "lower": m - half,
            "upper": m + half, "alpha": alpha, "df": n - 1}


def confidence_interval_proportion(k, n, alpha=0.05):
    """Wilson score interval for a binomial proportion.

    The Wald interval p_hat +- z*sqrt(p_hat(1-p_hat)/n) has terrible
    coverage when the true proportion is near 0 or 1, so this uses the
    Wilson interval instead, which comes from solving the quadratic for
    the p0 values the normal approximation would not reject.
    """
    if n < 1:
        raise ValueError("n must be at least one")
    if not 0 <= k <= n:
        raise ValueError("k must be between 0 and n")
    z = normal_cdf_inverse(1.0 - alpha / 2.0)
    p_hat = k / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n))
            / denom)
    lower = max(0.0, center - half)
    upper = min(1.0, center + half)
    if k == 0:
        lower = 0.0
    if k == n:
        upper = 1.0
    return {"p_hat": p_hat, "lower": lower, "upper": upper, "alpha": alpha}


def cohens_d(a, b):
    """Effect size for two groups, the standardized mean difference.

    Uses the pooled standard deviation so the result is comparable
    across studies that measure different scales.  Rules of thumb from
    Cohen: 0.2 small, 0.5 medium, 0.8 large.
    """
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        raise ValueError("each group needs at least two observations")
    m1, m2 = _mean(a), _mean(b)
    s1, s2 = _sample_stddev(a), _sample_stddev(b)
    sp = math.sqrt(((n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2) / (n1 + n2 - 2))
    return (m1 - m2) / sp


# ---------------------------------------------------------------------------
# Power and sample size (z test)
# ---------------------------------------------------------------------------

def power_z_test(effect_size, sigma, n, alpha=0.05, alternative="greater"):
    """Probability of rejecting a false null, for a known-sigma z test.

    effect_size is the true mean difference from mu0.  Under the
    alternative the test statistic is centered at effect_size*sqrt(n)/
    sigma, so the power is the probability that it clears the critical
    value.  The two-sided power is computed with both tails and is only
    an approximation, which is fine for planning sample sizes.
    """
    _check_alternative(alternative)
    if sigma <= 0.0 or n < 1:
        raise ValueError("sigma must be positive and n at least one")
    shift = effect_size * math.sqrt(n) / sigma
    if alternative == "greater":
        zcrit = normal_cdf_inverse(1.0 - alpha)
        return 1.0 - normal_cdf(zcrit - shift)
    if alternative == "less":
        zcrit = normal_cdf_inverse(alpha)
        return normal_cdf(zcrit - shift)
    zcrit = normal_cdf_inverse(1.0 - alpha / 2.0)
    return (1.0 - normal_cdf(zcrit - shift)
            + normal_cdf(-zcrit - shift))


def sample_size_z_test(effect_size, sigma, alpha=0.05, power=0.8,
                       alternative="greater"):
    """Smallest sample size giving the requested power for a z test.

    Solves the power equation by bisection over n and rounds up, so the
    returned number is the first integer with power at least the target.
    """
    _check_alternative(alternative)
    if not 0.0 < power < 1.0:
        raise ValueError("power must lie strictly between 0 and 1")
    if effect_size == 0.0:
        raise ValueError("effect size cannot be zero, power is meaningless")
    lo, hi = 1, 2
    while power_z_test(effect_size, sigma, hi, alpha, alternative) < power:
        hi *= 2
        if hi > 10 ** 9:
            raise ValueError("no feasible sample size found")
    while lo < hi:
        mid = (lo + hi) // 2
        if power_z_test(effect_size, sigma, mid, alpha, alternative) < power:
            lo = mid + 1
        else:
            hi = mid
    return lo


def decision(p, alpha=0.05):
    """Plain-language verdict for a p-value at a chosen significance level."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if p <= alpha:
        return "reject the null hypothesis"
    return "fail to reject the null hypothesis"
