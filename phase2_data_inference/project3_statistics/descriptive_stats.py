import math


def _as_numbers(data):
    """Coerce an iterable to a list of floats, rejecting empties."""
    values = [float(x) for x in data]
    if not values:
        raise ValueError("need at least one value, got an empty sequence")
    return values


def mean(data):
    """Arithmetic mean: sum of the values divided by their count."""
    values = _as_numbers(data)
    return sum(values) / len(values)


def median(data):
    """Middle value.  For an even count, the mean of the two middles."""
    values = _as_numbers(data)
    ordered = sorted(values)
    n = len(ordered)
    if n % 2 == 1:
        return ordered[n // 2]
    return (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0


def mode(data):
    """The most frequent value(s), as a list in order of first appearance.

    A list is returned because ties are possible: for [1, 1, 2, 2] both 1
    and 2 are modes.  With every value unique, every value is a mode.
    """
    values = _as_numbers(data)
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts.values())
    result = []
    for v in values:
        if counts[v] == best and v not in result:
            result.append(v)
    return result


def data_range(data):
    """The spread between the largest and smallest value."""
    values = _as_numbers(data)
    return max(values) - min(values)


def quantile(data, q):
    """The q-quantile, linear interpolation, q in [0, 1].

    Uses the same convention as numpy.percentile with its default 'linear'
    method, so results can be checked directly against numpy.
    """
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be between 0 and 1, got {}".format(q))
    values = _as_numbers(data)
    ordered = sorted(values)
    n = len(ordered)
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    frac = pos - lo
    if frac == 0.0:
        return ordered[lo]
    return ordered[lo] + (ordered[lo + 1] - ordered[lo]) * frac


def iqr(data):
    """Interquartile range: the 0.75 quantile minus the 0.25 quantile."""
    return quantile(data, 0.75) - quantile(data, 0.25)


def variance(data, sample=True):
    """Variance of the data.

    sample=True uses n-1 (Bessel's correction), the unbiased estimator of
    the population variance from a sample.  sample=False divides by n.
    Requires at least 2 values in sample mode.
    """
    values = _as_numbers(data)
    n = len(values)
    if sample and n < 2:
        raise ValueError("sample variance needs at least two values")
    m = sum(values) / n
    return sum((x - m) ** 2 for x in values) / (n - 1 if sample else n)


def stddev(data, sample=True):
    """Standard deviation: the square root of the variance."""
    return math.sqrt(variance(data, sample=sample))


def _central_moment(data, order):
    """m_order = sum((x - mean)^order) / n, the population central moment."""
    values = _as_numbers(data)
    m = sum(values) / len(values)
    return sum((x - m) ** order for x in values) / len(values)


def skewness(data, bias=True):
    """Moment coefficient of skewness.

    bias=True gives the population moment formula
        g1 = m3 / m2^(3/2)
    bias=False applies the small-sample adjustment
        sqrt(n(n-1)) / (n-2) * g1
    matching scipy.stats.skew.  Undefined (division by zero) when all
    values are equal.
    """
    values = _as_numbers(data)
    n = len(values)
    if bias is False and n < 3:
        raise ValueError("sample skewness needs at least three values")
    if min(values) == max(values):
        raise ValueError("data has zero variance; skewness is undefined")
    m2 = _central_moment(values, 2)
    m3 = _central_moment(values, 3)
    g1 = m3 / m2 ** 1.5
    if bias:
        return g1
    return math.sqrt(n * (n - 1)) / (n - 2) * g1


def kurtosis(data, bias=True, fisher=True):
    """Excess kurtosis by default, using the moment formulas.

    bias=True gives the population excess kurtosis
        g2 = m4 / m2^2 - 3
    bias=False applies the small-sample adjustment
        (n+1)(n-1) / ((n-2)(n-3)) * (m4/m2^2 - 3 + 6/(n+1))
    matching scipy.stats.kurtosis.  With fisher=False the raw fourth
    standardized moment m4 / m2^2 is returned instead of the excess.
    Undefined when all values are equal.
    """
    values = _as_numbers(data)
    n = len(values)
    if bias is False and n < 4:
        raise ValueError("sample kurtosis needs at least four values")
    if min(values) == max(values):
        raise ValueError("data has zero variance; kurtosis is undefined")
    m2 = _central_moment(values, 2)
    m4 = _central_moment(values, 4)
    if bias:
        val = m4 / m2 ** 2 - 3.0
    else:
        val = ((n + 1.0) * (n - 1.0) / ((n - 2.0) * (n - 3.0))
               * (m4 / m2 ** 2 - 3.0 + 6.0 / (n + 1.0)))
    if not fisher:
        val = val + 3.0
    return val


def _joint_pairs(x, y):
    a = _as_numbers(x)
    b = _as_numbers(y)
    if len(a) != len(b):
        raise ValueError("both series must have the same length, got {} and {}".format(
            len(a), len(b)))
    return a, b


def covariance(x, y, sample=True):
    """Covariance of two series.

    Population version divides by n; sample version by n-1, which is the
    unbiased estimate and what numpy.cov(..., ddof=1) returns.
    """
    a, b = _joint_pairs(x, y)
    n = len(a)
    if sample and n < 2:
        raise ValueError("sample covariance needs at least two pairs")
    ma = sum(a) / n
    mb = sum(b) / n
    return sum((u - ma) * (v - mb) for u, v in zip(a, b)) / (n - 1 if sample else n)


def pearson_correlation(x, y):
    """Pearson correlation coefficient, always in [-1, 1].

    Computed as
        sum((x-mx)(y-my)) / sqrt(sum((x-mx)^2) * sum((y-my)^2))
    which is the same number whether you think in population or sample
    terms, because the 1/n cancels.  Undefined when either series is
    constant.
    """
    a, b = _joint_pairs(x, y)
    n = len(a)
    if min(a) == max(a) or min(b) == max(b):
        raise ValueError("correlation is undefined when a series is constant")
    ma = sum(a) / n
    mb = sum(b) / n
    dx = [u - ma for u in a]
    dy = [v - mb for v in b]
    ssx = sum(d * d for d in dx)
    ssy = sum(d * d for d in dy)
    return sum(u * v for u, v in zip(dx, dy)) / math.sqrt(ssx * ssy)


def pct_change(values):
    """Period-over-period percentage change: v_i / v_{i-1} - 1.

    The first value has no predecessor and is skipped, so the result has
    one fewer element than the input.  Used to turn a price series into
    returns.
    """
    vals = _as_numbers(values)
    if len(vals) < 2:
        raise ValueError("pct_change needs at least two values")
    out = []
    prev = vals[0]
    for v in vals[1:]:
        if prev == 0.0:
            raise ValueError("cannot compute a percentage change off zero")
        out.append(v / prev - 1.0)
        prev = v
    return out


def summary(data):
    """One dict with every single-variable statistic for the data."""
    values = _as_numbers(data)
    return {
        'count': len(values),
        'mean': mean(values),
        'median': median(values),
        'mode': mode(values),
        'min': min(values),
        'max': max(values),
        'range': data_range(values),
        'q1': quantile(values, 0.25),
        'q3': quantile(values, 0.75),
        'iqr': iqr(values),
        'variance_population': variance(values, sample=False),
        'variance_sample': variance(values, sample=True),
        'stddev_population': stddev(values, sample=False),
        'stddev_sample': stddev(values, sample=True),
        'skewness': skewness(values, bias=True),
        'kurtosis_excess': kurtosis(values, bias=True, fisher=True),
    }
