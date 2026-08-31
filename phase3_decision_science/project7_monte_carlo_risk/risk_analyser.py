import math
import random


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def _mean(values):
    """Arithmetic mean of a non-empty sequence."""
    if not values:
        raise ValueError("need at least one value")
    return sum(values) / len(values)


def _variance_sample(values):
    """Sample variance (divides by n-1) of a non-empty sequence."""
    if len(values) < 2:
        raise ValueError("need at least two values for sample variance")
    mu = _mean(values)
    return sum((x - mu) ** 2 for x in values) / (len(values) - 1)


def _std_sample(values):
    return math.sqrt(_variance_sample(values))


def _sorted_copy(values):
    return sorted(values)


# ---------------------------------------------------------------------------
# Random number generation
# ---------------------------------------------------------------------------

def _uniform_pair(rng):
    """Two independent uniform draws on (0, 1) that are never exactly zero."""
    u1 = rng.random()
    u2 = rng.random()
    # Box-Muller needs the log of u1; guard against the degenerate zero draw
    while u1 == 0.0:
        u1 = rng.random()
    return u1, u2


def _box_muller(rng):
    """One standard normal draw via the exact Box-Muller transform.

    If U1 and U2 are independent uniform(0,1) random variables then
        Z = sqrt(-2 ln U1) * cos(2 pi U2)
    is standard normal.  The transform is exact, not an approximation.
    """
    u1, u2 = _uniform_pair(rng)
    magnitude = math.sqrt(-2.0 * math.log(u1))
    angle = 2.0 * math.pi * u2
    return magnitude * math.cos(angle)


def _standard_normal_sample(n, seed=None):
    """n independent standard normal draws using Box-Muller."""
    rng = random.Random(seed)
    return [_box_muller(rng) for _ in range(n)]


def _student_t_sample(n, df, seed=None):
    """n independent Student-t draws with df degrees of freedom.

    A t(df) variable is Z / sqrt(V/df) where Z is standard normal and
    V is chi-squared(df).  A chi-squared(df) variable is the sum of df
    squared standard normals.  Both pieces come from Box-Muller so the
    whole draw uses only uniform random numbers.
    """
    if df <= 0:
        raise ValueError("degrees of freedom must be positive")
    rng = random.Random(seed)
    draws = []
    for _ in range(n):
        z = _box_muller(rng)
        # chi-squared(df) = sum of df squared standard normals
        chi2 = sum(_box_muller(rng) ** 2 for _ in range(int(math.ceil(df))))
        # scale chi2 to match df exactly for integer df
        draws.append(z / math.sqrt(chi2 / df))
    return draws


# ---------------------------------------------------------------------------
# Return simulation
# ---------------------------------------------------------------------------

def simulate_normal_returns(mu, sigma, n_periods, n_paths, seed=None):
    """Simulate return paths from a normal distribution.

    Each path is a list of n_periods returns drawn from N(mu, sigma^2).
    Returns a list of n_paths lists.

    Parameters
    ----------
    mu        : mean return per period (e.g. daily mean)
    sigma     : standard deviation per period
    n_periods : length of each simulated path
    n_paths   : number of independent paths
    seed      : integer seed for reproducibility; None means unseeded
    """
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")
    if n_periods < 1 or n_paths < 1:
        raise ValueError("n_periods and n_paths must be at least 1")
    total = n_periods * n_paths
    zs = _standard_normal_sample(total, seed=seed)
    paths = []
    idx = 0
    for _ in range(n_paths):
        path = [mu + sigma * zs[idx + t] for t in range(n_periods)]
        idx += n_periods
        paths.append(path)
    return paths


def simulate_t_returns(mu, sigma, df, n_periods, n_paths, seed=None):
    """Simulate return paths from a scaled Student-t distribution.

    Fatter tails than the normal.  The t variable is scaled so that its
    standard deviation matches sigma when df > 2.

    Parameters
    ----------
    mu        : mean return per period
    sigma     : target standard deviation (matched when df > 2)
    df        : degrees of freedom; lower means fatter tails
    n_periods : length of each simulated path
    n_paths   : number of independent paths
    seed      : integer seed for reproducibility
    """
    if df <= 2:
        raise ValueError("df must be greater than 2 for finite variance")
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")
    if n_periods < 1 or n_paths < 1:
        raise ValueError("n_periods and n_paths must be at least 1")
    # The t(df) variable has variance df/(df-2), so scale to match sigma
    scale = sigma * math.sqrt((df - 2.0) / df)
    total = n_periods * n_paths
    ts = _student_t_sample(total, df, seed=seed)
    paths = []
    idx = 0
    for _ in range(n_paths):
        path = [mu + scale * ts[idx + t] for t in range(n_periods)]
        idx += n_periods
        paths.append(path)
    return paths


def fit_normal(returns):
    """Estimate mu and sigma from a return series.

    Returns (mu, sigma) as the sample mean and sample standard deviation.
    """
    return _mean(returns), _std_sample(returns)


def fit_t(returns, df=5):
    """Estimate mu and sigma for a t distribution with fixed df.

    The mean is the sample mean.  Sigma is chosen so that the fitted
    t distribution has the same variance as the sample.  df is fixed
    by the caller; estimating it properly requires maximum likelihood,
    which is outside the scope of this project.
    """
    if df <= 2:
        raise ValueError("df must be greater than 2 for finite variance")
    mu = _mean(returns)
    s = _std_sample(returns)
    # Var(scaled t) = sigma^2 * df/(df-2) = s^2, so sigma = s * sqrt((df-2)/df)
    sigma = s * math.sqrt((df - 2.0) / df)
    return mu, sigma


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------

def equity_curve(returns, initial=1.0):
    """Running product of (1 + r) for each return r.

    The first element is the initial wealth.  The i-th element is the
    wealth after i periods.  A return of -1 means total loss.
    """
    if not returns:
        raise ValueError("need at least one return")
    curve = [initial]
    for r in returns:
        if r <= -1.0:
            raise ValueError("a return of -1 or less means total loss; "
                             "check your data")
        curve.append(curve[-1] * (1.0 + r))
    return curve


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------

def drawdown_series(curve):
    """Drawdown at each point in an equity curve.

    The drawdown at time t is (peak up to t - value at t) / peak up to t.
    It is zero when the curve is at or above its previous peak and
    positive otherwise.  The returned list has the same length as curve.
    """
    if not curve:
        raise ValueError("curve must not be empty")
    peak = curve[0]
    dd = []
    for v in curve:
        if v > peak:
            peak = v
        dd.append((peak - v) / peak if peak > 0.0 else 0.0)
    return dd


def max_drawdown(curve):
    """Largest peak-to-trough decline as a fraction of the peak.

    Returns a value in [0, 1].  Zero means the curve never fell below
    its starting point.
    """
    dd = drawdown_series(curve)
    return max(dd)


def drawdown_duration(curve):
    """Number of periods spent below the previous peak.

    Counts every period where the drawdown is strictly positive.
    """
    dd = drawdown_series(curve)
    return sum(1 for d in dd if d > 0.0)


def max_drawdown_duration(curve):
    """Length of the longest single drawdown episode.

    An episode starts when the curve falls below its peak and ends when
    it recovers to the peak.  The duration is the number of periods in
    the longest such episode.
    """
    dd = drawdown_series(curve)
    max_run = 0
    current_run = 0
    for d in dd:
        if d > 0.0:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 0
    return max_run


def calmar_ratio(returns, periods_per_year=252):
    """Annualised return divided by maximum drawdown.

    A higher Calmar ratio means more return per unit of drawdown risk.
    Returns None when the maximum drawdown is zero.
    """
    if not returns:
        raise ValueError("need at least one return")
    curve = equity_curve(returns)
    mdd = max_drawdown(curve)
    if mdd == 0.0:
        return None
    mu = _mean(returns)
    annualised = (1.0 + mu) ** periods_per_year - 1.0
    return annualised / mdd


# ---------------------------------------------------------------------------
# Value at Risk
# ---------------------------------------------------------------------------

def _quantile(sorted_values, alpha):
    """The alpha-quantile of a sorted list by linear interpolation.

    alpha = 0.05 gives the 5th percentile.
    """
    n = len(sorted_values)
    if n == 0:
        raise ValueError("need at least one value")
    # map alpha to a fractional index
    pos = alpha * (n - 1)
    lo = int(pos)
    hi = lo + 1
    if hi >= n:
        return sorted_values[-1]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def historical_var(returns, alpha=0.05):
    """Historical Value at Risk at confidence level 1 - alpha.

    The loss (positive number) not exceeded in (1 - alpha) of past
    periods.  alpha = 0.05 gives the 95% VaR.

    The historical method makes no distributional assumption.  It reads
    the loss directly from the sorted return series.
    """
    if not returns:
        raise ValueError("need at least one return")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    s = _sorted_copy(returns)
    # The left tail is the worst returns; the alpha-quantile is the VaR
    return -_quantile(s, alpha)


def parametric_var(returns, alpha=0.05):
    """Parametric (variance-covariance) VaR assuming normally distributed returns.

    Fits a normal to the return series, then uses the inverse normal CDF
    to find the loss threshold.  The inverse normal CDF at alpha is
    approximated by the rational approximation of Peter Acklam, which
    is accurate to about 1.15e-9 across the full range.

    Parameters
    ----------
    returns : observed return series used to fit mu and sigma
    alpha   : tail probability (0.05 gives the 95% VaR)
    """
    if not returns:
        raise ValueError("need at least one return")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    mu, sigma = fit_normal(returns)
    z = _normal_ppf(alpha)
    return -(mu + sigma * z)


def _normal_ppf(p):
    """Inverse standard normal CDF at probability p.

    Uses the rational approximation of Peter Acklam (2002), accurate to
    about 1.15e-9.  This replaces scipy.stats.norm.ppf entirely.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be strictly between 0 and 1")
    # Coefficients from Acklam's approximation
    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]
    p_lo = 0.02425
    p_hi = 1.0 - p_lo
    if p < p_lo:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    elif p <= p_hi:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)


def normal_cdf(x):
    """Standard normal CDF using the math.erf identity.

    P(Z <= x) = 0.5 * (1 + erf(x / sqrt(2)))
    This is exact within floating-point arithmetic.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# CVaR (Expected Shortfall)
# ---------------------------------------------------------------------------

def historical_cvar(returns, alpha=0.05):
    """Historical CVaR (Expected Shortfall) at confidence level 1 - alpha.

    The mean loss in the worst alpha fraction of periods.  Always larger
    than the VaR at the same level because it averages the tail rather
    than just marking its boundary.

    CVaR = -E[X | X <= VaR_alpha]
    """
    if not returns:
        raise ValueError("need at least one return")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    s = _sorted_copy(returns)
    # Keep only the worst alpha fraction
    cutoff = max(1, int(math.floor(alpha * len(s))))
    tail = s[:cutoff]
    return -_mean(tail)


def parametric_cvar(returns, alpha=0.05):
    """Parametric CVaR assuming normally distributed returns.

    For a normal distribution the CVaR at level alpha is

        CVaR = -(mu - sigma * phi(z_alpha) / alpha)

    where phi is the standard normal PDF and z_alpha = Phi^{-1}(alpha).
    This is an exact closed-form result, not a simulation.
    """
    if not returns:
        raise ValueError("need at least one return")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    mu, sigma = fit_normal(returns)
    z = _normal_ppf(alpha)
    phi_z = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return -(mu - sigma * phi_z / alpha)


# ---------------------------------------------------------------------------
# Monte Carlo risk metrics
# ---------------------------------------------------------------------------

def simulate_risk_metrics(returns, n_paths=10000, n_periods=None,
                          distribution="normal", df=5, alpha=0.05, seed=None):
    """Run a Monte Carlo simulation and compute risk metrics across paths.

    Fits a distribution to returns, generates n_paths simulated paths of
    n_periods each, then computes VaR, CVaR, maximum drawdown, and ruin
    probability from the cross-section of path outcomes.

    Parameters
    ----------
    returns      : historical return series used to fit the distribution
    n_paths      : number of simulated paths
    n_periods    : length of each path; defaults to len(returns)
    distribution : 'normal' or 't'
    df           : degrees of freedom for the t distribution
    alpha        : tail probability for VaR and CVaR
    seed         : integer seed for reproducibility

    Returns a dict with keys:
        paths            list of equity curves (each a list of wealth values)
        final_wealth     final wealth of each path
        path_var         VaR from the distribution of final returns
        path_cvar        CVaR from the distribution of final returns
        path_max_dd      maximum drawdown of each path
        mean_max_dd      mean maximum drawdown across paths
        median_max_dd    median maximum drawdown across paths
        worst_max_dd     largest maximum drawdown across all paths
    """
    if not returns:
        raise ValueError("need at least one return")
    if n_periods is None:
        n_periods = len(returns)
    if distribution == "normal":
        mu, sigma = fit_normal(returns)
        paths = simulate_normal_returns(mu, sigma, n_periods, n_paths, seed=seed)
    elif distribution == "t":
        mu, sigma = fit_t(returns, df=df)
        paths = simulate_t_returns(mu, sigma, df, n_periods, n_paths, seed=seed)
    else:
        raise ValueError("distribution must be 'normal' or 't'")

    equity_curves = [equity_curve(p) for p in paths]

    # Final returns: total return of each path
    final_returns = [(ec[-1] - ec[0]) / ec[0] for ec in equity_curves]

    # VaR and CVaR from final return distribution
    s_final = _sorted_copy(final_returns)
    path_var = -_quantile(s_final, alpha)
    cutoff = max(1, int(math.floor(alpha * len(s_final))))
    path_cvar = -_mean(s_final[:cutoff])

    # Drawdown across paths
    mdd_list = [max_drawdown(ec) for ec in equity_curves]
    s_mdd = _sorted_copy(mdd_list)
    mean_mdd = _mean(mdd_list)
    median_mdd = _quantile(s_mdd, 0.5)
    worst_mdd = max(mdd_list)

    return {
        "paths": equity_curves,
        "final_wealth": [ec[-1] for ec in equity_curves],
        "path_var": path_var,
        "path_cvar": path_cvar,
        "path_max_dd": mdd_list,
        "mean_max_dd": mean_mdd,
        "median_max_dd": median_mdd,
        "worst_max_dd": worst_mdd,
    }


def ruin_probability(results, threshold=0.2):
    """Fraction of simulated paths where max drawdown exceeds threshold.

    threshold = 0.2 means any path that fell more than 20% from its peak
    at any point is counted as a ruin event.

    Parameters
    ----------
    results   : dict returned by simulate_risk_metrics
    threshold : drawdown fraction that defines ruin (0 < threshold <= 1)
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    mdd_list = results["path_max_dd"]
    if not mdd_list:
        raise ValueError("results contain no paths")
    ruined = sum(1 for d in mdd_list if d >= threshold)
    return ruined / len(mdd_list)


def ruin_probability_ci(results, threshold=0.2, confidence=0.95, n_boot=2000,
                        seed=None):
    """Bootstrap confidence interval around the ruin probability.

    Resamples the max-drawdown list with replacement n_boot times and
    computes the ruin probability on each resample.  Returns
    (point_estimate, lower, upper).
    """
    mdd_list = results["path_max_dd"]
    n = len(mdd_list)
    point = ruin_probability(results, threshold)
    rng = random.Random(seed)
    boot_estimates = []
    for _ in range(n_boot):
        resample = [mdd_list[rng.randrange(n)] for _ in range(n)]
        ruined = sum(1 for d in resample if d >= threshold)
        boot_estimates.append(ruined / n)
    boot_estimates.sort()
    tail = (1.0 - confidence) / 2.0
    lower = _quantile(boot_estimates, tail)
    upper = _quantile(boot_estimates, 1.0 - tail)
    return point, lower, upper


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals for any scalar metric
# ---------------------------------------------------------------------------

def bootstrap_ci(values, stat_fn, confidence=0.95, n_boot=2000, seed=None):
    """Percentile bootstrap confidence interval for stat_fn applied to values.

    stat_fn takes a list and returns a scalar.  Returns
    (point_estimate, lower, upper).
    """
    if not values:
        raise ValueError("need at least one value")
    n = len(values)
    point = stat_fn(values)
    rng = random.Random(seed)
    boot = []
    for _ in range(n_boot):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        boot.append(stat_fn(resample))
    boot.sort()
    tail = (1.0 - confidence) / 2.0
    lower = _quantile(boot, tail)
    upper = _quantile(boot, 1.0 - tail)
    return point, lower, upper


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def risk_report(returns, n_paths=10000, n_periods=None,
                distribution="normal", df=5, alpha=0.05,
                ruin_threshold=0.2, seed=None):
    """Full risk report for a return series.

    Runs the Monte Carlo simulation and assembles every metric into a
    single dict ready for printing or further analysis.

    Keys in the returned dict
    -------------------------
    n_returns        number of observed returns
    mu               sample mean return
    sigma            sample standard deviation
    hist_var         historical VaR at level alpha
    hist_cvar        historical CVaR at level alpha
    param_var        parametric VaR (normal assumption)
    param_cvar       parametric CVaR (normal assumption)
    obs_max_dd       maximum drawdown of the observed series
    obs_mdd_dur      max drawdown duration of the observed series
    obs_calmar       Calmar ratio of the observed series
    sim_var          VaR from Monte Carlo final returns
    sim_cvar         CVaR from Monte Carlo final returns
    sim_mean_mdd     mean max drawdown across simulated paths
    sim_median_mdd   median max drawdown across simulated paths
    sim_worst_mdd    worst max drawdown across simulated paths
    ruin_prob        fraction of paths breaching ruin_threshold
    ruin_lower       lower bound of 95% CI on ruin probability
    ruin_upper       upper bound of 95% CI on ruin probability
    alpha            tail probability used throughout
    ruin_threshold   drawdown threshold that defines ruin
    distribution     distribution used for simulation
    n_paths          number of simulated paths
    n_periods        length of each simulated path
    """
    mu, sigma = fit_normal(returns)
    obs_curve = equity_curve(returns)
    results = simulate_risk_metrics(
        returns, n_paths=n_paths, n_periods=n_periods,
        distribution=distribution, df=df, alpha=alpha, seed=seed
    )
    point, lower, upper = ruin_probability_ci(
        results, threshold=ruin_threshold, seed=seed
    )
    calmar = calmar_ratio(returns)
    return {
        "n_returns":       len(returns),
        "mu":              mu,
        "sigma":           sigma,
        "hist_var":        historical_var(returns, alpha),
        "hist_cvar":       historical_cvar(returns, alpha),
        "param_var":       parametric_var(returns, alpha),
        "param_cvar":      parametric_cvar(returns, alpha),
        "obs_max_dd":      max_drawdown(obs_curve),
        "obs_mdd_dur":     max_drawdown_duration(obs_curve),
        "obs_calmar":      calmar,
        "sim_var":         results["path_var"],
        "sim_cvar":        results["path_cvar"],
        "sim_mean_mdd":    results["mean_max_dd"],
        "sim_median_mdd":  results["median_max_dd"],
        "sim_worst_mdd":   results["worst_max_dd"],
        "ruin_prob":       point,
        "ruin_lower":      lower,
        "ruin_upper":      upper,
        "alpha":           alpha,
        "ruin_threshold":  ruin_threshold,
        "distribution":    distribution,
        "n_paths":         n_paths,
        "n_periods":       n_periods if n_periods is not None else len(returns),
    }
