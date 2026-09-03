import math
import random


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mean(values):
    if not values:
        raise ValueError("need at least one value")
    return sum(values) / len(values)


def _variance_sample(values):
    if len(values) < 2:
        raise ValueError("need at least two values for sample variance")
    mu = _mean(values)
    return sum((x - mu) ** 2 for x in values) / (len(values) - 1)


def _std_sample(values):
    return math.sqrt(_variance_sample(values))


def _sorted_copy(values):
    return sorted(values)


def _quantile(sorted_values, alpha):
    """Alpha-quantile of a sorted list by linear interpolation."""
    n = len(sorted_values)
    if n == 0:
        raise ValueError("need at least one value")
    pos = alpha * (n - 1)
    lo = int(pos)
    hi = lo + 1
    if hi >= n:
        return sorted_values[-1]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


# ---------------------------------------------------------------------------
# Trade log
# ---------------------------------------------------------------------------

def load_trade_log(records):
    """Validate and normalise a list of trade dicts.

    Each record must contain at least a 'return' key with a numeric
    value.  Optional keys are 'date', 'direction' ('long' or 'short'),
    'entry', and 'exit'.  Missing optional fields are filled with None.

    Returns a list of cleaned dicts, one per trade.

    Parameters
    ----------
    records : list of dict, each with at minimum {'return': float}
    """
    if not records:
        raise ValueError("trade log must contain at least one trade")
    cleaned = []
    for i, rec in enumerate(records):
        if "return" not in rec:
            raise ValueError(
                "record {} has no 'return' key".format(i)
            )
        try:
            r = float(rec["return"])
        except (TypeError, ValueError):
            raise ValueError(
                "record {} has a non-numeric return: {!r}".format(
                    i, rec["return"]
                )
            )
        direction = rec.get("direction", None)
        if direction is not None and direction not in ("long", "short"):
            raise ValueError(
                "record {} direction must be 'long' or 'short', "
                "got {!r}".format(i, direction)
            )
        cleaned.append({
            "date":      rec.get("date", None),
            "direction": direction,
            "entry":     rec.get("entry", None),
            "exit":      rec.get("exit", None),
            "return":    r,
        })
    return cleaned


def returns_from_log(trade_log):
    """Extract the list of per-trade returns from a cleaned trade log."""
    return [t["return"] for t in trade_log]


def filter_by_direction(trade_log, direction):
    """Return only the trades in a given direction ('long' or 'short')."""
    if direction not in ("long", "short"):
        raise ValueError("direction must be 'long' or 'short'")
    return [t for t in trade_log if t["direction"] == direction]


# ---------------------------------------------------------------------------
# Basic performance metrics
# ---------------------------------------------------------------------------

def expectancy(returns):
    """Mean return per trade.

    This is the single most important number in a backtest.  A positive
    expectancy means the strategy makes money on average.  Combined with
    the number of trades per year it gives the expected annual return.

    expectancy = (1/n) * sum(r_i)
    """
    return _mean(returns)


def win_rate(returns):
    """Fraction of trades with a strictly positive return."""
    if not returns:
        raise ValueError("need at least one return")
    return sum(1 for r in returns if r > 0.0) / len(returns)


def loss_rate(returns):
    """Fraction of trades with a strictly negative return."""
    if not returns:
        raise ValueError("need at least one return")
    return sum(1 for r in returns if r < 0.0) / len(returns)


def average_win(returns):
    """Mean return of winning trades.  Returns None if there are no wins."""
    wins = [r for r in returns if r > 0.0]
    return _mean(wins) if wins else None


def average_loss(returns):
    """Mean return of losing trades (a negative number).

    Returns None if there are no losses.
    """
    losses = [r for r in returns if r < 0.0]
    return _mean(losses) if losses else None


def profit_factor(returns):
    """Gross profit divided by gross loss (as a positive number).

    Values above 1 mean the winners outweigh the losers.  A value of 2
    means the strategy earns two dollars in winners for every dollar lost.
    Returns None when there are no losing trades.
    """
    gross_profit = sum(r for r in returns if r > 0.0)
    gross_loss   = sum(abs(r) for r in returns if r < 0.0)
    if gross_loss == 0.0:
        return None
    return gross_profit / gross_loss


def payoff_ratio(returns):
    """Average win divided by the absolute value of the average loss.

    Together with win_rate this fully determines expectancy:
        expectancy = win_rate * avg_win + loss_rate * avg_loss

    Returns None when there are no wins or no losses.
    """
    aw = average_win(returns)
    al = average_loss(returns)
    if aw is None or al is None:
        return None
    return aw / abs(al)


def expectancy_from_components(returns):
    """Expectancy decomposed into win rate, payoff ratio, and loss rate.

    Returns a dict with all components and the reconstructed expectancy.
    Useful for understanding which piece of the edge is doing the work.

    expectancy = win_rate * avg_win + loss_rate * avg_loss
    """
    wr  = win_rate(returns)
    lr  = loss_rate(returns)
    aw  = average_win(returns)
    al  = average_loss(returns)
    ev  = expectancy(returns)
    reconstructed = (wr * (aw or 0.0)) + (lr * (al or 0.0))
    return {
        "expectancy":         ev,
        "win_rate":           wr,
        "loss_rate":          lr,
        "average_win":        aw,
        "average_loss":       al,
        "payoff_ratio":       payoff_ratio(returns),
        "profit_factor":      profit_factor(returns),
        "reconstructed_ev":   reconstructed,
    }


# ---------------------------------------------------------------------------
# Risk-adjusted performance
# ---------------------------------------------------------------------------

def sharpe_ratio(returns, risk_free=0.0, periods_per_year=252):
    """Annualised Sharpe ratio.

    Measures mean excess return per unit of total return volatility.

    Sharpe = (mean(r) - r_f) / std(r) * sqrt(periods_per_year)

    A Sharpe above 1.0 is generally considered acceptable.  Above 2.0
    is strong.  The ratio assumes returns are i.i.d. and normally
    distributed, which is an approximation for any real strategy.

    Parameters
    ----------
    risk_free        : risk-free return per period (default 0)
    periods_per_year : trading periods in a year (252 for daily)
    """
    if len(returns) < 2:
        raise ValueError("need at least two returns for Sharpe ratio")
    excess = [r - risk_free for r in returns]
    if min(excess) == max(excess):
        return None
    mu = _mean(excess)
    sigma = _std_sample(excess)
    return (mu / sigma) * math.sqrt(periods_per_year)


def sortino_ratio(returns, risk_free=0.0, periods_per_year=252):
    """Annualised Sortino ratio.

    Like the Sharpe but uses only downside deviation in the denominator.
    Upside volatility is not penalised.

    Sortino = (mean(r) - r_f) / downside_std(r) * sqrt(periods_per_year)

    Downside deviation: std of returns that fall below the risk-free rate.

    Parameters
    ----------
    risk_free        : minimum acceptable return per period (default 0)
    periods_per_year : trading periods in a year (252 for daily)
    """
    if not returns:
        raise ValueError("need at least one return")
    excess_mean = _mean(returns) - risk_free
    downside = [r - risk_free for r in returns if r < risk_free]
    if not downside:
        return None
    if len(downside) < 2:
        downside_std = abs(downside[0])
    else:
        downside_std = math.sqrt(
            sum(d ** 2 for d in downside) / len(downside)
        )
    if downside_std == 0.0:
        return None
    return (excess_mean / downside_std) * math.sqrt(periods_per_year)


# ---------------------------------------------------------------------------
# Equity curve and drawdown
# ---------------------------------------------------------------------------

def equity_curve(returns, initial=1.0):
    """Compounded wealth curve starting from initial.

    Each element is the wealth after that many trades.  The first
    element is the initial wealth before any trades.
    """
    if not returns:
        raise ValueError("need at least one return")
    curve = [initial]
    for r in returns:
        if r <= -1.0:
            raise ValueError(
                "return of -1 or worse means total ruin; got {!r}".format(r)
            )
        curve.append(curve[-1] * (1.0 + r))
    return curve


def drawdown_series(curve):
    """Drawdown fraction at each point in the equity curve."""
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
    """Largest peak-to-trough decline as a fraction of the peak."""
    return max(drawdown_series(curve))


def max_drawdown_duration(curve):
    """Length of the longest uninterrupted drawdown episode."""
    dd = drawdown_series(curve)
    max_run = current_run = 0
    for d in dd:
        if d > 0.0:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


def calmar_ratio(returns, periods_per_year=252):
    """Annualised return divided by max drawdown.

    Returns None when max drawdown is zero.
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
# Statistical edge verification
# ---------------------------------------------------------------------------

def _normal_cdf(x):
    """Standard normal CDF via the math.erf identity."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _t_sf(t, df):
    """Survival function of the t distribution: P(T > t).

    Uses a continued-fraction approximation to the regularised incomplete
    beta function.  Accurate enough for all practical sample sizes.
    """
    x = df / (df + t * t)
    # regularised incomplete beta I_x(df/2, 1/2)
    a = df / 2.0
    b = 0.5
    # Lentz continued fraction for the incomplete beta
    tiny = 1e-300
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

    def _betacf(a, b, x):
        qab = a + b
        qap = a + 1.0
        qam = a - 1.0
        c = 1.0
        d = 1.0 - qab * x / qap
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        h = d
        for m in range(1, 200):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + aa / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + aa / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 3e-7:
                break
        return h

    if x < (a + 1.0) / (a + b + 2.0):
        bt = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta)
        ibeta = bt * _betacf(a, b, x) / a
    else:
        bt = math.exp(math.log(1.0 - x) * b + math.log(x) * a - lbeta)
        ibeta = 1.0 - bt * _betacf(b, a, 1.0 - x) / b
    # ibeta here is I_x(df/2, 1/2) which equals the two-tailed p-value
    # sf = p-value / 2  for the one-tailed test against the upper tail
    return ibeta / 2.0


def t_test_one_sample(returns, mu0=0.0):
    """Two-sided one-sample t-test: is mean(returns) different from mu0?

    The null hypothesis H0: mean return = mu0.
    For edge verification mu0 is 0 -- does the strategy earn anything?

    The t-statistic is

        t = (x_bar - mu0) / (s / sqrt(n))

    where x_bar is the sample mean, s is the sample standard deviation,
    and n is the number of trades.  Under H0 this follows a t(n-1)
    distribution.

    Returns a dict with t_stat, df, p_value, and a plain-English verdict.
    """
    if len(returns) < 2:
        raise ValueError("need at least two returns for a t-test")
    n    = len(returns)
    xbar = _mean(returns)
    if min(returns) == max(returns):
        raise ValueError(
            "all returns are identical; the t-test is undefined"
        )
    s    = _std_sample(returns)
    t_stat = (xbar - mu0) / (s / math.sqrt(n))
    df     = n - 1
    # two-sided p-value
    p_value = 2.0 * _t_sf(abs(t_stat), df)
    return {
        "t_stat":  t_stat,
        "df":      df,
        "p_value": p_value,
        "significant_at_5pct": p_value < 0.05,
        "significant_at_1pct": p_value < 0.01,
    }


def confidence_interval_mean(returns, alpha=0.05):
    """Two-sided confidence interval for the mean return per trade.

    Uses the t distribution so it is exact for any sample size.

    CI = x_bar +/- t_{alpha/2, n-1} * s / sqrt(n)

    Returns (lower, upper).
    """
    if len(returns) < 2:
        raise ValueError("need at least two returns")
    n    = len(returns)
    xbar = _mean(returns)
    s    = _std_sample(returns)
    df   = n - 1
    # find t critical value by bisection on the survival function
    tail = alpha / 2.0
    t_crit = _t_ppf(1.0 - tail, df)
    margin = t_crit * s / math.sqrt(n)
    return xbar - margin, xbar + margin


def _t_ppf(p, df):
    """Inverse t CDF at probability p with df degrees of freedom.

    Found by bisection on _t_sf.  Works for any p in (0, 1).
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    # t_sf(t, df) = 1 - p for the upper tail, so we want t where sf = 1-p
    target_sf = 1.0 - p
    # bracket
    lo, hi = 0.0, 1.0
    while _t_sf(hi, df) > target_sf:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _t_sf(mid, df) > target_sf:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_ci(values, stat_fn, confidence=0.95, n_boot=2000, seed=None):
    """Percentile bootstrap confidence interval for stat_fn(values).

    Returns (point_estimate, lower, upper).
    """
    if not values:
        raise ValueError("need at least one value")
    n     = len(values)
    point = stat_fn(values)
    rng   = random.Random(seed)
    boot  = []
    for _ in range(n_boot):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        boot.append(stat_fn(resample))
    boot.sort()
    tail  = (1.0 - confidence) / 2.0
    lower = _quantile(boot, tail)
    upper = _quantile(boot, 1.0 - tail)
    return point, lower, upper


def bootstrap_equity_curve(returns, n_boot=1000, seed=None):
    """Bootstrap the equity curve to get a distribution of outcomes.

    Resamples the trade returns with replacement n_boot times and builds
    an equity curve for each resample.  Returns a dict with:
        curves         list of n_boot equity curves
        final_wealth   final wealth of each resampled curve
        median_curve   element-wise median across all curves
        ci_lower       element-wise 5th percentile curve
        ci_upper       element-wise 95th percentile curve
    """
    if not returns:
        raise ValueError("need at least one return")
    n   = len(returns)
    rng = random.Random(seed)
    curves = []
    for _ in range(n_boot):
        resample = [returns[rng.randrange(n)] for _ in range(n)]
        curves.append(equity_curve(resample))
    # element-wise percentiles across curves
    n_points = len(curves[0])
    median_curve = []
    ci_lower     = []
    ci_upper     = []
    for t in range(n_points):
        col = _sorted_copy([c[t] for c in curves])
        median_curve.append(_quantile(col, 0.50))
        ci_lower.append(_quantile(col, 0.05))
        ci_upper.append(_quantile(col, 0.95))
    return {
        "curves":       curves,
        "final_wealth": [c[-1] for c in curves],
        "median_curve": median_curve,
        "ci_lower":     ci_lower,
        "ci_upper":     ci_upper,
    }


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def permutation_test(returns, stat_fn, n_permutations=5000, seed=None):
    """Test whether the observed statistic could arise from random ordering.

    The null hypothesis is that the order of returns does not matter --
    any permutation is equally likely.  This tests whether the strategy's
    edge is in the individual trade outcomes or in the sequencing of those
    outcomes (e.g. a momentum effect).

    Shuffles the returns n_permutations times and computes stat_fn on
    each shuffle.  The p-value is the fraction of shuffles that produce
    a statistic as extreme as or more extreme than the observed one.

    Returns a dict with the observed statistic, the null distribution,
    and the one-sided p-value (fraction of permutations >= observed).
    """
    if not returns:
        raise ValueError("need at least one return")
    observed = stat_fn(returns)
    rng      = random.Random(seed)
    null     = []
    r        = list(returns)
    for _ in range(n_permutations):
        rng.shuffle(r)
        null.append(stat_fn(r))
    p_value = sum(1 for v in null if v >= observed) / len(null)
    return {
        "observed":      observed,
        "null":          null,
        "p_value":       p_value,
        "significant_at_5pct": p_value < 0.05,
    }


# ---------------------------------------------------------------------------
# Full backtest report
# ---------------------------------------------------------------------------

def backtest_report(trade_log, risk_free=0.0, periods_per_year=252,
                    n_boot=1000, seed=None):
    """Full backtest report for a trade log.

    Runs every metric and returns a single dict.  The trade log should
    be the output of load_trade_log, or any list of dicts with a
    'return' key.

    Keys in the returned dict
    -------------------------
    n_trades           number of trades
    returns            list of per-trade returns

    expectancy         mean return per trade
    expectancy_ci      (lower, upper) 95% CI on expectancy
    win_rate           fraction of winning trades
    loss_rate          fraction of losing trades
    average_win        mean return of winning trades
    average_loss       mean return of losing trades
    profit_factor      gross profit / gross loss
    payoff_ratio       avg_win / abs(avg_loss)

    sharpe             annualised Sharpe ratio
    sortino            annualised Sortino ratio
    calmar             annualised Calmar ratio
    max_drawdown       max peak-to-trough decline
    max_dd_duration    longest drawdown episode in trades

    t_stat             t-statistic for H0: mean return = 0
    p_value            two-sided p-value
    significant_5pct   True if p < 0.05
    significant_1pct   True if p < 0.01

    boot_final_median  median final wealth from bootstrap
    boot_final_5pct    5th percentile final wealth from bootstrap
    boot_final_95pct   95th percentile final wealth from bootstrap
    """
    returns = returns_from_log(trade_log)
    n       = len(returns)

    comps   = expectancy_from_components(returns)
    ev      = comps["expectancy"]
    _, ev_lo, ev_hi = bootstrap_ci(
        returns, lambda v: sum(v) / len(v),
        confidence=0.95, n_boot=n_boot, seed=seed
    )

    sharpe  = sharpe_ratio(returns, risk_free, periods_per_year) \
              if n >= 2 else None
    sortino = sortino_ratio(returns, risk_free, periods_per_year)
    calmar  = calmar_ratio(returns, periods_per_year)

    curve   = equity_curve(returns)
    mdd     = max_drawdown(curve)
    mdd_dur = max_drawdown_duration(curve)

    ttest   = t_test_one_sample(returns, mu0=0.0) if n >= 2 else None

    boot    = bootstrap_equity_curve(returns, n_boot=n_boot, seed=seed)
    fw      = _sorted_copy(boot["final_wealth"])

    return {
        "n_trades":          n,
        "returns":           returns,

        "expectancy":        ev,
        "expectancy_ci":     (ev_lo, ev_hi),
        "win_rate":          comps["win_rate"],
        "loss_rate":         comps["loss_rate"],
        "average_win":       comps["average_win"],
        "average_loss":      comps["average_loss"],
        "profit_factor":     comps["profit_factor"],
        "payoff_ratio":      comps["payoff_ratio"],

        "sharpe":            sharpe,
        "sortino":           sortino,
        "calmar":            calmar,
        "max_drawdown":      mdd,
        "max_dd_duration":   mdd_dur,

        "t_stat":            ttest["t_stat"]    if ttest else None,
        "p_value":           ttest["p_value"]   if ttest else None,
        "significant_5pct":  ttest["significant_at_5pct"] if ttest else None,
        "significant_1pct":  ttest["significant_at_1pct"] if ttest else None,

        "boot_final_median": _quantile(fw, 0.50),
        "boot_final_5pct":   _quantile(fw, 0.05),
        "boot_final_95pct":  _quantile(fw, 0.95),
    }


# ---------------------------------------------------------------------------
# Simulate a strategy on a return series
# ---------------------------------------------------------------------------

def simulate_ma_crossover(returns, fast=10, slow=30):
    """Simulate a simple moving-average crossover strategy.

    Goes long when the fast MA is above the slow MA, flat otherwise.
    The trade return on day t is the next day's return when long, or
    zero when flat.

    This is included not as a recommended strategy but as a concrete
    example of generating a trade log from a raw return series so the
    backtester has something to work with without needing a separate
    data file of pre-computed trades.

    Parameters
    ----------
    returns : list of daily returns in chronological order
    fast    : lookback for the fast moving average
    slow    : lookback for the slow moving average (must be > fast)
    """
    if slow <= fast:
        raise ValueError("slow period must be greater than fast period")
    if len(returns) < slow + 1:
        raise ValueError(
            "need at least {} returns for these MA periods".format(slow + 1)
        )
    # compute running sums for the two MAs
    trades = []
    for i in range(slow - 1, len(returns) - 1):
        fast_ma = sum(returns[i - fast + 1: i + 1]) / fast
        slow_ma = sum(returns[i - slow + 1: i + 1]) / slow
        if fast_ma > slow_ma:
            # long signal: capture next day's return
            trades.append({
                "date":      None,
                "direction": "long",
                "entry":     None,
                "exit":      None,
                "return":    returns[i + 1],
            })
    return trades


def simulate_mean_reversion(returns, lookback=20, z_threshold=1.5):
    """Simulate a simple mean-reversion strategy.

    Goes long when the most recent return is more than z_threshold
    standard deviations below the rolling mean of the lookback window,
    expecting a bounce.  The trade return is the next day's return.

    Parameters
    ----------
    returns      : list of daily returns
    lookback     : rolling window for the mean and std
    z_threshold  : how many standard deviations below mean triggers a long
    """
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    if len(returns) < lookback + 1:
        raise ValueError(
            "need at least {} returns for this lookback".format(lookback + 1)
        )
    trades = []
    for i in range(lookback - 1, len(returns) - 1):
        window = returns[i - lookback + 1: i + 1]
        if min(window) == max(window):
            continue
        mu  = sum(window) / len(window)
        s   = math.sqrt(sum((x - mu) ** 2 for x in window) / (len(window) - 1))
        z = (returns[i] - mu) / s
        if z < -z_threshold:
            trades.append({
                "date":      None,
                "direction": "long",
                "entry":     None,
                "exit":      None,
                "return":    returns[i + 1],
            })
    return trades
