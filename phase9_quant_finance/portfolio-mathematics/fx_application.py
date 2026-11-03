import csv
import statistics
from typing import List, Dict

import mean_variance as mv
import tangency_portfolio as tp
import capm
import risk_decomposition as rd
import kelly_criterion as kc


def load_log_returns(csv_path: str, column: str = "log_return") -> List[float]:
    values = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row[column]))
    return values


def momentum_strategy(returns: List[float], lookback: int = 5) -> List[float]:
    signal = [sum(returns[i - lookback:i]) for i in range(lookback, len(returns))]
    return [(1 if signal[i - lookback] > 0 else -1) * returns[i] for i in range(lookback, len(returns))]


def mean_reversion_strategy(returns: List[float], lookback: int = 20) -> List[float]:
    """Uses a different (20-day) lookback than momentum's 5-day one,
    deliberately: using the identical window for both would make the
    two strategies exact negatives of each other by construction
    (correlation exactly -1, a degenerate covariance matrix), a real
    design trap this project family caught early elsewhere.
    """
    signal = [sum(returns[i - lookback:i]) for i in range(lookback, len(returns))]
    return [(-1 if signal[i - lookback] > 0 else 1) * returns[i] for i in range(lookback, len(returns))]


def vol_scaled_strategy(returns: List[float], window: int = 20, target_vol: float = 0.01) -> List[float]:
    vol = [statistics.pstdev(returns[i - window:i]) for i in range(window, len(returns))]
    return [returns[i] * (target_vol / vol[i - window]) if vol[i - window] > 0 else 0.0
            for i in range(window, len(returns))]


def build_strategy_returns(returns: List[float]) -> Dict[str, List[float]]:
    """Build four strategies from the same underlying real EUR/USD
    series and align them to a common length (starting once every
    strategy's own lookback window has data available), the same four
    strategies used elsewhere in this project family, rebuilt fully
    from scratch here rather than imported.
    """
    start = 20
    buy_hold = returns[start:]
    momentum = momentum_strategy(returns, 5)[start - 5:]
    mean_reversion = mean_reversion_strategy(returns, 20)
    vol_scaled = vol_scaled_strategy(returns, 20)

    min_len = min(len(buy_hold), len(momentum), len(mean_reversion), len(vol_scaled))
    return {
        "buy_hold": buy_hold[-min_len:],
        "momentum": momentum[-min_len:],
        "mean_reversion": mean_reversion[-min_len:],
        "vol_scaled": vol_scaled[-min_len:],
    }


def annualized_mean(returns: List[float], periods_per_year: int = 252) -> float:
    return statistics.mean(returns) * periods_per_year


def sample_covariance_matrix(series_list: List[List[float]], periods_per_year: int = 252) -> List[List[float]]:
    n = len(series_list)
    m = len(series_list[0])
    means = [statistics.mean(s) for s in series_list]
    cov = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            cov[i][j] = sum((series_list[i][t] - means[i]) * (series_list[j][t] - means[j])
                             for t in range(m)) / m * periods_per_year
    return cov


def run_full_analysis(strategies: Dict[str, List[float]], rf: float = 0.02) -> Dict:
    """Run the complete portfolio-mathematics toolkit on a set of
    strategy return series: the efficient frontier, the tangency
    (max-Sharpe) portfolio, CAPM betas of each strategy against
    buy_hold as the market factor, a risk decomposition of the
    tangency portfolio, and the Kelly-optimal leverage for that same
    portfolio.
    """
    names = list(strategies.keys())
    series_list = [strategies[n] for n in names]
    mu = [annualized_mean(s) for s in series_list]
    sigma = sample_covariance_matrix(series_list)

    gmv = mv.global_minimum_variance_portfolio(mu, sigma)
    tangency = tp.tangency_portfolio(mu, sigma, rf)

    betas = {}
    market = strategies["buy_hold"]
    for name in names:
        betas[name] = capm.risk_decomposition(strategies[name], market)

    risk_decomp = rd.verify_euler_decomposition(tangency["weights"], sigma)
    risk_pct = rd.risk_contribution_percentages(tangency["weights"], sigma)

    kelly_w = kc.kelly_weights_multi_asset(mu, sigma)
    kelly_normalized = [w / sum(kelly_w) for w in kelly_w]

    return {
        "names": names, "mu": mu, "sigma": sigma,
        "gmv": gmv, "tangency": tangency,
        "betas": betas,
        "risk_decomposition": risk_decomp, "risk_contribution_pct": risk_pct,
        "kelly_weights": kelly_w, "kelly_normalized": kelly_normalized,
    }
