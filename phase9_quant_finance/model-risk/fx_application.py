import csv
from typing import List, Dict

import var_backtesting as vb
import overfitting_demo as od
import parameter_uncertainty as pu


def load_log_returns(csv_path: str, column: str = "log_return") -> List[float]:
    values = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row[column]))
    return values


def var_model_comparison(returns: List[float], confidences: List[float] = (0.95, 0.99)) -> Dict:
    """Backtest both a Gaussian and a historical-simulation VaR model
    against real returns at several confidence levels, running
    Kupiec's POF test on each. The central capstone finding this
    project builds toward: at 95% confidence (well within the bulk of
    the distribution), both models tend to backtest reasonably; at 99%
    confidence (deep in the tail, exactly where the fat tails
    established throughout this entire project family via kurtosis,
    KL divergence from a fitted Gaussian, and direct extreme-value
    counts actually bite), the Gaussian model's normality assumption
    breaks down and it fails the backtest decisively, while the
    historical model, making no distributional assumption at all,
    continues to pass.
    """
    n = len(returns)
    results = {}
    for confidence in confidences:
        gaussian_var = vb.parametric_gaussian_var(returns, confidence)
        historical_var_est = vb.historical_var(returns, confidence)

        gaussian_exceptions = vb.count_exceptions(returns, gaussian_var)
        historical_exceptions = vb.count_exceptions(returns, historical_var_est)

        results[confidence] = {
            "gaussian": {"var": gaussian_var, "exceptions": gaussian_exceptions,
                         "kupiec": vb.kupiec_pof_test(n, gaussian_exceptions, confidence)},
            "historical": {"var": historical_var_est, "exceptions": historical_exceptions,
                           "kupiec": vb.kupiec_pof_test(n, historical_exceptions, confidence)},
        }
    return results


def overfitting_analysis(returns: List[float]) -> Dict:
    """The moving-average-crossover overfitting demonstration, applied
    to real EUR/USD returns: a reasonably rich grid of (short, long)
    window combinations is searched for the best in-sample Sharpe
    ratio on a training period, and that specific combination's own
    out-of-sample performance on a held-out test period is reported
    directly, alongside the average out-of-sample performance across
    every combination tested.
    """
    train, test = od.train_test_split(returns, 0.6)
    param_grid = [(s, l) for s in (3, 5, 8, 10, 15, 20)
                  for l in (20, 30, 40, 50, 60, 80, 100, 120) if s < l]
    return od.grid_search_overfitting_demo(train, test, od.moving_average_crossover_returns, param_grid)


def option_price_uncertainty(returns: List[float], s0: float, k: float, r: float, t: float,
                              n_bootstrap: int = 1000) -> Dict:
    """Bootstrap the volatility estimate used to price a real EUR/USD
    option and propagate that estimation uncertainty through to the
    option's own price: even under the assumption that Black-Scholes
    is exactly the right pricing model, the price itself is only as
    certain as the volatility estimate feeding it, and this quantifies
    exactly how much of that price is genuinely uncertain purely from
    finite historical sample size.
    """
    output_fn = lambda sigma: pu.black_scholes_call(s0, k, r, sigma, t)
    return pu.propagate_to_model_output(returns, pu.annualized_volatility_estimator, output_fn,
                                         n_bootstrap=n_bootstrap, seed=1)
