import math
import statistics
from typing import List, Callable, Dict, Tuple


def sharpe_ratio(returns: List[float], periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    mean_r = statistics.mean(returns)
    std_r = statistics.pstdev(returns)
    if std_r == 0:
        return 0.0
    return (mean_r / std_r) * math.sqrt(periods_per_year)


def train_test_split(data: List, train_fraction: float) -> Tuple[List, List]:
    split_idx = int(len(data) * train_fraction)
    return data[:split_idx], data[split_idx:]


def grid_search_overfitting_demo(train_returns: List[float], test_returns: List[float],
                                  strategy_fn: Callable, param_grid: List, periods_per_year: int = 252) -> Dict:
    """The central demonstration of backtest overfitting via
    parameter search: for every parameter combination in param_grid,
    build the corresponding strategy's return series on both the
    training and test periods via strategy_fn(returns, params), and
    record the in-sample (training) and out-of-sample (test) Sharpe
    ratio for each. Selecting the single combination with the best
    in-sample Sharpe and reporting that specific combination's own
    out-of-sample performance directly demonstrates the mechanism this
    project's own information-theory repository identified via
    multiple hypothesis testing (scanning many lags for "significant"
    mutual information and finding one by chance): searching over many
    parameter combinations is itself a multiple-comparisons procedure,
    and the single best-looking combination in-sample is, in
    expectation, the one whose in-sample performance was most
    flattered by noise, not necessarily the one with genuine
    out-of-sample skill. The average out-of-sample Sharpe across every
    combination tested (not just the in-sample winner) is reported
    alongside it as the honest baseline this specific, cherry-picked
    combination should be compared against.
    """
    results = []
    for params in param_grid:
        train_strategy_returns = strategy_fn(train_returns, params)
        test_strategy_returns = strategy_fn(test_returns, params)
        in_sample_sharpe = sharpe_ratio(train_strategy_returns, periods_per_year)
        out_sample_sharpe = sharpe_ratio(test_strategy_returns, periods_per_year)
        results.append({"params": params, "in_sample_sharpe": in_sample_sharpe,
                         "out_of_sample_sharpe": out_sample_sharpe})

    best = max(results, key=lambda r: r["in_sample_sharpe"])
    avg_out_of_sample = statistics.mean(r["out_of_sample_sharpe"] for r in results)

    return {
        "all_results": results,
        "best_in_sample_params": best["params"],
        "best_in_sample_sharpe": best["in_sample_sharpe"],
        "best_combo_out_of_sample_sharpe": best["out_of_sample_sharpe"],
        "average_out_of_sample_sharpe_across_all_combos": avg_out_of_sample,
        "n_combinations_tested": len(param_grid),
    }


def moving_average_crossover_returns(returns: List[float], params: Tuple[int, int]) -> List[float]:
    """A simple moving-average crossover trading rule: go long when
    the short-window trailing average of past returns exceeds the
    long-window trailing average, short otherwise, applied to the
    next day's return. This is exactly the kind of rule that a large
    parameter grid search (over many (short, long) window pairs) is
    liable to overfit: with enough parameter combinations tried, some
    will look good on any fixed training period purely by chance, even
    if the underlying return series has no genuine exploitable
    structure at all (as established elsewhere in this project family
    via entropy, mutual information, and direct backtesting on this
    same EUR/USD data).
    """
    short_window, long_window = params
    n = len(returns)
    if long_window >= n:
        return []
    strategy_returns = []
    for i in range(long_window, n):
        short_ma = sum(returns[i - short_window:i]) / short_window
        long_ma = sum(returns[i - long_window:i]) / long_window
        position = 1 if short_ma > long_ma else -1
        strategy_returns.append(position * returns[i])
    return strategy_returns
