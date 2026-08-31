"""Command-line interface for the strategy backtester.

Usage
-----
    python3 cli.py report   FILE.csv --col NAME [options]
    python3 cli.py stats    FILE.csv --col NAME [options]
    python3 cli.py ttest    FILE.csv --col NAME [--mu0 0.0]
    python3 cli.py boot     FILE.csv --col NAME [--n-boot 1000]
    python3 cli.py permute  FILE.csv --col NAME [--stat sharpe|expectancy|mdd]
    python3 cli.py simulate FILE.csv --col NAME [--strategy ma|mr]
    python3 cli.py verify

Options
-------
    --col          column name for per-trade returns
    --rf           risk-free rate per period, default 0.0
    --periods      trading periods per year, default 252
    --n-boot       number of bootstrap resamples, default 1000
    --n-perm       number of permutations, default 5000
    --mu0          null hypothesis mean for t-test, default 0.0
    --stat         statistic for permutation test: sharpe, expectancy, or mdd
    --strategy     ma (moving average crossover) or mr (mean reversion)
    --seed         integer seed for reproducibility

Examples
--------
    python3 cli.py report   data/DEXUSEU_returns.csv --col return
    python3 cli.py ttest    data/DEXUSEU_returns.csv --col return
    python3 cli.py simulate data/DEXUSEU_returns.csv --col return --strategy ma
    python3 cli.py permute  data/DEXUSEU_returns.csv --col return --stat sharpe
    python3 cli.py verify
"""

import argparse
import csv
import sys

import backtester as bt


def load_column(path, name):
    values = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if name not in (reader.fieldnames or []):
            sys.exit("column {!r} not found in {}".format(name, path))
        for row in reader:
            cell = row[name]
            if cell is None or cell == "":
                continue
            try:
                values.append(float(cell))
            except ValueError:
                continue
    if not values:
        sys.exit("column {!r} has no numeric values".format(name))
    return values


def returns_to_log(returns):
    """Wrap a plain return list into trade log format."""
    return bt.load_trade_log([{"return": r} for r in returns])


def add_common(p):
    p.add_argument("file")
    p.add_argument("--col", required=True)
    p.add_argument("--rf", type=float, default=0.0,
                   help="risk-free rate per period")
    p.add_argument("--periods", type=int, default=252,
                   help="trading periods per year")
    p.add_argument("--n-boot", type=int, default=1000, dest="n_boot")
    p.add_argument("--seed", type=int, default=None)


def fmt_pct(x, places=4):
    if x is None:
        return "n/a"
    return "{:+.{}f}%".format(x * 100.0, places)


def fmt_pos(x, places=4):
    if x is None:
        return "n/a"
    return "{:.{}f}%".format(x * 100.0, places)


def fmt_float(x, places=4):
    if x is None:
        return "n/a"
    return "{:.{}f}".format(x, places)


def cmd_report(args):
    returns = load_column(args.file, args.col)
    log     = returns_to_log(returns)
    report  = bt.backtest_report(log, risk_free=args.rf,
                                 periods_per_year=args.periods,
                                 n_boot=args.n_boot, seed=args.seed)
    w = 42
    print("=" * w)
    print("Backtest Report")
    print("=" * w)
    print("Trades:           {:,}".format(report["n_trades"]))
    print("-" * w)
    print("Edge")
    print("  expectancy:     {}  [{}, {}]".format(
        fmt_pct(report["expectancy"]),
        fmt_pct(report["expectancy_ci"][0]),
        fmt_pct(report["expectancy_ci"][1]),
    ))
    print("  win rate:       {}".format(fmt_pos(report["win_rate"])))
    print("  loss rate:      {}".format(fmt_pos(report["loss_rate"])))
    print("  avg win:        {}".format(fmt_pct(report["average_win"])))
    print("  avg loss:       {}".format(fmt_pct(report["average_loss"])))
    print("  profit factor:  {}".format(fmt_float(report["profit_factor"])))
    print("  payoff ratio:   {}".format(fmt_float(report["payoff_ratio"])))
    print("-" * w)
    print("Statistical significance")
    print("  t-statistic:    {}".format(fmt_float(report["t_stat"])))
    print("  p-value:        {}".format(fmt_float(report["p_value"], 6)))
    print("  significant:    5% = {}   1% = {}".format(
        "yes" if report["significant_5pct"] else "no",
        "yes" if report["significant_1pct"] else "no",
    ))
    print("-" * w)
    print("Risk-adjusted performance  (annualised, rf={})".format(args.rf))
    print("  Sharpe:         {}".format(fmt_float(report["sharpe"])))
    print("  Sortino:        {}".format(fmt_float(report["sortino"])))
    print("  Calmar:         {}".format(fmt_float(report["calmar"])))
    print("  max drawdown:   {}".format(fmt_pos(report["max_drawdown"])))
    print("  max dd dur:     {} trades".format(report["max_dd_duration"]))
    print("-" * w)
    print("Bootstrap ({:,} resamples)".format(args.n_boot))
    print("  final wealth 5th pct:    {:.4f}".format(report["boot_final_5pct"]))
    print("  final wealth median:     {:.4f}".format(report["boot_final_median"]))
    print("  final wealth 95th pct:   {:.4f}".format(report["boot_final_95pct"]))
    print("=" * w)


def cmd_stats(args):
    returns = load_column(args.file, args.col)
    comps   = bt.expectancy_from_components(returns)
    print("Performance breakdown")
    print("  expectancy:   {}".format(fmt_pct(comps["expectancy"])))
    print("  win rate:     {}".format(fmt_pos(comps["win_rate"])))
    print("  loss rate:    {}".format(fmt_pos(comps["loss_rate"])))
    print("  avg win:      {}".format(fmt_pct(comps["average_win"])))
    print("  avg loss:     {}".format(fmt_pct(comps["average_loss"])))
    print("  profit factor:{}".format(fmt_float(comps["profit_factor"])))
    print("  payoff ratio: {}".format(fmt_float(comps["payoff_ratio"])))


def cmd_ttest(args):
    returns = load_column(args.file, args.col)
    mu0     = args.mu0
    result  = bt.t_test_one_sample(returns, mu0=mu0)
    lo, hi  = bt.confidence_interval_mean(returns)
    print("One-sample t-test   H0: mean return = {:.6g}".format(mu0))
    print("  n:              {:,}".format(len(returns)))
    print("  sample mean:    {}".format(fmt_pct(bt._mean(returns))))
    print("  t-statistic:    {:.6f}".format(result["t_stat"]))
    print("  df:             {:,}".format(result["df"]))
    print("  p-value:        {:.6f}".format(result["p_value"]))
    print("  significant:    5% = {}   1% = {}".format(
        "yes" if result["significant_at_5pct"] else "no",
        "yes" if result["significant_at_1pct"] else "no",
    ))
    print("  95% CI mean:    [{}, {}]".format(fmt_pct(lo), fmt_pct(hi)))


def cmd_boot(args):
    returns = load_column(args.file, args.col)
    log     = returns_to_log(returns)
    boot    = bt.bootstrap_equity_curve(returns, n_boot=args.n_boot, seed=args.seed)
    fw      = bt._sorted_copy(boot["final_wealth"])
    print("Bootstrap equity curve  ({:,} resamples)".format(args.n_boot))
    print("  final wealth 5th pct:   {:.6f}".format(bt._quantile(fw, 0.05)))
    print("  final wealth 25th pct:  {:.6f}".format(bt._quantile(fw, 0.25)))
    print("  final wealth median:    {:.6f}".format(bt._quantile(fw, 0.50)))
    print("  final wealth 75th pct:  {:.6f}".format(bt._quantile(fw, 0.75)))
    print("  final wealth 95th pct:  {:.6f}".format(bt._quantile(fw, 0.95)))
    mdd_boot = [bt.max_drawdown(c) for c in boot["curves"]]
    mdd_s    = bt._sorted_copy(mdd_boot)
    print("  max dd median:          {}".format(fmt_pos(bt._quantile(mdd_s, 0.50))))
    print("  max dd 95th pct:        {}".format(fmt_pos(bt._quantile(mdd_s, 0.95))))


def cmd_permute(args):
    returns = load_column(args.file, args.col)
    stat_name = args.stat

    if stat_name == "sharpe":
        def stat_fn(v):
            s = bt.sharpe_ratio(v)
            return s if s is not None else 0.0
    elif stat_name == "expectancy":
        stat_fn = lambda v: bt._mean(v)
    elif stat_name == "mdd":
        # we want low max drawdown to be good, so negate it
        stat_fn = lambda v: -bt.max_drawdown(bt.equity_curve(v))
    else:
        sys.exit("unknown stat {!r}; choose sharpe, expectancy, or mdd".format(
            stat_name))

    result = bt.permutation_test(returns, stat_fn,
                                 n_permutations=args.n_perm, seed=args.seed)
    print("Permutation test   stat = {}   permutations = {:,}".format(
        stat_name, args.n_perm))
    print("  observed:        {:.6f}".format(result["observed"]))
    print("  null mean:       {:.6f}".format(
        sum(result["null"]) / len(result["null"])))
    print("  p-value:         {:.6f}".format(result["p_value"]))
    print("  significant:     5% = {}".format(
        "yes" if result["significant_at_5pct"] else "no"))


def cmd_simulate(args):
    returns  = load_column(args.file, args.col)
    strategy = args.strategy

    if strategy == "ma":
        trades = bt.simulate_ma_crossover(returns, fast=10, slow=30)
        label  = "MA crossover (fast=10, slow=30)"
    elif strategy == "mr":
        trades = bt.simulate_mean_reversion(returns, lookback=20, z_threshold=1.5)
        label  = "Mean reversion (lookback=20, z_threshold=1.5)"
    else:
        sys.exit("unknown strategy {!r}; choose ma or mr".format(strategy))

    if not trades:
        print("No trades generated.")
        return

    trade_returns = bt.returns_from_log(trades)
    print("{} on {} returns -> {:,} trades".format(
        label, len(returns), len(trades)))
    log    = bt.load_trade_log(trades)
    report = bt.backtest_report(log, risk_free=args.rf,
                                periods_per_year=args.periods,
                                n_boot=args.n_boot, seed=args.seed)
    print("  expectancy:   {}".format(fmt_pct(report["expectancy"])))
    print("  win rate:     {}".format(fmt_pos(report["win_rate"])))
    print("  Sharpe:       {}".format(fmt_float(report["sharpe"])))
    print("  max drawdown: {}".format(fmt_pos(report["max_drawdown"])))
    print("  p-value:      {:.6f}".format(report["p_value"]))
    print("  significant:  5% = {}".format(
        "yes" if report["significant_5pct"] else "no"))


def cmd_verify(args):
    import math
    try:
        import numpy as np
        from scipy import stats as spstats
    except ImportError:
        sys.exit("verify needs numpy and scipy")

    returns = load_column("data/DEXUSEU_returns.csv", "return")
    checks  = []

    # expectancy vs numpy mean
    our_ev   = bt.expectancy(returns)
    np_mean  = float(np.mean(returns))
    checks.append((abs(our_ev - np_mean) < 1e-12, "expectancy matches numpy mean"))

    # win rate
    our_wr = bt.win_rate(returns)
    np_wr  = float(np.mean([1.0 if r > 0.0 else 0.0 for r in returns]))
    checks.append((abs(our_wr - np_wr) < 1e-12, "win rate matches numpy"))

    # profit factor: gross profit / gross loss
    gp  = sum(r for r in returns if r > 0.0)
    gl  = sum(abs(r) for r in returns if r < 0.0)
    our_pf = bt.profit_factor(returns)
    checks.append((abs(our_pf - gp / gl) < 1e-12, "profit factor correct"))

    # payoff ratio
    our_pr   = bt.payoff_ratio(returns)
    avg_win  = float(np.mean([r for r in returns if r > 0.0]))
    avg_loss = float(np.mean([r for r in returns if r < 0.0]))
    checks.append((abs(our_pr - avg_win / abs(avg_loss)) < 1e-12,
                   "payoff ratio correct"))

    # expectancy decomposition
    comps = bt.expectancy_from_components(returns)
    reconstructed = comps["reconstructed_ev"]
    checks.append((abs(reconstructed - our_ev) < 1e-10,
                   "expectancy decomposition reconstructs correctly"))

    # Sharpe vs manual
    mu    = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    our_sharpe = bt.sharpe_ratio(returns)
    np_sharpe  = (mu / sigma) * math.sqrt(252)
    checks.append((abs(our_sharpe - np_sharpe) < 1e-10, "Sharpe matches manual"))

    # Sortino -- downside deviation of returns below 0
    down  = [r for r in returns if r < 0.0]
    dstd  = math.sqrt(sum(d ** 2 for d in down) / len(down))
    our_sortino = bt.sortino_ratio(returns)
    np_sortino  = (mu / dstd) * math.sqrt(252)
    checks.append((abs(our_sortino - np_sortino) < 1e-10,
                   "Sortino matches manual"))

    # equity curve
    curve    = bt.equity_curve(returns)
    np_curve = [1.0]
    for r in returns:
        np_curve.append(np_curve[-1] * (1.0 + r))
    checks.append((all(abs(a - b) < 1e-12 for a, b in zip(curve, np_curve)),
                   "equity curve matches running product"))

    # max drawdown manual check on small known sequence
    small_c = [1.0, 1.5, 1.2, 1.8, 0.9]
    our_mdd = bt.max_drawdown(small_c)
    # peak at 1.8, trough at 0.9: mdd = (1.8 - 0.9) / 1.8 = 0.5
    checks.append((abs(our_mdd - 0.5) < 1e-12,
                   "max drawdown on known sequence"))

    # t-test vs scipy
    result   = bt.t_test_one_sample(returns, mu0=0.0)
    sp_res   = spstats.ttest_1samp(returns, 0.0)
    checks.append((abs(result["t_stat"] - float(sp_res.statistic)) < 1e-8,
                   "t-statistic matches scipy"))
    checks.append((abs(result["p_value"] - float(sp_res.pvalue)) < 1e-6,
                   "p-value matches scipy"))

    # CI via scipy
    lo, hi   = bt.confidence_interval_mean(returns)
    n        = len(returns)
    sp_ci    = spstats.t.interval(0.95, df=n - 1, loc=mu, scale=sigma / math.sqrt(n))
    checks.append((abs(lo - sp_ci[0]) < 1e-6, "CI lower bound matches scipy"))
    checks.append((abs(hi - sp_ci[1]) < 1e-6, "CI upper bound matches scipy"))

    # bootstrap CI width shrinks with more data
    small_r = returns[:50]
    large_r = returns[:500]
    _, lo_s, hi_s = bt.bootstrap_ci(small_r, lambda v: sum(v)/len(v),
                                     n_boot=500, seed=1)
    _, lo_l, hi_l = bt.bootstrap_ci(large_r, lambda v: sum(v)/len(v),
                                     n_boot=500, seed=1)
    checks.append((hi_s - lo_s > hi_l - lo_l,
                   "bootstrap CI is narrower with more data"))

    # permutation test: null p-value should be near 0.5 for a random series
    import random as _random
    rng_r = _random.Random(42)
    random_r = [rng_r.gauss(0.0, 0.01) for _ in range(300)]
    perm_rand = bt.permutation_test(random_r, lambda v: sum(v)/len(v),
                                     n_permutations=500, seed=42)
    checks.append((perm_rand["p_value"] > 0.05,
                   "permutation test does not flag pure noise as significant"))

    # simulate_ma_crossover produces fewer trades than raw returns
    ma_trades = bt.simulate_ma_crossover(returns, fast=10, slow=30)
    checks.append((0 < len(ma_trades) < len(returns),
                   "MA crossover generates some trades but not all returns"))

    # simulate_mean_reversion produces trades
    mr_trades = bt.simulate_mean_reversion(returns, lookback=20, z_threshold=1.5)
    checks.append((len(mr_trades) > 0,
                   "mean reversion generates at least one trade"))

    # load_trade_log validates
    try:
        bt.load_trade_log([{"return": "abc"}])
        checks.append((False, "load_trade_log should reject non-numeric return"))
    except ValueError:
        checks.append((True, "load_trade_log rejects non-numeric return"))

    try:
        bt.load_trade_log([{"direction": "long"}])
        checks.append((False, "load_trade_log should require return key"))
    except ValueError:
        checks.append((True, "load_trade_log requires return key"))

    passed = sum(1 for ok, _ in checks if ok)
    for ok, label in checks:
        print("{:>5}  {}".format("PASS" if ok else "FAIL", label))
    print("-" * 50)
    print("{}/{} checks passed".format(passed, len(checks)))
    if passed != len(checks):
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub    = parser.add_subparsers(dest="command")

    p = sub.add_parser("report", help="full backtest report")
    add_common(p)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("stats", help="performance breakdown")
    add_common(p)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("ttest", help="t-test on mean return")
    add_common(p)
    p.add_argument("--mu0", type=float, default=0.0)
    p.set_defaults(func=cmd_ttest)

    p = sub.add_parser("boot", help="bootstrap equity curve")
    add_common(p)
    p.set_defaults(func=cmd_boot)

    p = sub.add_parser("permute", help="permutation test")
    add_common(p)
    p.add_argument("--stat", default="sharpe",
                   choices=["sharpe", "expectancy", "mdd"])
    p.add_argument("--n-perm", type=int, default=5000, dest="n_perm")
    p.set_defaults(func=cmd_permute)

    p = sub.add_parser("simulate", help="run a built-in strategy on a return series")
    add_common(p)
    p.add_argument("--strategy", default="ma", choices=["ma", "mr"])
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("verify", help="cross-check against numpy/scipy")
    p.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
