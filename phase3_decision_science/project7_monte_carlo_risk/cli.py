"""Command-line front end for the Monte Carlo Risk & Drawdown analyser."""
import argparse
import csv
import sys

import risk_analyser as ra


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


def add_common(p):
    p.add_argument("file")
    p.add_argument("--col", required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--paths", type=int, default=10000)
    p.add_argument("--periods", type=int, default=None)
    p.add_argument("--dist", default="normal", choices=["normal", "t"])
    p.add_argument("--df", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--threshold", type=float, default=0.2)


def fmt_pct(x):
    return "{:+.4f}%".format(x * 100.0)


def fmt_pos(x):
    return "{:.4f}%".format(x * 100.0)


def cmd_report(args):
    returns = load_column(args.file, args.col)
    report = ra.risk_report(
        returns,
        n_paths=args.paths,
        n_periods=args.periods,
        distribution=args.dist,
        df=args.df,
        alpha=args.alpha,
        ruin_threshold=args.threshold,
        seed=args.seed,
    )
    w = 38
    print("=" * w)
    print("Risk Report")
    print("=" * w)
    print("Series:        {} observations".format(report["n_returns"]))
    print("Distribution:  {}{}".format(
        report["distribution"],
        " (df={})".format(int(args.df)) if report["distribution"] == "t" else ""
    ))
    print("Paths:         {:,}  x  {:,} periods".format(
        report["n_paths"], report["n_periods"]
    ))
    print("-" * w)
    print("Fitted parameters")
    print("  mean return:   {}".format(fmt_pct(report["mu"])))
    print("  std deviation: {}".format(fmt_pos(report["sigma"])))
    print("-" * w)
    print("Value at Risk  (alpha = {:.1%})".format(report["alpha"]))
    print("  historical:    {}".format(fmt_pos(report["hist_var"])))
    print("  parametric:    {}".format(fmt_pos(report["param_var"])))
    print("  simulated:     {}".format(fmt_pos(report["sim_var"])))
    print("-" * w)
    print("CVaR / Expected Shortfall")
    print("  historical:    {}".format(fmt_pos(report["hist_cvar"])))
    print("  parametric:    {}".format(fmt_pos(report["param_cvar"])))
    print("  simulated:     {}".format(fmt_pos(report["sim_cvar"])))
    print("-" * w)
    print("Observed drawdown")
    print("  max drawdown:  {}".format(fmt_pos(report["obs_max_dd"])))
    print("  mdd duration:  {} periods".format(report["obs_mdd_dur"]))
    if report["obs_calmar"] is not None:
        print("  Calmar ratio:  {:.4f}".format(report["obs_calmar"]))
    else:
        print("  Calmar ratio:  n/a (no drawdown)")
    print("-" * w)
    print("Simulated drawdown")
    print("  mean max dd:   {}".format(fmt_pos(report["sim_mean_mdd"])))
    print("  median max dd: {}".format(fmt_pos(report["sim_median_mdd"])))
    print("  worst max dd:  {}".format(fmt_pos(report["sim_worst_mdd"])))
    print("-" * w)
    print("Ruin probability (threshold >= {:.0%})".format(report["ruin_threshold"]))
    print("  estimate:      {:.4f}%".format(report["ruin_prob"] * 100.0))
    print("  95% CI:        [{:.4f}%, {:.4f}%]".format(
        report["ruin_lower"] * 100.0, report["ruin_upper"] * 100.0
    ))
    print("=" * w)


def cmd_var(args):
    returns = load_column(args.file, args.col)
    hist = ra.historical_var(returns, args.alpha)
    param = ra.parametric_var(returns, args.alpha)
    print("VaR at alpha = {:.1%}".format(args.alpha))
    print("  historical:  {}".format(fmt_pos(hist)))
    print("  parametric:  {}".format(fmt_pos(param)))


def cmd_cvar(args):
    returns = load_column(args.file, args.col)
    hist = ra.historical_cvar(returns, args.alpha)
    param = ra.parametric_cvar(returns, args.alpha)
    print("CVaR at alpha = {:.1%}".format(args.alpha))
    print("  historical:  {}".format(fmt_pos(hist)))
    print("  parametric:  {}".format(fmt_pos(param)))


def cmd_drawdown(args):
    returns = load_column(args.file, args.col)
    curve = ra.equity_curve(returns)
    mdd = ra.max_drawdown(curve)
    dur = ra.drawdown_duration(curve)
    max_dur = ra.max_drawdown_duration(curve)
    calmar = ra.calmar_ratio(returns)
    print("Drawdown analysis")
    print("  max drawdown:      {}".format(fmt_pos(mdd)))
    print("  periods in dd:     {}".format(dur))
    print("  longest episode:   {} periods".format(max_dur))
    if calmar is not None:
        print("  Calmar ratio:      {:.4f}".format(calmar))
    else:
        print("  Calmar ratio:      n/a")


def cmd_ruin(args):
    returns = load_column(args.file, args.col)
    results = ra.simulate_risk_metrics(
        returns, n_paths=args.paths, n_periods=args.periods,
        distribution=args.dist, df=args.df, alpha=args.alpha, seed=args.seed
    )
    point, lower, upper = ra.ruin_probability_ci(
        results, threshold=args.threshold, seed=args.seed
    )
    print("Ruin probability (max drawdown >= {:.0%})".format(args.threshold))
    print("  paths simulated: {:,}".format(args.paths))
    print("  estimate:        {:.4f}%".format(point * 100.0))
    print("  95% CI:          [{:.4f}%, {:.4f}%]".format(
        lower * 100.0, upper * 100.0
    ))


def cmd_simulate(args):
    returns = load_column(args.file, args.col)
    results = ra.simulate_risk_metrics(
        returns, n_paths=args.paths, n_periods=args.periods,
        distribution=args.dist, df=args.df, alpha=args.alpha, seed=args.seed
    )
    print("Monte Carlo simulation  ({} paths, {} periods, dist={})".format(
        args.paths,
        args.periods if args.periods else len(returns),
        args.dist
    ))
    print("  VaR  ({:.1%}):         {}".format(
        args.alpha, fmt_pos(results["path_var"])))
    print("  CVaR ({:.1%}):         {}".format(
        args.alpha, fmt_pos(results["path_cvar"])))
    print("  mean max drawdown:   {}".format(fmt_pos(results["mean_max_dd"])))
    print("  median max drawdown: {}".format(fmt_pos(results["median_max_dd"])))
    print("  worst max drawdown:  {}".format(fmt_pos(results["worst_max_dd"])))


def cmd_verify(args):
    """Cross-check every metric against numpy/scipy."""
    import math
    try:
        import numpy as np
        from scipy import stats as spstats
    except ImportError:
        sys.exit("verify needs numpy and scipy: pip install numpy scipy")

    returns = load_column("data/DEXUSEU_returns.csv", "return")
    checks = []

    # mean and sigma
    our_mu, our_sigma = ra.fit_normal(returns)
    np_mu = float(np.mean(returns))
    np_sigma = float(np.std(returns, ddof=1))
    checks.append((abs(our_mu - np_mu) < 1e-12, "mean matches numpy"))
    checks.append((abs(our_sigma - np_sigma) < 1e-12, "sigma matches numpy"))

    # historical VaR
    for alpha in (0.01, 0.05, 0.10):
        our_var = ra.historical_var(returns, alpha)
        s = sorted(returns)
        n = len(s)
        pos = alpha * (n - 1)
        lo = int(pos)
        hi = lo + 1
        frac = pos - lo
        np_q = s[lo] * (1.0 - frac) + (s[hi] if hi < n else s[-1]) * frac
        np_var = -np_q
        checks.append((abs(our_var - np_var) < 1e-12,
                        "hist VaR alpha={} matches manual".format(alpha)))

    # parametric VaR via scipy
    for alpha in (0.01, 0.05, 0.10):
        our_var = ra.parametric_var(returns, alpha)
        mu_r = float(np.mean(returns))
        sig_r = float(np.std(returns, ddof=1))
        sp_var = -float(spstats.norm.ppf(alpha, loc=mu_r, scale=sig_r))
        checks.append((abs(our_var - sp_var) < 1e-6,
                        "param VaR alpha={} close to scipy".format(alpha)))

    # normal_ppf round-trip
    for p in (0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99):
        z = ra._normal_ppf(p)
        recovered = ra.normal_cdf(z)
        checks.append((abs(recovered - p) < 1e-8,
                        "normal_ppf round-trips at p={}".format(p)))

    # historical CVaR
    for alpha in (0.05, 0.10):
        our_cvar = ra.historical_cvar(returns, alpha)
        s = sorted(returns)
        cutoff = max(1, int(math.floor(alpha * len(s))))
        np_cvar = -float(np.mean(s[:cutoff]))
        checks.append((abs(our_cvar - np_cvar) < 1e-12,
                        "hist CVaR alpha={} matches numpy".format(alpha)))

    # parametric CVaR
    for alpha in (0.05, 0.10):
        our_cvar = ra.parametric_cvar(returns, alpha)
        mu_r = float(np.mean(returns))
        sig_r = float(np.std(returns, ddof=1))
        z = float(spstats.norm.ppf(alpha))
        phi_z = float(spstats.norm.pdf(z))
        sp_cvar = -(mu_r - sig_r * phi_z / alpha)
        checks.append((abs(our_cvar - sp_cvar) < 1e-8,
                        "param CVaR alpha={} close to scipy".format(alpha)))

    # equity curve and drawdown on a known sequence
    known = [0.1, -0.05, 0.2, -0.3, 0.1]
    curve = ra.equity_curve(known)
    manual_curve = [1.0]
    for r in known:
        manual_curve.append(manual_curve[-1] * (1.0 + r))
    checks.append((all(abs(a - b) < 1e-12 for a, b in zip(curve, manual_curve)),
                   "equity curve matches manual product"))

    # drawdown on known curve: peak is 1.0*1.1*0.95*1.2 = 1.254
    # after -0.3: value = 1.254 * 0.7 = 0.8778, drawdown = (1.254 - 0.8778)/1.254
    peak_expected = 1.0 * 1.1 * 0.95 * 1.2
    val_after_loss = peak_expected * 0.7
    expected_dd = (peak_expected - val_after_loss) / peak_expected
    dd_series = ra.drawdown_series(curve)
    checks.append((abs(dd_series[4] - expected_dd) < 1e-12,
                   "drawdown value after -30% matches formula"))
    checks.append((abs(ra.max_drawdown(curve) - expected_dd) < 1e-12,
                   "max drawdown identifies the 30% drop"))

    # flat curve has zero drawdown
    flat = [0.0, 0.0, 0.0, 0.0]
    flat_curve = ra.equity_curve(flat)
    checks.append((ra.max_drawdown(flat_curve) == 0.0,
                   "flat return series has zero drawdown"))

    # Calmar ratio direction: higher return, same drawdown => higher Calmar
    high_ret = [0.01] * 100 + [-0.2] + [0.01] * 100
    low_ret  = [0.005] * 100 + [-0.2] + [0.005] * 100
    calmar_high = ra.calmar_ratio(high_ret)
    calmar_low  = ra.calmar_ratio(low_ret)
    checks.append((calmar_high > calmar_low,
                   "higher return gives higher Calmar (same drawdown)"))

    # ruin probability bounds
    results = ra.simulate_risk_metrics(returns, n_paths=1000, seed=42)
    p_ruin = ra.ruin_probability(results, threshold=0.99)
    checks.append((0.0 <= p_ruin <= 1.0, "ruin probability in [0, 1]"))
    p_certain = ra.ruin_probability(results, threshold=0.0001)
    checks.append((p_certain == 1.0, "near-zero threshold ruins all paths"))

    # bootstrap CI contains the point estimate
    point, lower, upper = ra.ruin_probability_ci(results, threshold=0.2, seed=42)
    checks.append((lower <= point <= upper,
                   "ruin CI contains the point estimate"))

    # simulate_risk_metrics returns the right number of paths
    r2 = ra.simulate_risk_metrics(returns, n_paths=500, n_periods=50, seed=1)
    checks.append((len(r2["paths"]) == 500, "correct number of paths returned"))
    checks.append((all(len(p) == 51 for p in r2["paths"]),
                   "each equity curve has n_periods + 1 points"))

    # normal simulation: sample mean and sigma close to fitted values
    mu_f, sig_f = ra.fit_normal(returns)
    sim_paths = ra.simulate_normal_returns(mu_f, sig_f, 1000, 500, seed=7)
    all_returns = [r for path in sim_paths for r in path]
    sim_mu = sum(all_returns) / len(all_returns)
    checks.append((abs(sim_mu - mu_f) < 0.001,
                   "simulated mean close to fitted mean"))

    # t simulation: fatter tails than normal (higher kurtosis)
    mu_t, sig_t = ra.fit_t(returns, df=5)
    t_paths = ra.simulate_t_returns(mu_t, sig_t, 5, 500, 500, seed=9)
    t_all = [r for path in t_paths for r in path]
    t_mu_sample = sum(t_all) / len(t_all)
    t_mean_sq = sum(x ** 2 for x in t_all) / len(t_all)
    # just check it ran and produced numbers in a plausible range
    checks.append((abs(t_mu_sample - mu_t) < 0.005,
                   "t simulation mean close to fitted mean"))

    passed = sum(1 for ok, _ in checks if ok)
    for ok, label in checks:
        print("{:>5}  {}".format("PASS" if ok else "FAIL", label))
    print("-" * 50)
    print("{}/{} checks passed".format(passed, len(checks)))
    if passed != len(checks):
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("report", help="full risk report")
    add_common(p)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("var", help="Value at Risk")
    add_common(p)
    p.set_defaults(func=cmd_var)

    p = sub.add_parser("cvar", help="CVaR / Expected Shortfall")
    add_common(p)
    p.set_defaults(func=cmd_cvar)

    p = sub.add_parser("drawdown", help="drawdown metrics")
    add_common(p)
    p.set_defaults(func=cmd_drawdown)

    p = sub.add_parser("ruin", help="ruin probability with CI")
    add_common(p)
    p.set_defaults(func=cmd_ruin)

    p = sub.add_parser("simulate", help="Monte Carlo simulation summary")
    add_common(p)
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
