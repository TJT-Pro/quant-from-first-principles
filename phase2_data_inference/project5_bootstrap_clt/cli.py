"""Command-line front end for the resampling library.

    python3 cli.py bootstrap FILE.csv --col NAME [--statistic STAT]
                    [--n N] [--ci LEVEL] [--method METHOD] [--seed S]
    python3 cli.py jackknife FILE.csv --col NAME [--statistic STAT]
    python3 cli.py permtest FILE.csv --a COL --b COL [--n N] [--seed S]
    python3 cli.py permtest FILE.csv --col NAME --split N [--n N] [--seed S]
    python3 cli.py clt FILE.csv --col NAME --sample-sizes 1,5,30,100
                    [--n-samples N] [--seed S]
    python3 cli.py verify

statistic is one of mean, median, std or variance.  Every command takes
a --seed so runs can be reproduced exactly.  `verify` checks the
bootstrap, jackknife, permutation test and CLT machinery against scipy
on the real EUR/USD returns in data/.
"""

import argparse
import csv
import math
import sys

import bootstrap_sampler as bs

STATISTICS = {
    "mean": bs.mean,
    "median": bs.median,
    "std": bs.std,
    "variance": bs.variance,
}


def load_column(path, name):
    """Read one numeric column from a CSV with a header row."""
    values = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
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


def parse_list(text, name):
    try:
        return [int(x) for x in text.split(",")]
    except ValueError:
        sys.exit("{!r} is not a comma separated list of integers".format(text))


def cmd_bootstrap(args):
    data = load_column(args.file, args.col)
    stat = STATISTICS[args.statistic]
    ci = bs.bootstrap_ci(data, stat, n_boot=args.n, alpha=1.0 - args.ci,
                         method=args.method, seed=args.seed)
    name = "mean" if args.statistic == "mean" else args.statistic
    print("{} of {} (n = {}), bootstrap, {} resamples, seed {}".format(
        name, args.col, len(data), args.n, args.seed))
    print("point estimate:        {:+.6g}".format(ci["estimate"]))
    print("bootstrap SE:          {:.6g}".format(ci["se"]))
    print("{}% {} interval:  [{:.6g}, {:.6g}]".format(
        args.ci * 100, args.method, ci["lower"], ci["upper"]))


def cmd_jackknife(args):
    data = load_column(args.file, args.col)
    stat = STATISTICS[args.statistic]
    result = bs.jackknife(data, stat)
    print("jackknife of {} for {} (n = {}):".format(
        args.statistic, args.col, len(data)))
    print("estimate:   {:+.6g}".format(result["estimate"]))
    print("bias:       {:+.6g}".format(result["bias"]))
    print("standard error: {:.6g}".format(result["se"]))


def cmd_permtest(args):
    if args.col is not None:
        values = load_column(args.file, args.col)
        if args.split <= 0 or 2 * args.split > len(values):
            sys.exit("--split must be positive and at most half the column length")
        a = values[:args.split]
        b = values[args.split:2 * args.split]
        label = "first {} vs next {} of {}".format(args.split, args.split, args.col)
    else:
        a = load_column(args.file, args.a)
        b = load_column(args.file, args.b)
        label = "mean({}) - mean({})".format(args.a, args.b)
    total = len(a) + len(b)
    n_perm = args.n
    if n_perm is None:
        n_perm = min(10000, max(2000, 6_000_000 // total))
    result = bs.permutation_test(a, b, n_perm=n_perm, seed=args.seed)
    print("permutation test of {}, {} shuffles, seed {}".format(
        label, result["n_perm"], args.seed))
    print("observed statistic: {:+.6g}".format(result["statistic"]))
    print("p-value (two-sided): {:.6g}".format(result["p"]))
    print("shuffles reaching it: {}".format(result["count"]))
    verdict = ("evidence against the null"
               if result["p"] <= 0.05 else "no strong evidence against the null")
    print("at alpha 0.05: {}".format(verdict))


def cmd_clt(args):
    data = load_column(args.file, args.col)
    sizes = parse_list(args.sample_sizes, "sample-sizes")
    rows = bs.clt_convergence(data, sizes, args.n_samples, args.seed)
    print("central limit theorem on {} (population mean {:+.6g}, sd {:.6g}):".format(
        args.col, rows[0]["population_mean"], rows[0]["population_sd"]))
    print("  k   mean of means    sd of means    theor. sd    skewness  kurtosis")
    for row in rows:
        print("  {:<4} {:<15.6g} {:<13.6g} {:<11.6g} {:+.4f}   {:+.4f}".format(
            row["sample_size"], row["mean"], row["sd"],
            row["theoretical_sd"], row["skewness"], row["kurtosis"]))


def cmd_verify(args):
    import numpy as np
    from scipy import stats

    checks = []
    data = load_column("data/DEXUSEU_returns.csv", "return")
    checks.append((len(data) == 6930, "dataset has 6930 returns"))

    seed = 42
    boot_data = data[:1000]
    our = bs.bootstrap_ci(boot_data, bs.mean, n_boot=1999, alpha=0.05,
                          method="percentile", seed=seed)
    theirs = stats.bootstrap((boot_data,), np.mean, n_resamples=1999,
                             method="percentile",
                             random_state=np.random.default_rng(seed))
    checks.append((abs(our["lower"] - theirs.confidence_interval.low) < 1e-4,
                   "bootstrap percentile lower within tolerance"))
    checks.append((abs(our["upper"] - theirs.confidence_interval.high) < 1e-4,
                   "bootstrap percentile upper within tolerance"))

    our_bca = bs.bootstrap_ci(boot_data, bs.mean, n_boot=1999, alpha=0.05,
                              method="BCa", seed=seed)
    theirs_bca = stats.bootstrap((boot_data,), np.mean, n_resamples=1999,
                                 method="BCa",
                                 random_state=np.random.default_rng(seed))
    checks.append((abs(our_bca["lower"] - theirs_bca.confidence_interval.low) < 2e-4,
                   "bootstrap BCa lower within tolerance"))
    checks.append((abs(our_bca["upper"] - theirs_bca.confidence_interval.high) < 2e-4,
                   "bootstrap BCa upper within tolerance"))

    jk_data = data[:1000]
    jk = bs.jackknife(jk_data, bs.mean)
    expected_se = math.sqrt(bs.variance(jk_data) / len(jk_data))
    checks.append((abs(jk["se"] - expected_se) < 1e-12, "jackknife SE equals s/sqrt(n)"))
    checks.append((abs(jk["bias"]) < 1e-9, "jackknife bias of the mean is zero"))

    jk_var = bs.jackknife(jk_data, bs.variance)
    checks.append((abs(jk_var["bias"]) < 1e-9,
                   "jackknife bias of the sample variance is zero"))
    manual_loo = []
    n = len(jk_data)
    for i in range(n):
        rest = jk_data[:i] + jk_data[i + 1:]
        m = sum(rest) / (n - 1)
        manual_loo.append(sum((x - m) ** 2 for x in rest) / (n - 2))
    manual_se = math.sqrt((n - 1) / n * sum((v - sum(manual_loo) / n) ** 2
                                            for v in manual_loo))
    checks.append((abs(jk_var["se"] - manual_se) < 1e-12, "jackknife variance SE matches manual"))

    a, b = data[:500], data[500:1000]
    ours_perm = bs.permutation_test(a, b, n_perm=10000, seed=7)
    theirs_perm = stats.permutation_test((a, b), lambda x, y: np.mean(x) - np.mean(y),
                                         n_resamples=10000,
                                         random_state=np.random.default_rng(7))
    checks.append((abs(ours_perm["p"] - theirs_perm.pvalue) < 0.02,
                   "permutation test p-value within Monte Carlo tolerance"))

    for k in (1, 5, 30, 100):
        row = bs.clt_summary(data, k, n_samples=5000, seed=3)
        pop_sd = bs.population_std(data)
        checks.append((abs(row["theoretical_sd"] - pop_sd / math.sqrt(k)) < 1e-12,
                       "CLT theoretical sd formula at k={}".format(k)))
        checks.append((abs(row["sd"] - row["theoretical_sd"]) < 0.05 * row["theoretical_sd"],
                       "CLT empirical sd near theoretical at k={}".format(k)))

    inside = 0
    total = 0
    row_big = bs.clt_summary(data, 100, n_samples=5000, seed=3)
    for m in bs.sample_means(data, 100, 5000, seed=3):
        total += 1
        if abs(m - row_big["mean"]) <= 1.96 * row_big["theoretical_sd"]:
            inside += 1
    checks.append((0.93 <= inside / total <= 0.97,
                   "sample means are ~95% normal at k=100"))

    passed = sum(1 for ok, _ in checks if ok)
    print("real EUR/USD returns, bootstrap vs scipy (1999 resamples each)")
    for ok, label in checks:
        print("{:>5}  {}".format("PASS" if ok else "FAIL", label))
    print("-" * 44)
    print("{}/{} checks passed".format(passed, len(checks)))
    if passed != len(checks):
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("bootstrap", help="bootstrap confidence interval")
    p.add_argument("file")
    p.add_argument("--col", required=True)
    p.add_argument("--statistic", default="mean", choices=sorted(STATISTICS))
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--method", default="percentile",
                   choices=["percentile", "basic", "normal", "BCa"])
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=cmd_bootstrap)

    p = sub.add_parser("jackknife", help="jackknife bias and standard error")
    p.add_argument("file")
    p.add_argument("--col", required=True)
    p.add_argument("--statistic", default="mean", choices=sorted(STATISTICS))
    p.set_defaults(func=cmd_jackknife)

    p = sub.add_parser("permtest", help="two-sample permutation test")
    p.add_argument("file")
    p.add_argument("--a", default=None)
    p.add_argument("--b", default=None)
    p.add_argument("--col", default=None,
                   help="compare first --split values against the next --split")
    p.add_argument("--split", type=int, default=None)
    p.add_argument("--n", type=int, default=None,
                   help="permutations; defaults to a fast pure-Python count")
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=cmd_permtest)

    p = sub.add_parser("clt", help="central limit theorem demonstration")
    p.add_argument("file")
    p.add_argument("--col", required=True)
    p.add_argument("--sample-sizes", required=True)
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=cmd_clt)

    p = sub.add_parser("verify", help="cross-check everything against scipy")
    p.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
