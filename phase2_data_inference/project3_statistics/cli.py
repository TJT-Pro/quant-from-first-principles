#!/usr/bin/env python3
"""Command-line front end for the descriptive statistics library.

    python3 cli.py stats FILE.csv [--col NAME]
    python3 cli.py cov FILE.csv X [Y] [--lag N]
    python3 cli.py returns FILE.csv [--col NAME] [--out OUT.csv]
    python3 cli.py verify

The `stats` command prints the full summary for every numeric column (or
one, with --col).  `cov` needs two numeric columns, or one column with
--lag to measure autocovariance against its own past.  `returns` takes a
price column, computes period-over-period percentage changes, prints
their summary and optionally writes them to a CSV.  `verify` loads the
real EUR/USD returns dataset shipped in data/ and checks every function
against numpy and scipy.
"""

import argparse
import csv
import os
import sys

import descriptive_stats as ds


def load_numeric_columns(path):
    """Return (header, {column: [floats]}) for the numeric columns.

    A column counts as numeric when every non-empty cell parses as a
    float.  Empty cells (holidays in a price series) are skipped.
    """
    with open(path, 'r', encoding='utf-8') as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    raw = {h: [] for h in header}
    for row in rows[1:]:
        for h, cell in zip(header, row):
            if cell.strip():
                raw[h].append(cell)
    numeric = {}
    for h in header:
        try:
            numeric[h] = [float(c) for c in raw[h]]
        except ValueError:
            continue
        if not numeric[h]:
            del numeric[h]
    if not numeric:
        raise ValueError("no numeric columns found in {}".format(path))
    return header, numeric


def load_prices(path, col=None):
    """Return (column_name, [(date, price)]) for a price series.

    The first column is treated as the date/key, the chosen numeric column
    as the value.  Rows with an empty value are dropped.
    """
    with open(path, 'r', encoding='utf-8') as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    if col is None:
        candidates = []
        for idx, name in enumerate(header):
            vals = [r[idx] for r in rows[1:] if len(r) > idx and r[idx].strip()]
            try:
                for v in vals:
                    float(v)
            except ValueError:
                continue
            candidates.append((name, idx))
        if not candidates:
            raise ValueError("no numeric price column found in {}".format(path))
        name, idx = candidates[0]
    else:
        if col not in header:
            raise ValueError("column {!r} not in {}".format(col, path))
        idx = header.index(col)
        name = col
    pairs = [(r[0], float(r[idx])) for r in rows[1:]
             if len(r) > idx and r[idx].strip()]
    return name, pairs


def print_summary(label, data):
    s = ds.summary(data)
    print("{}: {} values".format(label, s['count']))
    print("  mean       {:+.6f}".format(s['mean']))
    print("  median     {:+.6f}".format(s['median']))
    modes = s['mode']
    freq = sum(1 for v in data if v == modes[0])
    if freq == 1:
        print("  mode       all values unique, so every value is a mode")
    else:
        print("  mode       {} (frequency {})".format(
            [round(m, 6) for m in modes[:8]], freq))
    print("  min/max    {:+.6f} / {:+.6f}".format(s['min'], s['max']))
    print("  range      {:.6f}".format(s['range']))
    print("  q1 / q3    {:+.6f} / {:+.6f}".format(s['q1'], s['q3']))
    print("  iqr        {:.6f}".format(s['iqr']))
    print("  variance   pop {:.10f}   sample {:.10f}".format(
        s['variance_population'], s['variance_sample']))
    print("  stddev     pop {:.8f}   sample {:.8f}".format(
        s['stddev_population'], s['stddev_sample']))
    print("  skewness   {:+.6f}".format(s['skewness']))
    print("  kurtosis   {:+.6f} (excess)".format(s['kurtosis_excess']))


def cmd_stats(args):
    header, numeric = load_numeric_columns(args.file)
    if args.col:
        if args.col not in numeric:
            sys.exit("column {!r} is not numeric or not present".format(args.col))
        print_summary(args.col, numeric[args.col])
    else:
        for name in numeric:
            print_summary(name, numeric[name])
            print()


def cmd_cov(args):
    header, numeric = load_numeric_columns(args.file)
    if args.y:
        if args.x not in numeric or args.y not in numeric:
            sys.exit("both columns must be numeric and present")
        a = numeric[args.x]
        b = numeric[args.y]
        if len(a) != len(b):
            sys.exit("columns have different lengths ({} vs {})".format(len(a), len(b)))
        label = "{} vs {}".format(args.x, args.y)
    else:
        if args.x not in numeric:
            sys.exit("column {!r} is not numeric or not present".format(args.x))
        lag = args.lag if args.lag is not None else 1
        col = numeric[args.x]
        if len(col) <= lag:
            sys.exit("column too short for lag {}".format(lag))
        a = col[lag:]
        b = col[:-lag]
        label = "{} vs its lag-{}".format(args.x, lag)
    cov_pop = ds.covariance(a, b, sample=False)
    cov_samp = ds.covariance(a, b, sample=True)
    corr = ds.pearson_correlation(a, b)
    print("{}: {} pairs".format(label, len(a)))
    print("covariance (population): {:.10f}".format(cov_pop))
    print("covariance (sample):     {:.10f}".format(cov_samp))
    print("pearson correlation:     {:+.6f}".format(corr))


def cmd_returns(args):
    name, pairs = load_prices(args.file, args.col)
    prices = [p for _, p in pairs]
    dates = [d for d, _ in pairs]
    returns = ds.pct_change(prices)
    print("prices: {} rows, returns: {}".format(len(prices), len(returns)))
    print_summary("daily returns ({})".format(name), returns)
    if args.out:
        with open(args.out, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['date', 'return'])
            for d, r in zip(dates[1:], returns):
                writer.writerow([d, '{:.10f}'.format(r)])
        print("wrote {}".format(args.out))


def cmd_verify(args):
    """Compare every function against numpy and scipy on the real data."""
    import numpy as np
    from scipy import stats as scipy_stats

    base = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(base, 'data', 'DEXUSEU_returns.csv')
    header, numeric = load_numeric_columns(data_file)
    returns = numeric['return']
    a = np.array(returns)
    n = len(returns)

    failures = []
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, ok, detail))
        if not ok:
            failures.append(name)

    def near(x, y, tol=1e-9):
        return abs(x - y) <= tol * max(1.0, abs(y))

    check("mean", near(ds.mean(returns), a.mean()))
    check("median", near(ds.median(returns), np.median(a)))
    check("quantile q1", near(ds.quantile(returns, 0.25), np.percentile(a, 25)))
    check("quantile q3", near(ds.quantile(returns, 0.75), np.percentile(a, 75)))
    check("iqr", near(ds.iqr(returns),
                      np.percentile(a, 75) - np.percentile(a, 25)))
    check("range", near(ds.data_range(returns), np.ptp(a)))
    check("variance population", near(ds.variance(returns, False), a.var(ddof=0)))
    check("variance sample", near(ds.variance(returns, True), a.var(ddof=1)))
    check("stddev population", near(ds.stddev(returns, False), a.std(ddof=0)))
    check("stddev sample", near(ds.stddev(returns, True), a.std(ddof=1)))
    check("skewness bias", near(ds.skewness(returns, True),
                                scipy_stats.skew(a, bias=True)))
    check("skewness unbiased", near(ds.skewness(returns, False),
                                    scipy_stats.skew(a, bias=False)))
    check("kurtosis excess", near(ds.kurtosis(returns, True, True),
                                  scipy_stats.kurtosis(a, fisher=True, bias=True)))
    check("kurtosis sample excess", near(ds.kurtosis(returns, False, True),
                                         scipy_stats.kurtosis(a, fisher=True, bias=False)))
    check("kurtosis non-excess", near(ds.kurtosis(returns, True, False),
                                      scipy_stats.kurtosis(a, fisher=False, bias=True)))

    lag = returns[1:]
    lead = returns[:-1]
    check("covariance population", near(
        ds.covariance(lead, lag, False),
        np.cov(np.array([lead, lag]), ddof=0)[0, 1]))
    check("covariance sample", near(
        ds.covariance(lead, lag, True),
        np.cov(np.array([lead, lag]), ddof=1)[0, 1]))
    check("pearson correlation", near(
        ds.pearson_correlation(lead, lag),
        np.corrcoef(np.array([lead, lag]))[0, 1]))

    rounded = [round(r, 4) for r in returns]
    my_modes = ds.mode(rounded)
    scipy_mode = float(scipy_stats.mode(rounded, keepdims=True).mode[0])
    check("mode contains scipy mode", scipy_mode in my_modes,
          "scipy {} not in mine".format(scipy_mode))
    max_freq = max({v: rounded.count(v) for v in set(rounded)}.values())
    check("all modes share the max frequency",
          all(rounded.count(m) == max_freq for m in my_modes))

    print("data: {} daily EUR/USD returns ({})".format(
        len(returns), os.path.basename(data_file)))
    width = 6
    for name, ok, detail in checks:
        marker = "PASS" if ok else "FAIL"
        print("{:<{w}} {}".format(marker, name, w=width))
        if not ok and detail:
            print("      {}".format(detail))
    print("-" * 44)
    print("{}/{} checks passed".format(len(checks) - len(failures), len(checks)))
    sys.exit(1 if failures else 0)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="descriptive-stats",
        description="Descriptive Statistics Library (Project 3)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_stats = sub.add_parser("stats", help="summary for numeric column(s)")
    p_stats.add_argument("file")
    p_stats.add_argument("--col", default=None)
    p_stats.set_defaults(func=cmd_stats)

    p_cov = sub.add_parser("cov", help="covariance and correlation")
    p_cov.add_argument("file")
    p_cov.add_argument("x")
    p_cov.add_argument("y", nargs="?", default=None)
    p_cov.add_argument("--lag", type=int, default=None,
                       help="compare x with its own lag when y is omitted")
    p_cov.set_defaults(func=cmd_cov)

    p_ret = sub.add_parser("returns", help="percentage changes of a price column")
    p_ret.add_argument("file")
    p_ret.add_argument("--col", default=None)
    p_ret.add_argument("--out", default=None)
    p_ret.set_defaults(func=cmd_returns)

    p_verify = sub.add_parser("verify", help="cross-check against numpy/scipy")
    p_verify.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, OSError) as exc:
        sys.exit("error: {}".format(exc))


if __name__ == '__main__':
    main()
