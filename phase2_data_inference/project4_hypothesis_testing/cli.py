"""Command-line front end for the Hypothesis Testing engine."""
import argparse
import csv
import math
import sys

import hypothesis_tests as ht


def load_columns(path):
    """Read a CSV with a header row, returning name -> list of floats.

    Empty cells are skipped, so holiday rows with no price never make it
    into the analysis.
    """
    columns = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for key, value in row.items():
                if value is None or value == "":
                    continue
                try:
                    number = float(value)
                except ValueError:
                    continue
                columns.setdefault(key, []).append(number)
    return columns


def parse_list(text, name):
    try:
        return [float(x) for x in text.split(",")]
    except ValueError:
        sys.exit("{!r} is not a comma separated list of numbers".format(name))


def parse_matrix(text):
    matrix = []
    for row_text in text.split(";"):
        try:
            matrix.append([float(x) for x in row_text.split(",")])
        except ValueError:
            sys.exit("{!r} is not a valid matrix".format(text))
    return matrix


def decision_line(p, alpha):
    return "decision: {}".format(ht.decision(p, alpha))


def cmd_ztest(args):
    result = ht.z_test(args.mean, args.mu0, args.sigma, args.n,
                       args.alternative)
    print("H0: mean = {:.6g}   Ha: mean {} {:.6g}".format(
        args.mu0, {"two-sided": "!=", "greater": ">", "less": "<"}[args.alternative],
        args.mu0))
    print("observed mean: {:.6g}   known sigma: {:.6g}   n: {}".format(
        args.mean, args.sigma, args.n))
    print("z statistic:  {:+.6f}".format(result["z"]))
    print("p-value:      {:.6g}".format(result["p"]))
    print(decision_line(result["p"], args.alpha))


def cmd_ttest_one(args):
    columns = load_columns(args.file)
    if args.col not in columns:
        sys.exit("column {!r} not found in {}".format(args.col, args.file))
    data = columns[args.col]
    result = ht.t_test_one_sample(data, args.mu0, args.alternative)
    print("H0: mean = {}   Ha: mean {} {}   n: {}".format(
        args.mu0, {"two-sided": "!=", "greater": ">", "less": "<"}[args.alternative],
        args.mu0, len(data)))
    print("sample mean:  {:.6g}   sample sd: {:.6g}".format(
        sum(data) / len(data), math.sqrt(
            sum((v - sum(data) / len(data)) ** 2 for v in data) / (len(data) - 1))))
    print("t statistic:  {:+.6f}   df: {:.1f}".format(result["t"], result["df"]))
    print("p-value:      {:.6g}".format(result["p"]))
    print(decision_line(result["p"], args.alpha))


def cmd_ttest_two(args):
    columns = load_columns(args.file)
    for name in (args.a, args.b):
        if name not in columns:
            sys.exit("column {!r} not found in {}".format(name, args.file))
    a, b = columns[args.a], columns[args.b]
    result = ht.t_test_two_sample(a, b, args.equal_var, args.alternative)
    print("H0: means equal   Ha: means {}   n1: {}   n2: {}".format(
        {"two-sided": "different", "greater": "a > b", "less": "a < b"}[args.alternative],
        len(a), len(b)))
    print("variance assumption: {}".format(
        "pooled" if args.equal_var else "unequal (Welch)"))
    print("t statistic:  {:+.6f}   df: {:.1f}".format(result["t"], result["df"]))
    print("p-value:      {:.6g}".format(result["p"]))
    print(decision_line(result["p"], args.alpha))


def cmd_ttest_paired(args):
    columns = load_columns(args.file)
    for name in (args.a, args.b):
        if name not in columns:
            sys.exit("column {!r} not found in {}".format(name, args.file))
    a, b = columns[args.a], columns[args.b]
    result = ht.t_test_paired(a, b, args.alternative)
    print("H0: mean difference = 0   Ha: difference {}   pairs: {}".format(
        {"two-sided": "!= 0", "greater": "> 0", "less": "< 0"}[args.alternative],
        len(a)))
    print("t statistic:  {:+.6f}   df: {:.1f}".format(result["t"], result["df"]))
    print("p-value:      {:.6g}".format(result["p"]))
    print(decision_line(result["p"], args.alpha))


def cmd_chisq_fit(args):
    observed = parse_list(args.observed, "observed")
    expected = parse_list(args.expected, "expected")
    result = ht.chi2_goodness_of_fit(observed, expected)
    print("H0: the counts follow the stated distribution   categories: {}".format(len(observed)))
    for i, (o, e, c) in enumerate(zip(observed, expected, result["contributions"])):
        print("  category {}  observed {:>10}  expected {:>10.4f}  chi2 {:>9.4f}".format(i, o, e, c))
    print("chi2:         {:.6f}   df: {}".format(result["chi2"], result["df"]))
    print("p-value:      {:.6g}".format(result["p"]))
    print(decision_line(result["p"], args.alpha))


def cmd_chisq_ind(args):
    matrix = parse_matrix(args.matrix)
    result = ht.chi2_independence(matrix)
    print("H0: the two variables are independent")
    print("  table rows: {}".format(len(matrix)))
    print("  table cols: {}".format(len(matrix[0])))
    print("chi2:         {:.6f}   df: {}".format(result["chi2"], result["df"]))
    print("p-value:      {:.6g}".format(result["p"]))
    print(decision_line(result["p"], args.alpha))


def cmd_prop_test(args):
    result = ht.binomial_proportion_test(args.k, args.n, args.p0,
                                         args.alternative,
                                         args.continuity)
    print("H0: proportion = {:.6g}   Ha: proportion {} {:.6g}   k/n: {}/{}".format(
        args.p0, {"two-sided": "!=", "greater": ">", "less": "<"}[args.alternative],
        args.p0, args.k, args.n))
    print("observed proportion: {:.6f}".format(result["p_hat"]))
    print("z statistic:  {:+.6f}".format(result["z"]))
    print("p-value:      {:.6g}".format(result["p"]))
    print(decision_line(result["p"], args.alpha))


def cmd_ci_mean(args):
    columns = load_columns(args.file)
    if args.col not in columns:
        sys.exit("column {!r} not found in {}".format(args.col, args.file))
    result = ht.confidence_interval_mean(columns[args.col], args.alpha)
    print("{}% confidence interval for the mean (n = {}):".format(
        (1 - args.alpha) * 100, len(columns[args.col])))
    print("  point estimate:  {:.6g}".format(result["mean"]))
    print("  standard error:  {:.6g}".format(result["se"]))
    print("  interval:        [{:.6g}, {:.6g}]".format(result["lower"], result["upper"]))


def cmd_ci_proportion(args):
    result = ht.confidence_interval_proportion(args.k, args.n, args.alpha)
    print("{}% Wilson interval for the proportion (k/n = {}/{}):".format(
        (1 - args.alpha) * 100, args.k, args.n))
    print("  point estimate:  {:.6f}".format(result["p_hat"]))
    print("  interval:        [{:.6f}, {:.6f}]".format(result["lower"], result["upper"]))


def cmd_power(args):
    result = ht.power_z_test(args.effect_size, args.sigma, args.n,
                             args.alpha, args.alternative)
    print("z-test power at effect size {:.6g}, sigma {:.6g}, n {}:".format(
        args.effect_size, args.sigma, args.n))
    print("  power:        {:.4f}".format(result))
    print("  alternative:  {}".format(args.alternative))


def cmd_sample_size(args):
    result = ht.sample_size_z_test(args.effect_size, args.sigma, args.alpha,
                                   args.power, args.alternative)
    print("sample size for z test, effect size {:.6g}, sigma {:.6g}:".format(
        args.effect_size, args.sigma))
    print("  target power: {:.4f}".format(args.power))
    print("  n required:   {}".format(result))


def verify_distributions():
    import numpy as np
    from scipy import stats
    checks = []
    for x in np.linspace(-6, 6, 400):
        checks.append((abs(ht.normal_cdf(x) - stats.norm.cdf(x)) < 1e-14,
                       "normal cdf at x={}".format(x)))
    for df in [1, 2, 3, 5, 10, 30, 100]:
        for t in np.linspace(-8, 8, 200):
            checks.append((abs(ht.t_cdf(t, df) - stats.t.cdf(t, df)) < 1e-9,
                           "t cdf at t={}, df={}".format(t, df)))
        for p in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]:
            checks.append((abs(ht.t_ppf(p, df) - stats.t.ppf(p, df)) < 1e-8,
                           "t ppf at p={}, df={}".format(p, df)))
    for df in [1, 2, 5, 10, 30, 100]:
        for x in np.linspace(0.01, 150, 300):
            checks.append((abs(ht.chi2_cdf(x, df) - stats.chi2.cdf(x, df)) < 1e-9,
                           "chi2 cdf at x={}, df={}".format(x, df)))
        for p in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]:
            checks.append((abs(ht.chi2_ppf(p, df) - stats.chi2.ppf(p, df)) < 1e-8,
                           "chi2 ppf at p={}, df={}".format(p, df)))
    for p in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]:
        checks.append((abs(ht.normal_cdf_inverse(p) - stats.norm.ppf(p)) < 1e-9,
                       "normal ppf at p={}".format(p)))
    return checks


def cmd_verify(args):
    import numpy as np
    from scipy import stats

    print("checking distributions against scipy ...")
    checks = verify_distributions()

    def report():
        passed = sum(1 for ok, _ in checks if ok)
        total = len(checks)
        print("-" * 44)
        print("{}/{} checks passed".format(passed, total))
        if passed != total:
            for ok, label in checks:
                if not ok:
                    print("FAIL  {}".format(label))
            sys.exit(1)

    data = load_columns("data/DEXUSEU_returns.csv")["return"]
    checks.append((len(data) == 6930, "dataset has 6930 returns"))

    mu, s = float(np.mean(data)), float(np.std(data, ddof=1))
    n = len(data)
    se = s / math.sqrt(n)
    tstat = mu / se
    our = ht.t_test_one_sample(data, 0.0)
    theirs = stats.ttest_1samp(data, 0.0)
    checks.append((abs(our["t"] - theirs.statistic) < 1e-9, "one-sample t statistic"))
    checks.append((abs(our["p"] - theirs.pvalue) < 1e-9, "one-sample t p-value"))
    checks.append((abs(our["t"] - tstat) < 1e-12, "one-sample t manual formula"))

    half = len(data) // 2
    a, b = data[:half], data[half:]
    for equal_var in (True, False):
        ours = ht.t_test_two_sample(a, b, equal_var=equal_var)
        theirs = stats.ttest_ind(a, b, equal_var=equal_var)
        checks.append((abs(ours["t"] - theirs.statistic) < 1e-9,
                       "two-sample t statistic pooled={}".format(equal_var)))
        checks.append((abs(ours["p"] - theirs.pvalue) < 1e-9,
                       "two-sample t p-value pooled={}".format(equal_var)))
        if not equal_var:
            checks.append((abs(ours["df"] - theirs.df) < 1e-9, "Welch degrees of freedom"))

    lagged = data[1:]
    ours = ht.t_test_paired(data[:-1], lagged)
    theirs = stats.ttest_rel(data[:-1], lagged)
    checks.append((abs(ours["t"] - theirs.statistic) < 1e-9, "paired t statistic"))
    checks.append((abs(ours["p"] - theirs.pvalue) < 1e-9, "paired t p-value"))

    observed = [100, 95, 105]
    expected = [100, 100, 100]
    ours = ht.chi2_goodness_of_fit(observed, expected)
    theirs = stats.chisquare(observed, expected)
    checks.append((abs(ours["chi2"] - theirs.statistic) < 1e-9, "goodness-of-fit chi2"))
    checks.append((abs(ours["p"] - theirs.pvalue) < 1e-9, "goodness-of-fit p-value"))

    table = [[30, 10], [20, 40]]
    ours = ht.chi2_independence(table)
    theirs = stats.chi2_contingency(table, correction=False)
    checks.append((abs(ours["chi2"] - theirs.statistic) < 1e-9, "independence chi2"))
    checks.append((abs(ours["p"] - theirs.pvalue) < 1e-9, "independence p-value"))

    zres = ht.z_test(mu, 0.0, s, n)
    ztheirs = (mu - 0.0) / (s / math.sqrt(n))
    ptheirs = 2 * (1 - stats.norm.cdf(abs(ztheirs)))
    checks.append((abs(zres["z"] - ztheirs) < 1e-12, "z statistic manual"))
    checks.append((abs(zres["p"] - ptheirs) < 1e-12, "z p-value manual"))

    k, nn, p0 = 60, 100, 0.5
    pres = ht.binomial_proportion_test(k, nn, p0)
    zman = (k / nn - p0) / math.sqrt(p0 * (1 - p0) / nn)
    pman = 2 * (1 - stats.norm.cdf(abs(zman)))
    checks.append((abs(pres["z"] - zman) < 1e-12, "proportion z manual"))
    checks.append((abs(pres["p"] - pman) < 1e-12, "proportion p manual"))

    ci = ht.confidence_interval_mean(data)
    tcrit = stats.t.ppf(1 - 0.05 / 2, n - 1)
    checks.append((abs(ci["lower"] - (mu - tcrit * se)) < 1e-9, "mean CI lower"))
    checks.append((abs(ci["upper"] - (mu + tcrit * se)) < 1e-9, "mean CI upper"))

    for kk, nn in [(0, 10), (1, 10), (5, 20), (10, 10), (99, 100)]:
        ci = ht.confidence_interval_proportion(kk, nn, 0.05)
        for bound in (ci["lower"], ci["upper"]):
            res = ht.binomial_proportion_test(kk, nn, bound)
            if ci["lower"] == 0.0 and bound == ci["lower"]:
                continue
            if ci["upper"] == 1.0 and bound == ci["upper"]:
                continue
            checks.append((abs(res["p"] - 0.05) < 1e-6,
                           "Wilson bound is a boundary: k={}, n={}".format(kk, nn)))

    eff, sigma, nn2 = 0.5, 1.0, 25
    ourp = ht.power_z_test(eff, sigma, nn2, 0.05, "greater")
    zcrit = stats.norm.ppf(0.95)
    manp = 1 - stats.norm.cdf(zcrit - eff * math.sqrt(nn2) / sigma)
    checks.append((abs(ourp - manp) < 1e-12, "z power manual"))

    need = ht.sample_size_z_test(eff, sigma, 0.05, 0.8, "greater")
    checks.append((need == 25, "sample size gives ~0.8 power (expected 25)"))

    report()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command")

    def add_common(p, include_alpha=True):
        p.add_argument("--alternative", default="two-sided",
                       choices=["two-sided", "greater", "less"])
        if include_alpha:
            p.add_argument("--alpha", type=float, default=0.05)

    p = sub.add_parser("ztest", help="one-sample z test, known sigma")
    p.add_argument("--mean", type=float, required=True)
    p.add_argument("--mu0", type=float, required=True)
    p.add_argument("--sigma", type=float, required=True)
    p.add_argument("--n", type=int, required=True)
    add_common(p)
    p.set_defaults(func=cmd_ztest)

    p = sub.add_parser("ttest-one", help="one-sample t test")
    p.add_argument("file")
    p.add_argument("--col", required=True)
    p.add_argument("--mu0", type=float, default=0.0)
    add_common(p)
    p.set_defaults(func=cmd_ttest_one)

    p = sub.add_parser("ttest-two", help="two-sample t test")
    p.add_argument("file")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--equal-var", action="store_true")
    add_common(p)
    p.set_defaults(func=cmd_ttest_two)

    p = sub.add_parser("ttest-paired", help="paired t test")
    p.add_argument("file")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    add_common(p)
    p.set_defaults(func=cmd_ttest_paired)

    p = sub.add_parser("chisq-fit", help="chi-squared goodness of fit")
    p.add_argument("--observed", required=True)
    p.add_argument("--expected", required=True)
    add_common(p)
    p.set_defaults(func=cmd_chisq_fit)

    p = sub.add_parser("chisq-ind", help="chi-squared test of independence")
    p.add_argument("--matrix", required=True)
    add_common(p)
    p.set_defaults(func=cmd_chisq_ind)

    p = sub.add_parser("prop-test", help="z test on a single proportion")
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--p0", type=float, required=True)
    p.add_argument("--continuity", action="store_true")
    add_common(p)
    p.set_defaults(func=cmd_prop_test)

    p = sub.add_parser("ci-mean", help="confidence interval for a mean")
    p.add_argument("file")
    p.add_argument("--col", required=True)
    add_common(p)
    p.set_defaults(func=cmd_ci_mean)

    p = sub.add_parser("ci-proportion", help="Wilson interval for a proportion")
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    add_common(p)
    p.set_defaults(func=cmd_ci_proportion)

    p = sub.add_parser("power", help="power of a z test")
    p.add_argument("--effect-size", type=float, required=True)
    p.add_argument("--sigma", type=float, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--alternative", default="greater",
                   choices=["two-sided", "greater", "less"])
    p.set_defaults(func=cmd_power)

    p = sub.add_parser("sample-size", help="sample size for a z test")
    p.add_argument("--effect-size", type=float, required=True)
    p.add_argument("--sigma", type=float, required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--power", type=float, default=0.8)
    p.add_argument("--alternative", default="greater",
                   choices=["two-sided", "greater", "less"])
    p.set_defaults(func=cmd_sample_size)

    p = sub.add_parser("verify", help="cross-check everything against scipy")
    p.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
