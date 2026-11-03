import argparse
import math
import os
import statistics
import sys

import poisson_processes as pp

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "DEXUSEU_returns.csv")


def cmd_pmf(args):
    print(f"Poisson process, rate={args.rate}, t={args.t}")
    for k in range(args.kmax + 1):
        print(f"  P(N({args.t})={k}) = {pp.pmf(args.rate, args.t, k):.6f}")
    print(f"mean = {pp.mean_count(args.rate, args.t):.4f}, "
          f"variance = {pp.variance_count(args.rate, args.t):.4f}")


def cmd_simulate(args):
    arrivals = pp.simulate_arrivals(args.rate, args.t, seed=args.seed)
    print(f"simulated {len(arrivals)} arrivals over [0, {args.t}] at rate {args.rate}")
    if len(arrivals) <= 40:
        print("arrival times:", [round(a, 4) for a in arrivals])
    gaps = pp.interarrival_times(arrivals)
    if gaps:
        moments = pp.exponential_moments_check(gaps)
        print(f"mean gap: {moments['mean']:.4f} (theory: {1.0 / args.rate:.4f})")
        print(f"coefficient of variation: {moments['coefficient_of_variation']:.4f} (theory: 1.0)")


def cmd_superpose(args):
    a = pp.simulate_arrivals(args.rate1, args.t, seed=args.seed)
    b = pp.simulate_arrivals(args.rate2, args.t, seed=None if args.seed is None else args.seed + 1)
    merged = pp.superpose(a, b)
    print(f"process A: rate={args.rate1}, {len(a)} arrivals")
    print(f"process B: rate={args.rate2}, {len(b)} arrivals")
    print(f"superposed: {len(merged)} arrivals (theoretical rate {args.rate1 + args.rate2})")


def cmd_thin(args):
    arrivals = pp.simulate_arrivals(args.rate, args.t, seed=args.seed)
    kept, discarded = pp.thin(arrivals, args.keep_prob,
                               seed=None if args.seed is None else args.seed + 1)
    print(f"original: {len(arrivals)} arrivals at rate {args.rate}")
    print(f"kept ({args.keep_prob}): {len(kept)} arrivals "
          f"(theoretical rate {args.rate * args.keep_prob})")
    print(f"discarded: {len(discarded)} arrivals "
          f"(theoretical rate {args.rate * (1 - args.keep_prob)})")


def cmd_inhomogeneous(args):
    rate_func = lambda t: args.base_rate + args.slope * t
    rate_max = rate_func(args.t)
    arrivals = pp.simulate_inhomogeneous(rate_func, args.t, rate_max, seed=args.seed)
    expected = pp.expected_inhomogeneous_count(rate_func, args.t)
    print(f"rate(t) = {args.base_rate} + {args.slope}*t on [0, {args.t}]")
    print(f"simulated arrivals: {len(arrivals)}")
    print(f"expected count (numerical integral): {expected:.4f}")


def cmd_fxpoisson(args):
    returns = pp.load_returns(DATA_PATH, column=args.col)
    result = pp.analyze_fx_poisson_process(returns, quantile=args.quantile, n_bins=args.bins)
    print(f"EUR/USD large-move arrival analysis, quantile={args.quantile}, "
          f"{result['n_observations']} observations")
    print(f"large moves found: {result['n_arrivals']}")
    print(f"estimated rate: {result['rate_hat']:.6f} per day")
    print(f"theoretical mean gap (exponential): {result['theoretical_mean_gap']:.4f} days")
    print(f"empirical mean gap:                  {result['empirical_mean_gap']:.4f} days")
    print(f"coefficient of variation of gaps:     {result['coefficient_of_variation']:.4f} "
          f"(1.0 expected under Poisson)")
    d = result["dispersion_test"]
    print(f"dispersion test: statistic={d['statistic']:.4f}, df={d['df']}, "
          f"index_of_dispersion={d['index_of_dispersion']:.4f}, p-value={d['p_value']:.3e}")


def cmd_verify(args):
    import numpy as np
    from scipy import stats as sstats

    checks = []

    def check(name, condition):
        checks.append((name, bool(condition)))
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {name}")

    # 1. pmf vs scipy poisson
    ok = True
    for rate, t in [(1.5, 2.0), (0.3, 10.0), (5.0, 0.5)]:
        mean = rate * t
        for k in range(0, 30):
            ours = pp.pmf(rate, t, k)
            theirs = sstats.poisson.pmf(k, mean)
            if not math.isclose(ours, theirs, abs_tol=1e-10):
                ok = False
    check("pmf vs scipy.stats.poisson.pmf", ok)

    # 2. cdf vs scipy
    ok = True
    for rate, t, k in [(2.0, 3.0, 5), (0.5, 4.0, 2), (10.0, 1.0, 15)]:
        ours = pp.cdf(rate, t, k)
        theirs = sstats.poisson.cdf(k, rate * t)
        if not math.isclose(ours, theirs, abs_tol=1e-9):
            ok = False
    check("cdf vs scipy.stats.poisson.cdf", ok)

    # 3. simulated arrival counts vs numpy Poisson sampler (matched means)
    rng = np.random.default_rng(0)
    rate, t_max, trials = 4.0, 25.0, 20000
    ours_counts = [len(pp.simulate_arrivals(rate, t_max, seed=i)) for i in range(trials)]
    numpy_counts = rng.poisson(rate * t_max, size=trials)
    check("simulated arrival counts match numpy.random.poisson (mean)",
          math.isclose(statistics.mean(ours_counts), float(np.mean(numpy_counts)), abs_tol=1.0))
    check("simulated arrival counts match numpy.random.poisson (variance)",
          math.isclose(statistics.variance(ours_counts), float(np.var(numpy_counts, ddof=1)), rel_tol=0.1))

    # 4. two independent simulation methods (arrivals vs Knuth direct) agree
    rate, t, trials = 3.5, 6.0, 20000
    arrival_based = [len(pp.simulate_arrivals(rate, t, seed=i)) for i in range(trials)]
    direct = [pp.simulate_count_direct(rate, t, seed=trials + i) for i in range(trials)]
    check("arrival-based simulation matches Knuth-direct simulation (mean)",
          math.isclose(statistics.mean(arrival_based), statistics.mean(direct), abs_tol=0.15))

    # 5. inter-arrival gaps are exponential: KS-style check against scipy's expon
    rate = 2.5
    arrivals = pp.simulate_arrivals(rate, 20000.0, seed=1)
    gaps = pp.interarrival_times(arrivals)
    ks_stat, ks_p = sstats.kstest(gaps, sstats.expon(scale=1.0 / rate).cdf)
    check("inter-arrival gaps pass KS test against Exponential(rate) (scipy)", ks_p > 0.01)

    # 6. superposition rate vs numpy
    rate_a, rate_b, t_max, trials = 3.0, 2.0, 15.0, 15000
    merged_counts = []
    for i in range(trials):
        a = pp.simulate_arrivals(rate_a, t_max, seed=2 * i)
        b = pp.simulate_arrivals(rate_b, t_max, seed=2 * i + 1)
        merged_counts.append(len(pp.superpose(a, b)))
    numpy_merged = rng.poisson((rate_a + rate_b) * t_max, size=trials)
    check("superposition rate matches sum of rates (numpy oracle mean)",
          math.isclose(statistics.mean(merged_counts), float(np.mean(numpy_merged)), abs_tol=1.0))

    # 7. thinning rates vs numpy binomial split of a Poisson oracle
    rate, t_max, keep_prob, trials = 6.0, 15.0, 0.35, 15000
    kept_counts = []
    for i in range(trials):
        arrivals = pp.simulate_arrivals(rate, t_max, seed=i)
        kept, _ = pp.thin(arrivals, keep_prob, seed=trials + i)
        kept_counts.append(len(kept))
    numpy_kept = rng.binomial(rng.poisson(rate * t_max, size=trials), keep_prob)
    check("thinning rate matches rate*keep_prob (numpy oracle mean)",
          math.isclose(statistics.mean(kept_counts), float(np.mean(numpy_kept)), abs_tol=1.0))

    # 8. inhomogeneous process expected count vs numpy quadrature-free
    #    Monte Carlo integral as an independent cross-check
    rate_func = lambda x: 2.0 + 0.3 * x
    t_max = 12.0
    our_integral = pp.expected_inhomogeneous_count(rate_func, t_max)
    mc_x = rng.uniform(0, t_max, size=200000)
    mc_integral = float(np.mean([rate_func(x) for x in mc_x])) * t_max
    check("trapezoidal integral matches Monte Carlo integral (inhomogeneous rate)",
          math.isclose(our_integral, mc_integral, rel_tol=0.02))

    # 9. chi2_sf vs scipy
    ok = True
    for x, df in [(1.0, 1), (5.0, 3), (20.0, 10)]:
        if not math.isclose(pp.chi2_sf(x, df), sstats.chi2.sf(x, df), abs_tol=1e-9):
            ok = False
    check("chi2_sf vs scipy.stats.chi2.sf", ok)

    # 10. dispersion test statistic vs a direct scipy chi-square goodness-of-fit
    #     computed from the same counts and the same expected value under
    #     equidispersion.
    rate, t_max, n_bins = 8.0, 300.0, 30
    arrivals = pp.simulate_arrivals(rate, t_max, seed=7)
    counts = pp.counts_in_bins(arrivals, t_max, n_bins)
    mean_c = statistics.mean(counts)
    scipy_stat = sum((c - mean_c) ** 2 / mean_c for c in counts)
    ours = pp.fisher_dispersion_test(counts)
    check("dispersion statistic matches direct chi-square recomputation",
          math.isclose(ours["statistic"], scipy_stat, abs_tol=1e-9))

    passed = sum(1 for _, ok in checks if ok)
    print(f"\n{passed}/{len(checks)} checks passed")
    if passed != len(checks):
        sys.exit(1)


def build_parser():
    parser = argparse.ArgumentParser(description="Poisson processes laboratory CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("pmf", help="print the Poisson count distribution")
    p1.add_argument("--rate", type=float, required=True)
    p1.add_argument("--t", type=float, required=True)
    p1.add_argument("--kmax", type=int, default=15)
    p1.set_defaults(func=cmd_pmf)

    p2 = sub.add_parser("simulate", help="simulate a homogeneous Poisson process")
    p2.add_argument("--rate", type=float, required=True)
    p2.add_argument("--t", type=float, required=True)
    p2.add_argument("--seed", type=int, default=None)
    p2.set_defaults(func=cmd_simulate)

    p3 = sub.add_parser("superpose", help="merge two independent Poisson processes")
    p3.add_argument("--rate1", type=float, required=True)
    p3.add_argument("--rate2", type=float, required=True)
    p3.add_argument("--t", type=float, required=True)
    p3.add_argument("--seed", type=int, default=None)
    p3.set_defaults(func=cmd_superpose)

    p4 = sub.add_parser("thin", help="thin a Poisson process by independent marking")
    p4.add_argument("--rate", type=float, required=True)
    p4.add_argument("--t", type=float, required=True)
    p4.add_argument("--keep-prob", type=float, required=True)
    p4.add_argument("--seed", type=int, default=None)
    p4.set_defaults(func=cmd_thin)

    p5 = sub.add_parser("inhomogeneous", help="simulate a linearly time-varying rate process")
    p5.add_argument("--base-rate", type=float, required=True)
    p5.add_argument("--slope", type=float, required=True)
    p5.add_argument("--t", type=float, required=True)
    p5.add_argument("--seed", type=int, default=None)
    p5.set_defaults(func=cmd_inhomogeneous)

    p6 = sub.add_parser("fxpoisson", help="test EUR/USD large moves against a Poisson process")
    p6.add_argument("--col", type=str, default="return")
    p6.add_argument("--quantile", type=float, default=0.95)
    p6.add_argument("--bins", type=int, default=20)
    p6.set_defaults(func=cmd_fxpoisson)

    p7 = sub.add_parser("verify", help="cross-check implementation against analytical/oracle values")
    p7.set_defaults(func=cmd_verify)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
