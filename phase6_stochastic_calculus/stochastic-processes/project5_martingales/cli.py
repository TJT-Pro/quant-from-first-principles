import argparse
import itertools
import math
import os
import statistics
import sys

import martingales as mg

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "DEXUSEU_returns.csv")


def cmd_ruin(args):
    prob = mg.ruin_probability(args.a, args.b, args.p)
    duration = mg.expected_duration(args.a, args.b, args.p)
    print(f"gambler's ruin, boundaries at {args.a} and -{args.b}, p={args.p}")
    print(f"P(hit {args.a} before -{args.b}) = {prob:.6f}")
    print(f"E[duration] = {duration:.4f}")


def cmd_doob(args):
    result = mg.doob_maximal_inequality_check(args.n, args.level, args.trials, seed=args.seed)
    print(f"Doob's maximal inequality, n={args.n}, level={args.level}, {args.trials} trials")
    print(f"empirical P(max S_k^2 >= level): {result['empirical_probability']:.6f}")
    print(f"Doob bound E[S_n^2]/level:        {result['doob_bound']:.6f}")
    print(f"bound holds: {result['bound_holds']}")


def cmd_polya(args):
    path = mg.polya_urn_path(args.r0, args.b0, args.draws, seed=args.seed)
    theory = mg.polya_urn_terminal_moments(args.r0, args.b0)
    print(f"Polya urn, r0={args.r0}, b0={args.b0}, {args.draws} draws")
    print(f"final fraction: {path[-1]:.6f}")
    print(f"theoretical limiting mean (Beta): {theory['mean']:.6f}")
    print(f"theoretical limiting variance (Beta): {theory['variance']:.6f}")


def cmd_doubling(args):
    import random
    rng = random.Random(args.seed)
    outcomes = [1 if rng.random() < args.p else -1 for _ in range(args.steps)]
    result = mg.simulate_doubling_strategy(outcomes, args.stake, args.cap)
    exact = mg.exact_expected_pnl(args.steps, args.p, args.stake, args.cap)
    print(f"doubling strategy, {args.steps} rounds, p={args.p}, cap={args.cap}")
    print(f"simulated final P&L: {result['final_pnl']:.2f}")
    print(f"forced stops: {result['forced_stops']}")
    print(f"exact expected P&L (DP, not Monte Carlo): {exact:.6f}")


def cmd_fxmartingale(args):
    returns = mg.load_returns(DATA_PATH, column=args.col)
    result = mg.martingale_difference_test(returns)
    print(f"martingale difference test: regress r_t on r_(t-1), n={result['n']}")
    print(f"slope (beta1): {result['beta1']:.6f}")
    print(f"SE (homoskedastic): {result['se_homoskedastic']:.6f}  "
          f"z={result['z_homoskedastic']:+.4f}  p={result['p_homoskedastic']:.4f}")
    print(f"SE (robust/White):   {result['se_robust']:.6f}  "
          f"z={result['z_robust']:+.4f}  p={result['p_robust']:.4f}")
    print(f"R-squared: {result['r_squared']:.6f}")

    print()
    outcomes = mg.sign_outcomes(returns, seed=args.seed)
    tails = mg.streak_tail_probabilities(outcomes, thresholds=[3, 5, 10, 15])
    print("down-streak tail probabilities: empirical vs i.i.d. Bernoulli(p_hat) model")
    for k, emp, theo in zip(tails["thresholds"], tails["empirical"], tails["theoretical"]):
        print(f"  P(streak>={k:2d}): empirical={emp:.5f}  iid_theory={theo:.5f}")

    print()
    p_hat = sum(1 for o in outcomes if o == 1) / len(outcomes)
    n_out, cap, trials = 1000, 10, args.trials
    iid_prediction = mg.exact_expected_pnl(n_out, p_hat, 1.0, cap)
    block_pnls = [mg.simulate_doubling_strategy(
        mg.block_bootstrap_resample(outcomes, 20, n_out, seed=5000 + i), 1.0, cap)["final_pnl"]
        for i in range(trials)]
    iid_pnls = [mg.simulate_doubling_strategy(
        mg.block_bootstrap_resample(outcomes, 1, n_out, seed=6000 + i), 1.0, cap)["final_pnl"]
        for i in range(trials)]
    print(f"capped doubling strategy (cap={cap}), {trials} resamples of {n_out} rounds")
    print(f"i.i.d. Bernoulli(p_hat) exact prediction: {iid_prediction:.2f}")
    print(f"block bootstrap (preserves dependence):   mean={statistics.mean(block_pnls):.2f}  "
          f"se={statistics.stdev(block_pnls) / trials ** 0.5:.2f}")
    print(f"i.i.d. resample (destroys dependence):    mean={statistics.mean(iid_pnls):.2f}  "
          f"se={statistics.stdev(iid_pnls) / trials ** 0.5:.2f}")


def cmd_verify(args):
    import numpy as np
    from scipy import stats as sstats

    checks = []

    def check(name, condition):
        checks.append((name, bool(condition)))
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {name}")

    # 1. exhaustive enumeration confirms exact zero-mean doubling strategy
    ok = True
    for n in (3, 6, 9, 11):
        total = sum(mg.simulate_doubling_strategy(list(seq), 1.0, None)["final_pnl"]
                    for seq in itertools.product([1, -1], repeat=n))
        if not math.isclose(total / (2 ** n), 0.0, abs_tol=1e-9):
            ok = False
    check("doubling strategy exact mean is zero (exhaustive enumeration, uncapped)", ok)

    ok = True
    for n, cap in [(6, 3), (9, 4), (11, 2)]:
        total = sum(mg.simulate_doubling_strategy(list(seq), 1.0, cap)["final_pnl"]
                    for seq in itertools.product([1, -1], repeat=n))
        if not math.isclose(total / (2 ** n), 0.0, abs_tol=1e-9):
            ok = False
    check("doubling strategy exact mean is zero (exhaustive enumeration, capped)", ok)

    # 2. exact_expected_pnl (DP) matches exhaustive enumeration exactly
    ok = True
    for n in (3, 7, 10):
        exhaustive = sum(mg.simulate_doubling_strategy(list(seq), 1.0, None)["final_pnl"]
                          for seq in itertools.product([1, -1], repeat=n)) / (2 ** n)
        dp = mg.exact_expected_pnl(n, 0.5, 1.0, None)
        if not math.isclose(exhaustive, dp, abs_tol=1e-9):
            ok = False
    check("exact_expected_pnl (DP) matches exhaustive enumeration", ok)

    # 3. exact_expected_pnl handles large n without overflow, stays exactly zero
    check("exact_expected_pnl stable and exactly zero at n=2000, p=0.5, uncapped",
          math.isclose(mg.exact_expected_pnl(2000, 0.5, 1.0, None), 0.0, abs_tol=1e-6))
    check("exact_expected_pnl stable and exactly zero at n=2000, p=0.5, capped",
          math.isclose(mg.exact_expected_pnl(2000, 0.5, 1.0, 10), 0.0, abs_tol=1e-6))

    # 4. probability_max_streak_at_least vs closed form for the boundary case
    for k in (5, 10):
        exact = mg.probability_max_streak_at_least(k, 0.5, k)
        closed_form = 0.5 ** k
        check(f"probability_max_streak_at_least matches closed form at n=k (k={k})",
              math.isclose(exact, closed_form, abs_tol=1e-12))

    # 5. Monte Carlo confirms uncapped doubling is severely misleading in
    #    finite samples despite the exact zero mean: the empirical average
    #    across a realistic number of trials stays far from zero because
    #    the compensating loss event is too rare to sample.
    rng = np.random.default_rng(0)
    n_steps, trials = 1000, 5000
    mc_pnls = []
    for i in range(trials):
        outcomes = list(rng.choice([1, -1], size=n_steps))
        mc_pnls.append(mg.simulate_doubling_strategy(outcomes, 1.0, None)["final_pnl"])
    mc_mean = statistics.mean(mc_pnls)
    exact_mean = mg.exact_expected_pnl(n_steps, 0.5, 1.0, None)
    check("Monte Carlo mean for uncapped doubling is far from the true exact mean "
          "(illustrates the sampling-bias mechanism, not a bug)",
          abs(mc_mean - exact_mean) > 50)

    # 6. gambler's ruin probability and expected duration vs simulation
    a, b, p, trials = 6, 9, 0.52, 30000
    outcomes = [mg.simulate_ruin(a, b, p, seed=i) for i in range(trials)]
    emp_prob = sum(o[0] for o in outcomes) / trials
    emp_dur = statistics.mean(o[1] for o in outcomes)
    check("ruin probability matches simulation",
          math.isclose(emp_prob, mg.ruin_probability(a, b, p), abs_tol=0.02))
    check("expected duration matches simulation",
          math.isclose(emp_dur, mg.expected_duration(a, b, p), rel_tol=0.05))

    # 7. Doob's maximal inequality never violated across many parameter sets
    ok = True
    for n, level in [(20, 30), (50, 80), (100, 150)]:
        result = mg.doob_maximal_inequality_check(n, level, trials=4000, seed=1)
        if not result["bound_holds"]:
            ok = False
    check("Doob's maximal inequality holds across multiple parameter sets", ok)

    # 8. Polya urn terminal distribution matches scipy.stats.beta (KS test)
    r0, b0, trials = 4, 6, 4000
    finals = [mg.polya_urn_path(r0, b0, 3000, seed=i)[-1] for i in range(trials)]
    ks_stat, ks_p = sstats.kstest(finals, sstats.beta(r0, b0).cdf)
    check("Polya urn terminal distribution passes KS test against scipy Beta(r0,b0)", ks_p > 0.01)

    # 9. OLS slope/intercept/homoskedastic SE vs scipy.stats.linregress
    x = list(rng.normal(0, 1, size=500))
    y = [2.0 + 0.5 * xi + n for xi, n in zip(x, rng.normal(0, 1, size=500))]
    ours = mg.ols_fit(x, y)
    scipy_result = sstats.linregress(x, y)
    check("OLS slope matches scipy.stats.linregress",
          math.isclose(ours["beta1"], scipy_result.slope, abs_tol=1e-9))
    check("OLS intercept matches scipy.stats.linregress",
          math.isclose(ours["beta0"], scipy_result.intercept, abs_tol=1e-9))
    check("OLS homoskedastic SE matches scipy.stats.linregress",
          math.isclose(ours["se_homoskedastic"], scipy_result.stderr, abs_tol=1e-9))

    # 10. White/HC0 robust SE vs an independent numpy sandwich-formula
    #     implementation (different code path, same estimator)
    x_arr = np.array(x)
    y_arr = np.array(y)
    X = np.column_stack([np.ones_like(x_arr), x_arr])
    beta_hat = np.linalg.lstsq(X, y_arr, rcond=None)[0]
    resid = y_arr - X @ beta_hat
    bread = np.linalg.inv(X.T @ X)
    meat = X.T @ np.diag(resid ** 2) @ X
    sandwich = bread @ meat @ bread
    numpy_robust_se = math.sqrt(sandwich[1, 1])
    check("White/HC0 robust SE matches independent numpy sandwich-formula recomputation",
          math.isclose(ours["se_robust"], numpy_robust_se, rel_tol=1e-6))

    passed = sum(1 for _, ok in checks if ok)
    print(f"\n{passed}/{len(checks)} checks passed")
    if passed != len(checks):
        sys.exit(1)


def build_parser():
    parser = argparse.ArgumentParser(description="Martingales laboratory CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("ruin", help="gambler's ruin via optional stopping")
    p1.add_argument("--a", type=int, required=True)
    p1.add_argument("--b", type=int, required=True)
    p1.add_argument("--p", type=float, required=True)
    p1.set_defaults(func=cmd_ruin)

    p2 = sub.add_parser("doob", help="Doob's maximal inequality check")
    p2.add_argument("--n", type=int, default=50)
    p2.add_argument("--level", type=float, default=100.0)
    p2.add_argument("--trials", type=int, default=10000)
    p2.add_argument("--seed", type=int, default=1)
    p2.set_defaults(func=cmd_doob)

    p3 = sub.add_parser("polya", help="simulate a Polya urn")
    p3.add_argument("--r0", type=int, required=True)
    p3.add_argument("--b0", type=int, required=True)
    p3.add_argument("--draws", type=int, default=1000)
    p3.add_argument("--seed", type=int, default=1)
    p3.set_defaults(func=cmd_polya)

    p4 = sub.add_parser("doubling", help="simulate the doubling betting system")
    p4.add_argument("--steps", type=int, default=1000)
    p4.add_argument("--p", type=float, default=0.5)
    p4.add_argument("--stake", type=float, default=1.0)
    p4.add_argument("--cap", type=int, default=None)
    p4.add_argument("--seed", type=int, default=1)
    p4.set_defaults(func=cmd_doubling)

    p5 = sub.add_parser("fxmartingale", help="test the martingale difference hypothesis on EUR/USD")
    p5.add_argument("--col", type=str, default="return")
    p5.add_argument("--trials", type=int, default=500)
    p5.add_argument("--seed", type=int, default=1)
    p5.set_defaults(func=cmd_fxmartingale)

    p6 = sub.add_parser("verify", help="cross-check implementation against analytical/oracle values")
    p6.set_defaults(func=cmd_verify)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
