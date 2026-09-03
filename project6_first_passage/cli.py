import argparse
import math
import os
import statistics
import sys

import first_passage as fp

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "DEXUSEU_returns.csv")


def cmd_ruin(args):
    prob = fp.absorbing_chain_ruin_probability(args.a, args.b, args.p)
    duration = fp.absorbing_chain_expected_duration(args.a, args.b, args.p)
    print(f"discrete two-boundary ruin (absorbing Markov chain method), a={args.a}, b={args.b}, p={args.p}")
    print(f"P(hit {args.a} before -{args.b}) = {prob:.6f}")
    print(f"E[duration] = {duration:.4f}")


def cmd_bmruin(args):
    prob = fp.bm_ruin_probability(args.a, args.b, args.mu, args.sigma)
    duration = fp.bm_expected_duration(args.a, args.b, args.mu, args.sigma)
    print(f"continuous two-boundary ruin (Brownian motion), a={args.a}, b={args.b}, mu={args.mu}, sigma={args.sigma}")
    print(f"P(hit {args.a} before -{args.b}) = {prob:.6f}")
    print(f"E[duration] = {duration:.4f}")


def cmd_barrier(args):
    prob = fp.bm_max_exceeds(args.a, args.t, args.mu, args.sigma)
    print(f"P(max_(0<=s<={args.t}) X_s >= {args.a}) = {prob:.6f}  (mu={args.mu}, sigma={args.sigma})")


def cmd_fxstop(args):
    returns = fp.load_log_returns(DATA_PATH, column=args.col)
    cal = fp.calibrate_gbm(returns)
    print(f"calibrated from {len(returns)} EUR/USD log returns: mu={cal['mu']:.8f}, sigma={cal['sigma']:.8f} per day")

    target = args.multiple * cal["sigma"]
    stop = args.multiple * cal["sigma"]
    print(f"\nstop/target set at {args.multiple} daily sigmas: target={target:.6f}, stop={stop:.6f}")

    model = fp.model_stop_target_prediction(target, stop, cal["mu"], cal["sigma"])
    print(f"\nmodel (infinite-horizon exact formula):")
    print(f"  P(hit target first) = {model['p_hit_target']:.4f}")
    print(f"  P(hit stop first)   = {model['p_hit_stop']:.4f}")
    print(f"  E[time to resolve]  = {model['expected_duration']:.2f} days")

    historical = fp.historical_stop_target_backtest(returns, target, stop, args.horizon)
    print(f"\nhistorical backtest over {args.horizon}-day horizon, {historical['n_trials']} overlapping windows:")
    print(f"  P(hit target first) = {historical['p_hit_target']:.4f}")
    print(f"  P(hit stop first)   = {historical['p_hit_stop']:.4f}")
    print(f"  P(unresolved)       = {historical['p_unresolved']:.4f}")
    print(f"  mean duration when resolved: {historical['mean_duration_when_resolved']:.2f} days")


def cmd_verify(args):
    import random
    import numpy as np
    from scipy import stats as sstats

    checks = []

    def check(name, condition):
        checks.append((name, bool(condition)))
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {name}")

    # 1. Triangulation: absorbing-chain method vs the closed-form
    #    difference-equation result (Project 1's method) for several
    #    symmetric and biased parameter sets.
    ok = True
    for a, b, p in [(8, 12, 0.5), (5, 5, 0.5), (3, 10, 0.5)]:
        chain = fp.absorbing_chain_ruin_probability(a, b, p)
        closed_form = b / (a + b)
        if not math.isclose(chain, closed_form, abs_tol=1e-9):
            ok = False
    check("absorbing-chain ruin probability matches closed-form difference-equation result (symmetric)", ok)

    ok = True
    for a, b, p in [(8, 12, 0.5), (5, 5, 0.5), (3, 10, 0.5)]:
        chain = fp.absorbing_chain_expected_duration(a, b, p)
        closed_form = float(a * b)
        if not math.isclose(chain, closed_form, abs_tol=1e-6):
            ok = False
    check("absorbing-chain expected duration matches closed-form a*b (symmetric)", ok)

    # 2. Triangulation: absorbing-chain method vs the martingale/OST
    #    formula (Project 5's method) for biased walks.
    ok = True
    for a, b, p in [(6, 9, 0.55), (4, 7, 0.4), (10, 3, 0.65)]:
        chain_prob = fp.absorbing_chain_ruin_probability(a, b, p)
        theta = (1 - p) / p
        martingale_prob = (1.0 - theta ** (-b)) / (theta ** a - theta ** (-b))
        if not math.isclose(chain_prob, martingale_prob, abs_tol=1e-9):
            ok = False
    check("absorbing-chain ruin probability matches martingale/OST formula (biased)", ok)

    # 3. Both discrete methods agree with direct Monte Carlo simulation
    a, b, p, trials = 6, 9, 0.55, 30000
    outcomes = [fp.simulate_ruin(a, b, p, seed=i) for i in range(trials)]
    emp_prob = sum(o[0] for o in outcomes) / trials
    emp_dur = statistics.mean(o[1] for o in outcomes)
    check("absorbing-chain ruin probability matches Monte Carlo",
          math.isclose(emp_prob, fp.absorbing_chain_ruin_probability(a, b, p), abs_tol=0.015))
    check("absorbing-chain expected duration matches Monte Carlo",
          math.isclose(emp_dur, fp.absorbing_chain_expected_duration(a, b, p), rel_tol=0.05))

    # 4. survival curve integrates consistently: P(T>n) - P(T>n+1) sums
    #    to 1 - P(T>N) for large N, and matches 1 minus a direct CDF
    #    built from repeated Monte Carlo simulation.
    a, b, p, n_steps = 5, 7, 0.5, 100
    curve = fp.absorbing_chain_survival_curve(a, b, p, n_steps)
    check("survival curve starts at 1 and is non-increasing",
          math.isclose(curve[0], 1.0) and all(x >= y - 1e-9 for x, y in zip(curve, curve[1:])))

    # 5. continuous double-barrier formula vs Monte Carlo (drifted)
    a, b, mu, sigma, dt, trials = 1.0, 1.5, 0.4, 1.0, 0.0005, 20000
    theoretical = fp.bm_ruin_probability(a, b, mu, sigma)
    outcomes = [fp.simulate_bm_ruin(a, b, mu, sigma, dt, seed=i)[0] for i in range(trials)]
    empirical = sum(outcomes) / trials
    check("Brownian two-boundary ruin probability matches Monte Carlo (drifted)",
          math.isclose(empirical, theoretical, abs_tol=0.02))

    theoretical_dur = fp.bm_expected_duration(a, b, mu, sigma)
    durations = [fp.simulate_bm_ruin(a, b, mu, sigma, dt, seed=i)[1] for i in range(trials)]
    empirical_dur = statistics.mean(durations)
    check("Brownian two-boundary expected duration matches Monte Carlo (drifted)",
          math.isclose(empirical_dur, theoretical_dur, rel_tol=0.1))

    # 6. drift-adjusted reflection principle vs scipy-based recomputation
    #    and vs Monte Carlo
    a, t, mu, sigma = 1.2, 2.0, 0.25, 1.0
    ours = fp.bm_max_exceeds(a, t, mu, sigma)
    scipy_recompute = (sstats.norm.cdf((mu * t - a) / (sigma * math.sqrt(t))) +
                        math.exp(2 * mu * a / sigma ** 2) *
                        sstats.norm.cdf((-mu * t - a) / (sigma * math.sqrt(t))))
    check("drift-adjusted reflection principle matches scipy-based recomputation",
          math.isclose(ours, scipy_recompute, abs_tol=1e-10))

    # Discrete-time simulation of a continuous barrier-crossing
    # probability is subject to the well-known discrete-monitoring
    # bias (Broadie-Glasserman-Kou 1997): checking the path only at
    # finitely many grid points systematically underestimates the true
    # continuous supremum, since the path can cross the barrier and
    # return between two monitoring points without being detected.
    # This bias shrinks slowly, at rate O(1/sqrt(n_steps)); enough
    # steps are used here for it to fall within the tolerance below.
    n_steps, trials = 5000, 20000
    dt = t / n_steps
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(trials):
        increments = rng.normal(mu * dt, sigma * math.sqrt(dt), size=n_steps)
        path = np.cumsum(increments)
        if path.max() >= a:
            hits += 1
    empirical = hits / trials
    check("drift-adjusted reflection principle matches Monte Carlo (numpy, fine discretization)",
          math.isclose(empirical, ours, abs_tol=0.015))

    # 7. zero-drift barrier formula recovers Project-4-style formula
    a, t, sigma = 1.5, 3.0, 1.0
    ours_zero_drift = fp.bm_max_exceeds(a, t, 0.0, sigma)
    project4_style = 2.0 * sstats.norm.sf(a, loc=0.0, scale=sigma * math.sqrt(t))
    check("zero-drift limit matches driftless reflection principle formula",
          math.isclose(ours_zero_drift, project4_style, abs_tol=1e-9))

    # 8. calibration and finite-horizon comparison sanity checks
    returns = fp.load_log_returns(DATA_PATH)
    cal = fp.calibrate_gbm(returns)
    check("calibrated sigma is positive and of plausible daily FX magnitude",
          0.001 < cal["sigma"] < 0.02)

    passed = sum(1 for _, ok in checks if ok)
    print(f"\n{passed}/{len(checks)} checks passed")
    if passed != len(checks):
        sys.exit(1)


def build_parser():
    parser = argparse.ArgumentParser(description="First-passage problems laboratory CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("ruin", help="discrete two-boundary ruin via absorbing Markov chain")
    p1.add_argument("--a", type=int, required=True)
    p1.add_argument("--b", type=int, required=True)
    p1.add_argument("--p", type=float, required=True)
    p1.set_defaults(func=cmd_ruin)

    p2 = sub.add_parser("bmruin", help="continuous two-boundary ruin for drifted Brownian motion")
    p2.add_argument("--a", type=float, required=True)
    p2.add_argument("--b", type=float, required=True)
    p2.add_argument("--mu", type=float, default=0.0)
    p2.add_argument("--sigma", type=float, default=1.0)
    p2.set_defaults(func=cmd_bmruin)

    p3 = sub.add_parser("barrier", help="drift-adjusted single-barrier exceedance probability")
    p3.add_argument("--a", type=float, required=True)
    p3.add_argument("--t", type=float, required=True)
    p3.add_argument("--mu", type=float, default=0.0)
    p3.add_argument("--sigma", type=float, default=1.0)
    p3.set_defaults(func=cmd_barrier)

    p4 = sub.add_parser("fxstop", help="stop-loss/take-profit: model vs historical backtest on EUR/USD")
    p4.add_argument("--col", type=str, default="log_return")
    p4.add_argument("--multiple", type=float, default=10.0, help="stop/target distance in daily sigmas")
    p4.add_argument("--horizon", type=int, default=200)
    p4.set_defaults(func=cmd_fxstop)

    p5 = sub.add_parser("verify", help="cross-check implementation against analytical/oracle values")
    p5.set_defaults(func=cmd_verify)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
