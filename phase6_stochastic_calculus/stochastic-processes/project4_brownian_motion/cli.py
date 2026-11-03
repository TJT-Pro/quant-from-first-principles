import argparse
import math
import os
import statistics
import sys

import brownian_motion as bm

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "DEXUSEU_returns.csv")


def cmd_simulate(args):
    times, values = bm.simulate_brownian_motion(args.t, args.steps, seed=args.seed)
    print(f"Brownian motion, t_max={args.t}, {args.steps} steps, seed={args.seed}")
    print(f"final value: {values[-1]:.6f}")
    print(f"quadratic variation: {bm.quadratic_variation(values):.6f} (theory: {args.t})")
    print(f"total variation:     {bm.total_variation(values):.6f}")


def cmd_reflection(args):
    p = bm.reflection_principle_bm(args.a, args.t)
    print(f"P(max_(0<=s<={args.t}) W_s >= {args.a}) = {p:.6f}")
    print(f"equivalently P(T_{args.a} <= {args.t}) = {p:.6f}")


def cmd_passage(args):
    density_points = [args.t * k / 10 for k in range(1, 11)]
    print(f"first-passage time density for level a={args.a}")
    for t in density_points:
        print(f"  f({t:.3f}) = {bm.first_passage_density_bm(args.a, t):.6f}")
    print(f"P(T_a <= {args.t}) = {bm.first_passage_cdf_bm(args.a, args.t):.6f}")


def cmd_donsker(args):
    times, values = bm.donsker_walk(args.steps, args.t, seed=args.seed)
    print(f"Donsker-scaled random walk, {args.steps} steps over [0, {args.t}]")
    print(f"final value: {values[-1]:.6f}")
    print(f"final variance (theory, Var=t): {args.t}")


def cmd_bridge(args):
    times, values = bm.brownian_bridge(args.t, args.steps, seed=args.seed)
    print(f"Brownian bridge on [0, {args.t}], {args.steps} steps")
    print(f"B(0) = {values[0]:.8f}, B(t_max) = {values[-1]:.8f}")
    mid = len(values) // 2
    print(f"B(t_max/2) = {values[mid]:.6f} (theoretical variance at midpoint: {args.t / 4:.6f})")


def cmd_fxscaling(args):
    returns = bm.load_log_returns(DATA_PATH, column=args.col)
    print(f"EUR/USD log returns, {len(returns)} observations")
    print(f"realized variance (full sample): {bm.realized_variance(returns):.8f}")
    print()
    horizons = [1, 2, 5, 10, 20, 50]
    scaling = bm.compare_scaling_to_theory(returns, horizons)
    print("horizon q | empirical Var(q-period) | theoretical q*Var(1-period)")
    for q, emp, theo in zip(scaling["horizons"], scaling["empirical_variance"], scaling["theoretical_variance"]):
        print(f"  {q:4d}    |  {emp:.8f}           |  {theo:.8f}")
    print()
    print("variance ratio test (VR(q) = 1 under Brownian scaling):")
    for q in [2, 5, 10, 20, 50]:
        result = bm.variance_ratio_test(returns, q)
        print(f"  q={q:3d}  VR={result['vr']:.4f}  z={result['z']:+.4f}  "
              f"p={result['p_value']:.4f}  (m={result['m']} blocks)")


def cmd_verify(args):
    import numpy as np
    from scipy import stats as sstats

    checks = []

    def check(name, condition):
        checks.append((name, bool(condition)))
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {name}")

    rng = np.random.default_rng(0)

    # 1. standard_normal_cdf vs scipy
    ok = True
    for x in (-3.0, -1.0, -0.3, 0.0, 0.7, 1.96, 2.5):
        if not math.isclose(bm.standard_normal_cdf(x), sstats.norm.cdf(x), abs_tol=1e-10):
            ok = False
    check("standard_normal_cdf vs scipy.stats.norm.cdf", ok)

    # 2. Box-Muller normals vs numpy standard normal (moment matching)
    import random as pyrandom
    py_rng = pyrandom.Random(0)
    ours = bm.generate_standard_normals(300000, py_rng)
    theirs = rng.standard_normal(300000)
    check("Box-Muller mean matches numpy standard_normal",
          math.isclose(statistics.mean(ours), float(np.mean(theirs)), abs_tol=0.02))
    check("Box-Muller variance matches numpy standard_normal",
          math.isclose(statistics.variance(ours), float(np.var(theirs, ddof=1)), abs_tol=0.02))
    ks_stat, ks_p = sstats.kstest(ours, "norm")
    check("Box-Muller sample passes KS test against N(0,1) (scipy)", ks_p > 0.01)

    # 3. terminal distribution of simulated BM vs scipy normal oracle
    t_max, n_steps, trials = 2.5, 300, 20000
    finals = [bm.simulate_brownian_motion(t_max, n_steps, seed=i)[1][-1] for i in range(trials)]
    ks_stat, ks_p = sstats.kstest(finals, sstats.norm(loc=0.0, scale=math.sqrt(t_max)).cdf)
    check(f"BM terminal value passes KS test against N(0, t) (scipy, t={t_max})", ks_p > 0.01)

    # 4. quadratic variation converges to t; variance shrinks with finer partitions
    t_max = 3.0
    means, variances = [], []
    for n_steps in (20, 200, 2000):
        qvs = [bm.quadratic_variation(bm.simulate_brownian_motion(t_max, n_steps, seed=i)[1])
               for i in range(2000)]
        means.append(statistics.mean(qvs))
        variances.append(statistics.variance(qvs))
    check("quadratic variation mean converges to t across partitions",
          all(math.isclose(m, t_max, abs_tol=0.15) for m in means))
    check("quadratic variation spread shrinks as partition is refined",
          variances[0] > variances[1] > variances[2])

    # 5. total variation grows with finer partitions (no finite limit)
    means_tv = []
    for n_steps in (20, 200, 2000):
        tvs = [bm.total_variation(bm.simulate_brownian_motion(t_max, n_steps, seed=i)[1])
               for i in range(1500)]
        means_tv.append(statistics.mean(tvs))
    check("total variation grows without bound as partition is refined",
          means_tv[0] < means_tv[1] < means_tv[2])

    # 6. reflection principle vs Monte Carlo, and vs scipy via the
    #    equivalent statement P(M_t >= a) = 2*P(W_t >= a)
    a, t = 1.2, 2.0
    ours = bm.reflection_principle_bm(a, t)
    scipy_equiv = 2.0 * sstats.norm.sf(a, loc=0.0, scale=math.sqrt(t))
    check("reflection principle matches scipy normal survival function",
          math.isclose(ours, scipy_equiv, abs_tol=1e-10))

    trials, n_steps = 30000, 400
    hits = sum(1 for i in range(trials)
               if max(bm.simulate_brownian_motion(t, n_steps, seed=i)[1]) >= a)
    empirical = hits / trials
    check("reflection principle matches Monte Carlo maximum",
          math.isclose(empirical, ours, abs_tol=0.015))

    # 7. first-passage density integrates (numerically) to the CDF
    a, t_end, n = 1.0, 3.0, 20000
    h = t_end / n
    total = sum(bm.first_passage_density_bm(a, i * h) for i in range(1, n)) * h
    check("first-passage density integrates to the first-passage CDF",
          math.isclose(total, bm.first_passage_cdf_bm(a, t_end), abs_tol=0.01))

    # 8. Donsker walk terminal distribution converges toward Gaussian
    #    as steps increase, measured by KS statistic against N(0,t)
    t_max, trials = 1.0, 8000
    coarse = [bm.donsker_walk(4, t_max, seed=i)[1][-1] for i in range(trials)]
    fine = [bm.donsker_walk(400, t_max, seed=i)[1][-1] for i in range(trials)]
    ks_coarse, _ = sstats.kstest(coarse, sstats.norm(0, math.sqrt(t_max)).cdf)
    ks_fine, _ = sstats.kstest(fine, sstats.norm(0, math.sqrt(t_max)).cdf)
    check("Donsker approximation to Brownian motion improves with more steps (KS distance)",
          ks_fine < ks_coarse)

    # 9. Brownian bridge variance at several checkpoints vs closed form
    t_max, n_steps, trials = 4.0, 400, 6000
    ok = True
    for frac in (0.25, 0.5, 0.75):
        idx = int(n_steps * frac)
        vals = [bm.brownian_bridge(t_max, n_steps, seed=i)[1][idx] for i in range(trials)]
        t = t_max * frac
        theoretical_var = t * (t_max - t) / t_max
        if not math.isclose(statistics.variance(vals), theoretical_var, abs_tol=0.2):
            ok = False
    check("Brownian bridge variance matches t(t_max-t)/t_max at multiple checkpoints", ok)

    # 10. variance ratio test calibration: under the true null (iid
    #     Gaussian increments), the test should reject at close to its
    #     nominal 5% rate across many independent replications, and the
    #     z-statistics should behave like a standard normal sample.
    rejections = 0
    z_values = []
    reps = 500
    for i in range(reps):
        sample = [rng.normal(0.0, 0.01) for _ in range(2000)]
        result = bm.variance_ratio_test(sample, q=5)
        z_values.append(result["z"])
        if result["p_value"] < 0.05:
            rejections += 1
    rejection_rate = rejections / reps
    check("variance ratio test rejection rate near nominal 5% under the true null",
          0.02 <= rejection_rate <= 0.09)
    check("variance ratio z-statistics have approximately unit variance under the null",
          math.isclose(statistics.variance(z_values), 1.0, abs_tol=0.25))

    # 11. variance ratio test detects strong induced autocorrelation
    py_rng2 = pyrandom.Random(3)
    base = [py_rng2.gauss(0.0, 0.01) for _ in range(4000)]
    momentum = [base[0]]
    for i in range(1, len(base)):
        momentum.append(0.5 * momentum[-1] + base[i])
    result = bm.variance_ratio_test(momentum, q=10)
    check("variance ratio test rejects strongly autocorrelated synthetic series",
          result["p_value"] < 0.01 and result["vr"] > 1.2)

    passed = sum(1 for _, ok in checks if ok)
    print(f"\n{passed}/{len(checks)} checks passed")
    if passed != len(checks):
        sys.exit(1)


def build_parser():
    parser = argparse.ArgumentParser(description="Brownian motion laboratory CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("simulate", help="simulate a Brownian motion path")
    p1.add_argument("--t", type=float, default=1.0)
    p1.add_argument("--steps", type=int, default=1000)
    p1.add_argument("--seed", type=int, default=None)
    p1.set_defaults(func=cmd_simulate)

    p2 = sub.add_parser("reflection", help="reflection principle / max distribution")
    p2.add_argument("--a", type=float, required=True)
    p2.add_argument("--t", type=float, required=True)
    p2.set_defaults(func=cmd_reflection)

    p3 = sub.add_parser("passage", help="first-passage time density and CDF")
    p3.add_argument("--a", type=float, required=True)
    p3.add_argument("--t", type=float, required=True)
    p3.set_defaults(func=cmd_passage)

    p4 = sub.add_parser("donsker", help="scaled random walk approximation to BM")
    p4.add_argument("--steps", type=int, default=1000)
    p4.add_argument("--t", type=float, default=1.0)
    p4.add_argument("--seed", type=int, default=None)
    p4.set_defaults(func=cmd_donsker)

    p5 = sub.add_parser("bridge", help="simulate a Brownian bridge")
    p5.add_argument("--t", type=float, default=1.0)
    p5.add_argument("--steps", type=int, default=1000)
    p5.add_argument("--seed", type=int, default=None)
    p5.set_defaults(func=cmd_bridge)

    p6 = sub.add_parser("fxscaling", help="test EUR/USD returns against Brownian scaling")
    p6.add_argument("--col", type=str, default="log_return")
    p6.set_defaults(func=cmd_fxscaling)

    p7 = sub.add_parser("verify", help="cross-check implementation against analytical/oracle values")
    p7.set_defaults(func=cmd_verify)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
