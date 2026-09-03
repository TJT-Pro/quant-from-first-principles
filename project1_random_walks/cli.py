import argparse
import math
import os
import statistics
import sys

import random_walks as rw

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "DEXUSEU_returns.csv")


def cmd_simulate1d(args):
    path = rw.simulate_walk_1d(args.steps, p=args.p, seed=args.seed)
    print(f"1D random walk, {args.steps} steps, p={args.p}, seed={args.seed}")
    print(f"final position: {path[-1]}")
    print(f"max position:   {max(path)}")
    print(f"min position:   {min(path)}")
    if args.steps <= 60:
        print("path:", path)


def cmd_simulate2d(args):
    path = rw.simulate_walk_2d(args.steps, seed=args.seed)
    disp = rw.displacement_2d(path)
    print(f"2D random walk, {args.steps} steps, seed={args.seed}")
    print(f"final position:     {path[-1]}")
    print(f"final displacement: {disp[-1]:.4f}")
    print(f"max displacement:   {max(disp):.4f}")


def cmd_reflection(args):
    prob = rw.reflection_principle_max_at_least(args.n, args.a, 0.5)
    print(f"P(max_(0<=k<={args.n}) S_k >= {args.a}) = {prob:.6f}   [symmetric walk]")


def cmd_hitting(args):
    prob_ever = rw.hitting_probability_ever(args.a, args.p)
    print(f"P(walk ever reaches level {args.a} | p={args.p}) = {prob_ever:.6f}")
    expected = rw.expected_hitting_time_biased(args.a, args.p)
    if math.isinf(expected):
        print(f"E[T_{args.a}] = infinity")
    else:
        print(f"E[T_{args.a}] = {expected:.4f}")


def cmd_gambler(args):
    times = []
    for i in range(args.trials):
        t = rw.simulate_first_passage_time(args.a, p=args.p, max_steps=args.max_steps, seed=i)
        if t is not None:
            times.append(t)
    hit_fraction = len(times) / args.trials
    print(f"Simulated first passage to level {args.a}, p={args.p}, {args.trials} trials")
    print(f"fraction reaching level within {args.max_steps} steps: {hit_fraction:.4f}")
    if times:
        print(f"empirical mean hitting time: {statistics.mean(times):.4f}")
    theoretical = rw.expected_hitting_time_biased(args.a, args.p)
    print(f"theoretical E[T_a]: {'infinity' if math.isinf(theoretical) else f'{theoretical:.4f}'}")


def cmd_fxwalk(args):
    returns = rw.load_returns(DATA_PATH, column=args.col)
    result = rw.fx_windowed_analysis(returns, window=args.window, seed=args.seed)
    print("EUR/USD sign random walk analysis")
    print(f"observations:        {result['n_observations']}")
    print(f"window size:         {result['window']}")
    print(f"number of windows:   {result['n_windows']}")
    print(f"p_hat (up fraction): {result['p_hat']:.6f}")
    print(f"theoretical mean:    {result['theoretical_mean']:.4f}")
    print(f"empirical mean:      {result['empirical_mean']:.4f}")
    print(f"theoretical var:     {result['theoretical_var']:.4f}")
    print(f"empirical var:       {result['empirical_var']:.4f}")
    print(f"threshold level a:   {result['threshold']}")
    print(f"P(max >= a) theory:  {result['theoretical_p_exceed']:.4f}   [p=0.5 null hypothesis]")
    print(f"P(max >= a) empirical: {result['empirical_p_exceed']:.4f}")


def cmd_verify(args):
    import numpy as np
    from scipy import stats as sstats

    checks = []

    def check(name, condition):
        checks.append((name, bool(condition)))
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {name}")

    rng = np.random.default_rng(0)

    # 1. prob_Sn_equals against scipy binomial pmf
    for n, p in [(20, 0.5), (37, 0.3), (50, 0.65)]:
        for m in range(-n, n + 1, 2 if n % 2 == 0 else 1):
            k = (n + m) // 2 if (n + m) % 2 == 0 else None
            if k is None or k < 0 or k > n:
                continue
            expected = sstats.binom.pmf(k, n, p)
            got = rw.prob_Sn_equals(n, m, p)
            if not math.isclose(expected, got, abs_tol=1e-10):
                check(f"prob_Sn_equals vs scipy.binom (n={n}, m={m}, p={p})", False)
                break
        else:
            check(f"prob_Sn_equals vs scipy.binom.pmf (n={n}, p={p})", True)

    # 2. reflection principle vs brute-force enumeration
    ok = True
    for n, a in [(6, 1), (8, 2), (10, 3), (14, 4)]:
        exact = rw.reflection_principle_max_at_least(n, a, 0.5)
        brute = rw.reflection_principle_max_at_least_bruteforce(n, a)
        if not math.isclose(exact, brute, abs_tol=1e-9):
            ok = False
    check("reflection principle vs brute-force enumeration", ok)

    # 3. reflection principle vs large-scale Monte Carlo
    n, a, trials = 300, 15, 40000
    hits = 0
    for i in range(trials):
        path = rw.simulate_walk_1d(n, p=0.5, seed=1000 + i)
        if max(path) >= a:
            hits += 1
    empirical = hits / trials
    theoretical = rw.reflection_principle_max_at_least(n, a, 0.5)
    check(f"reflection principle vs Monte Carlo (n={n}, a={a}, trials={trials})",
          math.isclose(empirical, theoretical, abs_tol=0.01))

    # 4. mean and variance vs numpy simulation
    n, p, trials = 500, 0.5, 20000
    steps = rng.choice([-1, 1], size=(trials, n), p=[1 - p, p])
    finals = steps.sum(axis=1)
    emp_mean = float(np.mean(finals))
    emp_var = float(np.var(finals))
    check("theoretical mean matches numpy simulation (p=0.5)",
          math.isclose(emp_mean, rw.theoretical_mean_1d(n, p), abs_tol=1.0))
    check("theoretical variance matches numpy simulation (p=0.5)",
          math.isclose(emp_var, rw.theoretical_variance_1d(n, p), rel_tol=0.05))

    n, p, trials = 500, 0.62, 20000
    steps = rng.choice([-1, 1], size=(trials, n), p=[1 - p, p])
    finals = steps.sum(axis=1)
    emp_mean = float(np.mean(finals))
    emp_var = float(np.var(finals))
    check("theoretical mean matches numpy simulation (p=0.62)",
          math.isclose(emp_mean, rw.theoretical_mean_1d(n, p), abs_tol=1.5))
    check("theoretical variance matches numpy simulation (p=0.62)",
          math.isclose(emp_var, rw.theoretical_variance_1d(n, p), rel_tol=0.08))

    # 5. first passage pmf vs simulation
    a, p, trials, max_steps = 3, 0.55, 30000, 400
    times = [rw.simulate_first_passage_time(a, p=p, max_steps=max_steps, seed=2000 + i)
             for i in range(trials)]
    ok = True
    for target_n in (3, 5, 7, 9):
        emp = sum(1 for t in times if t == target_n) / trials
        th = rw.first_passage_pmf(target_n, a, p)
        if abs(emp - th) > 0.015:
            ok = False
    check(f"first-passage pmf vs simulation (a={a}, p={p})", ok)

    # 6. hitting probability (biased down) vs simulation
    a, p, trials, max_steps = 2, 0.35, 30000, 2000
    reached = sum(1 for i in range(trials)
                  if rw.simulate_first_passage_time(a, p=p, max_steps=max_steps, seed=3000 + i) is not None)
    emp = reached / trials
    th = rw.hitting_probability_ever(a, p)
    check(f"hitting probability vs simulation (a={a}, p={p})",
          math.isclose(emp, th, abs_tol=0.02))

    # 7. expected hitting time (positive drift) vs simulation
    a, p, trials = 4, 0.7, 8000
    times = [rw.simulate_first_passage_time(a, p=p, max_steps=5000, seed=4000 + i) for i in range(trials)]
    times = [t for t in times if t is not None]
    emp_mean = statistics.mean(times)
    th_mean = rw.expected_hitting_time_biased(a, p)
    check(f"expected hitting time vs simulation (a={a}, p={p})",
          math.isclose(emp_mean, th_mean, rel_tol=0.1))

    # 8. 2D recurrence: return probability increases towards 1 with horizon
    short = rw.estimate_return_probability_2d(50, 8000, seed=1)
    long = rw.estimate_return_probability_2d(5000, 8000, seed=1)
    check("2D return probability increases with horizon (Polya recurrence)", long > short)

    # 9. distribution of S_n sums to one, cross-checked with numpy
    n, p = 40, 0.5
    total = sum(rw.prob_Sn_equals(n, m, p) for m in range(-n, n + 1))
    check("prob_Sn_equals sums to 1 over full support", math.isclose(total, 1.0, abs_tol=1e-9))

    passed = sum(1 for _, ok in checks if ok)
    print(f"\n{passed}/{len(checks)} checks passed")
    if passed != len(checks):
        sys.exit(1)


def build_parser():
    parser = argparse.ArgumentParser(description="Random walks laboratory CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("simulate1d", help="simulate a 1D random walk")
    p1.add_argument("--steps", type=int, default=100)
    p1.add_argument("--p", type=float, default=0.5)
    p1.add_argument("--seed", type=int, default=None)
    p1.set_defaults(func=cmd_simulate1d)

    p2 = sub.add_parser("simulate2d", help="simulate a 2D random walk")
    p2.add_argument("--steps", type=int, default=100)
    p2.add_argument("--seed", type=int, default=None)
    p2.set_defaults(func=cmd_simulate2d)

    p3 = sub.add_parser("reflection", help="reflection-principle max probability")
    p3.add_argument("--n", type=int, required=True)
    p3.add_argument("--a", type=int, required=True)
    p3.set_defaults(func=cmd_reflection)

    p4 = sub.add_parser("hitting", help="hitting probability and expected hitting time")
    p4.add_argument("--a", type=int, required=True)
    p4.add_argument("--p", type=float, required=True)
    p4.set_defaults(func=cmd_hitting)

    p5 = sub.add_parser("gambler", help="simulate first passage times")
    p5.add_argument("--a", type=int, required=True)
    p5.add_argument("--p", type=float, default=0.5)
    p5.add_argument("--trials", type=int, default=2000)
    p5.add_argument("--max-steps", type=int, default=5000)
    p5.set_defaults(func=cmd_gambler)

    p6 = sub.add_parser("fxwalk", help="analyse EUR/USD sign process as a random walk")
    p6.add_argument("--col", type=str, default="return")
    p6.add_argument("--window", type=int, default=252)
    p6.add_argument("--seed", type=int, default=1)
    p6.set_defaults(func=cmd_fxwalk)

    p7 = sub.add_parser("verify", help="cross-check implementation against analytical/oracle values")
    p7.set_defaults(func=cmd_verify)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
