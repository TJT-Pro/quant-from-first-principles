import argparse
import json
import math
import os
import sys

import markov_chains as mc

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "DEXUSEU_returns.csv")


def _parse_matrix(text: str):
    rows = text.strip().split(";")
    return [[float(x) for x in row.split(",")] for row in rows]


def cmd_stationary(args):
    p = _parse_matrix(args.matrix)
    mc.validate_transition_matrix(p)
    pi_linear = mc.stationary_distribution_linear(p)
    pi_power = mc.stationary_distribution_power_iteration(p)
    print("stationary distribution (linear solve):", [round(x, 6) for x in pi_linear])
    print("stationary distribution (power iteration):", [round(x, 6) for x in pi_power])
    print("irreducible:", mc.is_irreducible(p))
    print("aperiodic:", mc.is_aperiodic(p))


def cmd_convergence(args):
    p = _parse_matrix(args.matrix)
    initial = [float(x) for x in args.initial.split(",")]
    trace = mc.convergence_trace(p, initial, args.n)
    for step, d in enumerate(trace, start=1):
        if step % max(1, args.n // 10) == 0 or step == 1:
            print(f"n={step:4d}   TV distance to stationary = {d:.8f}")
    slope = mc.estimate_convergence_rate(trace)
    if slope is not None:
        print(f"estimated log-linear convergence rate: {slope:.6f} per step")
    else:
        print("could not estimate a convergence rate (too few informative points)")


def cmd_passage(args):
    p = _parse_matrix(args.matrix)
    h = mc.mean_first_passage_times(p, args.target)
    for i, hi in enumerate(h):
        print(f"E[time from state {i} to state {args.target}] = {hi:.6f}")
    pi = mc.stationary_distribution_linear(p)
    print(f"expected return time to state {args.target} (Kac's formula): "
          f"{mc.expected_return_time(pi, args.target):.6f}")


def cmd_simulate(args):
    p = _parse_matrix(args.matrix)
    states = mc.simulate_chain(p, args.initial, args.steps, seed=args.seed)
    print(f"simulated {args.steps} steps from state {args.initial}")
    if args.steps <= 100:
        print("states:", states)
    counts = {}
    for s in states:
        counts[s] = counts.get(s, 0) + 1
    print("visitation frequency:", {k: round(v / len(states), 4) for k, v in sorted(counts.items())})


def cmd_fxchain(args):
    returns = mc.load_returns(DATA_PATH, column=args.col)
    result = mc.analyze_fx_markov_chain(returns, n_bins=args.bins, seed=args.seed)
    print(f"EUR/USD discretised Markov chain, {args.bins} bins, {result['n_observations']} observations")
    print("breakpoints:", [round(x, 6) for x in result["breakpoints"]])
    print("transition matrix:")
    for row in result["transition_matrix"]:
        print("  ", [round(x, 4) for x in row])
    print("stationary distribution:  ", [round(x, 4) for x in result["stationary_distribution"]])
    print("empirical frequency:      ", [round(x, 4) for x in result["empirical_frequency"]])
    test = result["independence_test"]
    print(f"independence test: statistic={test['statistic']:.4f}, df={test['df']}, "
          f"p-value={test['p_value']:.6f}")
    print(f"mean passage, lowest bin -> highest bin: {result['mean_passage_low_to_high']:.2f} days")
    print(f"mean passage, highest bin -> lowest bin: {result['mean_passage_high_to_low']:.2f} days")

    print("\nSign chain (2-state up/down):")
    states = mc.sign_states(returns, seed=args.seed)
    sign_test = mc.markov_independence_test(states, 2)
    print(f"  lag-1 independence test: statistic={sign_test['statistic']:.4f}, "
          f"df={sign_test['df']}, p-value={sign_test['p_value']:.6f}")


def cmd_verify(args):
    import numpy as np
    from scipy import stats as sstats

    checks = []

    def check(name, condition):
        checks.append((name, bool(condition)))
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {name}")

    rng = np.random.default_rng(0)

    # 1. matrix multiply / power vs numpy
    p = [[0.2, 0.5, 0.3], [0.1, 0.7, 0.2], [0.4, 0.4, 0.2]]
    p_np = np.array(p)
    for n in (1, 2, 5, 10, 37):
        ours = mc.matrix_power(p, n)
        theirs = np.linalg.matrix_power(p_np, n)
        ok = all(math.isclose(ours[i][j], theirs[i][j], abs_tol=1e-8)
                 for i in range(3) for j in range(3))
        check(f"matrix_power vs numpy.linalg.matrix_power (n={n})", ok)

    # 2. stationary distribution vs numpy eigenvector of P^T for eigenvalue 1
    for trial, mat in enumerate([
        [[0.2, 0.5, 0.3], [0.1, 0.7, 0.2], [0.4, 0.4, 0.2]],
        [[0.9, 0.1], [0.05, 0.95]],
        [[0.25, 0.25, 0.25, 0.25], [0.1, 0.6, 0.2, 0.1],
         [0.3, 0.3, 0.2, 0.2], [0.0, 0.5, 0.0, 0.5]],
    ]):
        ours = mc.stationary_distribution_linear(mat)
        eigvals, eigvecs = np.linalg.eig(np.array(mat).T)
        idx = int(np.argmin(np.abs(eigvals - 1.0)))
        vec = np.real(eigvecs[:, idx])
        vec = vec / vec.sum()
        ok = all(math.isclose(ours[i], vec[i], abs_tol=1e-6) for i in range(len(ours)))
        check(f"stationary distribution vs numpy eigenvector (case {trial})", ok)

    # 3. power iteration vs linear solve
    mat = [[0.2, 0.5, 0.3], [0.1, 0.7, 0.2], [0.4, 0.4, 0.2]]
    exact = mc.stationary_distribution_linear(mat)
    iterative = mc.stationary_distribution_power_iteration(mat)
    check("power iteration matches linear solve",
          all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(exact, iterative)))

    # 4. chi-square survival function vs scipy
    ok = True
    for x, df in [(1.0, 1), (3.5, 2), (10.0, 5), (0.5, 3), (25.0, 10)]:
        ours = mc.chi2_sf(x, df)
        theirs = sstats.chi2.sf(x, df)
        if not math.isclose(ours, theirs, abs_tol=1e-9):
            ok = False
    check("chi2_sf vs scipy.stats.chi2.sf", ok)

    # 5. simulated long-run visitation frequency vs stationary distribution
    mat = [[0.2, 0.5, 0.3], [0.1, 0.7, 0.2], [0.4, 0.4, 0.2]]
    pi = mc.stationary_distribution_linear(mat)
    states = mc.simulate_chain(mat, initial_state=0, n_steps=200000, seed=1)
    counts = [0, 0, 0]
    for s in states:
        counts[s] += 1
    freq = [c / len(states) for c in counts]
    check("long-run simulated frequency matches stationary distribution",
          all(math.isclose(freq[i], pi[i], abs_tol=0.01) for i in range(3)))

    # 6. mean first passage time vs simulation
    mat = [[0.7, 0.3], [0.4, 0.6]]
    target = 1
    theoretical = mc.mean_first_passage_times(mat, target)[0]
    trials = 20000
    total = 0
    for i in range(trials):
        path = mc.simulate_chain(mat, initial_state=0, n_steps=500, seed=5000 + i)
        try:
            first_hit = path.index(target)
        except ValueError:
            first_hit = 500
        total += first_hit
    empirical = total / trials
    check("mean first-passage time vs simulation",
          math.isclose(empirical, theoretical, rel_tol=0.1))

    # 7. expected return time (Kac's formula) vs simulation
    mat = [[0.2, 0.5, 0.3], [0.1, 0.7, 0.2], [0.4, 0.4, 0.2]]
    pi = mc.stationary_distribution_linear(mat)
    state = 0
    n_steps = 300000
    path = mc.simulate_chain(mat, initial_state=state, n_steps=n_steps, seed=2)
    visits = [i for i, s in enumerate(path) if s == state]
    gaps = [visits[i + 1] - visits[i] for i in range(len(visits) - 1)]
    empirical_return = sum(gaps) / len(gaps)
    theoretical_return = mc.expected_return_time(pi, state)
    check("expected return time (Kac's formula) vs simulation",
          math.isclose(empirical_return, theoretical_return, rel_tol=0.05))

    # 8. period detection vs known cycle structures
    ok = True
    for cycle_len in (2, 3, 4, 5):
        mat = [[0.0] * cycle_len for _ in range(cycle_len)]
        for i in range(cycle_len):
            mat[i][(i + 1) % cycle_len] = 1.0
        if mc.period_of_state(mat, 0) != cycle_len:
            ok = False
    check("period detection matches known cycle lengths", ok)

    # 9. independence test rejects a synthetic strongly-dependent chain
    #    and fails to reject a synthetic i.i.d. sequence, cross-checked
    #    against scipy's own chi-square contingency test.
    dependent = [0, 1] * 2000
    counts = [[0, 0], [0, 0]]
    for i in range(len(dependent) - 1):
        counts[dependent[i]][dependent[i + 1]] += 1
    scipy_stat, scipy_p, _, _ = sstats.chi2_contingency(np.array(counts), correction=False)
    ours = mc.markov_independence_test(dependent, 2)
    check("markov_independence_test statistic matches scipy.stats.chi2_contingency",
          math.isclose(ours["statistic"], scipy_stat, abs_tol=1e-6) and
          math.isclose(ours["p_value"], scipy_p, abs_tol=1e-9))

    passed = sum(1 for _, ok in checks if ok)
    print(f"\n{passed}/{len(checks)} checks passed")
    if passed != len(checks):
        sys.exit(1)


def build_parser():
    parser = argparse.ArgumentParser(description="Markov chains laboratory CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("stationary", help="compute the stationary distribution of a chain")
    p1.add_argument("--matrix", type=str, required=True,
                     help="rows separated by ';', entries by ',', e.g. '0.5,0.5;0.3,0.7'")
    p1.set_defaults(func=cmd_stationary)

    p2 = sub.add_parser("convergence", help="trace convergence to stationarity")
    p2.add_argument("--matrix", type=str, required=True)
    p2.add_argument("--initial", type=str, required=True, help="e.g. '1,0,0'")
    p2.add_argument("--n", type=int, default=50)
    p2.set_defaults(func=cmd_convergence)

    p3 = sub.add_parser("passage", help="mean first-passage times to a target state")
    p3.add_argument("--matrix", type=str, required=True)
    p3.add_argument("--target", type=int, required=True)
    p3.set_defaults(func=cmd_passage)

    p4 = sub.add_parser("simulate", help="simulate a chain trajectory")
    p4.add_argument("--matrix", type=str, required=True)
    p4.add_argument("--initial", type=int, required=True)
    p4.add_argument("--steps", type=int, default=1000)
    p4.add_argument("--seed", type=int, default=None)
    p4.set_defaults(func=cmd_simulate)

    p5 = sub.add_parser("fxchain", help="fit a discretised Markov chain to EUR/USD returns")
    p5.add_argument("--col", type=str, default="return")
    p5.add_argument("--bins", type=int, default=5)
    p5.add_argument("--seed", type=int, default=1)
    p5.set_defaults(func=cmd_fxchain)

    p6 = sub.add_parser("verify", help="cross-check implementation against analytical/oracle values")
    p6.set_defaults(func=cmd_verify)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
