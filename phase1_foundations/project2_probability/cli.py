#!/usr/bin/env python3
"""Command-line front end for the probability simulator.

    python3 cli.py flip N [--seed S]
    python3 cli.py roll N [--sides 6] [--face 3] [--seed S]
    python3 cli.py joint FILE.json [--seed S]
    python3 cli.py cond FILE.json X Y [--seed S]
    python3 cli.py bayes --prior P1 P2 ... --likelihood L1 L2 ... [--labels A B ...]
    python3 cli.py bayes-medical
    python3 cli.py sample binomial --n 10 --p 0.3 --size 20000 [--plot PATH] [--seed S]
    python3 cli.py sample geometric --p 0.25 --size 20000 [--plot PATH] [--seed S]
    python3 cli.py sample poisson --lambda 2.5 --size 20000 [--plot PATH] [--seed S]
    python3 cli.py lln coin [--n 3000] [--seed S]
    python3 cli.py lln dice [--n 3000] [--seed S]
    python3 cli.py verify

The conditional and joint-table commands use the key order
(X, Y): the first key is the row variable, the second the column.
"""

import argparse
import json
import os
import random
import sys

import probability_simulator as ps


def _make_rng(seed):
    return random.Random(seed)


def _maybe_num(text):
    """Turn a CLI argument into int or float when it looks like one."""
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def cmd_flip(args):
    rng = _make_rng(args.seed)
    flips = ps.flip_coin(args.n, rng)
    heads = sum(1 for f in flips if f == 'H')
    tails = args.n - heads
    print("flips:    {}".format(args.n))
    print("heads:    {}  ({:.4f})".format(heads, heads / args.n))
    print("tails:    {}  ({:.4f})".format(tails, tails / args.n))
    print("theoretical P(H) = 0.5")


def cmd_roll(args):
    rng = _make_rng(args.seed)
    rolls = ps.roll_die(args.sides, args.n, rng)
    empirical = ps.empirical_probability(rolls, args.face)
    theoretical = ps.theoretical_die_probability(args.sides)
    counts = {}
    for r in rolls:
        counts[r] = counts.get(r, 0) + 1
    print("rolls:     {}".format(args.n))
    print("P(face {}) empirical   = {:.4f}".format(args.face, empirical))
    print("P(face {}) theoretical = {:.4f}".format(args.face, theoretical))
    print("per-face counts: {}".format(
        " ".join("{}:{}".format(k, counts.get(k, 0))
                 for k in sorted(counts))))


def cmd_joint(args):
    table = load_joint_table(args.file)
    print("joint table (X, Y):")
    for (x, y), p in sorted(table.items()):
        print("  P(X={!r}, Y={!r}) = {}".format(x, y, p))
    print("P(X): {}".format(ps.marginals(table, 0)))
    print("P(Y): {}".format(ps.marginals(table, 1)))
    print("conditional table P(X | Y):")
    for given in sorted(set(y for (_, y) in table)):
        row = "  Y={!r}: ".format(given)
        row += "  ".join(
            "P(X={!r}|Y)={:.4f}".format(x, ps.conditional_probability(table, x, given))
            for x in sorted(set(x for (x, _) in table)))
        print(row)


def cmd_cond(args):
    table = load_joint_table(args.file)
    value = _maybe_num(args.x)
    given = _maybe_num(args.y)
    computed = ps.conditional_probability(table, value, given)
    rng = _make_rng(args.seed)
    samples = ps.sample_joint(table, 50000, rng)
    simulated = ps.empirical_conditional(samples, value, given)
    print("P(X={!r} | Y={!r}) computed  = {:.5f}".format(value, given, computed))
    print("P(X={!r} | Y={!r}) simulated = {:.5f}".format(value, given, simulated))
    print("difference: {:.5f}".format(abs(computed - simulated)))


def load_joint_table(path):
    with open(path, 'r', encoding='utf-8') as handle:
        data = json.load(handle)
    table = {}
    if isinstance(data, dict) and 'table' in data:
        data = data['table']
    for raw_key, p in data.items():
        key = raw_key
        if isinstance(raw_key, str) and raw_key.startswith('[') and raw_key.endswith(']'):
            key = tuple(_maybe_num(part.strip())
                        for part in raw_key[1:-1].split(','))
        elif isinstance(raw_key, list):
            key = tuple(_maybe_num(part) for part in raw_key)
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(
                "joint table keys must be (X, Y) pairs, got {!r}".format(raw_key))
        table[key] = float(p)
    return table


def cmd_bayes(args):
    if len(args.prior) != len(args.likelihood):
        sys.exit("--prior and --likelihood need the same number of values")
    if args.labels and len(args.labels) != len(args.prior):
        sys.exit("--labels must match the number of hypotheses")
    labels = args.labels if args.labels else [
        "H{}".format(i + 1) for i in range(len(args.prior))]
    prior = {labels[i]: args.prior[i] for i in range(len(labels))}
    likelihood = {labels[i]: args.likelihood[i] for i in range(len(labels))}
    posterior = ps.bayes(prior, likelihood)
    evidence = sum(likelihood[h] * prior[h] for h in prior)
    print("prior:      {}".format(prior))
    print("likelihood: {}".format(likelihood))
    print("P(evidence) = {:.6f}".format(evidence))
    for h in labels:
        print("P({} | evidence) = {:.6f}".format(h, posterior[h]))


def cmd_bayes_medical(args):
    result = ps.medical_test_example()
    print("Classic base-rate problem")
    print("prevalence:  1% (prior P(disease) = 0.01)")
    print("sensitivity: P(positive | disease)  = 0.99")
    print("false alarm: P(positive | healthy)  = 0.05")
    posterior = result['posterior']
    print("P(disease | positive) = {:.6f}  ({:.2f}%)".format(
        posterior['disease'], 100 * posterior['disease']))
    print("P(healthy | positive) = {:.6f}  ({:.2f}%)".format(
        posterior['healthy'], 100 * posterior['healthy']))


def cmd_sample(args):
    rng = _make_rng(args.seed)
    if args.dist == 'binomial':
        n = args.n
        p = args.p
        if n < 0 or not (0.0 <= p <= 1.0):
            sys.exit("binomial needs --n >= 0 and --p in [0, 1]")
        samples = ps.binomial_sample(n, p, args.size, rng)
        pmf = lambda k: ps.binomial_pmf(n, p, k)
        k_max = n
        label = "Binomial(n={}, p={}) PMF".format(n, p)
    elif args.dist == 'geometric':
        p = args.p
        if not (0.0 < p <= 1.0):
            sys.exit("geometric needs --p in (0, 1]")
        samples = ps.geometric_sample(p, args.size, rng)
        pmf = lambda k: ps.geometric_pmf(p, k)
        k_max = min(max(samples) + 2, 25)
        label = "Geometric(p={}) PMF".format(p)
    elif args.dist == 'poisson':
        lam = args.lam
        if lam <= 0:
            sys.exit("poisson needs --lambda > 0")
        samples = ps.poisson_sample(lam, args.size, rng)
        pmf = lambda k: ps.poisson_pmf(lam, k)
        k_max = min(max(samples) + 2, 25)
        label = "Poisson(lambda={}) PMF".format(lam)
    else:
        sys.exit("unknown distribution {!r}".format(args.dist))

    counts = ps.empirical_counts(samples)
    print("sample mean: {:.4f}".format(sum(samples) / len(samples)))
    print("k    empirical  theoretical  diff")
    for k in range(0, k_max + 1):
        emp = counts.get(k, 0) / len(samples)
        theo = pmf(k)
        print("{:<5} {:.6f}  {:.6f}  {:.6f}".format(k, emp, theo, abs(emp - theo)))

    if args.plot:
        path = ps.plot_distribution_comparison(
            samples, pmf, args.dist.title(), k_max, args.plot, label)
        print("plot saved to {}".format(path))


def cmd_lln(args):
    rng = _make_rng(args.seed)
    if args.what == 'coin':
        path = ps.plot_coin_convergence(args.n, rng=rng)
        print("plot saved to {}".format(path))
    elif args.what == 'dice':
        path = ps.plot_dice_convergence(sides=6, n=args.n, rng=rng)
        print("plot saved to {}".format(path))
    else:
        sys.exit("unknown target {!r}".format(args.what))


def cmd_verify(args):
    failures = []
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, ok, detail))
        if not ok:
            failures.append(name)

    rng = random.Random(1234)

    heads = ps.empirical_probability(ps.flip_coin(20000, rng), 'H')
    check("coin P(H) ~ 0.5", abs(heads - 0.5) < 0.01, "got {:.4f}".format(heads))

    rolls = ps.roll_die(6, 30000, rng)
    p3 = ps.empirical_probability(rolls, 3)
    check("die P(3) ~ 1/6", abs(p3 - 1.0 / 6.0) < 0.01, "got {:.4f}".format(p3))

    base = os.path.dirname(os.path.abspath(__file__))
    table = load_joint_table(os.path.join(base, 'samples', 'joint_rain_umbrella.json'))
    check("joint table sums to 1",
          abs(sum(table.values()) - 1.0) < 1e-9)
    m_y = ps.marginals(table, 1)
    check("marginal P(no umbrella) = 0.4",
          abs(m_y.get('no', 0.0) - 0.4) < 1e-9, "got {}".format(m_y))
    cond = ps.conditional_probability(table, 'rain', 'no')
    check("P(rain | no umbrella) = 0.1",
          abs(cond - 0.1) < 1e-9, "got {:.4f}".format(cond))
    sim = ps.empirical_conditional(ps.sample_joint(table, 50000, rng), 'rain', 'no')
    check("P(rain | no) simulated ~ 0.1", abs(sim - 0.1) < 0.02,
          "got {:.4f}".format(sim))

    med = ps.medical_test_example()['posterior']
    check("bayes medical = 1/6", abs(med['disease'] - 1.0 / 6.0) < 1e-12,
          "got {:.6f}".format(med['disease']))
    check("posterior sums to 1",
          abs(sum(med.values()) - 1.0) < 1e-12)

    post = ps.bayes({"A": 0.5, "B": 0.5}, {"A": 1.0, "B": 0.0})
    check("bayes certain evidence", post["A"] == 1.0 and post["B"] == 0.0,
          "got {}".format(post))

    binom = ps.binomial_sample(10, 0.3, 50000, rng)
    mean_b = sum(binom) / len(binom)
    check("binomial mean ~ 3.0", abs(mean_b - 3.0) < 0.05,
          "got {:.4f}".format(mean_b))
    check("binomial pmf sums to 1",
          abs(sum(ps.binomial_pmf(10, 0.3, k) for k in range(0, 11)) - 1.0) < 1e-12)

    geom = ps.geometric_sample(0.25, 50000, rng)
    mean_g = sum(geom) / len(geom)
    check("geometric mean ~ 4.0", abs(mean_g - 4.0) < 0.08,
          "got {:.4f}".format(mean_g))
    check("geometric pmf sums to 1",
          abs(sum(ps.geometric_pmf(0.25, k) for k in range(1, 500)) - 1.0) < 1e-6)

    pois = ps.poisson_sample(2.5, 50000, rng)
    mean_p = sum(pois) / len(pois)
    check("poisson mean ~ 2.5", abs(mean_p - 2.5) < 0.05,
          "got {:.4f}".format(mean_p))
    check("poisson pmf sums to 1",
          abs(sum(ps.poisson_pmf(2.5, k) for k in range(0, 60)) - 1.0) < 1e-9)

    rm = ps.running_mean([1, 2, 3, 4, 5])
    check("running mean ends at arithmetic mean", abs(rm[-1] - 3.0) < 1e-12)

    width = 6
    for name, ok, detail in checks:
        marker = "PASS" if ok else "FAIL"
        print("{:<{w}} {}".format(marker, name, w=width))
        if not ok and detail:
            print("      {}".format(detail))
    print("-" * 40)
    print("{}/{} checks passed".format(len(checks) - len(failures), len(checks)))
    sys.exit(1 if failures else 0)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="probability-simulator",
        description="Probability Simulator & Bayes Engine (Project 2)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_flip = sub.add_parser("flip", help="simulate coin flips")
    p_flip.add_argument("n", type=int)
    p_flip.add_argument("--seed", type=int, default=1)
    p_flip.set_defaults(func=cmd_flip)

    p_roll = sub.add_parser("roll", help="simulate die rolls")
    p_roll.add_argument("n", type=int)
    p_roll.add_argument("--sides", type=int, default=6)
    p_roll.add_argument("--face", type=int, default=1)
    p_roll.add_argument("--seed", type=int, default=1)
    p_roll.set_defaults(func=cmd_roll)

    p_joint = sub.add_parser("joint", help="inspect a joint probability table")
    p_joint.add_argument("file")
    p_joint.set_defaults(func=cmd_joint)

    p_cond = sub.add_parser("cond", help="conditional probability, computed and simulated")
    p_cond.add_argument("file")
    p_cond.add_argument("x")
    p_cond.add_argument("y")
    p_cond.add_argument("--seed", type=int, default=1)
    p_cond.set_defaults(func=cmd_cond)

    p_bayes = sub.add_parser("bayes", help="posterior from prior and likelihood")
    p_bayes.add_argument("--prior", nargs="+", type=float, required=True)
    p_bayes.add_argument("--likelihood", nargs="+", type=float, required=True)
    p_bayes.add_argument("--labels", nargs="+", default=None)
    p_bayes.set_defaults(func=cmd_bayes)

    p_med = sub.add_parser("bayes-medical", help="the classic base-rate problem")
    p_med.set_defaults(func=cmd_bayes_medical)

    p_samp = sub.add_parser("sample", help="draw from a distribution and compare to its PMF")
    p_samp.add_argument("dist", help="binomial|geometric|poisson")
    p_samp.add_argument("--n", type=int, default=10)
    p_samp.add_argument("--p", type=float, default=0.3)
    p_samp.add_argument("--lambda", dest="lam", type=float, default=2.5)
    p_samp.add_argument("--size", type=int, default=20000)
    p_samp.add_argument("--plot", default=None)
    p_samp.add_argument("--seed", type=int, default=1)
    p_samp.set_defaults(func=cmd_sample)

    p_lln = sub.add_parser("lln", help="plot law of large numbers convergence")
    p_lln.add_argument("what", help="coin|dice")
    p_lln.add_argument("--n", type=int, default=3000)
    p_lln.add_argument("--seed", type=int, default=1)
    p_lln.set_defaults(func=cmd_lln)

    p_verify = sub.add_parser("verify", help="run self-checks")
    p_verify.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        sys.exit("error: {}".format(exc))


if __name__ == '__main__':
    main()
