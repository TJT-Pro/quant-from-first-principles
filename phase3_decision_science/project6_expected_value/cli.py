import argparse
import csv
import sys

import decision_theory as dt

RULES = ("expected-value", "maximax", "maximin", "minimax-regret",
         "laplace", "hurwicz")
UTILITIES = ("linear", "log", "power", "exponential", "quadratic")


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


def load_column(path, name):
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


def cmd_ev(args):
    outcomes = parse_list(args.outcomes, "outcomes")
    probs = parse_list(args.probs, "probs")
    g = {"identity": lambda x: x, "square": lambda x: x * x,
         "log": math_log}[args.g]
    ev = dt.expected_value(outcomes, probs)
    eg = dt.expected_function(outcomes, probs, g)
    print("outcomes: {}   probabilities: {}".format(outcomes, probs))
    print("expected value:    {:+.6g}".format(ev))
    print("variance:          {:.6g}".format(dt.variance(outcomes, probs)))
    print("standard deviation:{:.6g}".format(dt.stddev(outcomes, probs)))
    print("E[g(X)] with g={}: {:+.6g}".format(args.g, eg))


def math_log(x):
    import math
    if x <= 0.0:
        return float("nan")
    return math.log(x)


def cmd_decide(args):
    payoffs = parse_matrix(args.payoffs)
    probs = parse_list(args.probs, "probs") if args.probs else None
    scores = dt.decision_scores(payoffs, args.rule, probs, args.alpha)
    print("payoff matrix (actions x states):")
    for i, row in enumerate(payoffs):
        print("  action {}: {}".format(i, row))
    if args.rule == "minimax-regret":
        print("maximum regret per action:")
    else:
        print("score per action:")
    for i, score in enumerate(scores):
        print("  action {}: {:+.6g}".format(i, score))
    best, score = dt.best_action(payoffs, args.rule, probs, args.alpha)
    print("recommended action: {} (score {:+.6g})".format(best, score))


def cmd_dominates(args):
    payoffs = parse_matrix(args.payoffs)
    result = dt.dominates(args.a, args.b, payoffs)
    if result:
        print("action {} dominates action {}".format(args.a, args.b))
    else:
        print("action {} does not dominate action {}".format(args.a, args.b))


def cmd_ce(args):
    outcomes = parse_list(args.outcomes, "outcomes")
    probs = parse_list(args.probs, "probs")
    utility = dt.make_utility(args.utility, args.param)
    ev = dt.expected_value(outcomes, probs)
    ce = dt.certainty_equivalent(outcomes, probs, utility)
    print("lottery outcomes: {}   probabilities: {}".format(outcomes, probs))
    print("utility: {}".format(utility.name))
    print("expected value:          {:+.6g}".format(ev))
    print("expected utility:        {:+.6g}".format(
        dt.expected_utility(outcomes, probs, utility)))
    print("certainty equivalent:    {:+.6g}".format(ce))
    print("risk premium:            {:+.6g}".format(
        dt.risk_premium(outcomes, probs, utility)))


def cmd_risk(args):
    utility = dt.make_utility(args.utility, args.param)
    print("utility {} at x = {}".format(utility.name, args.x))
    print("absolute risk aversion A(x): {:.6g}".format(
        utility.absolute_risk(args.x)))
    print("relative risk aversion RRA(x): {:.6g}".format(
        utility.relative_risk(args.x)))


def cmd_stpetersburg(args):
    outcomes, probs = dt.st_petersburg(args.tosses)
    utility = dt.make_utility(args.utility, args.param)
    ev = dt.expected_value(outcomes, probs)
    ce = dt.certainty_equivalent(outcomes, probs, utility)
    print("St. Petersburg game, {} tosses".format(args.tosses))
    print("expected value:            {:+.6g}".format(ev))
    print("certainty equivalent under {}: {:+.6g}".format(
        utility.name, ce))
    print("a fair entry price, by log utility, stays near 4 dollars")


def cmd_kelly_binary(args):
    fraction = dt.kelly_fraction_binary(args.p, args.odds)
    print("binary bet, win probability {:.4f}, pays {:.2f}:1".format(args.p, args.odds))
    print("Kelly fraction: {:.4f}".format(fraction))
    print("half-Kelly:     {:.4f}".format(0.5 * fraction))
    if fraction <= 0.0:
        print("no bet: the odds do not justify the win probability")


def cmd_kelly(args):
    returns = load_column(args.file, args.col)
    best = dt.optimal_kelly_fraction(returns)
    growth_best = dt.log_growth_rate(returns, best)
    half = 0.5 * best
    growth_half = dt.log_growth_rate(returns, half)
    print("optimal Kelly fraction of {} daily returns: {:.4f}".format(
        len(returns), best))
    print("expected log growth at full Kelly:  {:+.6g} per round".format(growth_best))
    print("expected log growth at half Kelly:  {:+.6g} per round".format(growth_half))
    if args.fraction is not None:
        growth = dt.log_growth_rate(returns, args.fraction)
        wealth = dt.expected_log_wealth(returns, args.fraction, args.rounds)
        print("at your fraction {:.4f}: growth {:+.6g} per round".format(
            args.fraction, growth))
        print("expected log wealth after {} rounds: {:+.6g}".format(
            args.rounds, wealth))


def cmd_invest(args):
    returns = load_column(args.file, args.col)
    utility = dt.make_utility(args.utility, args.param)
    probs = [1.0 / len(returns)] * len(returns)
    ev = dt.expected_value(returns, probs)
    ce = dt.certainty_equivalent(returns, probs, utility)
    premium = dt.risk_premium(returns, probs, utility)
    print("one day in the market, {} empirical returns, utility {}".format(
        len(returns), utility.name))
    print("expected return:          {:+.6g}".format(ev))
    print("certainty equivalent:     {:+.6g}".format(ce))
    print("risk premium:             {:+.6g}".format(premium))
    if ce >= 0.0:
        print("decision: the lottery is worth taking")
    else:
        print("decision: the lottery is not worth taking")


def cmd_verify(args):
    import math
    import numpy as np

    checks = []

    outcomes = [1.0, 2.0, 5.0]
    probs = [0.2, 0.3, 0.5]
    our_ev = dt.expected_value(outcomes, probs)
    their_ev = float(np.average(outcomes, weights=probs))
    checks.append((abs(our_ev - their_ev) < 1e-12, "expected value matches numpy"))

    mu = dt.expected_value(outcomes, probs)
    their_var = float(sum(p * (x - mu) ** 2 for p, x in zip(probs, outcomes)))
    checks.append((abs(dt.variance(outcomes, probs) - their_var) < 1e-12,
                   "variance from definition"))

    eu1 = dt.expected_utility(outcomes, probs, dt.make_utility("exponential", 1.0))
    their_eu = sum(p * (1 - math.exp(-x)) for p, x in zip(probs, outcomes))
    checks.append((abs(eu1 - their_eu) < 1e-12, "expected utility from definition"))

    alpha = 5.0
    utility = dt.make_utility("exponential", alpha)
    eu = dt.expected_utility(outcomes, probs, utility)
    ce = dt.certainty_equivalent(outcomes, probs, utility)
    checks.append((abs(utility.u(ce) - eu) < 1e-12, "CE inverts the utility"))

    payoffs = [[2.0, -1.0, 4.0], [3.0, 2.0, 1.0], [1.0, 0.0, 5.0]]
    pmat = [0.3, 0.4, 0.3]
    ev_scores = dt.decision_scores(payoffs, "expected-value", pmat)
    manual = [sum(p * x for p, x in zip(pmat, row)) for row in payoffs]
    checks.append((all(abs(a - b) < 1e-12 for a, b in zip(ev_scores, manual)),
                   "expected-value rule matches manual sums"))
    checks.append((dt.decision_scores(payoffs, "maximax") == [4.0, 3.0, 5.0],
                   "maximax picks best cases"))
    checks.append((dt.decision_scores(payoffs, "maximin") == [-1.0, 1.0, 0.0],
                   "maximin picks worst cases"))
    checks.append((dt.decision_scores(payoffs, "laplace") == [5.0 / 3, 2.0, 2.0],
                   "laplace averages states"))
    hur = dt.decision_scores(payoffs, "hurwicz", alpha=0.5)
    checks.append((hur == [1.5, 2.0, 2.5], "hurwicz blends best and worst"))
    regrets = dt.regret_matrix(payoffs)
    mr = dt.decision_scores(payoffs, "minimax-regret")
    manual_regret = [max(max(x for x in col) - row[j]
                         for j, col in enumerate(zip(*payoffs)))
                     for row in payoffs]
    checks.append((all(abs(a - b) < 1e-12 for a, b in zip(mr, manual_regret)),
                   "minimax regret matches manual"))

    dom = dt.dominates(0, 1, [[1.0, 2.0, 3.0], [1.0, 2.0, 2.0]])
    checks.append((dom is True, "strict dominance detected"))
    dom2 = dt.dominates(0, 1, [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    checks.append((dom2 is False, "identical actions are not dominant"))

    logu = dt.make_utility("log")
    checks.append((abs(logu.relative_risk(100.0) - 1.0) < 1e-12,
                   "log utility has RRA exactly 1"))
    powu = dt.make_utility("power", 0.5)
    checks.append((abs(powu.relative_risk(50.0) - 0.5) < 1e-12,
                   "power utility has RRA = 1 - gamma"))
    expu = dt.make_utility("exponential", 3.0)
    checks.append((abs(expu.absolute_risk(10.0) - 3.0) < 1e-12,
                   "exponential utility has constant A"))

    risky = ([10.0, 30.0], [0.5, 0.5])
    premium_lin = dt.risk_premium(risky[0], risky[1], dt.make_utility("linear"))
    premium_log = dt.risk_premium(risky[0], risky[1], dt.make_utility("log"))
    checks.append((abs(premium_lin) < 1e-12, "linear utility has zero premium"))
    checks.append((premium_log > 0.0, "log utility has positive premium"))

    pv = dt.prospect_value([100.0, -100.0], [0.5, 0.5])
    checks.append((pv < 0.0, "loss aversion makes symmetric lottery negative"))
    checks.append((dt.probability_weight(0.1) > 0.1,
                   "low probabilities are overweighted"))
    checks.append((dt.probability_weight(0.9) < 0.9,
                   "high probabilities are underweighted"))

    n = 30
    spo, spp = dt.st_petersburg(n)
    checks.append((abs(dt.expected_value(spo, spp) - n) < 1e-6,
                   "St. Petersburg EV is n (renormalized)"))
    ce_sp = dt.certainty_equivalent(spo, spp, dt.make_utility("log"))
    checks.append((abs(ce_sp - 4.0) < 0.05, "log CE stays near 4"))

    returns = load_column("data/DEXUSEU_returns.csv", "return")
    checks.append((len(returns) == 6930, "dataset has 6930 returns"))
    probs_r = [1.0 / len(returns)] * len(returns)
    util = dt.make_utility("exponential", 10.0)
    our_eu = dt.expected_utility(returns, probs_r, util)
    their_eu_r = float(np.mean(1.0 - np.exp(-10.0 * np.array(returns))))
    checks.append((abs(our_eu - their_eu_r) < 1e-12,
                   "expected utility of returns matches numpy"))
    mu_r = float(np.mean(returns))
    sig_r = float(np.std(returns))
    closed = 1.0 - math.exp(-10.0 * mu_r + 0.5 * 100.0 * sig_r * sig_r)
    checks.append((abs(our_eu - closed) < 0.005,
                   "exponential EU near the normal closed form"))

    kelly_lin = dt.kelly_fraction_binary(0.6, 1.0)
    checks.append((abs(kelly_lin - 0.2) < 1e-12, "binary Kelly p=0.6 odds=1 is 0.2"))
    kelly_odds = dt.kelly_fraction_binary(0.4, 2.0)
    checks.append((abs(kelly_odds - 0.1) < 1e-12, "binary Kelly p=0.4 odds=2 is 0.1"))
    checks.append((dt.kelly_fraction_binary(0.4, 1.0) == 0.0,
                   "unfavorable bet gets no Kelly stake"))

    best_f = dt.optimal_kelly_fraction(returns)
    checks.append((0.0 <= best_f <= 1.0, "optimal Kelly lies in [0, 1]"))
    g_half = dt.log_growth_rate(returns, 0.5 * best_f)
    g_full = dt.log_growth_rate(returns, best_f)
    g_quarter = dt.log_growth_rate(returns, 0.25 * best_f)
    checks.append((g_full >= g_half >= g_quarter,
                   "Kelly maximizes log growth (monotone toward it)"))
    g_at_best = dt.log_growth_rate(returns, best_f)
    g_neighbor = max(dt.log_growth_rate(returns, best_f + 0.02),
                     dt.log_growth_rate(returns, max(0.0, best_f - 0.02)))
    checks.append((g_at_best >= g_neighbor - 1e-9,
                   "grid search actually found a maximum"))
    if best_f > 0.0:
        checks.append((g_full > 0.0, "full Kelly grows wealth on the returns"))
    checks.append((dt.expected_log_wealth(returns, best_f, 100)
                   == 100.0 * g_full, "expected log wealth is rounds times growth"))
    traj = dt.wealth_trajectory(returns, best_f, 1.0)
    manual = [1.0]
    for r in returns:
        manual.append(manual[-1] * (1.0 + best_f * r))
    checks.append((abs(traj[-1] - manual[-1]) < 1e-9,
                   "wealth trajectory matches the running product"))

    passed = sum(1 for ok, _ in checks if ok)
    for ok, label in checks:
        print("{:>5}  {}".format("PASS" if ok else "FAIL", label))
    print("-" * 44)
    print("{}/{} checks passed".format(passed, len(checks)))
    if passed != len(checks):
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("ev", help="expected value of a lottery")
    p.add_argument("--outcomes", required=True)
    p.add_argument("--probs", required=True)
    p.add_argument("--g", default="identity", choices=["identity", "square", "log"])
    p.set_defaults(func=cmd_ev)

    p = sub.add_parser("decide", help="decision rules on a payoff matrix")
    p.add_argument("--payoffs", required=True)
    p.add_argument("--probs", default=None)
    p.add_argument("--rule", default="expected-value", choices=RULES)
    p.add_argument("--alpha", type=float, default=0.5)
    p.set_defaults(func=cmd_decide)

    p = sub.add_parser("dominates", help="check strict dominance")
    p.add_argument("--payoffs", required=True)
    p.add_argument("--a", type=int, required=True)
    p.add_argument("--b", type=int, required=True)
    p.set_defaults(func=cmd_dominates)

    p = sub.add_parser("ce", help="certainty equivalent and risk premium")
    p.add_argument("--outcomes", required=True)
    p.add_argument("--probs", required=True)
    p.add_argument("--utility", default="log", choices=UTILITIES)
    p.add_argument("--param", type=float, default=None)
    p.set_defaults(func=cmd_ce)

    p = sub.add_parser("risk", help="Arrow-Pratt risk aversion at a point")
    p.add_argument("--utility", default="log", choices=UTILITIES)
    p.add_argument("--param", type=float, default=None)
    p.add_argument("--x", type=float, required=True)
    p.set_defaults(func=cmd_risk)

    p = sub.add_parser("stpetersburg", help="St. Petersburg paradox demo")
    p.add_argument("--tosses", type=int, default=20)
    p.add_argument("--utility", default="log", choices=UTILITIES)
    p.add_argument("--param", type=float, default=None)
    p.set_defaults(func=cmd_stpetersburg)

    p = sub.add_parser("kelly-binary", help="Kelly fraction for a binary bet")
    p.add_argument("--p", type=float, required=True)
    p.add_argument("--odds", type=float, required=True)
    p.set_defaults(func=cmd_kelly_binary)

    p = sub.add_parser("kelly", help="optimal Kelly fraction from empirical returns")
    p.add_argument("file")
    p.add_argument("--col", required=True)
    p.add_argument("--fraction", type=float, default=None)
    p.add_argument("--rounds", type=int, default=250)
    p.set_defaults(func=cmd_kelly)

    p = sub.add_parser("invest", help="is one day in the market worth it")
    p.add_argument("file")
    p.add_argument("--col", required=True)
    p.add_argument("--utility", default="exponential", choices=UTILITIES)
    p.add_argument("--param", type=float, default=10.0)
    p.set_defaults(func=cmd_invest)

    p = sub.add_parser("verify", help="cross-check everything against numpy")
    p.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
