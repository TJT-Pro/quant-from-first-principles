import argparse
import math
import os
import sys

import low_discrepancy as ld
import discrepancy as disc
import integration as intg
import gaussian_transform as gt
import fx_application as fx

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DEXUSEU_returns.csv")


def cmd_sequence(args):
    points = ld.halton_sequence(args.n, args.dim)
    print(f"first {args.n} Halton points in {args.dim}D:")
    for p in points:
        print("  ", [round(v, 6) for v in p])


def cmd_discrepancy(args):
    print(f"{'N':>6}  {'random D':>12}  {'Halton D':>12}  {'ratio':>8}")
    for n in (50, 200, 500, 1000):
        random_pts = disc.random_uniform_points(n, args.dim, seed=1)
        halton_pts = ld.halton_sequence(n, args.dim)
        d_random = disc.l2_star_discrepancy_squared(random_pts) ** 0.5
        d_halton = disc.l2_star_discrepancy_squared(halton_pts) ** 0.5
        print(f"{n:6d}  {d_random:12.6f}  {d_halton:12.6f}  {d_random / d_halton:8.2f}")


def cmd_integrate(args):
    f = eval("lambda x: " + args.f)
    exact = eval(args.exact) if args.exact else None

    print(f"integrating f(x) = {args.f} over [0,1]^{args.dim}")
    if exact is not None:
        print(f"exact value: {exact}\n")

    n_values = [200, 800, 3200, 12800]
    mc_errors, qmc_errors = [], []
    for n in n_values:
        mc_val = intg.monte_carlo_integrate(f, args.dim, n, seed=1)
        qmc_val = intg.qmc_integrate(f, args.dim, n)
        if exact is not None:
            mc_errors.append(abs(mc_val - exact))
            qmc_errors.append(abs(qmc_val - exact))
            print(f"n={n:6d}  MC={mc_val:.6f} (err={mc_errors[-1]:.2e})  "
                  f"QMC={qmc_val:.6f} (err={qmc_errors[-1]:.2e})")
        else:
            print(f"n={n:6d}  MC={mc_val:.6f}  QMC={qmc_val:.6f}")

    if exact is not None:
        print(f"\nMC empirical order:  {intg.estimate_convergence_order(mc_errors, n_values):.3f}")
        print(f"QMC empirical order: {intg.estimate_convergence_order(qmc_errors, n_values):.3f}")


def cmd_price(args):
    exact = fx.black_scholes_closed_form(args.s0, args.k, args.r, args.sigma, args.t)
    print(f"S0={args.s0}, K={args.k}, r={args.r}, sigma={args.sigma}, T={args.t}")
    print(f"closed-form price: {exact:.6f}\n")

    result = fx.convergence_comparison(args.s0, args.k, args.r, args.sigma, args.t,
                                        [200, 800, 3200, 12800], n_mc_repeats=20, seed=1)
    print(f"{'N':>6}  {'MC error':>12}  {'QMC error':>12}")
    for n, mc_e, qmc_e in zip(result["n_values"], result["mc_errors"], result["qmc_errors"]):
        print(f"{n:6d}  {mc_e:12.6f}  {qmc_e:12.6f}")
    print(f"\nMC empirical order:  {result['mc_order']:.3f}")
    print(f"QMC empirical order: {result['qmc_order']:.3f}")


def cmd_fxprice(args):
    data = fx.load_prices_and_returns(DATA_PATH)
    sigma = fx.calibrate_volatility(data["returns"])
    s0 = data["prices"][-1]
    k = round(s0, 2) if args.k is None else args.k
    print(f"real EUR/USD spot: {s0}")
    print(f"calibrated annualized volatility: {sigma:.6f}\n")

    result = fx.convergence_comparison(s0, k, args.r, sigma, args.t, [200, 800, 3200, 12800],
                                        n_mc_repeats=20, seed=1)
    print(f"pricing a {args.t}-year call at strike {k}: closed-form = {result['exact_price']:.6f}\n")
    print(f"{'N':>6}  {'MC error':>12}  {'QMC error':>12}")
    for n, mc_e, qmc_e in zip(result["n_values"], result["mc_errors"], result["qmc_errors"]):
        print(f"{n:6d}  {mc_e:12.6f}  {qmc_e:12.6f}")
    print(f"\nMC empirical order:  {result['mc_order']:.3f}")
    print(f"QMC empirical order: {result['qmc_order']:.3f}  "
          f"(lower than a smooth integrand's ~0.9-1.0 due to the payoff's kink at S=K)")


def cmd_verify(args):
    import numpy as np
    from scipy.stats import norm as scipy_norm

    checks = []

    def check(name, condition):
        checks.append((name, bool(condition)))
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {name}")

    expected = [0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875, 0.0625]
    computed = [ld.van_der_corput(n, 2) for n in range(1, 9)]
    check("van der Corput base-2 sequence matches known values",
          all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(computed, expected)))

    points = disc.random_uniform_points(150, 2, seed=1)
    closed_form = disc.l2_star_discrepancy_squared(points)
    mc_estimate = disc.monte_carlo_l2_star_discrepancy_estimate(points, n_test_boxes=200000, seed=2)
    check("L2-star discrepancy closed form matches independent Monte Carlo estimate",
          math.isclose(closed_form, mc_estimate, rel_tol=0.15))

    ratios = []
    for n in (50, 200, 1000):
        random_pts = disc.random_uniform_points(n, 2, seed=1)
        halton_pts = ld.halton_sequence(n, 2)
        d_random = disc.l2_star_discrepancy_squared(random_pts) ** 0.5
        d_halton = disc.l2_star_discrepancy_squared(halton_pts) ** 0.5
        ratios.append(d_random / d_halton)
    check("Halton's discrepancy advantage over random points grows with N",
          ratios[0] < ratios[1] < ratios[2] and all(r > 1 for r in ratios))

    ok = True
    for u in (0.001, 0.05, 0.3, 0.5, 0.7, 0.95, 0.999):
        ours = gt.inverse_normal_cdf(u)
        theirs = scipy_norm.ppf(u)
        if not math.isclose(ours, theirs, abs_tol=1e-6):
            ok = False
    check("inverse normal CDF matches scipy.stats.norm.ppf", ok)

    f = lambda x: math.exp(sum(x))
    exact = (math.e - 1) ** 2
    n_values = [200, 800, 3200, 12800]
    mc_errors = [intg.monte_carlo_repeated_error(f, 2, n, exact, n_repeats=20, seed=1) for n in n_values]
    qmc_errors = [abs(intg.qmc_integrate(f, 2, n) - exact) for n in n_values]
    mc_order = intg.estimate_convergence_order(mc_errors, n_values)
    qmc_order = intg.estimate_convergence_order(qmc_errors, n_values)
    check(f"Monte Carlo convergence order near 0.5 (measured {mc_order:.3f})", abs(mc_order - 0.5) < 0.2)
    check(f"QMC convergence order substantially exceeds Monte Carlo (measured {qmc_order:.3f} vs {mc_order:.3f})",
          qmc_order > mc_order + 0.2)

    halton_pts = ld.halton_sequence(5000, 2)
    ours_integral = sum(f(p) for p in halton_pts) / 5000
    check("QMC integral of a smooth function matches the known exact value",
          math.isclose(ours_integral, exact, abs_tol=0.01))

    data = fx.load_prices_and_returns(DATA_PATH)
    sigma_real = fx.calibrate_volatility(data["returns"])
    s0_real = data["prices"][-1]
    k_real = round(s0_real, 2)
    result = fx.convergence_comparison(s0_real, k_real, 0.03, sigma_real, 0.25,
                                        [200, 800, 3200, 12800], n_mc_repeats=20, seed=1)
    check("real EUR/USD QMC option pricing beats MC in convergence order",
          result["qmc_order"] > result["mc_order"])
    check("real EUR/USD QMC achieves smaller absolute error at every sample size tested",
          all(q < m for q, m in zip(result["qmc_errors"], result["mc_errors"])))
    check("real EUR/USD calibrated volatility is plausible (3%-25% annualized)",
          0.03 < sigma_real < 0.25)

    passed = sum(1 for _, ok in checks if ok)
    print(f"\n{passed}/{len(checks)} checks passed")
    if passed != len(checks):
        sys.exit(1)


def build_parser():
    parser = argparse.ArgumentParser(description="Quasi-Monte Carlo engine CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("sequence", help="generate a Halton sequence")
    p1.add_argument("--n", type=int, default=10)
    p1.add_argument("--dim", type=int, default=2)
    p1.set_defaults(func=cmd_sequence)

    p2 = sub.add_parser("discrepancy", help="compare random vs Halton point-set discrepancy")
    p2.add_argument("--dim", type=int, default=2)
    p2.set_defaults(func=cmd_discrepancy)

    p3 = sub.add_parser("integrate", help="compare MC vs QMC integration of a function")
    p3.add_argument("--f", type=str, required=True, help="Python expr in x (a list), e.g. 'x[0]**2'")
    p3.add_argument("--dim", type=int, default=1)
    p3.add_argument("--exact", type=str, default=None, help="Python expr for the exact integral value")
    p3.set_defaults(func=cmd_integrate)

    p4 = sub.add_parser("price", help="compare MC vs QMC European option pricing")
    p4.add_argument("--s0", type=float, default=100.0)
    p4.add_argument("--k", type=float, default=100.0)
    p4.add_argument("--r", type=float, default=0.05)
    p4.add_argument("--sigma", type=float, default=0.2)
    p4.add_argument("--t", type=float, default=1.0)
    p4.set_defaults(func=cmd_price)

    p5 = sub.add_parser("fxprice", help="price a real EUR/USD option via MC vs QMC")
    p5.add_argument("--k", type=float, default=None)
    p5.add_argument("--r", type=float, default=0.03)
    p5.add_argument("--t", type=float, default=0.25)
    p5.set_defaults(func=cmd_fxprice)

    p6 = sub.add_parser("verify", help="cross-check implementation against scipy/numpy oracles and known theory")
    p6.set_defaults(func=cmd_verify)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
