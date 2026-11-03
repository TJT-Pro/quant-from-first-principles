import argparse
import math
import os
import sys

import heat_equation as he
import stability_analysis as sa
import wave_equation as we
import black_scholes_pde as bs
import fx_application as fx

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DEXUSEU_returns.csv")


def cmd_heat(args):
    alpha, length, t_end = args.alpha, args.length, args.t_end
    mode = args.mode
    ic = lambda x: math.sin(mode * math.pi * x / length)
    left_bc = lambda t: 0.0
    right_bc = lambda t: 0.0
    exact = he.exact_sinusoidal_solution(mode, alpha, length)

    for name, solver in [("explicit", he.solve_heat_explicit), ("implicit", he.solve_heat_implicit),
                          ("crank-nicolson", he.solve_heat_crank_nicolson)]:
        result = solver(alpha, length, t_end, args.n_space, args.n_time, ic, left_bc, right_bc)
        exact_u = [exact(x, t_end) for x in result["x"]]
        max_err = max(abs(a - b) for a, b in zip(result["u"], exact_u))
        print(f"{name:16s} r={result['r']:.4f}  max error vs exact = {max_err:.6e}")


def cmd_wave(args):
    c, length, t_end, mode = args.c, args.length, args.t_end, args.mode
    ic_disp = lambda x: math.sin(mode * math.pi * x / length)
    ic_vel = lambda x: 0.0
    exact = we.exact_standing_wave_solution(mode, c, length)

    result = we.solve_wave_explicit(c, length, t_end, args.n_space, args.n_time, ic_disp, ic_vel,
                                     lambda t: 0.0, lambda t: 0.0)
    exact_u = [exact(x, t_end) for x in result["x"]]
    max_err = max(abs(a - b) for a, b in zip(result["u"], exact_u))
    print(f"Courant number: {result['courant']:.4f}")
    print(f"max error vs exact standing-wave solution: {max_err:.6e}")
    if result["courant"] > 1.0:
        print("(Courant number exceeds 1: CFL condition violated, expect instability)")


def cmd_stability(args):
    print(f"{'r':>8}  {'explicit max|xi|':>18}  {'implicit max|xi|':>18}  {'CN max|xi|':>12}")
    for r in [0.1, 0.3, 0.5, 0.6, 1.0, 5.0]:
        max_exp = sa.max_growth_factor_over_modes(sa.explicit_growth_factor, r)
        max_imp = sa.max_growth_factor_over_modes(sa.implicit_growth_factor, r)
        max_cn = sa.max_growth_factor_over_modes(sa.crank_nicolson_growth_factor, r)
        print(f"{r:8.2f}  {max_exp:18.4f}  {max_imp:18.4f}  {max_cn:12.4f}")


def cmd_price(args):
    result = fx.price_option_all_schemes(args.s0, args.k, args.r, args.sigma, args.t,
                                          option_type=args.option_type)
    print(f"S0={args.s0}, K={args.k}, r={args.r}, sigma={args.sigma}, T={args.t}, type={args.option_type}\n")
    print(f"closed-form:     {result['closed_form']:.6f}")
    print(f"explicit PDE:    {result['explicit']:.6f}  (error {abs(result['explicit']-result['closed_form']):.2e})")
    print(f"implicit PDE:    {result['implicit']:.6f}  (error {abs(result['implicit']-result['closed_form']):.2e})")
    print(f"Crank-Nicolson:  {result['crank_nicolson']:.6f}  "
          f"(error {abs(result['crank_nicolson']-result['closed_form']):.2e})")


def cmd_fxprice(args):
    data = fx.load_prices_and_returns(DATA_PATH)
    sigma = fx.calibrate_volatility(data["returns"])
    s0 = data["prices"][-1]
    k = round(s0, 2) if args.k is None else args.k
    print(f"real EUR/USD spot: {s0}")
    print(f"calibrated annualized volatility: {sigma:.6f}\n")

    result = fx.price_option_all_schemes(s0, k, args.r, sigma, args.t, option_type=args.option_type)
    print(f"pricing a {args.t}-year {args.option_type} at strike {k}:")
    print(f"  closed-form:    {result['closed_form']:.6f}")
    print(f"  explicit PDE:   {result['explicit']:.6f}")
    print(f"  implicit PDE:   {result['implicit']:.6f}")
    print(f"  Crank-Nicolson: {result['crank_nicolson']:.6f}\n")

    print("convergence as the spatial grid is refined (Crank-Nicolson):")
    conv = fx.pde_convergence_study(s0, k, args.r, sigma, args.t, [25, 50, 100, 200], n_time=1000,
                                     option_type=args.option_type)
    for c in conv:
        print(f"  n_space={c['n_space']:4d}  price={c['crank_nicolson_price']:.6f}  error={c['error']:.2e}")


def cmd_verify(args):
    import numpy as np
    from scipy.stats import norm as scipy_norm

    checks = []

    def check(name, condition):
        checks.append((name, bool(condition)))
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {name}")

    import random
    import linear_algebra as la
    rng = random.Random(1)
    n = 20
    lower = [0.0] + [rng.uniform(-2, 2) for _ in range(n - 1)]
    diag = [rng.uniform(8, 12) for _ in range(n)]
    upper = [rng.uniform(-2, 2) for _ in range(n - 1)] + [0.0]
    rhs = [rng.uniform(-5, 5) for _ in range(n)]
    x = la.thomas_algorithm(lower, diag, upper, rhs)
    a_np = np.zeros((n, n))
    for i in range(n):
        a_np[i, i] = diag[i]
        if i > 0:
            a_np[i, i - 1] = lower[i]
        if i < n - 1:
            a_np[i, i + 1] = upper[i]
    x_np = np.linalg.solve(a_np, np.array(rhs))
    check("Thomas algorithm matches numpy.linalg.solve",
          all(math.isclose(a, b, abs_tol=1e-8) for a, b in zip(x, x_np)))

    alpha, length, t_end, mode = 0.5, 1.0, 0.1, 1
    ic = lambda xx: math.sin(mode * math.pi * xx / length)
    exact = he.exact_sinusoidal_solution(mode, alpha, length)
    for name, solver, n_time in [("explicit", he.solve_heat_explicit, 300),
                                  ("implicit", he.solve_heat_implicit, 100),
                                  ("crank-nicolson", he.solve_heat_crank_nicolson, 100)]:
        result = solver(alpha, length, t_end, 50, n_time, ic, lambda t: 0.0, lambda t: 0.0)
        exact_u = [exact(xx, t_end) for xx in result["x"]]
        max_err = max(abs(a - b) for a, b in zip(result["u"], exact_u))
        check(f"heat equation {name} scheme matches exact sinusoidal solution", max_err < 0.01)

    unstable_result = he.solve_heat_explicit(alpha, length, t_end, 50, 20, ic, lambda t: 0.0, lambda t: 0.0)
    check("explicit heat scheme is unstable when r>0.5 (von Neumann prediction)",
          unstable_result["r"] > 0.5 and max(abs(v) for v in unstable_result["u"]) > 1e6)

    c, mode_w = 1.0, 1
    ic_disp = lambda xx: math.sin(mode_w * math.pi * xx / length)
    ic_vel = lambda xx: 0.0
    exact_w = we.exact_standing_wave_solution(mode_w, c, length)
    stable_wave = we.solve_wave_explicit(c, length, 0.5, 100, 200, ic_disp, ic_vel, lambda t: 0.0, lambda t: 0.0)
    exact_wu = [exact_w(xx, 0.5) for xx in stable_wave["x"]]
    check("wave equation explicit scheme matches exact standing wave solution",
          max(abs(a - b) for a, b in zip(stable_wave["u"], exact_wu)) < 0.001)

    unstable_wave = we.solve_wave_explicit(c, length, 0.5, 100, 40, ic_disp, ic_vel, lambda t: 0.0, lambda t: 0.0)
    check("wave equation blows up when Courant number exceeds 1",
          unstable_wave["courant"] > 1.0 and max(abs(v) for v in unstable_wave["u"]) > 1000.0)

    s0, k, r, sigma, t = 100.0, 100.0, 0.05, 0.2, 1.0
    d1 = (math.log(s0 / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    scipy_price = s0 * scipy_norm.cdf(d1) - k * math.exp(-r * t) * scipy_norm.cdf(d2)
    our_price = bs.black_scholes_closed_form(s0, k, r, sigma, t, "call")
    check("closed-form Black-Scholes matches scipy.stats.norm-based computation",
          math.isclose(our_price, scipy_price, abs_tol=1e-10))

    result_all = fx.price_option_all_schemes(s0, k, r, sigma, t)
    for scheme in ("explicit", "implicit", "crank_nicolson"):
        check(f"Black-Scholes PDE ({scheme}) matches closed form",
              abs(result_all[scheme] - result_all["closed_form"]) < 0.01)

    x_min = math.log(k) - 3 * sigma * math.sqrt(t) - 1
    x_max = math.log(k) + 3 * sigma * math.sqrt(t) + 1
    n_space = 400
    imp_result = bs.solve_bs_pde_implicit(k, r, sigma, t, x_min, x_max, n_space, 50, "call")
    cn_result = bs.solve_bs_pde_crank_nicolson(k, r, sigma, t, x_min, x_max, n_space, 50, "call")
    closed = bs.black_scholes_closed_form(s0, k, r, sigma, t, "call")
    imp_err = abs(bs.price_at_spot(imp_result, s0) - closed)
    cn_err = abs(bs.price_at_spot(cn_result, s0) - closed)
    check("Crank-Nicolson achieves lower error than implicit at matched (space, time) grid", cn_err < imp_err)

    data = fx.load_prices_and_returns(DATA_PATH)
    sigma_real = fx.calibrate_volatility(data["returns"])
    s0_real = data["prices"][-1]
    k_real = round(s0_real, 2)
    real_result = fx.price_option_all_schemes(s0_real, k_real, 0.03, sigma_real, 0.25)
    check("real EUR/USD PDE prices agree with closed form across all schemes",
          all(abs(real_result[s] - real_result["closed_form"]) < 0.001
              for s in ("explicit", "implicit", "crank_nicolson")))
    check("real EUR/USD calibrated volatility is plausible (3%-25% annualized)",
          0.03 < sigma_real < 0.25)

    passed = sum(1 for _, ok in checks if ok)
    print(f"\n{passed}/{len(checks)} checks passed")
    if passed != len(checks):
        sys.exit(1)


def build_parser():
    parser = argparse.ArgumentParser(description="Numerical PDE engine CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("heat", help="solve the heat equation via all three schemes, compare to exact solution")
    p1.add_argument("--alpha", type=float, default=0.5)
    p1.add_argument("--length", type=float, default=1.0)
    p1.add_argument("--t-end", type=float, default=0.1)
    p1.add_argument("--mode", type=int, default=1)
    p1.add_argument("--n-space", type=int, default=50)
    p1.add_argument("--n-time", type=int, default=200)
    p1.set_defaults(func=cmd_heat)

    p2 = sub.add_parser("wave", help="solve the wave equation, compare to exact standing-wave solution")
    p2.add_argument("--c", type=float, default=1.0)
    p2.add_argument("--length", type=float, default=1.0)
    p2.add_argument("--t-end", type=float, default=0.5)
    p2.add_argument("--mode", type=int, default=1)
    p2.add_argument("--n-space", type=int, default=100)
    p2.add_argument("--n-time", type=int, default=200)
    p2.set_defaults(func=cmd_wave)

    p3 = sub.add_parser("stability", help="von Neumann stability analysis across a range of r")
    p3.set_defaults(func=cmd_stability)

    p4 = sub.add_parser("price", help="price a European option via closed form and all three PDE schemes")
    p4.add_argument("--s0", type=float, default=100.0)
    p4.add_argument("--k", type=float, default=100.0)
    p4.add_argument("--r", type=float, default=0.05)
    p4.add_argument("--sigma", type=float, default=0.2)
    p4.add_argument("--t", type=float, default=1.0)
    p4.add_argument("--option-type", choices=["call", "put"], default="call")
    p4.set_defaults(func=cmd_price)

    p5 = sub.add_parser("fxprice", help="price a real EUR/USD option via the PDE engine")
    p5.add_argument("--k", type=float, default=None)
    p5.add_argument("--r", type=float, default=0.03)
    p5.add_argument("--t", type=float, default=0.25)
    p5.add_argument("--option-type", choices=["call", "put"], default="call")
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
