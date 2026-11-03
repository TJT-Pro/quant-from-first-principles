import argparse
import math
import os
import sys

import ode_solvers as ode
import error_analysis as ea
import fx_application as fx

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DEXUSEU_returns.csv")


def cmd_solve(args):
    f = eval("lambda t, y: " + args.rhs)
    y0 = float(args.y0) if "," not in args.y0 else [float(v) for v in args.y0.split(",")]
    result = ode.solve_ivp(f, args.t0, y0, args.t_end, args.h, args.method)
    print(f"solved dy/dt = {args.rhs}, y({args.t0}) = {y0}, method={args.method}, h={args.h}")
    print(f"y({args.t_end}) = {result['y'][-1]}")


def cmd_adaptive(args):
    f = eval("lambda t, y: " + args.rhs)
    y0 = float(args.y0) if "," not in args.y0 else [float(v) for v in args.y0.split(",")]
    result = ode.solve_ivp_adaptive(f, args.t0, y0, args.t_end, args.h_init, args.tol, method=args.method)
    print(f"adaptive {args.method}, tol={args.tol}")
    print(f"y({args.t_end}) = {result['y'][-1]}")
    print(f"steps taken: {len(result['t']) - 1}")


def cmd_convergence(args):
    f = eval("lambda t, y: " + args.rhs)
    true_solution = eval("lambda t: " + args.exact)
    h_values = [0.2, 0.1, 0.05, 0.025, 0.0125]
    print(f"convergence order for dy/dt = {args.rhs}, exact solution: {args.exact}\n")
    for method in ("euler", "improved_euler", "midpoint", "rk4"):
        result = ea.estimate_convergence_order(f, args.t0, args.y0, args.t_end, h_values, method, true_solution)
        print(f"{method:16s}  estimated order = {result['estimated_order']:.3f}  "
              f"(errors: {[f'{e:.2e}' for e in result['errors']]})")


def cmd_fxmeanreversion(args):
    returns = fx.load_log_returns(DATA_PATH)
    squared = [r ** 2 for r in returns]

    print(f"calibrating mean-reversion ODE dV/dt = kappa*(theta-V) on {len(returns)} EUR/USD returns\n")
    for max_lag in (30, 300):
        result = fx.calibrate_mean_reversion(squared, max_lag)
        print(f"max_lag={max_lag:4d}:  kappa={result['kappa']:+.6f}/day  "
              f"half-life={result['half_life_days']:.1f} days  R^2={result['r_squared']:.4f}")

    print()
    final = fx.calibrate_mean_reversion(squared, args.max_lag)
    theta, kappa = final["theta"], final["kappa"]
    v0 = args.shock_multiple * theta
    shock_result = fx.simulate_volatility_shock(theta, kappa, v0, t_end=args.horizon, h=1.0)
    print(f"simulating a {args.shock_multiple}x variance shock (V0={v0:.3e}) with kappa from max_lag={args.max_lag}:")
    for method in ("euler", "improved_euler", "midpoint", "rk4"):
        print(f"  {method:16s} V({args.horizon}) = {shock_result[method]['final_value']:.6e}  "
              f"error = {shock_result[method]['error']:.3e}")
    if "days_to_within_5pct" in shock_result:
        print(f"\ndays until shock decays to within 5% of long-run variance: "
              f"{shock_result['days_to_within_5pct']:.1f}")


def cmd_fxregimes(args):
    q = fx.two_state_generator_from_persistence(args.p_stay_low, args.p_stay_high)
    pi = fx.stationary_distribution_generator(q)
    print(f"continuous-time generator (from daily persistence {args.p_stay_low}, {args.p_stay_high}):")
    for row in q:
        print("  ", [round(x, 6) for x in row])
    print(f"stationary distribution: {[round(p, 4) for p in pi]}\n")

    rhs = fx.kolmogorov_forward_rhs(q)
    result = ode.solve_ivp(rhs, 0.0, [1.0, 0.0], args.horizon, h=1.0, method="rk4")
    print(f"starting certainly in state 0, probability after {args.horizon} days: "
          f"{[round(p, 4) for p in result['y'][-1]]}")
    print(f"(sum = {sum(result['y'][-1]):.8f}, should be exactly 1)")


def cmd_verify(args):
    import numpy as np
    from scipy.integrate import solve_ivp as scipy_solve_ivp

    checks = []

    def check(name, condition):
        checks.append((name, bool(condition)))
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {name}")

    # 1. RK4 vs scipy.integrate.solve_ivp on a nonlinear ODE with no
    #    simple closed form (logistic growth)
    f = lambda t, y: 0.5 * y * (1 - y / 10.0)
    ours = ode.solve_ivp(f, 0.0, 0.1, 20.0, h=0.01, method="rk4")
    scipy_result = scipy_solve_ivp(f, [0, 20.0], [0.1], method="RK45", rtol=1e-10, atol=1e-12)
    check("RK4 matches scipy.integrate.solve_ivp on the logistic equation",
          math.isclose(ours["y"][-1], scipy_result.y[0][-1], rel_tol=1e-4))

    # 2. empirical convergence orders match theory
    f2 = lambda t, y: -y
    exact = lambda t: math.exp(-t)
    h_values = [0.2, 0.1, 0.05, 0.025, 0.0125]
    expected_orders = {"euler": 1.0, "improved_euler": 2.0, "midpoint": 2.0, "rk4": 4.0}
    ok = True
    for method, expected in expected_orders.items():
        result = ea.estimate_convergence_order(f2, 0.0, 1.0, 2.0, h_values, method, exact)
        if abs(result["estimated_order"] - expected) > 0.15:
            ok = False
    check("empirical convergence order matches theory for all four fixed-step methods", ok)

    # 3. exact endpoint reached regardless of step size divisibility
    result = ode.solve_ivp(f2, 0.0, 1.0, math.pi, h=0.137, method="rk4")
    check("solve_ivp lands exactly on t_end even when not a multiple of h",
          math.isclose(result["t"][-1], math.pi, abs_tol=1e-9))

    # 4. harmonic oscillator system: energy conservation check
    osc = lambda t, y: [y[1], -y[0]]
    sol = ode.solve_ivp(osc, 0.0, [1.0, 0.0], 50.0, h=0.001, method="rk4")
    energies = [x ** 2 + v ** 2 for x, v in sol["y"]]
    check("RK4 conserves harmonic-oscillator energy to within 1e-6 over 50 time units",
          max(energies) - min(energies) < 1e-6)

    # 5. adaptive solver vs scipy RK45
    result_adaptive = ode.solve_ivp_adaptive(f2, 0.0, 1.0, 5.0, h_init=0.1, tol=1e-10, method="rk4")
    scipy_adaptive = scipy_solve_ivp(f2, [0, 5.0], [1.0], method="RK45", rtol=1e-10, atol=1e-12)
    check("adaptive RK4 matches scipy's adaptive RK45",
          math.isclose(result_adaptive["y"][-1], scipy_adaptive.y[0][-1], abs_tol=1e-6))

    # 6. Richardson extrapolation improves accuracy by roughly the
    #    expected extra order
    err_plain = ea.global_error(f2, 0.0, 1.0, 1.0, 0.1, "euler", exact)
    extrapolated = ea.richardson_extrapolate(f2, 0.0, 1.0, 1.0, 0.1, "euler", order=1)
    err_extrapolated = abs(extrapolated - exact(1.0))
    check("Richardson extrapolation improves on the plain Euler estimate by at least 10x",
          err_extrapolated < err_plain / 10)

    # 7. real EUR/USD calibration: short window is a poor fit, long
    #    window is a real but partial fit
    returns = fx.load_log_returns(DATA_PATH)
    squared = [r ** 2 for r in returns]
    short = fx.calibrate_mean_reversion(squared, 30)
    long = fx.calibrate_mean_reversion(squared, 300)
    check("short-window (30-lag) calibration R^2 is poor", short["r_squared"] < 0.1)
    check("long-window (300-lag) calibration R^2 is real but partial (0.5-0.9)",
          0.5 < long["r_squared"] < 0.9)

    # 8. Kolmogorov forward equation: probability conservation and
    #    convergence to the stationary distribution, cross-checked
    #    against a numpy null-space eigenvector recomputation
    q = fx.two_state_generator_from_persistence(0.97, 0.95)
    pi = fx.stationary_distribution_generator(q)
    q_np = np.array(q)
    eigvals, eigvecs = np.linalg.eig(q_np.T)
    idx = int(np.argmin(np.abs(eigvals)))
    vec = np.real(eigvecs[:, idx])
    vec = vec / vec.sum()
    check("stationary distribution of the generator matches numpy null-space eigenvector",
          all(math.isclose(pi[i], vec[i], abs_tol=1e-6) for i in range(2)))

    rhs = fx.kolmogorov_forward_rhs(q)
    sol = ode.solve_ivp(rhs, 0.0, [1.0, 0.0], 3000, h=1.0, method="rk4")
    check("Kolmogorov forward solution converges to the stationary distribution",
          all(math.isclose(sol["y"][-1][i], pi[i], abs_tol=1e-4) for i in range(2)))

    passed = sum(1 for _, ok in checks if ok)
    print(f"\n{passed}/{len(checks)} checks passed")
    if passed != len(checks):
        sys.exit(1)


def build_parser():
    parser = argparse.ArgumentParser(description="ODE engine CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("solve", help="solve an IVP with a fixed-step method")
    p1.add_argument("--rhs", type=str, required=True, help="Python expr in t,y, e.g. '-y'")
    p1.add_argument("--t0", type=float, default=0.0)
    p1.add_argument("--y0", type=str, required=True, help="scalar or comma-separated vector")
    p1.add_argument("--t-end", type=float, required=True)
    p1.add_argument("--h", type=float, default=0.01)
    p1.add_argument("--method", choices=["euler", "improved_euler", "midpoint", "rk4"], default="rk4")
    p1.set_defaults(func=cmd_solve)

    p2 = sub.add_parser("adaptive", help="solve an IVP with adaptive step-doubling")
    p2.add_argument("--rhs", type=str, required=True)
    p2.add_argument("--t0", type=float, default=0.0)
    p2.add_argument("--y0", type=str, required=True)
    p2.add_argument("--t-end", type=float, required=True)
    p2.add_argument("--h-init", type=float, default=0.1)
    p2.add_argument("--tol", type=float, default=1e-6)
    p2.add_argument("--method", choices=["euler", "improved_euler", "midpoint", "rk4"], default="rk4")
    p2.set_defaults(func=cmd_adaptive)

    p3 = sub.add_parser("convergence", help="empirically estimate each method's convergence order")
    p3.add_argument("--rhs", type=str, required=True)
    p3.add_argument("--exact", type=str, required=True, help="Python expr in t for the true solution")
    p3.add_argument("--t0", type=float, default=0.0)
    p3.add_argument("--y0", type=float, required=True)
    p3.add_argument("--t-end", type=float, required=True)
    p3.set_defaults(func=cmd_convergence)

    p4 = sub.add_parser("fxmeanreversion", help="calibrate and simulate a volatility mean-reversion ODE")
    p4.add_argument("--max-lag", type=int, default=300)
    p4.add_argument("--shock-multiple", type=float, default=4.0)
    p4.add_argument("--horizon", type=float, default=200.0)
    p4.set_defaults(func=cmd_fxmeanreversion)

    p5 = sub.add_parser("fxregimes", help="solve the Kolmogorov forward equations for a 2-state chain")
    p5.add_argument("--p-stay-low", type=float, default=0.99)
    p5.add_argument("--p-stay-high", type=float, default=0.99)
    p5.add_argument("--horizon", type=float, default=1000.0)
    p5.set_defaults(func=cmd_fxregimes)

    p6 = sub.add_parser("verify", help="cross-check implementation against analytical/oracle values")
    p6.set_defaults(func=cmd_verify)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
