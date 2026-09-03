import argparse
import math
import os
import sys

import linear_algebra as la
import gradient_methods as gm
import newtons_method as nm
import constrained_optimization as co
import duality as du
import fx_application as fx

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DEXUSEU_returns.csv")


def cmd_minimize(args):
    f = eval("lambda x: " + args.f)
    x0 = [float(v) for v in args.x0.split(",")]
    if args.method == "gradient":
        result = gm.gradient_descent(f, x0)
    else:
        result = nm.newtons_method(f, x0)
    print(f"method={args.method}")
    print(f"x* = {result['x']}")
    print(f"f(x*) = {result['f']:.8f}")
    print(f"iterations = {result['iterations']}, converged = {result['converged']}")


def cmd_portfolio(args):
    returns = fx.load_log_returns(DATA_PATH)
    strategies = fx.build_strategy_returns(returns)
    names = list(strategies.keys())
    sigma = fx.sample_covariance(list(strategies.values()))

    print("strategy annualized Sharpe ratios:")
    for name, series in strategies.items():
        print(f"  {name:16s} {fx.annualized_sharpe(series):+.4f}")
    print()

    unconstrained = fx.min_variance_portfolio_unconstrained(sigma)
    unconstrained_var = fx.portfolio_variance(sigma, unconstrained["x"])
    print("unconstrained minimum-variance portfolio:")
    for name, w in zip(names, unconstrained["x"]):
        print(f"  {name:16s} {w:+.4f}")
    print(f"  variance: {unconstrained_var:.6e}")
    kkt_u = fx.verify_unconstrained_solution_kkt(sigma, unconstrained)
    print(f"  KKT satisfied: {kkt_u['all_conditions_satisfied']}\n")

    long_only = fx.min_variance_portfolio_long_only(sigma)
    print("long-only minimum-variance portfolio:")
    for name, w in zip(names, long_only["x"]):
        print(f"  {name:16s} {w:+.4f}")
    print(f"  variance: {long_only['variance']:.6e}")
    kkt_lo = fx.verify_long_only_solution_kkt(sigma, long_only["x"])
    print(f"  KKT satisfied: {kkt_lo['all_conditions_satisfied']}\n")

    print(f"cost of the long-only constraint: {long_only['variance'] / unconstrained_var:.4f}x "
          f"the unconstrained variance")


def cmd_verify(args):
    import random
    import numpy as np
    from scipy.optimize import minimize as scipy_minimize

    checks = []

    def check(name, condition):
        checks.append((name, bool(condition)))
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {name}")

    f = lambda x: (x[0] - 3) ** 2 + (x[1] + 2) ** 2
    ours = gm.gradient_descent(f, [0.0, 0.0])
    theirs = scipy_minimize(f, [0.0, 0.0], method="BFGS")
    check("gradient descent matches scipy.optimize.minimize (BFGS)",
          math.isclose(ours["x"][0], theirs.x[0], abs_tol=1e-4) and
          math.isclose(ours["x"][1], theirs.x[1], abs_tol=1e-4))

    f2 = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    grad2 = lambda x: [-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2), 200 * (x[1] - x[0] ** 2)]

    def hess2(x):
        h11 = 2 - 400 * x[1] + 1200 * x[0] ** 2
        h12 = -400 * x[0]
        return [[h11, h12], [h12, 200.0]]

    ours2 = nm.newtons_method(f2, [-1.0, 1.0], grad_fn=grad2, hess_fn=hess2, tol=1e-10)
    theirs2 = scipy_minimize(f2, [-1.0, 1.0], method="Newton-CG",
                              jac=lambda x: np.array(grad2(list(x))),
                              hess=lambda x: np.array(hess2(list(x))))
    check("Newton's method matches scipy.optimize.minimize (Newton-CG) on Rosenbrock",
          math.isclose(ours2["x"][0], theirs2.x[0], abs_tol=1e-4) and
          math.isclose(ours2["x"][1], theirs2.x[1], abs_tol=1e-4))

    rng = random.Random(1)
    n = 4
    m_rand = [[rng.gauss(0, 1) for _ in range(n)] for _ in range(n)]
    sigma = la.matmul(m_rand, la.transpose(m_rand))
    qp_result = co.solve_equality_constrained_qp(sigma, [0.0] * n, [[1.0] * n], [1.0])

    f3 = lambda w: 0.5 * la.dot(list(w), la.mat_vec(sigma, list(w)))
    constraint = {"type": "eq", "fun": lambda w: sum(w) - 1.0}
    theirs3 = scipy_minimize(f3, [1.0 / n] * n, method="SLSQP", constraints=[constraint])
    check("equality-constrained QP matches scipy.optimize.minimize (SLSQP)",
          all(math.isclose(a, b, abs_tol=1e-4) for a, b in zip(qp_result["x"], theirs3.x)))

    v = [2.0, -1.0, 0.5, 3.0]
    ours4 = co.project_onto_simplex(v)
    f4 = lambda x: sum((xi - vi) ** 2 for xi, vi in zip(x, v))
    cons4 = [{"type": "eq", "fun": lambda x: sum(x) - 1.0}]
    bounds4 = [(0.0, None)] * len(v)
    theirs4 = scipy_minimize(f4, [1.0 / len(v)] * len(v), method="SLSQP", constraints=cons4, bounds=bounds4)
    check("simplex projection matches scipy-solved projection QP",
          all(math.isclose(a, b, abs_tol=1e-4) for a, b in zip(ours4, theirs4.x)))

    returns = fx.load_log_returns(DATA_PATH)
    strategies = fx.build_strategy_returns(returns)
    sigma_real = fx.sample_covariance(list(strategies.values()))
    n_real = len(sigma_real)
    long_only = fx.min_variance_portfolio_long_only(sigma_real)

    f5 = lambda w: 0.5 * la.dot(list(w), la.mat_vec(sigma_real, list(w)))
    cons5 = [{"type": "eq", "fun": lambda w: sum(w) - 1.0}]
    bounds5 = [(0.0, None)] * n_real
    # scipy's SLSQP uses default tolerances (ftol~1e-6) tuned for
    # objectives of order-1 scale; this project's covariance-matrix
    # objectives are ~1e-5 in magnitude (real daily-return data), the
    # same scale-sensitivity already found and fixed for this
    # project's own projected_gradient_descent default step size.
    # Without a tightened ftol, SLSQP falsely reports convergence at
    # its initial guess without taking a single meaningful step.
    theirs5 = scipy_minimize(f5, [1.0 / n_real] * n_real, method="SLSQP", constraints=cons5, bounds=bounds5,
                              options={"ftol": 1e-16, "maxiter": 1000})
    check("long-only portfolio matches scipy-solved constrained QP on real EUR/USD data",
          math.isclose(long_only["variance"], 2 * theirs5.fun, rel_tol=1e-3))

    unconstrained_real = fx.min_variance_portfolio_unconstrained(sigma_real)
    kkt_u = fx.verify_unconstrained_solution_kkt(sigma_real, unconstrained_real)
    kkt_lo = fx.verify_long_only_solution_kkt(sigma_real, long_only["x"])
    check("KKT conditions hold at the unconstrained real-data solution", kkt_u["all_conditions_satisfied"])
    check("KKT conditions hold at the long-only real-data solution", kkt_lo["all_conditions_satisfied"])

    unconstrained_var = fx.portfolio_variance(sigma_real, unconstrained_real["x"])
    check("long-only real-data variance is not lower than the unconstrained variance",
          long_only["variance"] >= unconstrained_var - 1e-9)

    f8 = lambda x: x[0] ** 2
    g8 = [lambda x: 1.0 - x[0]]
    dual_result = du.maximize_dual_via_projected_gradient_ascent(f8, [], g8, [], [0.0], [0.5],
                                                                   step=0.5, max_iter=200)
    strong = du.verify_strong_duality(primal_optimal_value=1.0, dual_optimal_value=dual_result["dual_value"])
    check("strong duality holds on the known analytical example (min x^2 s.t. x>=1)",
          strong["strong_duality_holds"])

    diag0, diag1 = sigma_real[1][1], sigma_real[2][2]
    correlation = sigma_real[1][2] / (diag0 ** 0.5 * diag1 ** 0.5)
    check("momentum and mean-reversion strategies are not degenerate exact opposites",
          correlation > -0.99)

    passed = sum(1 for _, ok in checks if ok)
    print(f"\n{passed}/{len(checks)} checks passed")
    if passed != len(checks):
        sys.exit(1)


def build_parser():
    parser = argparse.ArgumentParser(description="Convex optimization engine CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("minimize", help="minimize a function via gradient descent or Newton's method")
    p1.add_argument("--f", type=str, required=True, help="Python expr in x (a list), e.g. '(x[0]-3)**2'")
    p1.add_argument("--x0", type=str, required=True, help="comma-separated starting point")
    p1.add_argument("--method", choices=["gradient", "newton"], default="gradient")
    p1.set_defaults(func=cmd_minimize)

    p2 = sub.add_parser("portfolio", help="constrained portfolio optimization on real EUR/USD strategies")
    p2.set_defaults(func=cmd_portfolio)

    p3 = sub.add_parser("verify", help="cross-check implementation against scipy/numpy oracles and known theory")
    p3.set_defaults(func=cmd_verify)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
