"""Command-line interface for the gradient descent optimizer and LP solver.

Usage
-----
    python3 cli.py gd       --fn EXPR --x0 X,Y [--lr LR] [--iters N]
    python3 cli.py momentum --fn EXPR --x0 X,Y [--lr LR] [--beta B]
    python3 cli.py rmsprop  --fn EXPR --x0 X,Y [--lr LR]
    python3 cli.py adam     --fn EXPR --x0 X,Y [--lr LR]
    python3 cli.py neldermead --fn EXPR --x0 X,Y
    python3 cli.py lp       --c C1,C2 --A "[[a,b],[c,d]]" --b B1,B2
    python3 cli.py logistic --data PATH
    python3 cli.py verify

Function expressions use x0, x1 as variable names (for 2D functions).

Examples
--------
    python3 cli.py gd        --fn "(x0-3)**2+(x1+2)**2" --x0 0,0 --lr 0.1
    python3 cli.py adam      --fn "(1-x0)**2+100*(x1-x0**2)**2" --x0 0,0 --lr 0.01 --iters 10000
    python3 cli.py lp        --c "-1,-1" --A "[[1,1],[1,0],[0,1]]" --b "4,3,3"
    python3 cli.py logistic  --data data/DEXUSEU_returns.csv
    python3 cli.py verify
"""

import argparse
import ast
import math
import sys

import optimizer as op


# ---------------------------------------------------------------------------
# Expression helpers
# ---------------------------------------------------------------------------

_SAFE = {
    "sin": math.sin, "cos": math.cos, "exp": math.exp,
    "log": math.log, "sqrt": math.sqrt, "abs": abs,
    "pi": math.pi, "e": math.e,
}


def make_fn(expr):
    def f(x):
        env = dict(_SAFE)
        env["x0"] = x[0]
        env["x1"] = x[1] if len(x) > 1 else 0.0
        return float(eval(expr, {"__builtins__": {}}, env))
    return f


def parse_csv_floats(s):
    return [float(v) for v in s.split(",")]


def parse_matrix(s):
    A = ast.literal_eval(s)
    return [[float(v) for v in row] for row in A]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _print_result(r, name):
    print("{} result:".format(name))
    print("  x:          [{}]".format(", ".join("{:.8g}".format(v) for v in r["x"])))
    print("  f(x):       {:.10g}".format(r["f_val"]))
    print("  iterations: {}".format(r["iterations"]))
    print("  converged:  {}".format(r["converged"]))


def cmd_gd(args):
    f = make_fn(args.fn)
    r = op.gradient_descent(f, args.x0, learning_rate=args.lr,
                              max_iter=args.iters, tolerance=args.tol)
    _print_result(r, "Gradient Descent")


def cmd_momentum(args):
    f = make_fn(args.fn)
    r = op.momentum_gd(f, args.x0, learning_rate=args.lr,
                        beta=args.beta, max_iter=args.iters, tolerance=args.tol)
    _print_result(r, "Momentum GD")


def cmd_rmsprop(args):
    f = make_fn(args.fn)
    r = op.rmsprop(f, args.x0, learning_rate=args.lr,
                    max_iter=args.iters, tolerance=args.tol)
    _print_result(r, "RMSProp")


def cmd_adam(args):
    f = make_fn(args.fn)
    r = op.adam(f, args.x0, learning_rate=args.lr,
                 max_iter=args.iters, tolerance=args.tol)
    _print_result(r, "Adam")


def cmd_neldermead(args):
    f = make_fn(args.fn)
    r = op.nelder_mead(f, args.x0, max_iter=args.iters, tolerance=args.tol)
    _print_result(r, "Nelder-Mead")


def cmd_lp(args):
    c = parse_csv_floats(args.c)
    A = parse_matrix(args.A)
    b = parse_csv_floats(args.b)
    r = op.lp_simplex(c, A, b)
    print("LP result:")
    print("  status:     {}".format(r["status"]))
    print("  objective:  {:.10g}".format(r["obj"]))
    print("  x:          [{}]".format(", ".join("{:.8g}".format(v) for v in r["x"])))
    print("  pivots:     {}".format(r["iterations"]))


def cmd_logistic(args):
    import csv, os
    rows  = []
    with open(args.data, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row and row[1].strip():
                rows.append(float(row[1]))
    # Binary classification: positive vs negative return
    n  = min(len(rows) - 1, 500)
    X  = [[1.0, rows[i], rows[i-1]] for i in range(1, n+1)]
    y  = [1 if rows[i] > 0 else 0 for i in range(1, n+1)]
    w, hist = op.fit_logistic(X, y, learning_rate=0.1, max_iter=500)
    preds = op.predict_logistic(w, X)
    acc   = sum(p == yi for p, yi in zip(preds, y)) / len(y)
    print("Logistic regression on EUR/USD ({} samples):".format(n))
    print("  weights:    [{}]".format(", ".join("{:.6g}".format(wi) for wi in w)))
    print("  accuracy:   {:.2%}".format(acc))
    print("  loss (initial / final): {:.4f} / {:.4f}".format(hist[0], hist[-1]))


def cmd_verify(args):
    checks = []
    import math as _m

    # ---- Gradient descent ----
    f2 = lambda x: x[0]**2 + x[1]**2
    r = op.gradient_descent(f2, [3.0, 4.0], learning_rate=0.1, tolerance=1e-8)
    checks.append((r["converged"], "GD converges on quadratic"))
    checks.append((abs(r["x"][0]) < 1e-4 and abs(r["x"][1]) < 1e-4,
                   "GD reaches minimum at origin"))
    checks.append((r["f_val"] < 1e-8, "GD f_val < 1e-8 at minimum"))
    checks.append((r["history"][0] > r["history"][-1],
                   "GD history is non-increasing"))

    # Shifted minimum
    fs = lambda x: (x[0]-3.0)**2 + (x[1]+2.0)**2
    r2 = op.gradient_descent(fs, [0.0, 0.0], learning_rate=0.1, tolerance=1e-8)
    checks.append((abs(r2["x"][0] - 3.0) < 1e-3 and abs(r2["x"][1] + 2.0) < 1e-3,
                   "GD converges to shifted minimum (3,-2)"))

    # ---- Momentum ----
    rm = op.momentum_gd(f2, [3.0, 4.0], learning_rate=0.05, beta=0.9)
    checks.append((rm["converged"], "Momentum GD converges"))

    # ---- RMSProp ----
    rr = op.rmsprop(f2, [3.0, 4.0], learning_rate=0.1)
    checks.append((rr["converged"], "RMSProp converges"))
    checks.append((abs(rr["x"][0]) < 1e-4, "RMSProp reaches minimum"))

    # ---- Adam ----
    ra = op.adam(f2, [3.0, 4.0], learning_rate=0.1)
    checks.append((ra["converged"], "Adam converges"))
    checks.append((abs(ra["x"][0]) < 1e-3, "Adam reaches near origin"))

    # ---- Line search ----
    rls = op.gradient_descent_ls(fs, [0.0, 0.0], tolerance=1e-8)
    checks.append((rls["converged"], "GD with line search converges"))
    checks.append((abs(rls["x"][0] - 3.0) < 1e-4, "Line search reaches x=3"))

    # ---- Nelder-Mead ----
    rn = op.nelder_mead(fs, [0.0, 0.0])
    checks.append((abs(rn["x"][0] - 3.0) < 0.01, "Nelder-Mead finds x=3"))
    checks.append((abs(rn["x"][1] + 2.0) < 0.01, "Nelder-Mead finds y=-2"))

    # ---- LP simplex ----
    # max 5x+4y s.t. 6x+4y<=24, x+2y<=6
    r_lp = op.lp_simplex([-5.0,-4.0], [[6.0,4.0],[1.0,2.0]], [24.0,6.0])
    checks.append((r_lp["status"] == "optimal", "LP status = optimal"))
    checks.append((abs(r_lp["obj"] + 21.0) < 1e-4, "LP optimal value = -21 (max 21)"))
    x_lp = r_lp["x"]
    for i, (row, bi) in enumerate(zip([[6,4],[1,2]], [24,6])):
        lhs = sum(row[j] * x_lp[j] for j in range(2))
        checks.append((lhs <= bi + 1e-6, "LP constraint {} satisfied".format(i)))
    checks.append((all(v >= -1e-8 for v in x_lp), "LP solution non-negative"))

    # ---- Logistic loss ----
    X_lr = [[-1.0], [1.0]]
    y_lr = [0, 1]
    loss0 = op.logistic_loss([0.0], X_lr, y_lr)
    checks.append((abs(loss0 - _m.log(2.0)) < 1e-8,
                   "logistic loss at w=0 = log(2)"))

    # ---- Logistic gradient ----
    g_an = op.logistic_gradient([0.5], X_lr, y_lr)
    g_nu = op._grad(lambda w: op.logistic_loss(w, X_lr, y_lr), [0.5], h=1e-6)
    checks.append((abs(g_an[0] - g_nu[0]) < 1e-5,
                   "logistic gradient matches numerical gradient"))

    # ---- Fit logistic ----
    X_cl = [[-2.0], [-1.0], [1.0], [2.0]]
    y_cl = [0, 0, 1, 1]
    w, hist = op.fit_logistic(X_cl, y_cl, learning_rate=0.5, max_iter=500)
    preds = op.predict_logistic(w, X_cl)
    acc   = sum(p == yi for p, yi in zip(preds, y_cl)) / 4
    checks.append((acc == 1.0, "logistic regression perfect accuracy on separable data"))
    checks.append((hist[-1] < hist[0], "logistic loss decreases during training"))

    passed = sum(1 for ok, _ in checks if ok)
    for ok, label in checks:
        print("{:>5}  {}".format("PASS" if ok else "FAIL", label))
    print("-" * 60)
    print("{}/{} checks passed".format(passed, len(checks)))
    if passed != len(checks):
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub    = parser.add_subparsers(dest="command")

    for cmd in ("gd", "momentum", "rmsprop", "adam", "neldermead"):
        p = sub.add_parser(cmd)
        p.add_argument("--fn", required=True)
        p.add_argument("--x0", required=True, type=lambda s: [float(v) for v in s.split(",")])
        p.add_argument("--lr", type=float, default=0.01)
        p.add_argument("--iters", type=int, default=10000)
        p.add_argument("--tol", type=float, default=1e-8)
        p.add_argument("--beta", type=float, default=0.9)
        fn = {"gd": cmd_gd, "momentum": cmd_momentum, "rmsprop": cmd_rmsprop,
              "adam": cmd_adam, "neldermead": cmd_neldermead}[cmd]
        p.set_defaults(func=fn)

    p = sub.add_parser("lp")
    p.add_argument("--c", required=True)
    p.add_argument("--A", required=True)
    p.add_argument("--b", required=True)
    p.set_defaults(func=cmd_lp)

    p = sub.add_parser("logistic")
    p.add_argument("--data", required=True)
    p.set_defaults(func=cmd_logistic)

    p = sub.add_parser("verify")
    p.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
