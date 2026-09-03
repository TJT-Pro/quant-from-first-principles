"""Command-line front end for the Numerical Differentiation engine."""
import argparse
import math
import sys

import differentiator as df


# ---------------------------------------------------------------------------
# Expression parsing
# ---------------------------------------------------------------------------

_SAFE_NAMES = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
    "abs": abs, "pi": math.pi, "e": math.e,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "atan2": math.atan2, "ceil": math.ceil, "floor": math.floor,
}


def make_1d(expr):
    """Turn an expression string into a one-variable function f(x)."""
    def f(x):
        env = dict(_SAFE_NAMES)
        env["x"] = x
        try:
            return float(eval(expr, {"__builtins__": {}}, env))
        except Exception as e:
            sys.exit("error evaluating {!r}: {}".format(expr, e))
    return f


def make_nd(expr):
    """Turn an expression string into a multivariate function f(x_list)."""
    def f(x):
        env = dict(_SAFE_NAMES)
        env["x"] = x
        try:
            return float(eval(expr, {"__builtins__": {}}, env))
        except Exception as e:
            sys.exit("error evaluating {!r}: {}".format(expr, e))
    return f


def parse_floats(s):
    try:
        return [float(v) for v in s.split(",")]
    except ValueError:
        sys.exit("expected comma-separated floats, got {!r}".format(s))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_diff(args):
    f = make_1d(args.f)
    x = float(args.x)
    h = float(args.h) if args.h else None
    result = df.differentiate(f, x, method=args.method, h=h)
    print("f(x)   = {}".format(args.f))
    print("x      = {}".format(x))
    print("method = {}".format(args.method))
    print("f'(x)  = {:.15g}".format(result))


def cmd_gradient(args):
    f  = make_nd(args.f)
    x  = parse_floats(args.x)
    h  = float(args.h) if args.h else None
    g  = df.gradient(f, x, h=h, method=args.method)
    print("f(x)     = {}".format(args.f))
    print("x        = {}".format(x))
    print("gradient = [{}]".format(", ".join("{:.10g}".format(v) for v in g)))
    mag = math.sqrt(sum(v * v for v in g))
    print("|grad|   = {:.10g}".format(mag))


def cmd_second(args):
    f  = make_1d(args.f)
    x  = float(args.x)
    h  = float(args.h) if args.h else None
    d2 = df.second_derivative(f, x, h=h)
    print("f(x)   = {}".format(args.f))
    print("x      = {}".format(x))
    print("f''(x) = {:.15g}".format(d2))


def cmd_error(args):
    f     = make_1d(args.f)
    x     = float(args.x)
    exact = float(args.exact)
    # generate h values spanning many orders of magnitude
    h_values = [10.0 ** (-k/2) for k in range(2, 30)]
    print("f(x) = {}   x = {}   exact f'(x) = {}".format(
        args.f, x, exact))
    print("{:>14}  {:>14}  {:>14}  {:>14}".format(
        "h", "forward err", "central err", "5-point err"))
    for h in h_values:
        e_fwd = abs(df.forward_difference(f, x, h=h) - exact)
        e_cen = abs(df.central_difference(f, x, h=h) - exact)
        try:
            e_5pt = abs(df.five_point_stencil(f, x, h=h) - exact)
        except Exception:
            e_5pt = float("inf")
        print("{:>14.2e}  {:>14.2e}  {:>14.2e}  {:>14.2e}".format(
            h, e_fwd, e_cen, e_5pt))


def cmd_product(args):
    f = make_1d(args.f)
    g = make_1d(args.g)
    x = float(args.x)
    r = df.verify_product_rule(f, g, x)
    print("Product rule check at x = {}".format(x))
    print("  f(x)   = {}".format(args.f))
    print("  g(x)   = {}".format(args.g))
    print("  (fg)'  = {:.15g}  (numerical, direct)".format(r["lhs"]))
    print("  f'g+fg'= {:.15g}  (product rule)".format(r["rhs"]))
    print("  error  = {:.3e}".format(r["error"]))


def cmd_chain(args):
    f = make_1d(args.f)
    g = make_1d(args.g)
    x = float(args.x)
    r = df.verify_chain_rule(f, g, x)
    print("Chain rule check at x = {}".format(x))
    print("  f(x)     = {}".format(args.f))
    print("  g(x)     = {}".format(args.g))
    print("  (fog)'   = {:.15g}  (numerical, direct)".format(r["lhs"]))
    print("  f'(g)*g' = {:.15g}  (chain rule)".format(r["rhs"]))
    print("  error    = {:.3e}".format(r["error"]))


def cmd_descent(args):
    f  = make_nd(args.f)
    x0 = parse_floats(args.x)
    lr = args.lr
    result = df.gradient_descent(f, x0, learning_rate=lr,
                                 max_iterations=args.iters,
                                 tolerance=1e-8)
    print("Gradient descent on f(x) = {}".format(args.f))
    print("  start:      {}  f = {:.6g}".format(x0, f(x0)))
    print("  end:        {}  f = {:.6g}".format(
        [round(v, 8) for v in result["x"]], result["f_x"]))
    print("  iterations: {}".format(result["iterations"]))
    print("  converged:  {}".format(result["converged"]))


def cmd_bond(args):
    y      = args.yield_rate
    c      = args.coupon
    f      = args.face
    n      = args.n
    price  = df.bond_price(y, c, f, n)
    dur    = df.bond_duration(y, c, f, n)
    conv   = df.bond_convexity(y, c, f, n)
    w = 42
    print("=" * w)
    print("Bond Analytics")
    print("=" * w)
    print("  Yield:       {:.4%}".format(y))
    print("  Coupon:      {:.2f} per period".format(c))
    print("  Face value:  {:.2f}".format(f))
    print("  Periods:     {}".format(n))
    print("-" * w)
    print("  Price:           {:.6f}".format(price))
    print("  Mod duration:    {:.6f}  (numerical)".format(dur["numerical"]))
    print("  Mod duration:    {:.6f}  (analytical)".format(dur["analytical"]))
    print("  Convexity:       {:.6f}".format(conv))
    if args.dy is not None:
        chg = df.bond_price_change(y, c, f, n, args.dy)
        print("-" * w)
        print("  Yield shift:       {:+.4%}".format(args.dy))
        print("  Exact dP/P:        {:+.6%}".format(chg["exact_pct_change"]))
        print("  Duration approx:   {:+.6%}  "
              "(error {:.2e})".format(chg["duration_approx"],
                                     chg["duration_error"]))
        print("  Dur+conv approx:   {:+.6%}  "
              "(error {:.2e})".format(chg["duration_convexity_approx"],
                                     chg["duration_convexity_error"]))
    print("=" * w)


def cmd_verify(args):
    """Cross-check all numerical results against known exact values."""
    checks = []

    # Forward difference: f(x) = x^2, f'(x) = 2x
    fwd = df.forward_difference(lambda x: x*x, 3.0, h=1e-7)
    checks.append((abs(fwd - 6.0) < 1e-5, "forward difference x^2 at x=3"))

    # Backward difference
    bwd = df.backward_difference(lambda x: x*x, 3.0, h=1e-7)
    checks.append((abs(bwd - 6.0) < 1e-5, "backward difference x^2 at x=3"))

    # Central difference -- much more accurate
    cen = df.central_difference(lambda x: x*x, 3.0)
    checks.append((abs(cen - 6.0) < 1e-10, "central difference x^2 at x=3"))

    # Five-point stencil
    fp = df.five_point_stencil(lambda x: x*x, 3.0)
    checks.append((abs(fp - 6.0) < 1e-10, "five-point x^2 at x=3"))

    # sin(x): f'(x) = cos(x)
    for x_val in [0.0, 0.5, 1.0, math.pi/4, math.pi/3]:
        approx = df.central_difference(math.sin, x_val)
        exact  = math.cos(x_val)
        checks.append((abs(approx - exact) < 1e-10,
                       "sin'({:.4f}) = cos({:.4f})".format(x_val, x_val)))

    # exp(x): f'(x) = exp(x)
    for x_val in [0.0, 1.0, -1.0]:
        approx = df.central_difference(math.exp, x_val)
        exact  = math.exp(x_val)
        checks.append((abs(approx - exact) < 1e-8,
                       "exp'({}) = exp({})".format(x_val, x_val)))

    # log(x): f'(x) = 1/x
    for x_val in [1.0, 2.0, math.e]:
        approx = df.central_difference(math.log, x_val)
        exact  = 1.0 / x_val
        checks.append((abs(approx - exact) < 1e-8,
                       "log'({:.4f}) = 1/x".format(x_val)))

    # second derivative: exp(x)'' = exp(x)
    d2 = df.second_derivative(math.exp, 1.0)
    checks.append((abs(d2 - math.exp(1.0)) < 1e-6, "exp'' at x=1 = e"))

    # second derivative: sin(x)'' = -sin(x)
    d2s = df.second_derivative(math.sin, 1.0)
    checks.append((abs(d2s - (-math.sin(1.0))) < 1e-6,
                   "sin'' at x=1 = -sin(1)"))

    # five-point is more accurate than central for smooth functions
    h = 1e-3
    err_cen = abs(df.central_difference(math.sin, 1.0, h=h) - math.cos(1.0))
    err_5pt = abs(df.five_point_stencil(math.sin, 1.0, h=h) - math.cos(1.0))
    checks.append((err_5pt < err_cen,
                   "five-point more accurate than central for sin at h=1e-3"))

    # gradient of f(x,y) = x^2 + y^2 is [2x, 2y]
    f_2d = lambda x: x[0]**2 + x[1]**2
    g    = df.gradient(f_2d, [3.0, 4.0])
    checks.append((abs(g[0] - 6.0) < 1e-8, "grad(x^2+y^2) df/dx at (3,4) = 6"))
    checks.append((abs(g[1] - 8.0) < 1e-8, "grad(x^2+y^2) df/dy at (3,4) = 8"))

    # gradient of f(x,y,z) = x*y*z is [yz, xz, xy]
    f_3d = lambda x: x[0] * x[1] * x[2]
    g3   = df.gradient(f_3d, [2.0, 3.0, 4.0])
    checks.append((abs(g3[0] - 12.0) < 1e-7, "grad(xyz) df/dx at (2,3,4) = 12"))
    checks.append((abs(g3[1] - 8.0)  < 1e-7, "grad(xyz) df/dy at (2,3,4) = 8"))
    checks.append((abs(g3[2] - 6.0)  < 1e-7, "grad(xyz) df/dz at (2,3,4) = 6"))

    # directional derivative along (1,0) is df/dx
    f_2d2 = lambda x: x[0]**2 + x[1]**2
    dd = df.directional_derivative(f_2d2, [3.0, 4.0], [1.0, 0.0])
    checks.append((abs(dd - 6.0) < 1e-8,
                   "directional derivative along x-axis = df/dx"))

    # directional derivative along (1,1)/sqrt(2) = (df/dx + df/dy)/sqrt(2)
    dd2 = df.directional_derivative(f_2d2, [3.0, 4.0], [1.0, 1.0])
    expected_dd2 = (6.0 + 8.0) / math.sqrt(2.0)
    checks.append((abs(dd2 - expected_dd2) < 1e-7,
                   "directional derivative along (1,1) correct"))

    # second partial of x^2 + y^2: d^2f/dx^2 = 2, d^2f/dy^2 = 2
    d2xx = df.second_partial(lambda x: x[0]**2 + x[1]**2, [1.0, 1.0], 0, 0)
    d2yy = df.second_partial(lambda x: x[0]**2 + x[1]**2, [1.0, 1.0], 1, 1)
    checks.append((abs(d2xx - 2.0) < 1e-4, "d^2/dx^2 of x^2+y^2 = 2"))
    checks.append((abs(d2yy - 2.0) < 1e-4, "d^2/dy^2 of x^2+y^2 = 2"))

    # mixed partial of x*y: d^2/dxdy = 1
    d2xy = df.second_partial(lambda x: x[0] * x[1], [2.0, 3.0], 0, 1)
    checks.append((abs(d2xy - 1.0) < 1e-4, "d^2(xy)/dxdy = 1"))

    # Jacobian of f(x) = [x[0]^2, x[0]*x[1]] at [2, 3]
    # J = [[2*x0, 0], [x1, x0]] = [[4, 0], [3, 2]]
    f_vec = lambda x: [x[0]**2, x[0]*x[1]]
    J = df.jacobian(f_vec, [2.0, 3.0])
    checks.append((abs(J[0][0] - 4.0) < 1e-7, "Jacobian J[0][0] = 2x0 = 4"))
    checks.append((abs(J[0][1] - 0.0) < 1e-7, "Jacobian J[0][1] = 0"))
    checks.append((abs(J[1][0] - 3.0) < 1e-7, "Jacobian J[1][0] = x1 = 3"))
    checks.append((abs(J[1][1] - 2.0) < 1e-7, "Jacobian J[1][1] = x0 = 2"))

    # product rule: f=sin, g=exp at x=1
    pr = df.verify_product_rule(math.sin, math.exp, 1.0)
    checks.append((pr["error"] < 1e-8, "product rule sin*exp at x=1"))

    # chain rule: f=sin, g=x^2 at x=1
    cr = df.verify_chain_rule(math.sin, lambda x: x*x, 1.0)
    checks.append((cr["error"] < 1e-8, "chain rule sin(x^2) at x=1"))

    # gradient descent: minimum of x^2+y^2 is at (0,0)
    result = df.gradient_descent(lambda x: x[0]**2 + x[1]**2,
                                  [3.0, 4.0], learning_rate=0.1)
    checks.append((result["converged"], "gradient descent converges on x^2+y^2"))
    checks.append((abs(result["x"][0]) < 1e-5 and abs(result["x"][1]) < 1e-5,
                   "gradient descent finds minimum at (0,0)"))

    # gradient descent on (x-2)^2 + (y-3)^2 converges to (2,3)
    res2 = df.gradient_descent(
        lambda x: (x[0]-2)**2 + (x[1]-3)**2,
        [0.0, 0.0], learning_rate=0.1
    )
    checks.append((abs(res2["x"][0] - 2.0) < 1e-4 and
                   abs(res2["x"][1] - 3.0) < 1e-4,
                   "gradient descent finds minimum at (2,3)"))

    # bond price: 5% annual coupon, 10-year, 5% yield => par
    p = df.bond_price(0.05, 50, 1000, 10)
    checks.append((abs(p - 1000.0) < 1e-6,
                   "bond at par when coupon rate = yield"))

    # bond below par when yield > coupon rate
    p_above = df.bond_price(0.06, 50, 1000, 10)
    checks.append((p_above < 1000.0, "bond below par when yield > coupon rate"))

    # bond above par when yield < coupon rate
    p_below = df.bond_price(0.04, 50, 1000, 10)
    checks.append((p_below > 1000.0, "bond above par when yield < coupon rate"))

    # duration: numerical matches analytical within 1e-6
    dur = df.bond_duration(0.05, 50, 1000, 10)
    checks.append((abs(dur["numerical"] - dur["analytical"]) < 1e-6,
                   "numerical duration matches analytical"))

    # duration positive
    checks.append((dur["numerical"] > 0.0, "bond duration is positive"))

    # convexity positive
    conv = df.bond_convexity(0.05, 50, 1000, 10)
    checks.append((conv > 0.0, "bond convexity is positive"))

    # duration+convexity approximation closer to exact than duration alone
    chg = df.bond_price_change(0.05, 50, 1000, 10, 0.01)
    checks.append((chg["duration_convexity_error"] < chg["duration_error"],
                   "duration+convexity approximation better than duration alone"))

    passed = sum(1 for ok, _ in checks if ok)
    for ok, label in checks:
        print("{:>5}  {}".format("PASS" if ok else "FAIL", label))
    print("-" * 56)
    print("{}/{} checks passed".format(passed, len(checks)))
    if passed != len(checks):
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub    = parser.add_subparsers(dest="command")

    p = sub.add_parser("diff", help="differentiate a 1D function")
    p.add_argument("--f", required=True)
    p.add_argument("--x", required=True)
    p.add_argument("--method", default="central",
                   choices=["forward", "backward", "central", "five-point"])
    p.add_argument("--h", default=None)
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser("gradient", help="gradient of a multivariate function")
    p.add_argument("--f", required=True)
    p.add_argument("--x", required=True)
    p.add_argument("--method", default="central",
                   choices=["forward", "backward", "central", "five-point"])
    p.add_argument("--h", default=None)
    p.set_defaults(func=cmd_gradient)

    p = sub.add_parser("second", help="second derivative of a 1D function")
    p.add_argument("--f", required=True)
    p.add_argument("--x", required=True)
    p.add_argument("--h", default=None)
    p.set_defaults(func=cmd_second)

    p = sub.add_parser("error", help="approximation error vs step size table")
    p.add_argument("--f", required=True)
    p.add_argument("--x", required=True)
    p.add_argument("--exact", required=True, type=float)
    p.add_argument("--method", default="central",
                   choices=["forward", "backward", "central", "five-point"])
    p.set_defaults(func=cmd_error)

    p = sub.add_parser("product", help="verify the product rule")
    p.add_argument("--f", required=True)
    p.add_argument("--g", required=True)
    p.add_argument("--x", required=True)
    p.set_defaults(func=cmd_product)

    p = sub.add_parser("chain", help="verify the chain rule")
    p.add_argument("--f", required=True)
    p.add_argument("--g", required=True)
    p.add_argument("--x", required=True)
    p.set_defaults(func=cmd_chain)

    p = sub.add_parser("descent", help="gradient descent to find a minimum")
    p.add_argument("--f", required=True)
    p.add_argument("--x", required=True)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--iters", type=int, default=10000)
    p.set_defaults(func=cmd_descent)

    p = sub.add_parser("bond", help="bond price, duration, and convexity")
    p.add_argument("--yield", dest="yield_rate", type=float, required=True)
    p.add_argument("--coupon", type=float, required=True)
    p.add_argument("--face", type=float, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--dy", type=float, default=None,
                   help="yield shift for price change approximation")
    p.set_defaults(func=cmd_bond)

    p = sub.add_parser("verify", help="cross-check all results against exact values")
    p.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
