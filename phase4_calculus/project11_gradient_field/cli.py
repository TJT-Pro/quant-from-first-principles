"""Command-line interface for the gradient field analyser.

Usage
-----
    python3 cli.py partial   --f EXPR --point X,Y[,Z] --i I
    python3 cli.py gradient  --f EXPR --point X,Y[,Z]
    python3 cli.py directional --f EXPR --point X,Y --dir DX,DY
    python3 cli.py hessian   --f EXPR --point X,Y
    python3 cli.py classify  --f EXPR --point X,Y
    python3 cli.py descent   --f EXPR --x0 X --y0 Y [--lr LR] [--iters N]
    python3 cli.py compare   --f EXPR --x0 X --y0 Y --rates R1,R2,...
    python3 cli.py field     --f EXPR --xmin A --xmax B --ymin C --ymax D
    python3 cli.py divergence --fx EXPR --fy EXPR --point X,Y
    python3 cli.py curl      --fx EXPR --fy EXPR --point X,Y
    python3 cli.py verify

Function expressions use x, y, z as variables with standard math
functions: sin, cos, exp, log, sqrt, pi, e, abs, tan.

Examples
--------
    python3 cli.py partial   --f "x**2+y**2" --point 3,4 --i 0
    python3 cli.py gradient  --f "sin(x)*cos(y)" --point 0.5,1.0
    python3 cli.py directional --f "x**2+y**2" --point 3,4 --dir 1,0
    python3 cli.py hessian   --f "x**2+y**2" --point 1,1
    python3 cli.py classify  --f "x**2-y**2" --point 0,0
    python3 cli.py descent   --f "x**2+y**2" --x0 3 --y0 4 --lr 0.1
    python3 cli.py compare   --f "x**2+y**2" --x0 5 --y0 5 --rates 0.01,0.1,0.5
    python3 cli.py field     --f "x**2+y**2" --xmin -2 --xmax 2 --ymin -2 --ymax 2
    python3 cli.py divergence --fx "x" --fy "y" --point 1,2
    python3 cli.py curl      --fx "-y" --fy "x" --point 1,2
    python3 cli.py verify
"""

import argparse
import math
import sys

import gradient_field as gf


# ---------------------------------------------------------------------------
# Expression evaluation
# ---------------------------------------------------------------------------

_SAFE = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
    "abs": abs, "pi": math.pi, "e": math.e,
    "sinh": math.sinh, "cosh": math.cosh,
}


def make_f2(expr):
    """Build a 2D function f([x, y]) from a string expression."""
    def f(p):
        env = dict(_SAFE)
        env["x"] = p[0]
        env["y"] = p[1]
        try:
            return float(eval(expr, {"__builtins__": {}}, env))
        except Exception as err:
            sys.exit("error evaluating {!r}: {}".format(expr, err))
    return f


def make_f3(expr):
    """Build a 3D function f([x, y, z]) from a string expression."""
    def f(p):
        env = dict(_SAFE)
        env["x"] = p[0]
        env["y"] = p[1]
        env["z"] = p[2] if len(p) > 2 else 0.0
        try:
            return float(eval(expr, {"__builtins__": {}}, env))
        except Exception as err:
            sys.exit("error evaluating {!r}: {}".format(expr, err))
    return f


def parse_point(s):
    try:
        return [float(v) for v in s.split(",")]
    except Exception:
        sys.exit("cannot parse point {!r} -- use comma-separated numbers".format(s))


def parse_floats(s):
    try:
        return [float(v) for v in s.split(",")]
    except Exception:
        sys.exit("cannot parse {!r} as a list of numbers".format(s))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_partial(args):
    f     = make_f3(args.f)
    point = parse_point(args.point)
    i     = args.i
    val   = gf.partial_derivative(f, point, i)
    print("df/dx_{} at {} = {:.12g}".format(i, point, val))


def cmd_gradient(args):
    f     = make_f3(args.f)
    point = parse_point(args.point)
    g     = gf.gradient(f, point)
    mag   = math.sqrt(sum(gi*gi for gi in g))
    print("∇f at {} = {}".format(point, ["{:.10g}".format(gi) for gi in g]))
    print("||∇f|| = {:.10g}".format(mag))


def cmd_directional(args):
    f     = make_f2(args.f)
    point = parse_point(args.point)
    direc = parse_floats(args.dir)
    dd    = gf.directional_derivative(f, point, direc)
    print("D_v f at {} in direction {} = {:.12g}".format(point, direc, dd))


def cmd_hessian(args):
    f     = make_f2(args.f)
    point = parse_point(args.point)
    H     = gf.hessian(f, point)
    lap   = gf.laplacian(f, point)
    print("Hessian at {}:".format(point))
    for row in H:
        print("  [{}]".format("  ".join("{:12.8g}".format(v) for v in row)))
    print("Laplacian: {:.10g}".format(lap))


def cmd_classify(args):
    f      = make_f2(args.f)
    point  = parse_point(args.point)
    result = gf.classify_critical_2d(f, point)
    print("Critical point analysis at {}:".format(point))
    print("  fxx = {:.8g}".format(result["fxx"]))
    print("  fyy = {:.8g}".format(result["fyy"]))
    print("  fxy = {:.8g}".format(result["fxy"]))
    print("  det = {:.8g}".format(result["det"]))
    print("  ||∇f|| = {:.4e}".format(result["grad_norm"]))
    print("  → {}".format(result["classification"].upper()))


def cmd_descent(args):
    f      = make_f2(args.f)
    result = gf.gradient_descent_2d(
        f, args.x0, args.y0,
        learning_rate=args.lr,
        max_iterations=args.iters,
        tolerance=args.tol,
    )
    print("Gradient descent on '{}' from ({}, {})".format(args.f, args.x0, args.y0))
    print("  learning rate:  {}".format(args.lr))
    print("  iterations:     {}".format(result["iterations"]))
    print("  converged:      {}".format(result["converged"]))
    print("  final x:        {:.10g}".format(result["x"]))
    print("  final y:        {:.10g}".format(result["y"]))
    print("  final f(x,y):   {:.10g}".format(result["f_val"]))
    if args.verbose and result["trajectory"]:
        print("  trajectory (every 10th step):")
        for k, (x, y, fv) in enumerate(result["trajectory"]):
            if k % 10 == 0 or k == len(result["trajectory"]) - 1:
                print("    step {:4d}  ({:.6g}, {:.6g})  f={:.6g}".format(k, x, y, fv))


def cmd_compare(args):
    f       = make_f2(args.f)
    rates   = parse_floats(args.rates)
    results = gf.compare_learning_rates(
        f, args.x0, args.y0, rates,
        max_iterations=args.iters,
        tolerance=args.tol,
    )
    print("Learning rate comparison for '{}' from ({}, {})".format(
        args.f, args.x0, args.y0))
    print("{:>10}  {:>10}  {:>12}  {:>10}  {:>12}".format(
        "rate", "iters", "converged", "f_val", "grad_norm"))
    for r in results:
        print("{:>10.4g}  {:>10d}  {:>12}  {:>10.6g}  {:>12.4e}".format(
            r["learning_rate"],
            r["iterations"],
            str(r["converged"]),
            r["f_val"],
            r["final_grad_norm"],
        ))


def cmd_field(args):
    f    = make_f2(args.f)
    data = gf.gradient_field_grid(
        f,
        (args.xmin, args.xmax),
        (args.ymin, args.ymax),
        nx=args.nx, ny=args.ny,
    )
    print("Gradient field of '{}' on [{},{}] x [{},{}]".format(
        args.f, args.xmin, args.xmax, args.ymin, args.ymax))
    print("  grid: {}×{}".format(args.nx, args.ny))
    mags = data["magnitudes"]
    flat = [mags[j][i] for j in range(len(mags)) for i in range(len(mags[0]))]
    print("  ||∇f|| min: {:.6g}".format(min(flat)))
    print("  ||∇f|| max: {:.6g}".format(max(flat)))
    print("  ||∇f|| mean:{:.6g}".format(sum(flat) / len(flat)))


def cmd_divergence(args):
    fx    = make_f2(args.fx)
    fy    = make_f2(args.fy)
    point = parse_point(args.point)
    div   = gf.divergence(fx, fy, point)
    print("div(F) at {} = {:.12g}".format(point, div))


def cmd_curl(args):
    fx    = make_f2(args.fx)
    fy    = make_f2(args.fy)
    point = parse_point(args.point)
    curl  = gf.curl_z(fx, fy, point)
    print("curl_z(F) at {} = {:.12g}".format(point, curl))


def cmd_verify(args):
    import math as _m

    checks = []

    # Partial derivatives
    f2 = lambda p: p[0]**2 + p[1]**2

    val = gf.partial_derivative(f2, [3.0, 4.0], 0)
    checks.append((abs(val - 6.0) < 1e-7, "df/dx of x²+y² at (3,4) = 6"))

    val = gf.partial_derivative(f2, [3.0, 4.0], 1)
    checks.append((abs(val - 8.0) < 1e-7, "df/dy of x²+y² at (3,4) = 8"))

    val = gf.partial_derivative(lambda p: _m.sin(p[0])*_m.cos(p[1]),
                                 [0.5, 1.0], 0)
    checks.append((abs(val - _m.cos(0.5)*_m.cos(1.0)) < 1e-7,
                   "d/dx sin(x)cos(y) at (0.5,1)"))

    # 3D partial derivatives
    f3 = lambda p: p[0]**2 + 2*p[1]**2 + 3*p[2]**2
    for i, expected in enumerate([2.0, 8.0, 18.0]):
        val = gf.partial_derivative(f3, [1.0, 2.0, 3.0], i)
        checks.append((abs(val - expected) < 1e-6,
                       "3D partial d/dx_{} of x²+2y²+3z² at (1,2,3)".format(i)))

    # Gradient
    g = gf.gradient(f2, [3.0, 4.0])
    checks.append((abs(g[0] - 6.0) < 1e-7 and abs(g[1] - 8.0) < 1e-7,
                   "gradient of x²+y² at (3,4) = [6,8]"))

    mag = gf.gradient_magnitude(f2, [3.0, 4.0])
    checks.append((abs(mag - 10.0) < 1e-6, "||∇(x²+y²)|| at (3,4) = 10"))

    # Directional derivative
    dd = gf.directional_derivative(f2, [3.0, 4.0], [1.0, 0.0])
    checks.append((abs(dd - 6.0) < 1e-6,
                   "D_x (x²+y²) at (3,4) = 6"))

    dd_grad = gf.directional_derivative(f2, [3.0, 4.0], [6.0, 8.0])
    checks.append((abs(dd_grad - 10.0) < 1e-5,
                   "D_∇f f at (3,4) = ||∇f|| = 10"))

    # Second partials and Hessian
    d2 = gf.second_partial(f2, [1.0, 2.0], 0, 0)
    checks.append((abs(d2 - 2.0) < 1e-5, "d²/dx² (x²+y²) = 2"))

    d2mix = gf.second_partial(f2, [1.0, 2.0], 0, 1)
    checks.append((abs(d2mix) < 1e-5, "d²/dxdy (x²+y²) = 0"))

    d2xy = gf.second_partial(lambda p: p[0]*p[1], [1.0, 2.0], 0, 1)
    checks.append((abs(d2xy - 1.0) < 1e-5, "d²/dxdy (xy) = 1"))

    # Laplacian
    lap_quad = gf.laplacian(f2, [3.0, 4.0])
    checks.append((abs(lap_quad - 4.0) < 1e-5, "∇²(x²+y²) = 4"))

    lap_saddle = gf.laplacian(lambda p: p[0]**2 - p[1]**2, [1.0, 2.0])
    checks.append((abs(lap_saddle) < 1e-5, "∇²(x²-y²) = 0 (harmonic)"))

    # Critical point classification
    r_min = gf.classify_critical_2d(f2, [0.0, 0.0])
    checks.append((r_min["classification"] == "minimum",
                   "x²+y² at (0,0) classified as minimum"))

    r_saddle = gf.classify_critical_2d(lambda p: p[0]**2 - p[1]**2, [0.0, 0.0])
    checks.append((r_saddle["classification"] == "saddle",
                   "x²-y² at (0,0) classified as saddle"))

    r_max = gf.classify_critical_2d(lambda p: -(p[0]**2 + p[1]**2), [0.0, 0.0])
    checks.append((r_max["classification"] == "maximum",
                   "-(x²+y²) at (0,0) classified as maximum"))

    # Gradient descent
    res = gf.gradient_descent_2d(f2, 3.0, 4.0, learning_rate=0.1, tolerance=1e-8)
    checks.append((res["converged"], "gradient descent on x²+y² converges"))
    checks.append((abs(res["x"]) < 1e-4 and abs(res["y"]) < 1e-4,
                   "gradient descent reaches near origin"))
    checks.append((res["f_val"] < 1e-8,
                   "gradient descent f_val near zero at minimum"))

    # Learning rate comparison: faster rate → fewer iterations on quadratic
    results = gf.compare_learning_rates(f2, 5.0, 5.0, [0.01, 0.1],
                                         max_iterations=2000, tolerance=1e-6)
    checks.append((results[0]["iterations"] > results[1]["iterations"],
                   "lr=0.1 converges faster than lr=0.01 on quadratic"))

    # Gradient field grid
    data = gf.gradient_field_grid(f2, (0, 2), (0, 2), nx=3, ny=3)
    # At (2, 2), gradient should be (4, 4)
    checks.append((abs(data["gx"][2][2] - 4.0) < 1e-5 and
                   abs(data["gy"][2][2] - 4.0) < 1e-5,
                   "gradient field at (2,2) = (4,4)"))

    # Divergence
    div = gf.divergence(lambda p: p[0], lambda p: p[1], [1.0, 2.0])
    checks.append((abs(div - 2.0) < 1e-6, "div(x, y) = 2"))

    div0 = gf.divergence(lambda p: 1.0, lambda p: 1.0, [0.0, 0.0])
    checks.append((abs(div0) < 1e-6, "div(1, 1) = 0"))

    # Curl
    curl_rot = gf.curl_z(lambda p: -p[1], lambda p: p[0], [1.0, 2.0])
    checks.append((abs(curl_rot - 2.0) < 1e-6, "curl_z(-y, x) = 2"))

    curl_irr = gf.curl_z(lambda p: 2*p[0], lambda p: 2*p[1], [1.0, 2.0])
    checks.append((abs(curl_irr) < 1e-6, "curl_z(∇f) = 0 (irrotational)"))

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

    p = sub.add_parser("partial", help="partial derivative")
    p.add_argument("--f", required=True)
    p.add_argument("--point", required=True)
    p.add_argument("--i", type=int, required=True)
    p.set_defaults(func=cmd_partial)

    p = sub.add_parser("gradient", help="gradient vector")
    p.add_argument("--f", required=True)
    p.add_argument("--point", required=True)
    p.set_defaults(func=cmd_gradient)

    p = sub.add_parser("directional", help="directional derivative")
    p.add_argument("--f", required=True)
    p.add_argument("--point", required=True)
    p.add_argument("--dir", required=True)
    p.set_defaults(func=cmd_directional)

    p = sub.add_parser("hessian", help="Hessian matrix and Laplacian")
    p.add_argument("--f", required=True)
    p.add_argument("--point", required=True)
    p.set_defaults(func=cmd_hessian)

    p = sub.add_parser("classify", help="classify a critical point")
    p.add_argument("--f", required=True)
    p.add_argument("--point", required=True)
    p.set_defaults(func=cmd_classify)

    p = sub.add_parser("descent", help="gradient descent with trajectory")
    p.add_argument("--f", required=True)
    p.add_argument("--x0", type=float, required=True)
    p.add_argument("--y0", type=float, required=True)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_descent)

    p = sub.add_parser("compare", help="compare learning rates")
    p.add_argument("--f", required=True)
    p.add_argument("--x0", type=float, required=True)
    p.add_argument("--y0", type=float, required=True)
    p.add_argument("--rates", required=True)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--tol", type=float, default=1e-8)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("field", help="gradient field on a grid")
    p.add_argument("--f", required=True)
    p.add_argument("--xmin", type=float, required=True)
    p.add_argument("--xmax", type=float, required=True)
    p.add_argument("--ymin", type=float, required=True)
    p.add_argument("--ymax", type=float, required=True)
    p.add_argument("--nx", type=int, default=20)
    p.add_argument("--ny", type=int, default=20)
    p.set_defaults(func=cmd_field)

    p = sub.add_parser("divergence", help="divergence of a vector field")
    p.add_argument("--fx", required=True)
    p.add_argument("--fy", required=True)
    p.add_argument("--point", required=True)
    p.set_defaults(func=cmd_divergence)

    p = sub.add_parser("curl", help="z-component of curl of a vector field")
    p.add_argument("--fx", required=True)
    p.add_argument("--fy", required=True)
    p.add_argument("--point", required=True)
    p.set_defaults(func=cmd_curl)

    p = sub.add_parser("verify", help="cross-check all results against exact values")
    p.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
