"""Command-line interface for the numerical integrator and Taylor series expander.

Usage
-----
    python3 cli.py integrate  --f EXPR --a A --b B [--method METHOD] [--n N]
    python3 cli.py compare    --f EXPR --a A --b B --exact EXACT [--n N]
    python3 cli.py adaptive   --f EXPR --a A --b B [--tol TOL]
    python3 cli.py gauss      --f EXPR --a A --b B [--n N]
    python3 cli.py improper   --f EXPR --a A --b B [--n N]
    python3 cli.py taylor     --f EXPR --x0 X0 --degree D [--at X,X,...]
    python3 cli.py normal-cdf --x X [--mu MU] [--sigma SIGMA]
    python3 cli.py bs-call    --S S --K K --r R --sigma SIGMA --T T
    python3 cli.py bond-dur   --yield Y --coupon C --T T
    python3 cli.py verify

Function expressions
--------------------
EXPR uses x as the variable with standard math functions available:
sin, cos, exp, log, sqrt, pi, e, abs, tan, sinh, cosh.

Use inf or -inf for infinite limits in the improper command.

Examples
--------
    python3 cli.py integrate --f "sin(x)" --a 0 --b "pi" --method simpsons --n 100
    python3 cli.py compare   --f "exp(x)" --a 0 --b 1 --exact 1.718281828459045
    python3 cli.py adaptive  --f "sin(x)" --a 0 --b "pi"
    python3 cli.py gauss     --f "exp(x)" --a 0 --b 1 --n 5
    python3 cli.py improper  --f "exp(-x**2)" --a 0 --b inf
    python3 cli.py taylor    --f "sin(x)" --x0 0 --degree 9 --at "-1,0,0.5,1"
    python3 cli.py normal-cdf --x 1.96
    python3 cli.py bs-call   --S 100 --K 100 --r 0.05 --sigma 0.2 --T 1
    python3 cli.py bond-dur  --yield 0.05 --coupon 0.06 --T 10
    python3 cli.py verify
"""

import argparse
import math
import sys

import integrator as ig


# ---------------------------------------------------------------------------
# Expression parser
# ---------------------------------------------------------------------------

_SAFE = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
    "abs": abs, "pi": math.pi, "e": math.e,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "inf": math.inf,
}


def make_f(expr):
    def f(x):
        env = dict(_SAFE)
        env["x"] = x
        try:
            return float(eval(expr, {"__builtins__": {}}, env))
        except Exception as err:
            sys.exit("error evaluating {!r} at x={}: {}".format(expr, x, err))
    return f


def parse_float(s):
    env = dict(_SAFE)
    try:
        return float(eval(s, {"__builtins__": {}}, env))
    except Exception:
        sys.exit("cannot parse {!r} as a number".format(s))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_integrate(args):
    f      = make_f(args.f)
    a      = parse_float(args.a)
    b      = parse_float(args.b)
    n      = args.n
    method = args.method

    methods = {
        "left":        ig.left_riemann,
        "right":       ig.right_riemann,
        "midpoint":    ig.midpoint_riemann,
        "trapezoidal": ig.trapezoidal,
        "simpsons":    ig.simpsons,
    }
    if method not in methods:
        sys.exit("unknown method {!r}".format(method))
    if method == "simpsons" and n % 2 != 0:
        n += 1

    result = methods[method](f, a, b, n)
    print("integral of {} from {} to {}".format(args.f, args.a, args.b))
    print("  method:   {}".format(method))
    print("  n:        {:,}".format(n))
    print("  result:   {:.15g}".format(result))


def cmd_compare(args):
    f     = make_f(args.f)
    a     = parse_float(args.a)
    b     = parse_float(args.b)
    exact = args.exact
    n     = args.n
    if n % 2 != 0:
        n += 1

    methods = [
        ("left",        ig.left_riemann),
        ("right",       ig.right_riemann),
        ("midpoint",    ig.midpoint_riemann),
        ("trapezoidal", ig.trapezoidal),
        ("simpsons",    ig.simpsons),
    ]
    print("integral of {} from {} to {}".format(args.f, args.a, args.b))
    print("exact = {:.15g}".format(exact))
    print("{:>14}  {:>20}  {:>14}".format("method", "estimate", "error"))
    for name, fn in methods:
        est = fn(f, a, b, n)
        err = abs(est - exact)
        print("{:>14}  {:>20.15g}  {:>14.3e}".format(name, est, err))


def cmd_adaptive(args):
    f   = make_f(args.f)
    a   = parse_float(args.a)
    b   = parse_float(args.b)
    tol = args.tol
    val, err = ig.adaptive_simpsons(f, a, b, tolerance=tol)
    print("adaptive Simpson's  {} from {} to {}".format(args.f, args.a, args.b))
    print("  tolerance:        {:.2e}".format(tol))
    print("  estimate:         {:.15g}".format(val))
    print("  error estimate:   {:.3e}".format(err))


def cmd_gauss(args):
    f = make_f(args.f)
    a = parse_float(args.a)
    b = parse_float(args.b)
    n = args.n
    result = ig.gauss_legendre(f, a, b, n)
    print("Gauss-Legendre  {} from {} to {}  n={}".format(args.f, args.a, args.b, n))
    print("  result: {:.15g}".format(result))


def cmd_improper(args):
    f = make_f(args.f)
    a = parse_float(args.a)
    b = parse_float(args.b)
    n = args.n
    if n % 2 != 0:
        n += 1
    result = ig.improper_integral(f, a, b, n)
    print("improper integral  {} from {} to {}".format(args.f, args.a, args.b))
    print("  n:      {:,}".format(n))
    print("  result: {:.15g}".format(result))


def cmd_taylor(args):
    f      = make_f(args.f)
    x0     = parse_float(args.x0)
    degree = args.degree
    coeffs = ig.taylor_coefficients(f, x0, degree)

    print("Taylor series of {} at x0={}  degree={}".format(args.f, x0, degree))
    print("Coefficients:")
    for k, c in enumerate(coeffs):
        print("  a_{:2d} = {:+.10g}".format(k, c))

    r = ig.radius_of_convergence(coeffs)
    if math.isinf(r):
        print("Radius of convergence: infinity")
    else:
        print("Radius of convergence estimate: {:.6g}".format(r))

    if args.at:
        xs = [parse_float(s.strip()) for s in args.at.split(",")]
        print("\n{:>10}  {:>18}  {:>18}  {:>12}".format(
            "x", "exact", "taylor approx", "error"))
        for x in xs:
            results = ig.taylor_error(f, coeffs, x0, [x])
            _, exact, approx, err = results[0]
            print("{:>10.4f}  {:>18.10g}  {:>18.10g}  {:>12.3e}".format(
                x, exact, approx, err))


def cmd_normal_cdf(args):
    x     = args.x
    mu    = args.mu
    sigma = args.sigma
    cdf   = ig.normal_cdf(x, mu=mu, sigma=sigma)
    exact = 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))
    print("Normal CDF  mu={} sigma={}  at x={}".format(mu, sigma, x))
    print("  numerical:  {:.12f}".format(cdf))
    print("  exact (erf):{:.12f}".format(exact))
    print("  error:      {:.3e}".format(abs(cdf - exact)))


def cmd_bs_call(args):
    C = ig.black_scholes_call(args.S, args.K, args.r, args.sigma, args.T)
    P = ig.black_scholes_put(args.S, args.K, args.r, args.sigma, args.T)
    print("Black-Scholes  S={} K={} r={} sigma={} T={}".format(
        args.S, args.K, args.r, args.sigma, args.T))
    print("  call price: {:.6f}".format(C))
    print("  put price:  {:.6f}".format(P))
    intrinsic = max(args.S - args.K, 0.0)
    print("  intrinsic:  {:.6f}  (call)".format(intrinsic))
    print("  time value: {:.6f}  (call - intrinsic)".format(C - intrinsic))


def cmd_bond_dur(args):
    result = ig.continuous_bond_duration(args.yield_rate, args.coupon, args.T)
    print("Continuous bond  yield={:.4%}  coupon={:.4%}  T={}".format(
        args.yield_rate, args.coupon, args.T))
    print("  price:    {:.8f}".format(result["price"]))
    print("  duration: {:.8f} years".format(result["duration"]))


def cmd_verify(args):
    import math as _math

    checks = []

    # Exact integrals to test against
    # integral of x^2 from 0 to 1 = 1/3
    exact_x2 = 1.0 / 3.0
    for fn, name in [
        (ig.left_riemann, "left"),
        (ig.right_riemann, "right"),
        (ig.midpoint_riemann, "midpoint"),
        (ig.trapezoidal, "trapezoidal"),
    ]:
        val = fn(lambda x: x*x, 0.0, 1.0, 1000)
        # left and right are O(h): error ~ h/2 ~ 5e-4 at n=1000
        # midpoint and trapezoidal are O(h^2): error ~ h^2/12 ~ 8e-8
        tol = 1e-3 if name in ("left", "right") else 1e-5
        checks.append((abs(val - exact_x2) < tol,
                       "{} integral of x^2 accurate".format(name)))

    val = ig.simpsons(lambda x: x*x, 0.0, 1.0, 1000)
    checks.append((abs(val - exact_x2) < 1e-10,
                   "simpsons integral of x^2 exact (polynomial)"))

    # Simpson's is exact for polynomials of degree <= 3
    # integral of x^3 from 0 to 1 = 1/4
    val3 = ig.simpsons(lambda x: x**3, 0.0, 1.0, 100)
    checks.append((abs(val3 - 0.25) < 1e-12,
                   "simpsons integral of x^3 exact"))

    # integral of sin(x) from 0 to pi = 2
    for fn, name, tol in [
        (ig.midpoint_riemann, "midpoint", 1e-5),
        (ig.trapezoidal, "trapezoidal", 1e-5),
        (ig.simpsons, "simpsons", 1e-10),
    ]:
        val = fn(_math.sin, 0.0, _math.pi, 1000)
        checks.append((abs(val - 2.0) < tol,
                       "{} integral of sin(x) from 0 to pi".format(name)))

    # integral of exp(x) from 0 to 1 = e - 1
    exact_exp = _math.e - 1.0
    val_exp = ig.simpsons(_math.exp, 0.0, 1.0, 100)
    checks.append((abs(val_exp - exact_exp) < 1e-9,
                   "simpsons integral of exp(x) from 0 to 1"))

    # convergence rates: midpoint and trapezoidal O(h^2), simpsons O(h^4)
    def rate(fn, f, a, b, n1, n2):
        e1 = abs(fn(f, a, b, n1) - 2.0)  # sin integral = 2
        e2 = abs(fn(f, a, b, n2) - 2.0)
        return e1 / e2 if e2 > 0 else float('inf')

    r_mid  = rate(ig.midpoint_riemann, _math.sin, 0.0, _math.pi, 100, 200)
    r_trap = rate(ig.trapezoidal,      _math.sin, 0.0, _math.pi, 100, 200)
    r_simp = rate(ig.simpsons,         _math.sin, 0.0, _math.pi, 100, 200)
    checks.append((3.5 < r_mid < 4.5,
                   "midpoint converges at O(h^2): ratio ~4"))
    checks.append((3.5 < r_trap < 4.5,
                   "trapezoidal converges at O(h^2): ratio ~4"))
    checks.append((14.0 < r_simp < 18.0,
                   "simpsons converges at O(h^4): ratio ~16"))

    # Gauss-Legendre exact for polynomials of degree <= 2n-1
    # n=3 should be exact for degree 5 poly: integral of x^5 from -1 to 1 = 0
    val_g = ig.gauss_legendre(lambda x: x**5, -1.0, 1.0, n=3)
    checks.append((abs(val_g) < 1e-14,
                   "gauss-legendre n=3 exact for x^5 (odd)"))
    # integral of x^4 from 0 to 1 = 1/5
    val_g4 = ig.gauss_legendre(lambda x: x**4, 0.0, 1.0, n=3)
    checks.append((abs(val_g4 - 0.2) < 1e-12,
                   "gauss-legendre n=3 exact for x^4"))

    # Adaptive Simpson's on sin(x)
    val_a, err_a = ig.adaptive_simpsons(_math.sin, 0.0, _math.pi, tolerance=1e-10)
    checks.append((abs(val_a - 2.0) < 1e-10,
                   "adaptive simpsons achieves 1e-10 tolerance on sin"))
    checks.append((err_a < 1e-10, "adaptive error estimate is below tolerance"))

    # Degenerate interval
    checks.append((ig.simpsons(lambda x: x, 3.0, 3.0, 10) == 0.0,
                   "integral over zero-width interval is 0"))

    # Improper integral: integral of exp(-x^2) from 0 to inf = sqrt(pi)/2
    exact_gauss = _math.sqrt(_math.pi) / 2.0
    val_imp = ig.improper_integral(lambda x: _math.exp(-x*x), 0.0, _math.inf, 2000)
    checks.append((abs(val_imp - exact_gauss) < 1e-4,
                   "improper integral of exp(-x^2) from 0 to inf"))

    # Improper integral: integral of exp(-x) from 0 to inf = 1
    val_exp_inf = ig.improper_integral(lambda x: _math.exp(-x), 0.0, _math.inf, 2000)
    checks.append((abs(val_exp_inf - 1.0) < 1e-4,
                   "improper integral of exp(-x) from 0 to inf = 1"))

    # Taylor series of sin(x) at x0=0
    # Coefficients: [0, 1, 0, -1/6, 0, 1/120, 0, -1/5040, 0, 1/362880]
    coeffs_sin = ig.taylor_coefficients(_math.sin, 0.0, 9)
    checks.append((abs(coeffs_sin[0]) < 1e-10, "sin Taylor a0 = 0"))
    checks.append((abs(coeffs_sin[1] - 1.0) < 1e-8, "sin Taylor a1 = 1"))
    checks.append((abs(coeffs_sin[2]) < 1e-8, "sin Taylor a2 = 0"))
    checks.append((abs(coeffs_sin[3] - (-1.0/6.0)) < 1e-6,
                   "sin Taylor a3 = -1/6"))
    checks.append((abs(coeffs_sin[5] - 1.0/120.0) < 1e-5,
                   "sin Taylor a5 = 1/120"))

    # Taylor series of exp(x) at x0=0: a_k = 1/k!
    coeffs_exp = ig.taylor_coefficients(_math.exp, 0.0, 6)
    fact = 1
    for k in range(7):
        if k > 0:
            fact *= k
        checks.append((abs(coeffs_exp[k] - 1.0/fact) < 1e-6,
                       "exp Taylor a{} = 1/{}!".format(k, k)))

    # Taylor evaluate via Horner matches direct evaluation
    coeffs_cos = ig.taylor_coefficients(_math.cos, 0.0, 8)
    for x_test in [-0.5, 0.0, 0.5, 1.0]:
        approx = ig.taylor_evaluate(coeffs_cos, 0.0, x_test)
        exact_cos = _math.cos(x_test)
        checks.append((abs(approx - exact_cos) < 1e-5,
                       "cos Taylor at x={} accurate".format(x_test)))

    # Radius of convergence of sin(x) should be large (converges everywhere)
    r = ig.radius_of_convergence(coeffs_sin)
    checks.append((r > 3.0, "sin Taylor radius of convergence > 3"))

    # Normal CDF matches math.erf
    for x_val in [-2.0, -1.0, 0.0, 1.0, 1.96, 2.0]:
        num  = ig.normal_cdf(x_val)
        exact_n = 0.5 * (1.0 + _math.erf(x_val / _math.sqrt(2.0)))
        checks.append((abs(num - exact_n) < 1e-8,
                       "normal CDF at x={} matches erf".format(x_val)))

    # Exponential CDF: compare against exact 1 - exp(-x)
    for x_val in [0.5, 1.0, 2.0, 3.0]:
        num   = ig.exponential_cdf(x_val, rate=1.0)
        exact_e = 1.0 - _math.exp(-x_val)
        checks.append((abs(num - exact_e) < 1e-8,
                       "exponential CDF at x={} accurate".format(x_val)))

    # Black-Scholes put-call parity: C - P = S - K*exp(-rT)
    S, K, r_bs, sig, T_bs = 100.0, 100.0, 0.05, 0.2, 1.0
    C = ig.black_scholes_call(S, K, r_bs, sig, T_bs)
    P = ig.black_scholes_put(S, K, r_bs, sig, T_bs)
    parity = C - P - (S - K * _math.exp(-r_bs * T_bs))
    checks.append((abs(parity) < 1e-8, "Black-Scholes put-call parity holds"))

    # ATM call is positive
    checks.append((C > 0.0, "ATM call price is positive"))

    # Deep ITM call approaches intrinsic
    C_deep = ig.black_scholes_call(200.0, 100.0, 0.05, 0.2, 1.0)
    checks.append((C_deep > 95.0, "deep ITM call close to intrinsic"))

    # Continuous bond duration: at yield=coupon, duration = (1 - exp(-y*T)) / y + T*exp(-y*T) / price
    res = ig.continuous_bond_duration(0.05, 0.05, 10.0)
    checks.append((0.0 < res["duration"] < 10.0,
                   "bond duration between 0 and maturity"))
    checks.append((res["price"] > 0.0, "bond price is positive"))

    # Present value of constant cash flow: PV = c/r * (1 - exp(-rT))
    c, r_pv, T_pv = 100.0, 0.05, 10.0
    pv_exact = c / r_pv * (1.0 - _math.exp(-r_pv * T_pv))
    pv_num   = ig.present_value_continuous(lambda t: c, r_pv, T_pv, 1000)
    checks.append((abs(pv_num - pv_exact) < 1e-6,
                   "PV of constant cash flow matches formula"))

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

    p = sub.add_parser("integrate", help="integrate using a chosen method")
    p.add_argument("--f", required=True)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--method", default="simpsons",
                   choices=["left", "right", "midpoint", "trapezoidal", "simpsons"])
    p.add_argument("--n", type=int, default=1000)
    p.set_defaults(func=cmd_integrate)

    p = sub.add_parser("compare", help="compare all methods at the same n")
    p.add_argument("--f", required=True)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--exact", type=float, required=True)
    p.add_argument("--n", type=int, default=1000)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("adaptive", help="adaptive Simpson's rule")
    p.add_argument("--f", required=True)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--tol", type=float, default=1e-8)
    p.set_defaults(func=cmd_adaptive)

    p = sub.add_parser("gauss", help="Gauss-Legendre quadrature")
    p.add_argument("--f", required=True)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--n", type=int, default=5,
                   choices=[2, 3, 4, 5, 6, 7, 10])
    p.set_defaults(func=cmd_gauss)

    p = sub.add_parser("improper", help="integrate over an infinite interval")
    p.add_argument("--f", required=True)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--n", type=int, default=10000)
    p.set_defaults(func=cmd_improper)

    p = sub.add_parser("taylor", help="Taylor series expansion")
    p.add_argument("--f", required=True)
    p.add_argument("--x0", required=True)
    p.add_argument("--degree", type=int, required=True)
    p.add_argument("--at", default=None,
                   help="comma-separated x values to evaluate the series at")
    p.set_defaults(func=cmd_taylor)

    p = sub.add_parser("normal-cdf", help="normal CDF by numerical integration")
    p.add_argument("--x", type=float, required=True)
    p.add_argument("--mu", type=float, default=0.0)
    p.add_argument("--sigma", type=float, default=1.0)
    p.set_defaults(func=cmd_normal_cdf)

    p = sub.add_parser("bs-call", help="Black-Scholes option pricing")
    p.add_argument("--S", type=float, required=True)
    p.add_argument("--K", type=float, required=True)
    p.add_argument("--r", type=float, required=True)
    p.add_argument("--sigma", type=float, required=True)
    p.add_argument("--T", type=float, required=True)
    p.set_defaults(func=cmd_bs_call)

    p = sub.add_parser("bond-dur", help="continuous bond duration by integration")
    p.add_argument("--yield", dest="yield_rate", type=float, required=True)
    p.add_argument("--coupon", type=float, required=True)
    p.add_argument("--T", type=float, required=True)
    p.set_defaults(func=cmd_bond_dur)

    p = sub.add_parser("verify", help="cross-check all results against exact values")
    p.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
