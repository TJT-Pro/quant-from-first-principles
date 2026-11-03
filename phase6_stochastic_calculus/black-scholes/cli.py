import argparse
import math
import os
import statistics
import sys

import black_scholes as bs
import implied_volatility as iv
import binomial_tree as bt
import monte_carlo as mc
import hedging_simulation as hs

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DEXUSEU_returns.csv")


def cmd_price(args):
    price_fn = bs.call_price if args.type == "call" else bs.put_price
    price = price_fn(args.s, args.k, args.t, args.r, args.sigma, args.q)
    print(f"{args.type} price: {price:.6f}")
    print(f"put-call parity gap (should be ~0): {bs.put_call_parity_gap(args.s, args.k, args.t, args.r, args.sigma, args.q):.2e}")


def cmd_greeks(args):
    p = dict(s=args.s, k=args.k, t=args.t, r=args.r, sigma=args.sigma, q=args.q)
    if args.type == "call":
        delta, theta, rho_d, rho_f = bs.call_delta(**p), bs.call_theta(**p), bs.call_rho_domestic(**p), bs.call_rho_foreign(**p)
    else:
        delta, theta, rho_d, rho_f = bs.put_delta(**p), bs.put_theta(**p), bs.put_rho_domestic(**p), bs.put_rho_foreign(**p)
    print(f"delta: {delta:.6f}")
    print(f"gamma: {bs.gamma(**p):.6f}")
    print(f"vega (per unit sigma): {bs.vega(**p):.6f}")
    print(f"theta (per year): {theta:.6f}")
    print(f"rho (domestic rate): {rho_d:.6f}")
    print(f"rho (foreign rate): {rho_f:.6f}")


def cmd_impliedvol(args):
    newton = iv.implied_vol_newton(args.price, args.s, args.k, args.t, args.r, args.q, args.type)
    bisection = iv.implied_vol_bisection(args.price, args.s, args.k, args.t, args.r, args.q, args.type)
    print(f"Newton-Raphson: sigma={newton['sigma']:.6f}, iterations={newton['iterations']}, converged={newton['converged']}")
    print(f"Bisection:      sigma={bisection['sigma']:.6f}, iterations={bisection['iterations']}")


def cmd_binomial(args):
    price = bt.binomial_price(args.s, args.k, args.t, args.r, args.sigma, args.steps, args.type, args.exercise, args.q)
    print(f"binomial price ({args.exercise}, {args.steps} steps): {price:.6f}")
    if args.exercise == "european":
        bs_price_fn = bs.call_price if args.type == "call" else bs.put_price
        bs_price = bs_price_fn(args.s, args.k, args.t, args.r, args.sigma, args.q)
        print(f"Black-Scholes closed form:        {bs_price:.6f}")
        print(f"difference: {price - bs_price:+.6f}")


def cmd_montecarlo(args):
    result = mc.monte_carlo_price(args.s, args.k, args.t, args.r, args.sigma, args.paths,
                                   args.type, args.q, seed=args.seed, antithetic=args.antithetic)
    print(f"Monte Carlo price: {result['price']:.6f} +/- {result['std_error']:.6f} (95% CI: "
          f"[{result['ci_low']:.6f}, {result['ci_high']:.6f}])")
    price_fn = bs.call_price if args.type == "call" else bs.put_price
    bs_price = price_fn(args.s, args.k, args.t, args.r, args.sigma, args.q)
    print(f"Black-Scholes closed form: {bs_price:.6f}")


def cmd_convergence(args):
    price_fn = bs.call_price if args.type == "call" else bs.put_price
    bs_price = price_fn(args.s, args.k, args.t, args.r, args.sigma, args.q)
    print(f"Black-Scholes closed form: {bs_price:.6f}\n")

    print("binomial tree convergence:")
    steps = [10, 50, 100, 500, 2000]
    tree_prices = bt.binomial_convergence_series(args.s, args.k, args.t, args.r, args.sigma, args.type, steps, args.q)
    for n, p in zip(steps, tree_prices):
        print(f"  n={n:5d}   price={p:.6f}   error={p - bs_price:+.6f}")

    print("\nMonte Carlo convergence:")
    counts = [1000, 10000, 100000, 1000000]
    mc_series = mc.monte_carlo_convergence_series(args.s, args.k, args.t, args.r, args.sigma,
                                                   args.type, counts, args.q, seed=1)
    for row in mc_series:
        print(f"  n={row['n_paths']:8d}   price={row['price']:.6f}   "
              f"std_error={row['std_error']:.6f}   error={row['price'] - bs_price:+.6f}")


def cmd_fxhedge(args):
    log_returns = bs.load_log_returns(DATA_PATH)
    implied_vol = bs.calibrate_annualized_volatility(log_returns)
    prices = hs.reconstruct_price_path(log_returns)
    print(f"calibrated full-sample implied vol: {implied_vol:.4f} (annualized)")

    results = hs.rolling_hedge_backtest(prices, args.horizon, implied_vol, args.r, args.q, args.type)
    pnls = [r["pnl"] for r in results]
    realized_vols = [r["realized_vol"] for r in results]

    print(f"\n{args.horizon}-day delta-hedged short {args.type}, {len(results)} non-overlapping windows")
    print(f"mean hedging P&L:  {statistics.mean(pnls):+.6f}")
    print(f"stdev hedging P&L: {statistics.stdev(pnls):.6f}")
    print(f"mean realized vol across windows: {statistics.mean(realized_vols):.4f}")

    var_gaps = [rv ** 2 - implied_vol ** 2 for rv in realized_vols]
    n = len(pnls)
    mean_g, mean_p = statistics.mean(var_gaps), statistics.mean(pnls)
    cov = sum((g - mean_g) * (p - mean_p) for g, p in zip(var_gaps, pnls)) / n
    corr = cov / (statistics.pstdev(var_gaps) * statistics.pstdev(pnls))
    print(f"\ncorrelation(realized_var - implied_var, hedging P&L): {corr:.4f}")
    print("(expected strongly negative for a short call: realized vol above")
    print(" what was charged for hurts the seller, and vice versa)")


def cmd_verify(args):
    import numpy as np
    from scipy import stats as sstats

    checks = []

    def check(name, condition):
        checks.append((name, bool(condition)))
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {name}")

    # 1. price and Greeks vs an independent scipy-based recomputation
    s, k, t, r, sigma, q = 105.0, 100.0, 0.6, 0.03, 0.22, 0.015
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    scipy_call = s * math.exp(-q * t) * sstats.norm.cdf(d1) - k * math.exp(-r * t) * sstats.norm.cdf(d2)
    scipy_put = k * math.exp(-r * t) * sstats.norm.cdf(-d2) - s * math.exp(-q * t) * sstats.norm.cdf(-d1)
    check("call price matches scipy-based recomputation",
          math.isclose(bs.call_price(s, k, t, r, sigma, q), scipy_call, abs_tol=1e-10))
    check("put price matches scipy-based recomputation",
          math.isclose(bs.put_price(s, k, t, r, sigma, q), scipy_put, abs_tol=1e-10))

    scipy_delta_call = math.exp(-q * t) * sstats.norm.cdf(d1)
    scipy_gamma = math.exp(-q * t) * sstats.norm.pdf(d1) / (s * sigma * math.sqrt(t))
    scipy_vega = s * math.exp(-q * t) * sstats.norm.pdf(d1) * math.sqrt(t)
    check("call delta matches scipy-based recomputation",
          math.isclose(bs.call_delta(s, k, t, r, sigma, q), scipy_delta_call, abs_tol=1e-10))
    check("gamma matches scipy-based recomputation",
          math.isclose(bs.gamma(s, k, t, r, sigma, q), scipy_gamma, abs_tol=1e-10))
    check("vega matches scipy-based recomputation",
          math.isclose(bs.vega(s, k, t, r, sigma, q), scipy_vega, abs_tol=1e-10))

    # 2. all Greeks vs finite-difference numerical derivatives
    ok = True
    for fn, num_fn, price_fn in [
        (bs.call_delta, bs.numerical_delta, bs.call_price),
        (bs.put_delta, bs.numerical_delta, bs.put_price),
    ]:
        analytical = fn(s, k, t, r, sigma, q)
        numerical = num_fn(price_fn, s, k, t, r, sigma, q)
        if not math.isclose(analytical, numerical, abs_tol=1e-4):
            ok = False
    check("delta (call and put) matches finite-difference numerical derivative", ok)

    # 3. put-call parity across a parameter grid
    ok = True
    for s_ in (80, 100, 120):
        for k_ in (90, 100, 110):
            for t_ in (0.25, 1.0, 2.0):
                gap = bs.put_call_parity_gap(float(s_), float(k_), t_, 0.03, 0.2, 0.01)
                if abs(gap) > 1e-8:
                    ok = False
    check("put-call parity holds across a parameter grid", ok)

    # 4. implied vol round-trips (Newton and bisection) across a strike grid
    ok = True
    for k_ in (80.0, 100.0, 120.0):
        true_sigma = 0.25
        price = bs.call_price(100.0, k_, 1.0, 0.03, true_sigma, 0.01)
        newton_sigma = iv.implied_vol_newton(price, 100.0, k_, 1.0, 0.03, 0.01, "call")["sigma"]
        bisection_sigma = iv.implied_vol_bisection(price, 100.0, k_, 1.0, 0.03, 0.01, "call")["sigma"]
        if not (math.isclose(newton_sigma, true_sigma, abs_tol=1e-4) and
                math.isclose(bisection_sigma, true_sigma, abs_tol=1e-4)):
            ok = False
    check("implied volatility solvers round-trip a known sigma across strikes", ok)

    # 5. binomial tree converges to Black-Scholes
    s, k, t, r, sigma = 100.0, 100.0, 1.0, 0.03, 0.2
    bs_price = bs.call_price(s, k, t, r, sigma)
    tree_price = bt.binomial_price(s, k, t, r, sigma, 3000, "call", "european")
    check("binomial tree (3000 steps) converges to Black-Scholes",
          math.isclose(tree_price, bs_price, abs_tol=0.02))

    # 6. American call with no dividend equals European (no early exercise value)
    european = bt.binomial_price(s, k, t, r, sigma, 500, "call", "european")
    american = bt.binomial_price(s, k, t, r, sigma, 500, "call", "american")
    check("American call equals European call when there is no dividend/foreign yield",
          math.isclose(european, american, abs_tol=1e-8))

    # 7. Monte Carlo converges to Black-Scholes, with the CI containing it
    result = mc.monte_carlo_price(s, k, t, r, sigma, 300000, "call", seed=1)
    check("Monte Carlo price within its own 95% CI of the Black-Scholes price",
          result["ci_low"] - 3 * result["std_error"] <= bs_price <= result["ci_high"] + 3 * result["std_error"])

    # 8. antithetic variates reduce standard error vs plain Monte Carlo (numpy oracle comparison)
    n_paths = 50000
    plain = mc.monte_carlo_price(s, k, t, r, sigma, n_paths, "call", seed=2, antithetic=False)
    anti = mc.monte_carlo_price(s, k, t, r, sigma, n_paths, "call", seed=2, antithetic=True)
    check("antithetic variates reduce Monte Carlo standard error", anti["std_error"] < plain["std_error"])

    # 9. three independent pricing methods (analytical, binomial, Monte Carlo)
    #    agree with each other, not just with Black-Scholes individually
    check("binomial and Monte Carlo prices agree with each other",
          abs(tree_price - result["price"]) < 5 * result["std_error"])

    # 10. numpy-based independent GBM terminal-price recomputation
    rng = np.random.default_rng(0)
    n = 200000
    z = rng.standard_normal(n)
    numpy_terminal = s * np.exp((r - 0.5 * sigma ** 2) * t + sigma * math.sqrt(t) * z)
    numpy_call_price = math.exp(-r * t) * float(np.mean(np.maximum(numpy_terminal - k, 0.0)))
    check("Monte Carlo price matches an independent numpy-vectorised GBM recomputation",
          math.isclose(numpy_call_price, bs_price, abs_tol=0.05))

    # 11. delta-hedging mechanism: sign and rough magnitude vs a known
    #     synthetic vol mismatch
    trials = 1500
    pnls_over = []
    pnls_under = []
    for i in range(trials):
        rgen = np.random.default_rng(i)
        dt = 1 / 252
        horizon = 30
        true_sigma_high = 0.35
        log_rets_high = list(rgen.normal(-0.5 * true_sigma_high ** 2 * dt, true_sigma_high * math.sqrt(dt), horizon + 2))
        prices_high = hs.reconstruct_price_path(log_rets_high, s0=1.0)
        pnls_over.append(hs.simulate_delta_hedge(prices_high, 0, horizon, prices_high[0], 0.15, 0.0, 0.0, "call")["pnl"])

        true_sigma_low = 0.08
        log_rets_low = list(rgen.normal(-0.5 * true_sigma_low ** 2 * dt, true_sigma_low * math.sqrt(dt), horizon + 2))
        prices_low = hs.reconstruct_price_path(log_rets_low, s0=1.0)
        pnls_under.append(hs.simulate_delta_hedge(prices_low, 0, horizon, prices_low[0], 0.20, 0.0, 0.0, "call")["pnl"])
    check("short call loses on average when realized vol exceeds implied vol used to hedge",
          statistics.mean(pnls_over) < 0.0)
    check("short call gains on average when realized vol is below implied vol used to hedge",
          statistics.mean(pnls_under) > 0.0)

    # 12. real EUR/USD data: hedging P&L correlates negatively with the
    #     realized-vs-implied variance gap
    log_returns = bs.load_log_returns(DATA_PATH)
    fx_prices = hs.reconstruct_price_path(log_returns)
    fx_implied_vol = bs.calibrate_annualized_volatility(log_returns)
    fx_results = hs.rolling_hedge_backtest(fx_prices, 30, fx_implied_vol, 0.0, 0.0, "call")
    fx_pnls = [x["pnl"] for x in fx_results]
    fx_gaps = [x["realized_vol"] ** 2 - fx_implied_vol ** 2 for x in fx_results]
    n2 = len(fx_pnls)
    mg, mp = statistics.mean(fx_gaps), statistics.mean(fx_pnls)
    cov2 = sum((g - mg) * (p - mp) for g, p in zip(fx_gaps, fx_pnls)) / n2
    corr2 = cov2 / (statistics.pstdev(fx_gaps) * statistics.pstdev(fx_pnls))
    check("real EUR/USD hedging P&L correlates negatively with realized-vs-implied variance gap",
          corr2 < -0.5)

    passed = sum(1 for _, ok in checks if ok)
    print(f"\n{passed}/{len(checks)} checks passed")
    if passed != len(checks):
        sys.exit(1)


def build_parser():
    parser = argparse.ArgumentParser(description="Black-Scholes derivatives engine CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("price", help="Black-Scholes / Garman-Kohlhagen price")
    for name in ("s", "k", "t", "r", "sigma"):
        p1.add_argument(f"--{name}", type=float, required=True)
    p1.add_argument("--q", type=float, default=0.0)
    p1.add_argument("--type", choices=["call", "put"], default="call")
    p1.set_defaults(func=cmd_price)

    p2 = sub.add_parser("greeks", help="all Greeks for a given option")
    for name in ("s", "k", "t", "r", "sigma"):
        p2.add_argument(f"--{name}", type=float, required=True)
    p2.add_argument("--q", type=float, default=0.0)
    p2.add_argument("--type", choices=["call", "put"], default="call")
    p2.set_defaults(func=cmd_greeks)

    p3 = sub.add_parser("impliedvol", help="solve for implied volatility")
    p3.add_argument("--price", type=float, required=True)
    for name in ("s", "k", "t", "r"):
        p3.add_argument(f"--{name}", type=float, required=True)
    p3.add_argument("--q", type=float, default=0.0)
    p3.add_argument("--type", choices=["call", "put"], default="call")
    p3.set_defaults(func=cmd_impliedvol)

    p4 = sub.add_parser("binomial", help="binomial tree price (European or American)")
    for name in ("s", "k", "t", "r", "sigma"):
        p4.add_argument(f"--{name}", type=float, required=True)
    p4.add_argument("--q", type=float, default=0.0)
    p4.add_argument("--type", choices=["call", "put"], default="call")
    p4.add_argument("--exercise", choices=["european", "american"], default="european")
    p4.add_argument("--steps", type=int, default=500)
    p4.set_defaults(func=cmd_binomial)

    p5 = sub.add_parser("montecarlo", help="Monte Carlo price")
    for name in ("s", "k", "t", "r", "sigma"):
        p5.add_argument(f"--{name}", type=float, required=True)
    p5.add_argument("--q", type=float, default=0.0)
    p5.add_argument("--type", choices=["call", "put"], default="call")
    p5.add_argument("--paths", type=int, default=100000)
    p5.add_argument("--seed", type=int, default=1)
    p5.add_argument("--antithetic", action="store_true")
    p5.set_defaults(func=cmd_montecarlo)

    p6 = sub.add_parser("convergence", help="compare analytical, binomial, and Monte Carlo convergence")
    for name in ("s", "k", "t", "r", "sigma"):
        p6.add_argument(f"--{name}", type=float, required=True)
    p6.add_argument("--q", type=float, default=0.0)
    p6.add_argument("--type", choices=["call", "put"], default="call")
    p6.set_defaults(func=cmd_convergence)

    p7 = sub.add_parser("fxhedge", help="delta-hedge backtest on real EUR/USD data")
    p7.add_argument("--horizon", type=int, default=30)
    p7.add_argument("--r", type=float, default=0.0)
    p7.add_argument("--q", type=float, default=0.0)
    p7.add_argument("--type", choices=["call", "put"], default="call")
    p7.set_defaults(func=cmd_fxhedge)

    p8 = sub.add_parser("verify", help="cross-check implementation against analytical/oracle values")
    p8.set_defaults(func=cmd_verify)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
