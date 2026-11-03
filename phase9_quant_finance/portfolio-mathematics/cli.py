import argparse
import math
import os
import sys

import mean_variance as mv
import tangency_portfolio as tp
import capm
import risk_decomposition as rd
import kelly_criterion as kc
import fx_application as fx

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DEXUSEU_returns.csv")


def cmd_frontier(args):
    mu = [float(x) for x in args.mu.split(",")]
    sigma_flat = [float(x) for x in args.sigma.split(",")]
    n = len(mu)
    sigma = [sigma_flat[i * n:(i + 1) * n] for i in range(n)]

    gmv = mv.global_minimum_variance_portfolio(mu, sigma)
    print(f"global minimum variance portfolio: weights={[round(w, 4) for w in gmv['weights']]}")
    print(f"  return={gmv['target_return']:.4f}  variance={gmv['variance']:.6f}\n")

    print("efficient frontier:")
    targets = [gmv["target_return"] + i * 0.02 for i in range(6)]
    for t in targets:
        result = mv.efficient_frontier_portfolio(mu, sigma, t)
        print(f"  target={t:.4f}  std={math.sqrt(result['variance']):.4f}  "
              f"weights={[round(w, 4) for w in result['weights']]}")


def cmd_tangency(args):
    mu = [float(x) for x in args.mu.split(",")]
    sigma_flat = [float(x) for x in args.sigma.split(",")]
    n = len(mu)
    sigma = [sigma_flat[i * n:(i + 1) * n] for i in range(n)]

    tan = tp.tangency_portfolio(mu, sigma, args.rf)
    print(f"tangency portfolio: weights={[round(w, 4) for w in tan['weights']]}")
    print(f"  return={tan['return']:.4f}  std={math.sqrt(tan['variance']):.4f}  sharpe={tan['sharpe']:.4f}\n")

    print("Capital Market Line:")
    for target_std in (0.05, 0.10, 0.15, 0.20):
        cml_return = tp.capital_market_line(tan, args.rf, target_std)
        print(f"  std={target_std:.2f}  return={cml_return:.4f}")


def cmd_kelly(args):
    f_star = kc.kelly_fraction_single_asset(args.mu, args.sigma)
    print(f"Kelly-optimal leverage: {f_star:.4f}\n")

    print("fractional Kelly comparison (simulated, 252 periods, 3000 paths):")
    result = kc.fractional_kelly_comparison(f_star, args.mu, args.sigma, n_periods=252, n_paths=3000,
                                             fractions=[0.25, 0.5, 1.0, 1.5, 2.0], seed=1)
    for r in result["results"]:
        print(f"  {r['fraction_of_kelly']:.2f}x Kelly (leverage={r['leverage']:.2f}): "
              f"growth={r['mean_log_growth_per_period']:.6f}  std={r['std_terminal_log_wealth']:.4f}")


def cmd_fxanalysis(args):
    returns = fx.load_log_returns(DATA_PATH)
    strategies = fx.build_strategy_returns(returns)
    result = fx.run_full_analysis(strategies, rf=args.rf)

    print(f"strategies: {result['names']}")
    print(f"annualized mean returns: {[round(m, 4) for m in result['mu']]}\n")

    print(f"global minimum variance portfolio: {[round(w, 4) for w in result['gmv']['weights']]}")
    print(f"  return={result['gmv']['target_return']:.4f}  std={math.sqrt(result['gmv']['variance']):.4f}\n")

    print(f"tangency portfolio: {[round(w, 4) for w in result['tangency']['weights']]}")
    print(f"  return={result['tangency']['return']:.4f}  sharpe={result['tangency']['sharpe']:.4f}\n")

    print("CAPM betas vs buy_hold (the market factor):")
    for name in result["names"]:
        b = result["betas"][name]
        print(f"  {name:16s} beta={b['beta']:.4f}  r_squared={b['r_squared']:.4f}")

    print(f"\nrisk decomposition of the tangency portfolio (Euler check: {result['risk_decomposition']['matches']}):")
    for name, pct in zip(result["names"], result["risk_contribution_pct"]):
        print(f"  {name:16s} {pct*100:6.2f}% of total risk")

    print(f"\nKelly-normalized weights (rf=0 direction): {[round(w, 4) for w in result['kelly_normalized']]}")


def cmd_verify(args):
    import random
    import numpy as np

    checks = []

    def check(name, condition):
        checks.append((name, bool(condition)))
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {name}")

    mu = [0.05, 0.10, 0.15]
    sigma = [[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]]

    target = 0.08
    result = mv.efficient_frontier_portfolio(mu, sigma, target)
    sigma_np = np.array(sigma)
    mu_np = np.array(mu)
    ones_np = np.ones(3)
    sigma_inv_np = np.linalg.inv(sigma_np)
    a_np = ones_np @ sigma_inv_np @ ones_np
    b_np = ones_np @ sigma_inv_np @ mu_np
    c_np = mu_np @ sigma_inv_np @ mu_np
    d_np = a_np * c_np - b_np ** 2
    l1 = (a_np * target - b_np) / d_np
    l2 = (c_np - b_np * target) / d_np
    w_np = sigma_inv_np @ (l1 * mu_np + l2 * ones_np)
    check("efficient frontier portfolio matches independent numpy computation",
          all(math.isclose(a, b, abs_tol=1e-8) for a, b in zip(result["weights"], w_np)))

    gmv = mv.global_minimum_variance_portfolio(mu, sigma)
    gmv_np = (sigma_inv_np @ ones_np) / a_np
    check("global minimum variance portfolio matches independent numpy computation",
          all(math.isclose(a, b, abs_tol=1e-8) for a, b in zip(gmv["weights"], gmv_np)))

    rf = 0.02
    tan = tp.tangency_portfolio(mu, sigma, rf)
    rng = random.Random(1)
    best_random_sharpe = -999.0
    for _ in range(50000):
        raw = [rng.uniform(-1, 2) for _ in range(3)]
        s = sum(raw)
        w = [x / s for x in raw]
        best_random_sharpe = max(best_random_sharpe, tp.sharpe_ratio(w, mu, sigma, rf))
    check("tangency portfolio Sharpe ratio is not beaten by 50,000 random portfolios",
          tan["sharpe"] + 1e-6 >= best_random_sharpe)

    rng2 = random.Random(2)
    market = [rng2.gauss(0.001, 0.01) for _ in range(2000)]
    asset = [1.5 * m + rng2.gauss(0.0005, 0.005) for m in market]
    decomp = capm.risk_decomposition(asset, market)
    check("CAPM variance decomposition (systematic+idiosyncratic=total) holds exactly",
          math.isclose(decomp["total_variance"],
                       decomp["systematic_variance"] + decomp["idiosyncratic_variance"], rel_tol=1e-9))

    w_test = [0.5, 0.3, 0.2]
    euler = rd.verify_euler_decomposition(w_test, sigma)
    check("Euler risk decomposition sums exactly to portfolio volatility", euler["matches"])

    kelly_mu, kelly_sigma = 0.0005, 0.01
    f_star = kc.kelly_fraction_single_asset(kelly_mu, kelly_sigma)
    growth_full = kc.simulate_compound_growth(f_star, kelly_mu, kelly_sigma, 252, 4000, seed=1)
    growth_half = kc.simulate_compound_growth(0.5 * f_star, kelly_mu, kelly_sigma, 252, 4000, seed=1)
    growth_over = kc.simulate_compound_growth(1.5 * f_star, kelly_mu, kelly_sigma, 252, 4000, seed=1)
    check("full Kelly leverage beats half and 1.5x Kelly in simulated compound growth",
          growth_full > growth_half and growth_full > growth_over)

    kelly_w = kc.kelly_weights_multi_asset(mu, sigma)
    kelly_normalized = [w / sum(kelly_w) for w in kelly_w]
    tan_rf0 = tp.tangency_portfolio(mu, sigma, 0.0)
    check("Kelly direction matches tangency portfolio direction exactly at rf=0",
          all(math.isclose(a, b, abs_tol=1e-8) for a, b in zip(kelly_normalized, tan_rf0["weights"])))

    returns = fx.load_log_returns(DATA_PATH)
    strategies = fx.build_strategy_returns(returns)
    real_result = fx.run_full_analysis(strategies, rf=0.02)
    check("real EUR/USD buy_hold beta against itself is exactly 1",
          math.isclose(real_result["betas"]["buy_hold"]["beta"], 1.0, abs_tol=1e-8))
    check("real EUR/USD Euler risk decomposition holds", real_result["risk_decomposition"]["matches"])

    real_result_rf0 = fx.run_full_analysis(strategies, rf=0.0)
    check("real EUR/USD Kelly-tangency equivalence holds at rf=0",
          all(math.isclose(a, b, abs_tol=1e-6)
              for a, b in zip(real_result_rf0["kelly_normalized"], real_result_rf0["tangency"]["weights"])))

    passed = sum(1 for _, ok in checks if ok)
    print(f"\n{passed}/{len(checks)} checks passed")
    if passed != len(checks):
        sys.exit(1)


def build_parser():
    parser = argparse.ArgumentParser(description="Portfolio mathematics engine CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("frontier", help="construct the efficient frontier")
    p1.add_argument("--mu", type=str, default="0.05,0.10,0.15")
    p1.add_argument("--sigma", type=str, default="0.04,0.01,0.0,0.01,0.09,0.02,0.0,0.02,0.16",
                     help="flattened covariance matrix, row-major")
    p1.set_defaults(func=cmd_frontier)

    p2 = sub.add_parser("tangency", help="compute the tangency (max-Sharpe) portfolio and CML")
    p2.add_argument("--mu", type=str, default="0.05,0.10,0.15")
    p2.add_argument("--sigma", type=str, default="0.04,0.01,0.0,0.01,0.09,0.02,0.0,0.02,0.16")
    p2.add_argument("--rf", type=float, default=0.02)
    p2.set_defaults(func=cmd_tangency)

    p3 = sub.add_parser("kelly", help="Kelly criterion and fractional Kelly comparison")
    p3.add_argument("--mu", type=float, default=0.0005)
    p3.add_argument("--sigma", type=float, default=0.01)
    p3.set_defaults(func=cmd_kelly)

    p4 = sub.add_parser("fxanalysis", help="full portfolio analysis of real EUR/USD-derived strategies")
    p4.add_argument("--rf", type=float, default=0.02)
    p4.set_defaults(func=cmd_fxanalysis)

    p5 = sub.add_parser("verify", help="cross-check implementation against numpy oracles and known theory")
    p5.set_defaults(func=cmd_verify)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
