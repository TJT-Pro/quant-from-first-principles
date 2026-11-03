import argparse
import math
import os
import sys

import var_backtesting as vb
import model_comparison as mc
import overfitting_demo as od
import fx_application as fx

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DEXUSEU_returns.csv")


def _load_prices():
    import csv
    prices = []
    with open(DATA_PATH, newline="") as f:
        for row in csv.DictReader(f):
            prices.append(float(row["price"]))
    return prices


def cmd_varbacktest(args):
    returns = fx.load_log_returns(DATA_PATH)
    result = fx.var_model_comparison(returns, confidences=[args.confidence])
    data = result[args.confidence]

    print(f"VaR backtest at {args.confidence*100:.0f}% confidence, {len(returns)} observations\n")
    for name in ("gaussian", "historical"):
        d = data[name]
        k = d["kupiec"]
        status = "REJECTED" if k["rejected_at_5_percent"] else "passed"
        print(f"{name:12s} VaR={d['var']:.5f}  exceptions={d['exceptions']:4d}  "
              f"expected={k['expected_exceptions']:.1f}  p-value={k['p_value']:.6f}  [{status}]")


def cmd_overfit(args):
    returns = fx.load_log_returns(DATA_PATH)
    result = fx.overfitting_analysis(returns)
    print(f"tested {result['n_combinations_tested']} (short, long) moving-average window combinations\n")
    print(f"best in-sample combo: {result['best_in_sample_params']}")
    print(f"  in-sample Sharpe:     {result['best_in_sample_sharpe']:+.4f}")
    print(f"  out-of-sample Sharpe: {result['best_combo_out_of_sample_sharpe']:+.4f}")
    print(f"average out-of-sample Sharpe across all combos: "
          f"{result['average_out_of_sample_sharpe_across_all_combos']:+.4f}")


def cmd_priceuncertainty(args):
    returns = fx.load_log_returns(DATA_PATH)
    prices = _load_prices()
    s0 = prices[-1]
    k = round(s0, 2) if args.k is None else args.k

    result = fx.option_price_uncertainty(returns, s0, k, args.r, args.t, n_bootstrap=args.n_bootstrap)
    print(f"spot={s0}, strike={k}, r={args.r}, T={args.t}\n")
    print(f"volatility estimate: {result['parameter_estimate']:.6f}")
    print(f"  90% CI: [{result['parameter_5th_percentile']:.6f}, {result['parameter_95th_percentile']:.6f}]\n")
    print(f"option price point estimate: {result['output_point_estimate']:.6f}")
    print(f"  90% CI: [{result['output_5th_percentile']:.6f}, {result['output_95th_percentile']:.6f}]")
    relative = result['output_std'] / result['output_point_estimate'] * 100
    print(f"  relative uncertainty from volatility estimation alone: {relative:.2f}%")


def cmd_modelselect(args):
    import random
    rng = random.Random(args.seed)
    n = 200
    x_vals = [rng.uniform(-3, 3) for _ in range(n)]
    true_coeffs = [1.0, -2.0, 0.5]
    y_vals = [sum(c * x ** i for i, c in enumerate(true_coeffs)) + rng.gauss(0, 1.0) for x in x_vals]

    def fit_polynomial(degree):
        a_mat = [[x ** p for p in range(degree + 1)] for x in x_vals]
        at = list(zip(*a_mat))
        n_p = degree + 1
        ata = [[sum(at[i][kk] * a_mat[kk][j] for kk in range(n)) for j in range(n_p)] for i in range(n_p)]
        aty = [sum(at[i][kk] * y_vals[kk] for kk in range(n)) for i in range(n_p)]
        aug = [ata[i][:] + [aty[i]] for i in range(n_p)]
        for col in range(n_p):
            piv = max(range(col, n_p), key=lambda r: abs(aug[r][col]))
            aug[col], aug[piv] = aug[piv], aug[col]
            pivot_val = aug[col][col]
            for j in range(col, n_p + 1):
                aug[col][j] /= pivot_val
            for row in range(n_p):
                if row != col:
                    factor = aug[row][col]
                    for j in range(col, n_p + 1):
                        aug[row][j] -= factor * aug[col][j]
        return [aug[i][n_p] for i in range(n_p)]

    print("true generating polynomial: degree 2 (coefficients [1.0, -2.0, 0.5])\n")
    print(f"{'degree':>8}  {'AIC':>10}  {'BIC':>10}")
    for degree in range(7):
        coeffs = fit_polynomial(degree)
        residuals = [y_vals[i] - sum(c * x_vals[i] ** p for p, c in enumerate(coeffs)) for i in range(n)]
        rss = sum(r ** 2 for r in residuals)
        sigma2_hat = rss / n
        log_likelihood = -0.5 * n * math.log(2 * math.pi * sigma2_hat) - 0.5 * rss / sigma2_hat
        aic_val = mc.aic(log_likelihood, degree + 2)
        bic_val = mc.bic(log_likelihood, degree + 2, n)
        marker = "  <- true degree" if degree == 2 else ""
        print(f"{degree:8d}  {aic_val:10.2f}  {bic_val:10.2f}{marker}")


def cmd_verify(args):
    import random
    import numpy as np
    from scipy.stats import chi2 as scipy_chi2, norm as scipy_norm

    checks = []

    def check(name, condition):
        checks.append((name, bool(condition)))
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {name}")

    ok = all(math.isclose(vb.inverse_normal_cdf(p), scipy_norm.ppf(p), abs_tol=1e-6)
             for p in (0.01, 0.05, 0.5, 0.95, 0.99))
    check("inverse normal CDF matches scipy.stats.norm.ppf", ok)

    ok = all(math.isclose(vb.chi_squared_1_cdf(x), scipy_chi2.cdf(x, 1), abs_tol=1e-6)
             for x in (1.0, 3.841, 6.635, 10.0))
    check("chi-squared(1) CDF matches scipy.stats.chi2", ok)

    ok = all(math.isclose(mc.chi_squared_cdf(x, df), scipy_chi2.cdf(x, df), abs_tol=1e-6)
             for df in (1, 2, 5, 10) for x in (2.0, 5.0, 15.0))
    check("general chi-squared CDF (incomplete gamma) matches scipy.stats.chi2", ok)

    rng = random.Random(1)
    n, confidence = 1000, 0.95
    p_true = 1 - confidence
    n_rejected = 0
    n_trials = 500
    for _ in range(n_trials):
        exceptions = sum(1 for _ in range(n) if rng.random() < p_true)
        result = vb.kupiec_pof_test(n, exceptions, confidence)
        if result["rejected_at_5_percent"]:
            n_rejected += 1
    check(f"Kupiec test rejects a correctly specified model near the nominal 5% rate "
          f"(measured {n_rejected/n_trials*100:.1f}%)",
          n_rejected / n_trials < 0.12)

    rng2 = random.Random(2)
    n_obs = 200
    x_vals = [rng2.uniform(-3, 3) for _ in range(n_obs)]
    true_coeffs = [1.0, -2.0, 0.5]
    y_vals = [sum(c * x ** i for i, c in enumerate(true_coeffs)) + rng2.gauss(0, 1.0) for x in x_vals]

    def fit_poly(degree):
        a_mat = [[x ** p for p in range(degree + 1)] for x in x_vals]
        at = list(zip(*a_mat))
        n_p = degree + 1
        ata = [[sum(at[i][kk] * a_mat[kk][j] for kk in range(n_obs)) for j in range(n_p)] for i in range(n_p)]
        aty = [sum(at[i][kk] * y_vals[kk] for kk in range(n_obs)) for i in range(n_p)]
        a_np = np.array(ata)
        b_np = np.array(aty)
        return list(np.linalg.solve(a_np, b_np))

    aics, bics = [], []
    for degree in range(7):
        coeffs = fit_poly(degree)
        residuals = [y_vals[i] - sum(c * x_vals[i] ** p for p, c in enumerate(coeffs)) for i in range(n_obs)]
        rss = sum(r ** 2 for r in residuals)
        sigma2_hat = rss / n_obs
        log_likelihood = -0.5 * n_obs * math.log(2 * math.pi * sigma2_hat) - 0.5 * rss / sigma2_hat
        aics.append(mc.aic(log_likelihood, degree + 2))
        bics.append(mc.bic(log_likelihood, degree + 2, n_obs))
    check("AIC is minimized at the true polynomial degree (2) on synthetic data", aics.index(min(aics)) == 2)
    check("BIC is minimized at the true polynomial degree (2) on synthetic data", bics.index(min(bics)) == 2)

    rng3 = random.Random(3)
    noise_returns = [rng3.gauss(0, 0.01) for _ in range(6000)]
    train, test = od.train_test_split(noise_returns, 0.6)
    param_grid = [(s, l) for s in (3, 5, 8, 10, 15, 20) for l in (20, 30, 40, 50, 60, 80, 100, 120) if s < l]
    overfit_result = od.grid_search_overfitting_demo(train, test, od.moving_average_crossover_returns, param_grid)
    check("best in-sample Sharpe exceeds its own out-of-sample Sharpe on pure-noise data",
          overfit_result["best_in_sample_sharpe"] > overfit_result["best_combo_out_of_sample_sharpe"])

    returns = fx.load_log_returns(DATA_PATH)
    var_result = fx.var_model_comparison(returns, confidences=[0.95, 0.99])
    check("real EUR/USD: both VaR models pass at 95% confidence",
          not var_result[0.95]["gaussian"]["kupiec"]["rejected_at_5_percent"]
          and not var_result[0.95]["historical"]["kupiec"]["rejected_at_5_percent"])
    check("real EUR/USD: Gaussian VaR model fails at 99% confidence (fat tails)",
          var_result[0.99]["gaussian"]["kupiec"]["rejected_at_5_percent"])
    check("real EUR/USD: historical VaR model passes at 99% confidence",
          not var_result[0.99]["historical"]["kupiec"]["rejected_at_5_percent"])

    real_overfit = fx.overfitting_analysis(returns)
    check("real EUR/USD: best in-sample Sharpe exceeds its own out-of-sample Sharpe",
          real_overfit["best_in_sample_sharpe"] > real_overfit["best_combo_out_of_sample_sharpe"])

    prices = _load_prices()
    s0 = prices[-1]
    price_result = fx.option_price_uncertainty(returns, s0, round(s0, 2), 0.03, 0.25, n_bootstrap=500)
    check("real EUR/USD: option price has genuine, positive uncertainty from volatility estimation alone",
          price_result["output_std"] > 0.0)

    passed = sum(1 for _, ok in checks if ok)
    print(f"\n{passed}/{len(checks)} checks passed")
    if passed != len(checks):
        sys.exit(1)


def build_parser():
    parser = argparse.ArgumentParser(description="Model risk engine CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("varbacktest", help="backtest Gaussian vs historical VaR on real EUR/USD data")
    p1.add_argument("--confidence", type=float, default=0.99)
    p1.set_defaults(func=cmd_varbacktest)

    p2 = sub.add_parser("overfit", help="demonstrate backtest overfitting via parameter grid search")
    p2.set_defaults(func=cmd_overfit)

    p3 = sub.add_parser("priceuncertainty", help="propagate volatility estimation uncertainty to option price")
    p3.add_argument("--k", type=float, default=None)
    p3.add_argument("--r", type=float, default=0.03)
    p3.add_argument("--t", type=float, default=0.25)
    p3.add_argument("--n-bootstrap", type=int, default=1000)
    p3.set_defaults(func=cmd_priceuncertainty)

    p4 = sub.add_parser("modelselect", help="AIC/BIC model selection sanity check on synthetic data")
    p4.add_argument("--seed", type=int, default=1)
    p4.set_defaults(func=cmd_modelselect)

    p5 = sub.add_parser("verify", help="cross-check implementation against scipy/numpy oracles and known theory")
    p5.set_defaults(func=cmd_verify)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
