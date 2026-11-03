import argparse
import math
import os
import sys

import autocorrelation as ac
import ar_models as ar
import ma_models as ma
import arma_models as arma
import stationarity as st
import model_selection as ms
import optimization as opt

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DEXUSEU_returns.csv")


def cmd_acf(args):
    x = ac.load_log_returns(DATA_PATH) if args.fx else [float(v) for v in args.series.split(",")]
    acf_vals = ac.acf(x, args.max_lag)
    pacf_vals = ac.pacf_durbin_levinson(x, args.max_lag)
    band = ac.bartlett_confidence_band(len(x))
    print(f"n={len(x)}, 95% significance band: +/-{band:.4f}")
    print("lag   acf      pacf")
    for k in range(1, args.max_lag + 1):
        print(f"{k:3d}   {acf_vals[k]:+.4f}   {pacf_vals[k-1]:+.4f}")


def cmd_adf(args):
    x = ac.load_log_returns(DATA_PATH) if args.fx else [float(v) for v in args.series.split(",")]
    result = st.adf_test(x, n_lags=args.lags, n_trials=args.trials, seed=1)
    print(f"ADF statistic: {result['statistic']:.4f}")
    print(f"5% critical value (Monte Carlo, n={len(x)}): {result['critical_value_5pct']:.4f}")
    print(f"p-value: {result['p_value']:.4f}")
    conclusion = "reject unit root (stationary)" if result["p_value"] < 0.05 else "fail to reject unit root"
    print(f"conclusion at 5%: {conclusion}")


def cmd_fitar(args):
    x = ac.load_log_returns(DATA_PATH) if args.fx else [float(v) for v in args.series.split(",")]
    yw = ar.fit_yule_walker(x, args.order)
    ols = ar.fit_ols(x, args.order)
    print(f"Yule-Walker: c={yw['c']:.6f}, phis={[round(p,6) for p in yw['phis']]}, sigma2={yw['sigma2']:.6f}")
    print(f"OLS:         c={ols['c']:.6f}, phis={[round(p,6) for p in ols['phis']]}, sigma2={ols['sigma2']:.6f}")
    print(f"stationary: {ar.is_stationary(yw['phis'])}")


def cmd_fitma(args):
    x = ac.load_log_returns(DATA_PATH) if args.fx else [float(v) for v in args.series.split(",")]
    result = ma.fit_css(x, args.order)
    print(f"CSS: mu={result['mu']:.6f}, thetas={[round(t,6) for t in result['thetas']]}, "
          f"sigma2={result['sigma2']:.6f}")
    print(f"invertible: {ma.is_invertible(result['thetas'])}")


def cmd_fitarma(args):
    x = ac.load_log_returns(DATA_PATH) if args.fx else [float(v) for v in args.series.split(",")]
    result = arma.fit_css(x, args.p, args.q)
    print(f"c={result['c']:.6f}, phis={[round(p,6) for p in result['phis']]}, "
          f"thetas={[round(t,6) for t in result['thetas']]}, sigma2={result['sigma2']:.6f}")
    print(f"stationary: {result['stationary']}, invertible: {result['invertible']}")


def cmd_selectorder(args):
    x = ac.load_log_returns(DATA_PATH) if args.fx else [float(v) for v in args.series.split(",")]
    result = ms.select_ar_order(x, args.max_order)
    print("order   AIC          BIC          phis")
    for r in result["results"]:
        print(f"{r['order']:3d}   {r['aic']:10.2f}   {r['bic']:10.2f}   {[round(p,4) for p in r['phis']]}")
    print(f"\nbest by AIC: order {result['best_aic_order']}")
    print(f"best by BIC: order {result['best_bic_order']}")


def cmd_ljungbox(args):
    x = ac.load_log_returns(DATA_PATH) if args.fx else [float(v) for v in args.series.split(",")]
    result = ac.ljung_box_test(x, args.max_lag, args.n_params)
    print(f"Q statistic: {result['statistic']:.4f}, df={result['df']}, p-value={result['p_value']:.6f}")
    conclusion = "reject white noise" if result["p_value"] < 0.05 else "fail to reject white noise"
    print(f"conclusion at 5%: {conclusion}")


def cmd_volclustering(args):
    returns = ac.load_log_returns(DATA_PATH)
    squared = [r ** 2 for r in returns]
    band = ac.bartlett_confidence_band(len(returns))
    acf_raw = ac.acf(returns, args.max_lag)
    acf_sq = ac.acf(squared, args.max_lag)
    print(f"n={len(returns)}, 95% significance band: +/-{band:.4f}\n")
    print("lag   raw_returns_acf   squared_returns_acf")
    for k in range(1, args.max_lag + 1):
        flag_raw = "*" if abs(acf_raw[k]) > band else " "
        flag_sq = "*" if abs(acf_sq[k]) > band else " "
        print(f"{k:3d}   {acf_raw[k]:+.4f} {flag_raw}          {acf_sq[k]:+.4f} {flag_sq}")
    print("\n(* = exceeds the white-noise significance band)")

    lb_raw = ac.ljung_box_test(returns, args.max_lag)
    lb_sq = ac.ljung_box_test(squared, args.max_lag)
    print(f"\nLjung-Box on raw returns:     Q={lb_raw['statistic']:.2f}, p={lb_raw['p_value']:.4f}")
    print(f"Ljung-Box on squared returns: Q={lb_sq['statistic']:.2f}, p={lb_sq['p_value']:.3e}")


def cmd_verify(args):
    import random
    import numpy as np
    from scipy import stats as sstats

    checks = []

    def check(name, condition):
        checks.append((name, bool(condition)))
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {name}")

    # 1. ACF/PACF against theoretical AR(1)/MA(1) values
    x_ar1 = ar.simulate_ar(20000, 0.0, [0.6], seed=1)
    rho1 = ac.autocorrelation(x_ar1, 1)
    check("AR(1) empirical lag-1 ACF matches theory (rho_1 = phi)",
          math.isclose(rho1, 0.6, abs_tol=0.03))
    rho2 = ac.autocorrelation(x_ar1, 2)
    check("AR(1) empirical lag-2 ACF matches theory (rho_2 = phi^2)",
          math.isclose(rho2, 0.36, abs_tol=0.03))

    # 2. inverse normal CDF vs scipy
    ok = True
    for p in (0.025, 0.5, 0.8, 0.975, 0.99):
        if not math.isclose(ac._inverse_normal_cdf(p), sstats.norm.ppf(p), abs_tol=1e-6):
            ok = False
    check("inverse normal CDF matches scipy.stats.norm.ppf", ok)

    # 3. chi2_sf vs scipy
    ok = True
    for x, df in [(1.0, 1), (5.0, 3), (20.0, 10)]:
        if not math.isclose(ac.chi2_sf(x, df), sstats.chi2.sf(x, df), abs_tol=1e-9):
            ok = False
    check("chi2_sf matches scipy.stats.chi2.sf", ok)

    # 4. Durand-Kerner roots vs numpy.roots
    coeffs = [-6.0, 11.0, -6.0, 1.0]  # (z-1)(z-2)(z-3)
    our_roots = sorted(r.real for r in ar.durand_kerner_roots(coeffs))
    numpy_roots = sorted(np.roots(list(reversed(coeffs))).real)
    check("Durand-Kerner roots match numpy.roots",
          all(math.isclose(a, b, abs_tol=1e-4) for a, b in zip(our_roots, numpy_roots)))

    # 5. AR OLS fit vs numpy lstsq
    x = ar.simulate_ar(3000, 1.0, [0.4, -0.2], seed=2)
    our_fit = ar.fit_ols(x, order=2)
    design = np.array([[1.0, x[t - 1], x[t - 2]] for t in range(2, len(x))])
    target = np.array(x[2:])
    numpy_beta = np.linalg.lstsq(design, target, rcond=None)[0]
    check("AR(2) OLS fit matches numpy.linalg.lstsq",
          math.isclose(our_fit["c"], numpy_beta[0], abs_tol=1e-6) and
          math.isclose(our_fit["phis"][0], numpy_beta[1], abs_tol=1e-6) and
          math.isclose(our_fit["phis"][1], numpy_beta[2], abs_tol=1e-6))

    # 6. Nelder-Mead vs scipy.optimize.minimize on a shared objective
    f = lambda v: (v[0] - 3.0) ** 2 + (v[1] + 1.5) ** 2
    our_result = opt.nelder_mead(f, [0.0, 0.0])
    from scipy.optimize import minimize as scipy_minimize
    scipy_min = scipy_minimize(f, [0.0, 0.0], method="Nelder-Mead")
    check("Nelder-Mead result matches scipy.optimize.minimize",
          math.isclose(our_result["x"][0], scipy_min.x[0], abs_tol=1e-3) and
          math.isclose(our_result["x"][1], scipy_min.x[1], abs_tol=1e-3))

    # 7. MA(1) and ARMA(1,1) round-trip recovery
    x_ma = ma.simulate_ma(6000, 0.0, [0.5], seed=3)
    ma_fit = ma.fit_css(x_ma, q=1)
    check("MA(1) CSS fit recovers true theta", math.isclose(ma_fit["thetas"][0], 0.5, abs_tol=0.08))

    x_arma = arma.simulate_arma(6000, 0.5, [0.4], [0.3], seed=4)
    arma_fit = arma.fit_css(x_arma, p=1, q=1)
    check("ARMA(1,1) CSS fit recovers true phi and theta",
          math.isclose(arma_fit["phis"][0], 0.4, abs_tol=0.12) and
          math.isclose(arma_fit["thetas"][0], 0.3, abs_tol=0.12))

    # 8. AIC/BIC order selection recovers the true order
    x_ar2 = arma.simulate_arma(4000, 0.0, [0.5, -0.3], [], seed=5)
    selection = ms.select_ar_order(x_ar2, max_order=5)
    check("AIC/BIC order selection recovers the true AR(2) order",
          selection["best_bic_order"] == 2)

    # 9. ADF test calibration: rejection rate near nominal level under
    #    the true null (many independent random walks)
    rejections = 0
    trials = 100
    for i in range(trials):
        rng = random.Random(2000 + i)
        walk = [0.0]
        for _ in range(300):
            walk.append(walk[-1] + rng.gauss(0, 1))
        result = st.adf_test(walk, n_trials=400, seed=1)
        if result["p_value"] < 0.05:
            rejections += 1
    rejection_rate = rejections / trials
    check("ADF test rejection rate near nominal 5% under the true unit-root null",
          rejection_rate < 0.15)

    # 10. real EUR/USD data: prices show a unit root, returns do not
    prices = ac.load_prices(DATA_PATH)
    returns = ac.load_log_returns(DATA_PATH)
    price_result = st.adf_test(prices, n_lags=1, n_trials=1000, seed=1)
    returns_result = st.adf_test(returns, n_lags=1, n_trials=1000, seed=1)
    check("real EUR/USD price level fails to reject the unit root",
          price_result["p_value"] > 0.10)
    check("real EUR/USD returns strongly reject the unit root",
          returns_result["p_value"] < 0.01)

    # 11. real EUR/USD data: squared returns show significant
    #     autocorrelation that raw returns do not
    squared = [r ** 2 for r in returns]
    band = ac.bartlett_confidence_band(len(returns))
    acf_raw = ac.acf(returns, 10)
    acf_sq = ac.acf(squared, 10)
    n_raw_exceed = sum(1 for k in range(1, 11) if abs(acf_raw[k]) > band)
    n_sq_exceed = sum(1 for k in range(1, 11) if abs(acf_sq[k]) > band)
    check("squared EUR/USD returns show far more significant ACF spikes than raw returns",
          n_sq_exceed > n_raw_exceed)

    passed = sum(1 for _, ok in checks if ok)
    print(f"\n{passed}/{len(checks)} checks passed")
    if passed != len(checks):
        sys.exit(1)


def build_parser():
    parser = argparse.ArgumentParser(description="Time series mathematics engine CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_series_args(p):
        p.add_argument("--series", type=str, help="comma-separated series")
        p.add_argument("--fx", action="store_true", help="use real EUR/USD log returns instead")

    p1 = sub.add_parser("acf", help="ACF and PACF")
    add_series_args(p1)
    p1.add_argument("--max-lag", type=int, default=10)
    p1.set_defaults(func=cmd_acf)

    p2 = sub.add_parser("adf", help="Augmented Dickey-Fuller unit root test")
    add_series_args(p2)
    p2.add_argument("--lags", type=int, default=1)
    p2.add_argument("--trials", type=int, default=1000)
    p2.set_defaults(func=cmd_adf)

    p3 = sub.add_parser("fitar", help="fit AR(p) by Yule-Walker and OLS")
    add_series_args(p3)
    p3.add_argument("--order", type=int, default=1)
    p3.set_defaults(func=cmd_fitar)

    p4 = sub.add_parser("fitma", help="fit MA(q) by conditional sum of squares")
    add_series_args(p4)
    p4.add_argument("--order", type=int, default=1)
    p4.set_defaults(func=cmd_fitma)

    p5 = sub.add_parser("fitarma", help="fit ARMA(p,q) by conditional sum of squares")
    add_series_args(p5)
    p5.add_argument("--p", type=int, default=1)
    p5.add_argument("--q", type=int, default=1)
    p5.set_defaults(func=cmd_fitarma)

    p6 = sub.add_parser("selectorder", help="AIC/BIC AR order selection")
    add_series_args(p6)
    p6.add_argument("--max-order", type=int, default=5)
    p6.set_defaults(func=cmd_selectorder)

    p7 = sub.add_parser("ljungbox", help="Ljung-Box white-noise test")
    add_series_args(p7)
    p7.add_argument("--max-lag", type=int, default=10)
    p7.add_argument("--n-params", type=int, default=0)
    p7.set_defaults(func=cmd_ljungbox)

    p8 = sub.add_parser("volclustering", help="ACF comparison: raw vs squared EUR/USD returns")
    p8.add_argument("--max-lag", type=int, default=10)
    p8.set_defaults(func=cmd_volclustering)

    p9 = sub.add_parser("verify", help="cross-check implementation against analytical/oracle values")
    p9.set_defaults(func=cmd_verify)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
