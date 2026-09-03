import argparse
import math
import os
import sys

import brownian_motion as bm
import sde_solvers as sde
import models
import convergence_analysis as ca
import ito_calculus as ito
import fx_application as fx

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DEXUSEU_returns.csv")


def cmd_brownian(args):
    result = bm.verify_terminal_distribution(args.t_end, args.n_steps, args.n_paths, seed=1)
    print(f"terminal distribution of W_{args.t_end}, {args.n_paths} paths:")
    print(f"  empirical mean:     {result['empirical_mean']:.6f}  (theory: {result['theoretical_mean']})")
    print(f"  empirical variance: {result['empirical_variance']:.6f}  (theory: {result['theoretical_variance']})")

    qv = bm.verify_quadratic_variation_converges_to_t(args.t_end, [10, 100, 1000, 10000], seed=1)
    print(f"\nquadratic variation, converging to t={qv['theoretical_limit']}:")
    for r in qv["results"]:
        print(f"  n_steps={r['n_steps']:6d}  QV={r['quadratic_variation']:.6f}")


def cmd_convergence(args):
    mu_param, sigma_param = args.mu, args.sigma
    x0, t_end = args.x0, args.t_end
    mu_fn = lambda x, t: mu_param * x
    sigma_fn = lambda x, t: sigma_param * x
    sigma_x_fn = lambda x, t: sigma_param
    exact_fn = lambda x0_, w_t: x0_ * math.exp((mu_param - 0.5 * sigma_param ** 2) * t_end + sigma_param * w_t)

    n_fine = 4096
    step_counts = [16, 32, 64, 128]
    step_sizes = [t_end / n for n in step_counts]

    print("Euler-Maruyama strong error:")
    em_errors = []
    for n in step_counts:
        err = ca.strong_error(sde.euler_maruyama, mu_fn, sigma_fn, None, x0, t_end, n, n_fine,
                               args.n_paths, exact_fn, seed=1)
        em_errors.append(err)
        print(f"  n={n:4d}  error={err:.6f}")
    print(f"  empirical strong order: {ca.estimate_convergence_order(em_errors, step_sizes):.3f}  (theory: 0.5)\n")

    print("Milstein strong error:")
    mil_errors = []
    for n in step_counts:
        err = ca.strong_error(sde.milstein, mu_fn, sigma_fn, sigma_x_fn, x0, t_end, n, n_fine,
                               args.n_paths, exact_fn, seed=1)
        mil_errors.append(err)
        print(f"  n={n:4d}  error={err:.6f}")
    print(f"  empirical strong order: {ca.estimate_convergence_order(mil_errors, step_sizes):.3f}  (theory: 1.0)")


def cmd_ito(args):
    result = ito.gbm_log_transform_check(args.mu, args.sigma, args.x0, args.t_end, args.n_steps, seed=1)
    print("Ito's lemma check: d(ln X_t) = (mu-0.5*sigma^2)dt + sigma*dW_t for GBM")
    print(f"  exact log change (closed form):      {result['exact_log_change']:.10f}")
    print(f"  simulated log change (via ln of X_T): {result['simulated_log_change']:.10f}")
    print(f"  difference: {result['difference']:.2e}")


def cmd_fxcalibrate(args):
    data = fx.load_prices_and_returns(DATA_PATH)
    gbm_params = fx.calibrate_gbm(data["returns"])
    print(f"GBM calibration on {len(data['returns'])} EUR/USD daily returns:")
    print(f"  mu (annualized drift):       {gbm_params['mu']:+.6f}")
    print(f"  sigma (annualized vol):      {gbm_params['sigma']:.6f}\n")

    vol_series = fx.rolling_realized_volatility(data["returns"], window=20)
    ou_params = fx.calibrate_ou_via_ar1(vol_series)
    print(f"OU calibration on the 20-day rolling realized volatility series ({len(vol_series)} obs):")
    print(f"  theta (mean-reversion speed): {ou_params['theta']:.4f}")
    print(f"  mu (long-run vol level):      {ou_params['mu']:.6f}")
    print(f"  sigma (vol-of-vol):           {ou_params['sigma']:.6f}")
    print(f"  half-life:                    {ou_params['half_life_days']:.1f} days\n")

    print("real vs GBM-theoretical dispersion at several horizons:")
    for horizon in (5, 20, 60):
        comparison = fx.compare_simulated_vs_real_terminal_distribution(data["prices"], gbm_params, horizon)
        print(f"  horizon={horizon:3d}d  real_std={comparison['real_std']:.6f}  "
              f"gbm_std={comparison['gbm_theoretical_std']:.6f}  ratio={comparison['std_ratio']:.4f}")


def cmd_verify(args):
    import random
    import numpy as np
    from scipy import stats as sstats

    checks = []

    def check(name, condition):
        checks.append((name, bool(condition)))
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {name}")

    rng = random.Random(1)
    n_paths = 5000
    terminal_values = []
    for i in range(n_paths):
        _, ws = bm.simulate_brownian_path(1.0, 100, seed=i)
        terminal_values.append(ws[-1])
    numpy_samples = np.random.default_rng(0).normal(0, 1, n_paths)
    ks_stat, ks_p = sstats.ks_2samp(terminal_values, numpy_samples)
    check("Brownian motion terminal distribution passes KS test vs numpy N(0,1) samples", ks_p > 0.01)

    qv = bm.verify_quadratic_variation_converges_to_t(1.0, [100, 10000], seed=1)
    check("quadratic variation converges toward t as resolution increases",
          abs(qv["results"][-1]["quadratic_variation"] - 1.0) < abs(qv["results"][0]["quadratic_variation"] - 1.0))

    x0, mu_p, sigma_p, t_end = 100.0, 0.05, 0.2, 1.0
    gbm_terminals = []
    for i in range(20000):
        _, xs = models.simulate_gbm_exact(x0, mu_p, sigma_p, t_end, 10, seed=i)
        gbm_terminals.append(xs[-1])
    scipy_lognorm_samples = sstats.lognorm(s=sigma_p * math.sqrt(t_end),
                                            scale=x0 * math.exp((mu_p - 0.5 * sigma_p ** 2) * t_end)).rvs(
        20000, random_state=1)
    ks_stat2, ks_p2 = sstats.ks_2samp(gbm_terminals, scipy_lognorm_samples)
    check("GBM terminal distribution matches scipy.stats.lognorm via KS test", ks_p2 > 0.01)

    mu_fn = lambda x, t: mu_p * x
    sigma_fn = lambda x, t: sigma_p * x
    sigma_x_fn = lambda x, t: sigma_p
    exact_fn = lambda x0_, w_t: x0_ * math.exp((mu_p - 0.5 * sigma_p ** 2) * t_end + sigma_p * w_t)
    n_fine = 2048
    step_counts = [16, 32, 64, 128]
    step_sizes = [t_end / n for n in step_counts]

    em_errors = [ca.strong_error(sde.euler_maruyama, mu_fn, sigma_fn, None, x0, t_end, n, n_fine, 3000,
                                  exact_fn, seed=1) for n in step_counts]
    em_order = ca.estimate_convergence_order(em_errors, step_sizes)
    check(f"Euler-Maruyama strong order near 0.5 (measured {em_order:.3f})", abs(em_order - 0.5) < 0.25)

    mil_errors = [ca.strong_error(sde.milstein, mu_fn, sigma_fn, sigma_x_fn, x0, t_end, n, n_fine, 3000,
                                   exact_fn, seed=1) for n in step_counts]
    mil_order = ca.estimate_convergence_order(mil_errors, step_sizes)
    check(f"Milstein strong order near 1.0 (measured {mil_order:.3f})", abs(mil_order - 1.0) < 0.3)

    step_counts_weak = [4, 8, 16, 32, 64, 128]
    step_sizes_weak = [t_end / n for n in step_counts_weak]
    exact_mean = x0 * math.exp(mu_p * t_end)
    weak_errors = [abs(ca.linear_drift_exact_mean_error(mu_p, x0, t_end, n) - exact_mean)
                   for n in step_counts_weak]
    weak_order = ca.estimate_convergence_order(weak_errors, step_sizes_weak)
    check(f"Euler-Maruyama weak order near 1.0 on linear drift (measured {weak_order:.3f})",
          abs(weak_order - 1.0) < 0.1)

    ito_check = ito.gbm_log_transform_check(mu_p, sigma_p, x0, t_end, 1000, seed=1)
    check("Ito's lemma log-transform of GBM matches exact closed form", ito_check["difference"] < 1e-9)

    ito_general_check = ito.verify_ito_transform_matches_direct_simulation(
        math.log, lambda x: 1.0 / x, lambda x: -1.0 / x ** 2, mu_fn, sigma_fn, x0, t_end, 5000, 300, seed=2)
    check("Ito's lemma general verification (log transform) matches direct simulation",
          ito_general_check["mean_absolute_difference"] < 0.01)

    theta_ou, mu_ou, sigma_ou = 2.0, 0.05, 0.3
    ou_terminals = []
    for i in range(20000):
        _, xs = models.simulate_ou_exact(0.0, theta_ou, mu_ou, sigma_ou, 20.0, 50, seed=i)
        ou_terminals.append(xs[-1])
    import statistics
    stat_mean, stat_var = models.ou_stationary_distribution(theta_ou, mu_ou, sigma_ou)
    check("OU stationary mean matches theory", abs(statistics.mean(ou_terminals) - stat_mean) < 0.02)
    check("OU stationary variance matches theory", abs(statistics.variance(ou_terminals) - stat_var) < 0.005)

    data = fx.load_prices_and_returns(DATA_PATH)
    gbm_params = fx.calibrate_gbm(data["returns"])
    check("real EUR/USD GBM-calibrated drift is small (near-zero, consistent with efficient market)",
          abs(gbm_params["mu"]) < 0.05)

    vol_series = fx.rolling_realized_volatility(data["returns"], window=20)
    ou_params = fx.calibrate_ou_via_ar1(vol_series)
    check("real EUR/USD OU-calibrated half-life is a plausible number of weeks to months",
          10 < ou_params["half_life_days"] < 200)

    dispersion_result = fx.compare_simulated_vs_real_terminal_distribution(data["prices"], gbm_params, 60)
    check("real EUR/USD dispersion at 60-day horizon meets or exceeds simple GBM prediction",
          dispersion_result["std_ratio"] >= 0.95)

    passed = sum(1 for _, ok in checks if ok)
    print(f"\n{passed}/{len(checks)} checks passed")
    if passed != len(checks):
        sys.exit(1)


def build_parser():
    parser = argparse.ArgumentParser(description="Stochastic differential equations engine CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("brownian", help="verify Brownian motion's defining properties")
    p1.add_argument("--t-end", type=float, default=1.0)
    p1.add_argument("--n-steps", type=int, default=100)
    p1.add_argument("--n-paths", type=int, default=10000)
    p1.set_defaults(func=cmd_brownian)

    p2 = sub.add_parser("convergence", help="measure strong convergence order for EM and Milstein on GBM")
    p2.add_argument("--mu", type=float, default=0.05)
    p2.add_argument("--sigma", type=float, default=0.3)
    p2.add_argument("--x0", type=float, default=1.0)
    p2.add_argument("--t-end", type=float, default=1.0)
    p2.add_argument("--n-paths", type=int, default=3000)
    p2.set_defaults(func=cmd_convergence)

    p3 = sub.add_parser("ito", help="verify Ito's lemma on the GBM log-transform")
    p3.add_argument("--mu", type=float, default=0.05)
    p3.add_argument("--sigma", type=float, default=0.3)
    p3.add_argument("--x0", type=float, default=1.0)
    p3.add_argument("--t-end", type=float, default=1.0)
    p3.add_argument("--n-steps", type=int, default=1000)
    p3.set_defaults(func=cmd_ito)

    p4 = sub.add_parser("fxcalibrate", help="calibrate GBM and OU models to real EUR/USD data")
    p4.set_defaults(func=cmd_fxcalibrate)

    p5 = sub.add_parser("verify", help="cross-check implementation against scipy/numpy oracles and known theory")
    p5.set_defaults(func=cmd_verify)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
