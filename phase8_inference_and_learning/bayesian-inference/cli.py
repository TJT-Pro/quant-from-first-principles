import argparse
import math
import os
import statistics
import sys

import conjugate_models as cm
import grid_approximation as ga
import metropolis_hastings as mh
import model_comparison as mcomp
import fx_application as fx

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DEXUSEU_returns.csv")


def cmd_betabinomial(args):
    prior = cm.BetaBinomial(args.a, args.b)
    posterior = prior.update(args.successes, args.trials)
    lo, hi = posterior.credible_interval(0.95)
    print(f"prior: Beta({args.a}, {args.b})")
    print(f"posterior: Beta({posterior.a}, {posterior.b})")
    print(f"posterior mean: {posterior.mean():.6f}")
    print(f"95% credible interval: [{lo:.6f}, {hi:.6f}]")
    print(f"log evidence: {prior.log_evidence(args.successes, args.trials):.4f}")


def cmd_gammapoisson(args):
    prior = cm.GammaPoisson(args.a, args.b)
    posterior = prior.update(args.count, args.intervals)
    print(f"prior: Gamma({args.a}, rate={args.b})")
    print(f"posterior: Gamma({posterior.a}, rate={posterior.b})")
    print(f"posterior mean rate: {posterior.mean():.6f}")
    print(f"posterior sd: {math.sqrt(posterior.variance()):.6f}")


def cmd_normalnormal(args):
    prior = cm.NormalNormal(args.mu0, args.tau0_sq)
    posterior = prior.update(args.sample_mean, args.sigma_sq, args.n)
    print(f"prior: N({args.mu0}, {args.tau0_sq})")
    print(f"posterior: N({posterior.mean():.6f}, {posterior.variance():.8f})")
    pred_mean, pred_var = posterior.posterior_predictive(args.sigma_sq)
    print(f"posterior predictive: N({pred_mean:.6f}, {pred_var:.6f})")


def cmd_mcmc(args):
    log_posterior = lambda theta: -0.5 * ((theta - args.target_mean) / args.target_sd) ** 2
    result = mh.metropolis_hastings(log_posterior, args.initial, args.proposal_sd,
                                     args.n_samples, args.burn_in, seed=1)
    samples = result["samples"]
    print(f"target: N({args.target_mean}, {args.target_sd}^2)")
    print(f"sample mean: {statistics.mean(samples):.4f}, sample sd: {statistics.stdev(samples):.4f}")
    print(f"acceptance rate: {result['acceptance_rate']:.4f}")
    print(f"effective sample size: {mh.effective_sample_size(samples):.1f} / {len(samples)}")


def cmd_fxbeta(args):
    returns = fx.load_log_returns(DATA_PATH)
    checkpoints = [30, 100, 500, 2000, len(returns)]
    result = fx.sequential_beta_binomial_updating(returns, checkpoints)
    print("sequential Bayesian estimation of P(up day), Beta(1,1) prior")
    print(f"{'n':>6}  {'mean':>8}  {'95% CI'}")
    for c in result["checkpoints"]:
        print(f"{c['n']:6d}  {c['mean']:.6f}  [{c['ci_low']:.4f}, {c['ci_high']:.4f}]")


def cmd_fxdrift(args):
    returns = fx.load_log_returns(DATA_PATH)
    result = fx.bayesian_drift_estimate(returns)
    print(f"n={result['n']}")
    print(f"posterior mean daily drift: {result['posterior_mean']:.8f}")
    print(f"posterior sd:               {result['posterior_sd']:.8f}")
    print(f"zero drift is {abs(result['posterior_mean']) / result['posterior_sd']:.3f} "
          f"posterior standard deviations from the posterior mean")


def cmd_fxpoisson(args):
    returns = fx.load_log_returns(DATA_PATH)
    result = fx.gamma_poisson_posterior_predictive_check(returns, args.quantile, args.window)
    print(f"n_windows={result['n_windows']}, window={args.window} days, quantile={args.quantile}")
    print(f"posterior rate mean: {result['posterior_rate_mean']:.4f} +/- {result['posterior_rate_sd']:.4f}")
    print(f"empirical mean/variance across windows: {result['empirical_mean']:.4f} / "
          f"{result['empirical_variance']:.4f}")
    print(f"(equidispersion would give variance approx equal to mean)\n")
    print("count   empirical   Poisson-predicted")
    for i, (e, p) in enumerate(zip(result["empirical_counts"], result["predictive_expected_counts"])):
        print(f"{i:5d}   {e:9d}   {p:10.2f}")


def cmd_fxstudentt(args):
    returns = fx.load_log_returns(DATA_PATH)
    print(f"fitting Student-t(df, scale) via MCMC on {len(returns)} EUR/USD log returns")
    print(f"({args.n_samples + args.burn_in} total steps; this can take several minutes on the "
          f"full dataset, see the README)")
    result = fx.fit_student_t_via_mcmc(returns, args.n_samples, args.burn_in, seed=1)
    print(f"\ndf:    mean={result['df_mean']:.3f}  sd={result['df_sd']:.3f}")
    print(f"scale: mean={result['scale_mean']:.6f}  sd={result['scale_sd']:.6f}")
    print(f"acceptance rate: {result['acceptance_rate']:.4f}")
    print(f"effective sample size (df): {mh.effective_sample_size(result['df_samples']):.1f}")
    print(f"\nlog-likelihood, Student-t at posterior mean: {result['log_lik_t_at_posterior_mean']:.2f}")
    print(f"log-likelihood, Normal at its own MLE:        {result['log_lik_normal_at_mle']:.2f}")


def cmd_verify(args):
    import random
    import numpy as np
    from scipy import stats as sstats

    checks = []

    def check(name, condition):
        checks.append((name, bool(condition)))
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {name}")

    # 1. Beta-Binomial vs scipy
    a, b, k, n = 3.0, 5.0, 12, 20
    posterior = cm.BetaBinomial(a, b).update(k, n)
    check("BetaBinomial posterior mean matches scipy.stats.beta",
          math.isclose(posterior.mean(), sstats.beta(posterior.a, posterior.b).mean(), abs_tol=1e-9))
    check("BetaBinomial posterior variance matches scipy.stats.beta",
          math.isclose(posterior.variance(), sstats.beta(posterior.a, posterior.b).var(), abs_tol=1e-9))
    lo, hi = posterior.credible_interval(0.95)
    scipy_lo, scipy_hi = sstats.beta(posterior.a, posterior.b).ppf([0.025, 0.975])
    check("BetaBinomial credible interval matches scipy.stats.beta.ppf",
          math.isclose(lo, scipy_lo, abs_tol=1e-3) and math.isclose(hi, scipy_hi, abs_tol=1e-3))

    ok = True
    for y in range(0, 15):
        ours = cm.BetaBinomial(a, b).posterior_predictive_pmf(y, 15)
        theirs = sstats.betabinom.pmf(y, 15, a, b)
        if not math.isclose(ours, theirs, abs_tol=1e-9):
            ok = False
    check("Beta-Binomial posterior predictive matches scipy.stats.betabinom", ok)

    # 2. Gamma-Poisson vs scipy
    a2, b2, s, n2 = 4.0, 2.0, 30, 10
    gp_posterior = cm.GammaPoisson(a2, b2).update(s, n2)
    check("GammaPoisson posterior mean matches scipy.stats.gamma",
          math.isclose(gp_posterior.mean(), sstats.gamma(gp_posterior.a, scale=1 / gp_posterior.b).mean(),
                       abs_tol=1e-9))
    ok = True
    r, p = gp_posterior.a, gp_posterior.b / (gp_posterior.b + 1)
    for y in range(0, 20):
        ours = gp_posterior.posterior_predictive_pmf(y, n_intervals=1.0)
        theirs = sstats.nbinom.pmf(y, r, p)
        if not math.isclose(ours, theirs, abs_tol=1e-9):
            ok = False
    check("Gamma-Poisson posterior predictive matches scipy.stats.nbinom", ok)

    # 3. Normal-Normal vs a direct numpy recomputation of the
    #    precision-weighted update formula
    mu0, tau0_sq, sigma_sq, nobs, xbar = 1.0, 4.0, 9.0, 25, 2.5
    ours = cm.NormalNormal(mu0, tau0_sq).update(xbar, sigma_sq, nobs)
    numpy_precision = 1 / tau0_sq + nobs / sigma_sq
    numpy_var = 1 / numpy_precision
    numpy_mean = numpy_var * (mu0 / tau0_sq + nobs * xbar / sigma_sq)
    check("Normal-Normal posterior matches independent numpy recomputation",
          math.isclose(ours.mean(), numpy_mean, abs_tol=1e-9) and
          math.isclose(ours.variance(), numpy_var, abs_tol=1e-9))

    # 4. grid approximation matches conjugate closed form
    grid = [i / 20000 for i in range(1, 20000)]
    log_prior = lambda theta: (a - 1) * math.log(theta) + (b - 1) * math.log(1 - theta)
    log_lik = lambda theta: k * math.log(theta) + (n - k) * math.log(1 - theta)
    probs = ga.grid_posterior(log_prior, log_lik, grid)
    grid_mean = ga.posterior_mean(grid, probs)
    conjugate = cm.BetaBinomial(a, b).update(k, n)
    check("grid approximation posterior mean matches conjugate closed form",
          math.isclose(grid_mean, conjugate.mean(), abs_tol=1e-3))

    # 5. Metropolis-Hastings recovers a known Normal target and agrees
    #    with a numpy-based independent Monte Carlo check
    true_mean, true_sd = 4.0, 1.5
    log_posterior = lambda theta: -0.5 * ((theta - true_mean) / true_sd) ** 2
    mcmc_result = mh.metropolis_hastings(log_posterior, 0.0, 1.5, 30000, 3000, seed=1)
    samples = mcmc_result["samples"]
    check("Metropolis-Hastings recovers known Normal mean and sd",
          math.isclose(statistics.mean(samples), true_mean, abs_tol=0.05) and
          math.isclose(statistics.stdev(samples), true_sd, abs_tol=0.05))

    rng = np.random.default_rng(0)
    numpy_samples = rng.normal(true_mean, true_sd, size=30000)
    ks_stat, ks_p = sstats.ks_2samp(samples, numpy_samples)
    check("Metropolis-Hastings sample distribution passes KS test vs numpy oracle samples", ks_p > 0.01)

    # 6. Gelman-Rubin: converged chains near 1, deliberately
    #    non-converged chains far above 1
    chains = [mh.metropolis_hastings(log_posterior, start, 1.5, 5000, 2000, seed=seed)["samples"]
              for seed, start in enumerate([-10.0, 0.0, 10.0, 20.0])]
    check("Gelman-Rubin R-hat near 1 for converged chains", mh.gelman_rubin(chains) < 1.05)

    bad_chains = [[offset + rng.normal(0, 0.01) for _ in range(500)] for offset in (0.0, 5.0, 10.0)]
    check("Gelman-Rubin R-hat far above 1 for non-converged chains", mh.gelman_rubin(bad_chains) > 2.0)

    # 7. harmonic mean estimator: in the right ballpark of, but
    #    systematically above, the true evidence
    hm_posterior = cm.BetaBinomial(a, b).update(k, n)
    true_log_evidence = cm.BetaBinomial(a, b).log_evidence(k, n)
    log_lik_fn = lambda theta: (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                                 + k * math.log(theta) + (n - k) * math.log(1 - theta))
    hm_estimates = []
    for trial in range(15):
        r = random.Random(trial)
        hm_samples = [r.betavariate(hm_posterior.a, hm_posterior.b) for _ in range(2000)]
        hm_estimates.append(mcomp.harmonic_mean_log_evidence([log_lik_fn(s) for s in hm_samples]))
    check("harmonic mean estimator is within 1.5 nats of the true evidence on average",
          abs(statistics.mean(hm_estimates) - true_log_evidence) < 1.5)
    check("harmonic mean estimator is systematically biased above the true evidence",
          statistics.mean(hm_estimates) > true_log_evidence)

    # 8. real EUR/USD application checks
    returns = fx.load_log_returns(DATA_PATH)
    check("real EUR/USD data loads with expected count", len(returns) == 6930)

    beta_result = fx.sequential_beta_binomial_updating(returns, [len(returns)])
    check("real EUR/USD up-probability posterior mean close to 0.5",
          abs(beta_result["checkpoints"][0]["mean"] - 0.5) < 0.03)

    pp_check = fx.gamma_poisson_posterior_predictive_check(returns, 0.95, 20)
    check("real EUR/USD large-move counts show overdispersion vs Poisson",
          pp_check["empirical_variance"] > 1.5 * pp_check["empirical_mean"])

    # 9. Student-t via MCMC recovers known parameters on synthetic data
    rng2 = random.Random(2)
    true_df, true_scale = 6.0, 0.02
    synthetic = []
    for _ in range(1200):
        z = rng2.gauss(0, 1)
        v = sum(rng2.gauss(0, 1) ** 2 for _ in range(int(true_df)))
        synthetic.append(true_scale * z / math.sqrt(v / true_df))
    t_result = fx.fit_student_t_via_mcmc(synthetic, n_samples=1800, burn_in=700, seed=1)
    check("Student-t MCMC fit recovers known df and scale from synthetic data",
          abs(t_result["df_mean"] - true_df) < 3.0 and abs(t_result["scale_mean"] - true_scale) < 0.004)

    passed = sum(1 for _, ok in checks if ok)
    print(f"\n{passed}/{len(checks)} checks passed")
    if passed != len(checks):
        sys.exit(1)


def build_parser():
    parser = argparse.ArgumentParser(description="Bayesian inference engine CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("betabinomial", help="Beta-Binomial conjugate update")
    p1.add_argument("--a", type=float, default=1.0)
    p1.add_argument("--b", type=float, default=1.0)
    p1.add_argument("--successes", type=int, required=True)
    p1.add_argument("--trials", type=int, required=True)
    p1.set_defaults(func=cmd_betabinomial)

    p2 = sub.add_parser("gammapoisson", help="Gamma-Poisson conjugate update")
    p2.add_argument("--a", type=float, default=1.0)
    p2.add_argument("--b", type=float, default=1.0)
    p2.add_argument("--count", type=int, required=True)
    p2.add_argument("--intervals", type=float, required=True)
    p2.set_defaults(func=cmd_gammapoisson)

    p3 = sub.add_parser("normalnormal", help="Normal-Normal conjugate update")
    p3.add_argument("--mu0", type=float, default=0.0)
    p3.add_argument("--tau0-sq", type=float, default=1.0)
    p3.add_argument("--sample-mean", type=float, required=True)
    p3.add_argument("--sigma-sq", type=float, required=True)
    p3.add_argument("--n", type=int, required=True)
    p3.set_defaults(func=cmd_normalnormal)

    p4 = sub.add_parser("mcmc", help="Metropolis-Hastings on a Normal target")
    p4.add_argument("--target-mean", type=float, default=0.0)
    p4.add_argument("--target-sd", type=float, default=1.0)
    p4.add_argument("--initial", type=float, default=0.0)
    p4.add_argument("--proposal-sd", type=float, default=1.0)
    p4.add_argument("--n-samples", type=int, default=10000)
    p4.add_argument("--burn-in", type=int, default=1000)
    p4.set_defaults(func=cmd_mcmc)

    p5 = sub.add_parser("fxbeta", help="sequential Bayesian up-probability estimation on EUR/USD")
    p5.set_defaults(func=cmd_fxbeta)

    p6 = sub.add_parser("fxdrift", help="Bayesian drift estimation on EUR/USD")
    p6.set_defaults(func=cmd_fxdrift)

    p7 = sub.add_parser("fxpoisson", help="Gamma-Poisson posterior predictive check on EUR/USD large moves")
    p7.add_argument("--quantile", type=float, default=0.95)
    p7.add_argument("--window", type=int, default=20)
    p7.set_defaults(func=cmd_fxpoisson)

    p8 = sub.add_parser("fxstudentt", help="fit Student-t(df,scale) to EUR/USD returns via MCMC")
    p8.add_argument("--n-samples", type=int, default=2000)
    p8.add_argument("--burn-in", type=int, default=500)
    p8.set_defaults(func=cmd_fxstudentt)

    p9 = sub.add_parser("verify", help="cross-check implementation against analytical/oracle values")
    p9.set_defaults(func=cmd_verify)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
