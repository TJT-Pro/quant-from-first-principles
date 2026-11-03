import csv
import math
import statistics
from typing import List, Dict

import conjugate_models as cm
import metropolis_hastings as mh


def load_log_returns(csv_path: str, column: str = "log_return") -> List[float]:
    values = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row[column]))
    return values


def sequential_beta_binomial_updating(returns: List[float], checkpoints: List[int]) -> Dict:
    """Update a Beta(1,1) (uniform, maximally uninformative) prior on
    the daily up-probability theta sequentially through the real
    return history, recording the posterior at each requested
    checkpoint. This is the Bayesian counterpart to the frequentist
    up-fraction estimates computed throughout `stochastic-processes`:
    rather than a single point estimate with a confidence interval
    bolted on afterward, the posterior distribution itself narrows in
    directly as evidence accumulates.
    """
    model = cm.BetaBinomial(1.0, 1.0)
    successes = 0
    trials = 0
    history = []
    checkpoint_set = set(checkpoints)
    for i, r in enumerate(returns, start=1):
        trials += 1
        if r > 0:
            successes += 1
        if i in checkpoint_set:
            posterior = cm.BetaBinomial(1.0 + successes, 1.0 + (trials - successes))
            lo, hi = posterior.credible_interval(0.95)
            history.append({"n": i, "mean": posterior.mean(), "ci_low": lo, "ci_high": hi})
    return {"checkpoints": history, "final_posterior": model.update(successes, trials)}


def bayesian_drift_estimate(returns: List[float], prior_mu0: float = 0.0,
                             prior_tau0_sq: float = 1.0) -> Dict:
    """A Normal-Normal estimate of the mean daily log return (the
    drift), using the sample variance as the "known" variance
    sigma^2 (the standard practical approximation when sigma is
    actually unknown but n is large enough that estimating it first
    and treating it as fixed introduces negligible extra uncertainty
    compared to the uncertainty in the mean itself).
    """
    n = len(returns)
    sample_mean = statistics.mean(returns)
    sample_var = statistics.variance(returns)
    prior = cm.NormalNormal(prior_mu0, prior_tau0_sq)
    posterior = prior.update(sample_mean, sample_var, n)
    return {
        "posterior_mean": posterior.mean(),
        "posterior_sd": math.sqrt(posterior.variance()),
        "sample_mean": sample_mean,
        "n": n,
    }


def large_move_indicator(returns: List[float], quantile: float = 0.95) -> List[int]:
    """Days whose absolute return exceeds the given quantile of the
    absolute-return distribution, the same "large move" definition
    used in `stochastic-processes/project3_poisson_processes`, kept
    identical here (rather than redefined some other way) so the
    posterior predictive check below is a genuine continuation of that
    project's finding rather than a coincidentally similar but
    different question.
    """
    abs_returns = sorted(abs(r) for r in returns)
    n = len(abs_returns)
    pos = quantile * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    threshold = abs_returns[lo] + frac * (abs_returns[hi] - abs_returns[lo])
    return [1 if abs(r) > threshold else 0 for r in returns]


def gamma_poisson_posterior_predictive_check(returns: List[float], quantile: float = 0.95,
                                              window: int = 20) -> Dict:
    """Fit a Gamma-Poisson model to the rate of large moves per
    `window`-day block, then compare its posterior predictive
    distribution of window counts against the empirical distribution
    of window counts actually observed. This is the Bayesian
    counterpart to `stochastic-processes/project3_poisson_processes`'s
    frequentist Fisher dispersion test: instead of asking "is the
    variance-to-mean ratio consistent with Poisson equidispersion,"
    ask "how well does a fitted Poisson model's own posterior
    predictive distribution match what actually happened," which is a
    posterior predictive check in the most literal sense the term is
    normally used.
    """
    indicators = large_move_indicator(returns, quantile)
    n_windows = len(indicators) // window
    window_counts = [sum(indicators[i * window:(i + 1) * window]) for i in range(n_windows)]

    total_events = sum(window_counts)
    prior = cm.GammaPoisson(1.0, 1.0)
    posterior = prior.update(total_events, n_windows)

    max_count = max(window_counts) + 3
    predictive_pmf = [posterior.posterior_predictive_pmf(y, n_intervals=1.0) for y in range(max_count + 1)]
    predictive_expected_counts = [p * n_windows for p in predictive_pmf]

    empirical_counts = [0] * (max_count + 1)
    for c in window_counts:
        if c <= max_count:
            empirical_counts[c] += 1

    return {
        "n_windows": n_windows,
        "posterior_rate_mean": posterior.mean(),
        "posterior_rate_sd": math.sqrt(posterior.variance()),
        "empirical_counts": empirical_counts,
        "predictive_expected_counts": predictive_expected_counts,
        "empirical_variance": statistics.variance(window_counts),
        "empirical_mean": statistics.mean(window_counts),
    }


def student_t_log_pdf(x: float, df: float, scale: float) -> float:
    """Log-density of a zero-location Student-t distribution with
    `df` degrees of freedom and scale `scale`:

        ln f(x) = ln Gamma((df+1)/2) - ln Gamma(df/2) - 0.5*ln(df*pi)
                  - ln(scale) - (df+1)/2 * ln(1 + (x/scale)^2/df)

    used as the likelihood for the fat-tailed model in
    `fit_student_t_via_mcmc`; no conjugate prior exists for either the
    degrees of freedom or the scale of a Student-t likelihood (jointly
    or separately), which is exactly why this application needs MCMC
    rather than a closed form.
    """
    if scale <= 0 or df <= 0:
        return float("-inf")
    z = x / scale
    return (math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
            - math.log(scale) - (df + 1) / 2 * math.log(1 + z ** 2 / df))


def fit_student_t_via_mcmc(returns: List[float], n_samples: int = 8000,
                            burn_in: int = 2000, seed: int = 1) -> Dict:
    """Jointly estimate the degrees of freedom and scale of a
    zero-mean Student-t model for the returns via Metropolis-Hastings,
    with weakly informative priors (log-normal on df to keep it
    positive and favour moderate values a priori, half-normal-like on
    scale via a positivity constraint) reflecting genuine prior
    uncertainty about both. A low degrees-of-freedom posterior
    (heavier tails than Gaussian) would be the Bayesian counterpart to
    the well-known "financial returns are leptokurtic" stylised fact.
    """
    n = len(returns)

    def log_posterior(params: List[float]) -> float:
        df, scale = params
        if df <= 2.0 or scale <= 0:
            return float("-inf")
        log_prior_df = -0.5 * ((math.log(df) - math.log(10.0)) / 1.0) ** 2 - math.log(df)
        log_prior_scale = -0.5 * ((math.log(scale) - math.log(0.006)) / 1.0) ** 2 - math.log(scale)
        log_lik = sum(student_t_log_pdf(r, df, scale) for r in returns)
        return log_lik + log_prior_df + log_prior_scale

    result = mh.metropolis_hastings_vector(log_posterior, initial=[10.0, 0.006],
                                            proposal_sds=[0.5, 0.0001],
                                            n_samples=n_samples, burn_in=burn_in, seed=seed)
    samples = result["samples"]
    df_samples = [s[0] for s in samples]
    scale_samples = [s[1] for s in samples]

    total_log_lik_normal = sum(-0.5 * math.log(2 * math.pi * statistics.variance(returns))
                                - r ** 2 / (2 * statistics.variance(returns)) for r in returns)
    posterior_mean_df = statistics.mean(df_samples)
    posterior_mean_scale = statistics.mean(scale_samples)
    total_log_lik_t = sum(student_t_log_pdf(r, posterior_mean_df, posterior_mean_scale) for r in returns)

    return {
        "df_mean": posterior_mean_df,
        "df_sd": statistics.stdev(df_samples),
        "scale_mean": posterior_mean_scale,
        "scale_sd": statistics.stdev(scale_samples),
        "acceptance_rate": result["acceptance_rate"],
        "df_samples": df_samples,
        "scale_samples": scale_samples,
        "log_lik_t_at_posterior_mean": total_log_lik_t,
        "log_lik_normal_at_mle": total_log_lik_normal,
    }
