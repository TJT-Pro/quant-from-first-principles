import math
from typing import List, Dict

import arma_models as arma


def aic(log_likelihood: float, n_params: int) -> float:
    """Akaike Information Criterion: 2*k - 2*ln(L). Penalises added
    parameters at a constant rate of 2 per parameter regardless of
    sample size, which makes AIC comparatively permissive toward
    larger models as the sample grows, favouring predictive accuracy
    over parsimony.
    """
    return 2 * n_params - 2 * log_likelihood


def bic(log_likelihood: float, n_params: int, n_obs: int) -> float:
    """Bayesian Information Criterion: k*ln(n) - 2*ln(L). The penalty
    per parameter grows with ln(n), so BIC penalises additional
    parameters more heavily than AIC does once n is at all large
    (ln(n) > 2 for n > e^2 ~ 7.4), making BIC the more conservative,
    parsimony-favouring criterion of the two; the two criteria
    disagreeing on the selected order is itself informative about
    whether the extra parameters are earning their keep in a
    predictive sense (AIC) or a fit-parsimony sense (BIC).
    """
    return n_params * math.log(n_obs) - 2 * log_likelihood


def select_ar_order(x: List[float], max_order: int) -> Dict:
    """Fit AR(p) for p = 0, 1, ..., max_order by conditional least
    squares (via `arma_models.fit_css` with q=0, which reduces to
    ordinary least squares on the lagged design matrix), and report
    AIC/BIC for each. To keep the comparison fair across different
    orders, every model is fit and scored on the *same* subset of
    observations (those available once the largest order's lags are
    accounted for), since a higher-order model's regression otherwise
    starts on a later observation and would be scored on a different
    (and generally easier or harder) stretch of data.
    """
    n_total = len(x)
    results = []
    usable_n = n_total - max_order
    for p in range(max_order + 1):
        fit = arma.fit_css(x, p=p, q=0)
        residuals = arma.css_residuals(x, fit["c"], fit["phis"], [])
        residuals_common = residuals[-usable_n:]
        sigma2_common = sum(e ** 2 for e in residuals_common) / len(residuals_common)
        ll = arma.log_likelihood_gaussian(residuals_common, sigma2_common)
        n_params = 1 + p + 1  # constant + phis + sigma^2
        results.append({
            "order": p,
            "aic": aic(ll, n_params),
            "bic": bic(ll, n_params, usable_n),
            "log_likelihood": ll,
            "phis": fit["phis"],
            "c": fit["c"],
        })

    best_aic = min(results, key=lambda r: r["aic"])
    best_bic = min(results, key=lambda r: r["bic"])
    return {"results": results, "best_aic_order": best_aic["order"], "best_bic_order": best_bic["order"]}


def select_arma_order(x: List[float], max_p: int, max_q: int) -> Dict:
    """Grid search over (p, q) in [0, max_p] x [0, max_q], fitting each
    candidate ARMA(p,q) model by conditional least squares and scoring
    it by AIC and BIC. Every candidate is scored on the same usable
    observation count (n - max_p), for the same reason as in
    `select_ar_order`.
    """
    n_total = len(x)
    usable_n = n_total - max_p
    results = []
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            fit = arma.fit_css(x, p=p, q=q)
            residuals = arma.css_residuals(x, fit["c"], fit["phis"], fit["thetas"])
            residuals_common = residuals[-usable_n:] if len(residuals) >= usable_n else residuals
            sigma2_common = sum(e ** 2 for e in residuals_common) / len(residuals_common)
            ll = arma.log_likelihood_gaussian(residuals_common, sigma2_common)
            n_params = 1 + p + q + 1
            results.append({
                "p": p, "q": q,
                "aic": aic(ll, n_params),
                "bic": bic(ll, n_params, usable_n),
                "log_likelihood": ll,
                "stationary": fit["stationary"],
                "invertible": fit["invertible"],
            })

    best_aic = min(results, key=lambda r: r["aic"])
    best_bic = min(results, key=lambda r: r["bic"])
    return {"results": results, "best_aic": (best_aic["p"], best_aic["q"]),
            "best_bic": (best_bic["p"], best_bic["q"])}
