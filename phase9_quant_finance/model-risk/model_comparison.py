import math
from typing import List, Callable, Dict


def aic(log_likelihood: float, n_params: int) -> float:
    """Akaike Information Criterion: 2*k - 2*ln(L), penalizing model
    complexity (k parameters) directly against fit quality (log
    likelihood). Lower is better. AIC approximates the expected
    out-of-sample prediction error (via an information-theoretic
    argument relating log-likelihood to KL divergence from the true
    generating process), making it a proxy for genuine predictive
    performance rather than in-sample fit alone, which a more
    flexible model can always improve by definition.
    """
    return 2 * n_params - 2 * log_likelihood


def bic(log_likelihood: float, n_params: int, n_obs: int) -> float:
    """Bayesian Information Criterion: k*ln(n) - 2*ln(L). The
    complexity penalty scales with ln(n) rather than AIC's constant 2,
    making BIC penalize additional parameters more heavily as the
    sample size grows; BIC is derived from a Bayesian model-selection
    argument (an asymptotic approximation to the log Bayes factor
    under a unit-information prior) and is consistent for selecting
    the true model as n->infinity among a finite set of candidates, a
    property AIC does not share (AIC is asymptotically efficient for
    prediction, not consistent for identifying the true model).
    """
    return n_params * math.log(n_obs) - 2 * log_likelihood


def lower_incomplete_gamma_regularized(a: float, x: float) -> float:
    """The regularized lower incomplete gamma function P(a,x), needed
    to compute a general chi-squared CDF (chi-squared with k degrees
    of freedom has CDF P(k/2, x/2)) for likelihood-ratio tests
    comparing models differing by more than one parameter. Computed
    via the standard two-regime numerical approach: a direct series
    expansion for x < a+1 (where it converges quickly), and a
    continued-fraction expansion for the complementary regularized
    upper incomplete gamma function Q(a,x)=1-P(a,x) otherwise (where
    the series would converge too slowly to be numerically reliable).
    """
    if x < 0 or a <= 0:
        raise ValueError("lower_incomplete_gamma_regularized requires x>=0, a>0")
    if x == 0:
        return 0.0

    if x < a + 1:
        term = 1.0 / a
        total = term
        n = a
        for _ in range(500):
            n += 1
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))

    tiny = 1e-300
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    q = math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
    return 1 - q


def chi_squared_cdf(x: float, df: int) -> float:
    """The CDF of a chi-squared distribution with df degrees of
    freedom, P(df/2, x/2), via the regularized incomplete gamma
    function above.
    """
    if x <= 0:
        return 0.0
    return lower_incomplete_gamma_regularized(df / 2, x / 2)


def likelihood_ratio_test(log_likelihood_restricted: float, log_likelihood_full: float,
                           df: int) -> Dict:
    """The likelihood-ratio test for nested models: LR =
    -2*(ln(L_restricted) - ln(L_full)), asymptotically chi-squared
    distributed with degrees of freedom equal to the number of
    additional parameters the full model has over the restricted one,
    under the null hypothesis that the restricted (simpler) model is
    correct. A significant LR (small p-value) rejects the restricted
    model in favour of the full one: the extra parameters buy a
    genuinely better fit, not merely a fit that looks better because
    there are more knobs to turn.
    """
    lr_stat = -2 * (log_likelihood_restricted - log_likelihood_full)
    p_value = 1 - chi_squared_cdf(lr_stat, df)
    return {"lr_statistic": lr_stat, "df": df, "p_value": p_value, "rejects_restricted_at_5_percent": p_value < 0.05}


def k_fold_cross_validation(data: List, k: int, fit_fn: Callable, predict_fn: Callable,
                             error_fn: Callable) -> float:
    """Standard k-fold cross-validation: partition the data into k
    roughly equal folds, repeatedly fit on k-1 folds and evaluate
    prediction error on the held-out fold, and average the resulting
    out-of-sample errors across folds. This estimates a model's
    genuine out-of-sample performance directly, by construction never
    evaluating a fold's error using a model that was fit including
    that same fold's data, unlike in-sample error, which can always be
    driven arbitrarily low by a sufficiently flexible model regardless
    of whether that flexibility reflects genuine predictive structure.
    """
    n = len(data)
    fold_size = n // k
    errors = []
    for fold in range(k):
        start, end = fold * fold_size, (fold + 1) * fold_size if fold < k - 1 else n
        test_set = data[start:end]
        train_set = data[:start] + data[end:]
        model = fit_fn(train_set)
        fold_errors = [error_fn(predict_fn(model, x), y) for x, y in test_set]
        errors.append(sum(fold_errors) / len(fold_errors))
    return sum(errors) / len(errors)
