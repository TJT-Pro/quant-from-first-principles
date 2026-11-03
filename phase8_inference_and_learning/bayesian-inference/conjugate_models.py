import math
from typing import Tuple


def log_beta(a: float, b: float) -> float:
    """ln B(a,b) = ln Gamma(a) + ln Gamma(b) - ln Gamma(a+b), computed
    in log-space via math.lgamma throughout so that the Beta function
    itself, which underflows or overflows a plain float for even
    moderately large arguments, never needs to be formed directly.
    """
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def beta_pdf(theta: float, a: float, b: float) -> float:
    if not 0.0 <= theta <= 1.0:
        return 0.0
    if theta in (0.0, 1.0) and (a < 1 or b < 1):
        return float("inf")
    if theta == 0.0 or theta == 1.0:
        return 0.0
    return math.exp((a - 1) * math.log(theta) + (b - 1) * math.log(1 - theta) - log_beta(a, b))


class BetaBinomial:
    """The conjugate pair for a fixed but unknown success probability
    theta observed through binomial (or repeated Bernoulli) trials.

    Prior: theta ~ Beta(a, b), with density proportional to
    theta^(a-1) * (1-theta)^(b-1). Likelihood of k successes in n
    trials: Binomial(n, theta), proportional to
    theta^k * (1-theta)^(n-k). Bayes' theorem multiplies prior and
    likelihood: the posterior is proportional to
    theta^(a+k-1) * (1-theta)^(b+n-k-1), which is exactly the Beta(a+k,
    b+n-k) kernel. Conjugacy means no normalising integral ever needs
    to be evaluated: the posterior's family is known in advance and
    only its parameters update, by simply adding the observed
    successes and failures to the prior's own pseudo-counts a and b
    (which is the standard interpretation of a Beta(a,b) prior: "as if"
    a-1 prior successes and b-1 prior failures had already been seen).
    """

    def __init__(self, a: float, b: float):
        if a <= 0 or b <= 0:
            raise ValueError("a and b must be positive")
        self.a = a
        self.b = b

    def update(self, successes: int, trials: int) -> "BetaBinomial":
        if successes < 0 or trials < successes:
            raise ValueError("require 0 <= successes <= trials")
        return BetaBinomial(self.a + successes, self.b + (trials - successes))

    def mean(self) -> float:
        return self.a / (self.a + self.b)

    def variance(self) -> float:
        total = self.a + self.b
        return (self.a * self.b) / (total ** 2 * (total + 1))

    def credible_interval(self, level: float = 0.95, n_grid: int = 20000) -> Tuple[float, float]:
        """An equal-tailed credible interval, found by inverting the
        Beta CDF numerically on a fine grid (rather than via a
        closed-form Beta quantile function, which does not exist in
        elementary terms) since this project's core modules avoid
        depending on scipy for anything used outside `verify`.
        """
        tail = (1 - level) / 2
        step = 1.0 / n_grid
        cdf = 0.0
        lower = 0.0
        for i in range(1, n_grid + 1):
            theta = i / n_grid
            cdf += beta_pdf(theta, self.a, self.b) * step
            if cdf >= tail:
                lower = theta
                break
        cdf = 0.0
        upper = 1.0
        for i in range(n_grid - 1, 0, -1):
            theta = i / n_grid
            cdf += beta_pdf(theta, self.a, self.b) * step
            if cdf >= tail:
                upper = theta
                break
        return lower, upper

    def posterior_predictive_pmf(self, y: int, m: int) -> float:
        """P(Y=y | data), the probability of y successes in m future
        trials, marginalising over the posterior uncertainty in theta
        rather than plugging in a single point estimate. This is the
        Beta-Binomial compound distribution, obtained by integrating
        the Binomial(m, theta) likelihood against the Beta(a,b)
        posterior density over theta:

            P(Y=y) = C(m,y) * B(y+a, m-y+b) / B(a,b),

        the ratio of Beta functions falling out directly from
        combining the theta^(...) terms in the integrand into a single
        Beta-function integral.
        """
        if not 0 <= y <= m:
            return 0.0
        log_binom = math.lgamma(m + 1) - math.lgamma(y + 1) - math.lgamma(m - y + 1)
        return math.exp(log_binom + log_beta(y + self.a, m - y + self.b) - log_beta(self.a, self.b))

    def log_evidence(self, successes: int, trials: int) -> float:
        """ln P(D | model): the marginal likelihood of the observed
        data, integrating the Binomial likelihood against the *prior*
        (not posterior) over theta. This is the same Beta-function
        identity as the posterior predictive above, evaluated at the
        prior's own (a,b) rather than the updated ones, and is exactly
        what Bayesian model comparison (see `model_comparison.py`)
        needs to compare this prior/likelihood specification against
        an alternative one.
        """
        n, k = trials, successes
        log_binom = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        return log_binom + log_beta(k + self.a, n - k + self.b) - log_beta(self.a, self.b)


class GammaPoisson:
    """The conjugate pair for an unknown Poisson rate lambda.

    Prior: lambda ~ Gamma(a, rate=b), density proportional to
    lambda^(a-1) * exp(-b*lambda). Likelihood of counts y_1,...,y_n
    (or equivalently their sum S over n intervals): Poisson(lambda),
    proportional (in lambda) to lambda^S * exp(-n*lambda). The product
    is proportional to lambda^(a+S-1) * exp(-(b+n)*lambda), the
    Gamma(a+S, b+n) kernel: as with Beta-Binomial, the prior's shape
    parameter a acts like "a-1 prior events" and its rate parameter b
    acts like "b prior intervals of exposure," both simply incremented
    by the observed data.
    """

    def __init__(self, a: float, b: float):
        if a <= 0 or b <= 0:
            raise ValueError("a and b must be positive")
        self.a = a
        self.b = b

    def update(self, total_count: int, n_intervals: float) -> "GammaPoisson":
        if total_count < 0 or n_intervals <= 0:
            raise ValueError("total_count must be non-negative and n_intervals positive")
        return GammaPoisson(self.a + total_count, self.b + n_intervals)

    def mean(self) -> float:
        return self.a / self.b

    def variance(self) -> float:
        return self.a / (self.b ** 2)

    def posterior_predictive_pmf(self, y: int, n_intervals: float = 1.0) -> float:
        """P(Y=y | data): the Gamma-Poisson compound distribution is a
        Negative Binomial, obtained by integrating the Poisson(lambda *
        n_intervals) likelihood against the Gamma(a,b) posterior
        density over lambda:

            P(Y=y) = Gamma(a+y)/(Gamma(a)*y!) * (b/(b+n))^a * (n/(b+n))^y,

        for a future observation spanning `n_intervals` (n below) worth
        of the same exposure as the historical counts were measured
        over. Written here directly in terms of the standard
        Negative-Binomial-with-overdispersion parameterisation that
        falls out of the Poisson-Gamma mixture identity.
        """
        if y < 0:
            return 0.0
        n = n_intervals
        log_pmf = (math.lgamma(self.a + y) - math.lgamma(self.a) - math.lgamma(y + 1)
                   + self.a * math.log(self.b / (self.b + n))
                   + y * math.log(n / (self.b + n)))
        return math.exp(log_pmf)

    def log_evidence(self, total_count: int, n_intervals: float) -> float:
        """ln P(D | model), the marginal likelihood integrating the
        Poisson likelihood against the *prior* over lambda, needed for
        Bayesian model comparison exactly as in BetaBinomial.log_evidence.
        """
        s, n = total_count, n_intervals
        return (self.a * math.log(self.b) - math.lgamma(self.a)
                + math.lgamma(self.a + s) - (self.a + s) * math.log(self.b + n))


class NormalNormal:
    """The conjugate pair for an unknown mean, known variance.

    Prior: mu ~ N(mu0, tau0^2). Likelihood of n observations with
    known variance sigma^2 and sample mean xbar: proportional (in mu)
    to exp(-n*(mu-xbar)^2 / (2*sigma^2)). Completing the square in the
    product of the two Gaussian kernels (prior and likelihood, both
    quadratic in mu inside the exponent) gives a third Gaussian, with
    precision (inverse variance) equal to the *sum* of the prior's and
    the data's precisions:

        1/tau_n^2 = 1/tau0^2 + n/sigma^2
        mu_n = tau_n^2 * (mu0/tau0^2 + n*xbar/sigma^2)

    i.e. the posterior mean is a precision-weighted average of the
    prior mean and the sample mean: more data, or a smaller sigma^2,
    shifts the posterior toward the data; a tighter prior (smaller
    tau0^2) resists that shift more.
    """

    def __init__(self, mu0: float, tau0_sq: float):
        if tau0_sq <= 0:
            raise ValueError("tau0_sq must be positive")
        self.mu0 = mu0
        self.tau0_sq = tau0_sq

    def update(self, sample_mean: float, sigma_sq: float, n: int) -> "NormalNormal":
        if sigma_sq <= 0 or n <= 0:
            raise ValueError("sigma_sq and n must be positive")
        posterior_precision = 1.0 / self.tau0_sq + n / sigma_sq
        posterior_var = 1.0 / posterior_precision
        posterior_mean = posterior_var * (self.mu0 / self.tau0_sq + n * sample_mean / sigma_sq)
        return NormalNormal(posterior_mean, posterior_var)

    def mean(self) -> float:
        return self.mu0

    def variance(self) -> float:
        return self.tau0_sq

    def posterior_predictive(self, sigma_sq: float) -> Tuple[float, float]:
        """Mean and variance of a single new observation's predictive
        distribution, N(mu_n, tau_n^2 + sigma^2): the posterior's own
        uncertainty about mu (tau_n^2) adds to the data's inherent
        observation noise (sigma^2), since a new draw is
        X_new = mu + noise with mu itself uncertain, and the variance
        of a sum of independent sources of uncertainty is the sum of
        their variances.
        """
        return self.mu0, self.tau0_sq + sigma_sq
