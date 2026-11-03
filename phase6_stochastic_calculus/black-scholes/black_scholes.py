import csv
import math
import statistics
from typing import List, Dict


def standard_normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def standard_normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def simulate_gbm_terminal(s0: float, mu: float, sigma: float, t: float, z: float) -> float:
    """S_t for geometric Brownian motion dS = mu*S*dt + sigma*S*dW.

    Ito's lemma applied to ln(S) removes the S-dependence from the
    diffusion coefficient: d(ln S) = (mu - sigma^2/2) dt + sigma dW,
    which integrates directly (it is now an ordinary drift-diffusion
    in ln S, not a multiplicative one) to give the lognormal solution

        S_t = S_0 * exp((mu - sigma^2/2) t + sigma * sqrt(t) * Z),

    for Z a standard normal variate. The sigma^2/2 correction is
    exactly the Ito correction term picked up by the second-order term
    in Ito's lemma; ignoring it (using ordinary calculus) would
    overstate the drift of S itself, since E[S_t] = S_0 exp(mu*t)
    still holds, but the median and the drift of ln(S_t) do not equal
    the naive (mu*t) that ordinary calculus would suggest.
    """
    return s0 * math.exp((mu - 0.5 * sigma ** 2) * t + sigma * math.sqrt(t) * z)


def d1_d2(s: float, k: float, t: float, r: float, sigma: float, q: float = 0.0):
    """The two standardised distances used throughout the Black-Scholes
    / Garman-Kohlhagen formula.

    d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    These arise directly from evaluating the risk-neutral expectation
    E^Q[max(S_T-K, 0)] with S_T lognormal under the risk-neutral
    measure Q (drift r-q instead of the real-world mu): d2 is the
    probability (in standardised units) that the option finishes in
    the money, and d1 is the same distance shifted by one factor of
    sigma*sqrt(T), which appears because differentiating S_T under the
    lognormal density shifts the effective mean by sigma^2*T.
    """
    if s <= 0 or k <= 0 or t <= 0 or sigma <= 0:
        raise ValueError("s, k, t, sigma must be positive")
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return d1, d2


def call_price(s: float, k: float, t: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Garman-Kohlhagen call price (Black-Scholes with a continuous
    yield q on the underlying; for FX options, q is the foreign
    risk-free rate and r is the domestic risk-free rate, since holding
    the foreign currency earns interest exactly the way a dividend-
    paying stock does).

    Derived from discounting the risk-neutral expectation of the
    payoff: e^{-rT} E^Q[max(S_T-K,0)]. Splitting the expectation at
    S_T=K and evaluating each piece against the lognormal risk-neutral
    density gives the closed form

        C = S e^{-qT} N(d1) - K e^{-rT} N(d2).

    The first term is the present value of receiving S_T conditional
    on exercise (with N(d1), not N(d2), because of the sigma^2*T shift
    in expectation under the S_T-weighted measure); the second is the
    present value of paying the strike, weighted by the risk-neutral
    probability of exercise, N(d2).
    """
    d1, d2 = d1_d2(s, k, t, r, sigma, q)
    return s * math.exp(-q * t) * standard_normal_cdf(d1) - k * math.exp(-r * t) * standard_normal_cdf(d2)


def put_price(s: float, k: float, t: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Garman-Kohlhagen put price, by the same argument as the call
    with the payoff max(K-S_T, 0):

        P = K e^{-rT} N(-d2) - S e^{-qT} N(-d1).
    """
    d1, d2 = d1_d2(s, k, t, r, sigma, q)
    return k * math.exp(-r * t) * standard_normal_cdf(-d2) - s * math.exp(-q * t) * standard_normal_cdf(-d1)


def put_call_parity_gap(s: float, k: float, t: float, r: float, sigma: float, q: float = 0.0) -> float:
    """C - P - (S e^{-qT} - K e^{-rT}), which must be exactly zero for
    any correctly implemented call and put under the same model
    parameters. Put-call parity itself follows from a static
    replication argument (a forward contract equals a long call plus
    a short put at the same strike) that holds regardless of the
    stochastic model used to price either leg, so this is an
    unconditional internal consistency check, not merely a property of
    the Black-Scholes formula specifically.
    """
    c = call_price(s, k, t, r, sigma, q)
    p = put_price(s, k, t, r, sigma, q)
    forward_value = s * math.exp(-q * t) - k * math.exp(-r * t)
    return c - p - forward_value


def call_delta(s: float, k: float, t: float, r: float, sigma: float, q: float = 0.0) -> float:
    """dC/dS = e^{-qT} N(d1). This is the number of units of the
    underlying a delta-hedged short call position must hold; it comes
    directly from differentiating the call formula, using the fact
    that the terms involving dN(d1)/dS and dN(d2)/dS cancel exactly
    (a standard, slightly delicate identity: S*phi(d1) = K*e^{-(r-q)T}*phi(d2)
    following from the definitions of d1, d2), leaving only the N(d1)
    term.
    """
    d1, _ = d1_d2(s, k, t, r, sigma, q)
    return math.exp(-q * t) * standard_normal_cdf(d1)


def put_delta(s: float, k: float, t: float, r: float, sigma: float, q: float = 0.0) -> float:
    """dP/dS = -e^{-qT} N(-d1) = e^{-qT}(N(d1) - 1), by the same
    cancellation as call_delta applied to the put formula, or
    equivalently by differentiating put-call parity: delta_put =
    delta_call - e^{-qT}.
    """
    d1, _ = d1_d2(s, k, t, r, sigma, q)
    return math.exp(-q * t) * (standard_normal_cdf(d1) - 1.0)


def gamma(s: float, k: float, t: float, r: float, sigma: float, q: float = 0.0) -> float:
    """d^2C/dS^2 = d^2P/dS^2 = e^{-qT} * phi(d1) / (S * sigma * sqrt(T)).
    Gamma is identical for calls and puts at the same strike, since
    put-call parity's right-hand side (a forward) is linear in S and
    so contributes nothing to the second derivative.
    """
    d1, _ = d1_d2(s, k, t, r, sigma, q)
    return math.exp(-q * t) * standard_normal_pdf(d1) / (s * sigma * math.sqrt(t))


def vega(s: float, k: float, t: float, r: float, sigma: float, q: float = 0.0) -> float:
    """dC/dsigma = dP/dsigma = S * e^{-qT} * phi(d1) * sqrt(T), also
    shared between calls and puts for the same reason as gamma. This
    is the raw partial derivative with respect to sigma expressed as a
    decimal (e.g. 0.20 for 20% volatility); trading desks often quote
    vega per one-percentage-point move, i.e. this value divided by 100.
    """
    d1, _ = d1_d2(s, k, t, r, sigma, q)
    return s * math.exp(-q * t) * standard_normal_pdf(d1) * math.sqrt(t)


def call_theta(s: float, k: float, t: float, r: float, sigma: float, q: float = 0.0) -> float:
    """dC/d(calendar time) = -dC/dT (theta is usually quoted as the
    rate of value loss as time passes, the negative of the derivative
    with respect to remaining maturity T):

        theta = -S e^{-qT} phi(d1) sigma / (2 sqrt(T))
                - r K e^{-rT} N(d2) + q S e^{-qT} N(d1).
    """
    d1, d2 = d1_d2(s, k, t, r, sigma, q)
    term1 = -s * math.exp(-q * t) * standard_normal_pdf(d1) * sigma / (2.0 * math.sqrt(t))
    term2 = -r * k * math.exp(-r * t) * standard_normal_cdf(d2)
    term3 = q * s * math.exp(-q * t) * standard_normal_cdf(d1)
    return term1 + term2 + term3


def put_theta(s: float, k: float, t: float, r: float, sigma: float, q: float = 0.0) -> float:
    """theta = -S e^{-qT} phi(d1) sigma/(2 sqrt(T)) + r K e^{-rT} N(-d2)
    - q S e^{-qT} N(-d1), by the same derivation as call_theta applied
    to the put formula.
    """
    d1, d2 = d1_d2(s, k, t, r, sigma, q)
    term1 = -s * math.exp(-q * t) * standard_normal_pdf(d1) * sigma / (2.0 * math.sqrt(t))
    term2 = r * k * math.exp(-r * t) * standard_normal_cdf(-d2)
    term3 = -q * s * math.exp(-q * t) * standard_normal_cdf(-d1)
    return term1 + term2 + term3


def call_rho_domestic(s: float, k: float, t: float, r: float, sigma: float, q: float = 0.0) -> float:
    """dC/dr = K T e^{-rT} N(d2), the sensitivity to the domestic
    (discounting) rate."""
    _, d2 = d1_d2(s, k, t, r, sigma, q)
    return k * t * math.exp(-r * t) * standard_normal_cdf(d2)


def put_rho_domestic(s: float, k: float, t: float, r: float, sigma: float, q: float = 0.0) -> float:
    """dP/dr = -K T e^{-rT} N(-d2)."""
    _, d2 = d1_d2(s, k, t, r, sigma, q)
    return -k * t * math.exp(-r * t) * standard_normal_cdf(-d2)


def call_rho_foreign(s: float, k: float, t: float, r: float, sigma: float, q: float = 0.0) -> float:
    """dC/dq = -T S e^{-qT} N(d1), the sensitivity to the foreign
    (dividend-like) rate, relevant for FX options where q is the
    foreign currency's own risk-free rate."""
    d1, _ = d1_d2(s, k, t, r, sigma, q)
    return -t * s * math.exp(-q * t) * standard_normal_cdf(d1)


def put_rho_foreign(s: float, k: float, t: float, r: float, sigma: float, q: float = 0.0) -> float:
    """dP/dq = T S e^{-qT} N(-d1)."""
    d1, _ = d1_d2(s, k, t, r, sigma, q)
    return t * s * math.exp(-q * t) * standard_normal_cdf(-d1)


def numerical_delta(price_fn, s: float, k: float, t: float, r: float, sigma: float,
                     q: float = 0.0, h: float = 1e-4) -> float:
    """Central-difference delta, used to check the closed-form Greeks
    above against a completely different (and much less efficient)
    computation that does not rely on having differentiated the
    formula correctly by hand.
    """
    return (price_fn(s + h, k, t, r, sigma, q) - price_fn(s - h, k, t, r, sigma, q)) / (2 * h)


def numerical_gamma(price_fn, s: float, k: float, t: float, r: float, sigma: float,
                     q: float = 0.0, h: float = 1e-3) -> float:
    return (price_fn(s + h, k, t, r, sigma, q) - 2 * price_fn(s, k, t, r, sigma, q)
            + price_fn(s - h, k, t, r, sigma, q)) / (h ** 2)


def numerical_vega(price_fn, s: float, k: float, t: float, r: float, sigma: float,
                    q: float = 0.0, h: float = 1e-5) -> float:
    return (price_fn(s, k, t, r, sigma + h, q) - price_fn(s, k, t, r, sigma - h, q)) / (2 * h)


def numerical_theta(price_fn, s: float, k: float, t: float, r: float, sigma: float,
                     q: float = 0.0, h: float = 1e-5) -> float:
    return -(price_fn(s, k, t + h, r, sigma, q) - price_fn(s, k, t - h, r, sigma, q)) / (2 * h)


def load_log_returns(csv_path: str, column: str = "log_return") -> List[float]:
    values = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row[column]))
    return values


def calibrate_annualized_volatility(log_returns: List[float], trading_days: int = 252) -> float:
    """Annualised volatility from daily log returns: since Var(sum of
    n iid increments) = n * Var(one increment) under the GBM model's
    constant-volatility assumption, the daily variance scales up by
    the number of trading days in a year to annualise it, and taking
    the square root converts back to a volatility (standard deviation)
    scale. This is the direct discrete-data analogue of Project 4's
    Brownian-motion scaling property.
    """
    daily_sigma = statistics.pstdev(log_returns)
    return daily_sigma * math.sqrt(trading_days)


def calibrate_annualized_drift(log_returns: List[float], trading_days: int = 252) -> float:
    """Annualised drift of log-price: the daily mean log return, scaled
    up linearly (not by sqrt, since means of iid sums scale linearly
    with the number of terms, unlike standard deviations).
    """
    return statistics.mean(log_returns) * trading_days
