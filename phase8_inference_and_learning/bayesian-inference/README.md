# bayesian-inference

A Bayesian inference engine built from scratch: the three classic
conjugate families (Beta-Binomial, Gamma-Poisson, Normal-Normal),
brute-force grid approximation for the general non-conjugate case,
a Metropolis-Hastings MCMC sampler with proper convergence diagnostics,
and Bayesian model comparison via Bayes factors, applied throughout to
real EUR/USD data.

This repository is independent of the sibling repositories and reuses
none of their code.

## Project structure

```
bayesian-inference/
    README.md
    data/DEXUSEU_returns.csv
    conjugate_models.py       Beta-Binomial, Gamma-Poisson, Normal-Normal
    grid_approximation.py     brute-force numerical Bayesian inference
    metropolis_hastings.py    MCMC sampler, ESS, Gelman-Rubin R-hat
    model_comparison.py       Bayes factors, harmonic mean estimator
    fx_application.py         all four methods applied to EUR/USD data
    test_*.py
    cli.py
```

## Status

76 tests, 18/18 verify checks, all passing.

## Setup

Python 3.11+. All inference code uses only the standard library.
`cli.py verify` uses numpy and scipy as independent oracles.

```bash
pip install --break-system-packages numpy scipy
```

## Running tests

```bash
python3 -m pytest test_conjugate_models.py test_grid_approximation.py \
    test_metropolis_hastings.py test_model_comparison.py test_fx_application.py -v
```

## Running the oracle

```bash
python3 cli.py verify
```

## The data

`data/DEXUSEU_returns.csv`: daily EUR/USD exchange rates from the
Federal Reserve (FRED series DEXUSEU), 1999-01-04 to 2026-08-21
(6,931 price levels, 6,930 daily returns), the same file used in the
sibling repositories.

## Mathematical background

### Bayes' theorem and the three conjugate families

Bayes' theorem, $P(\theta\mid D) \propto P(D\mid\theta)P(\theta)$, says
the posterior is proportional to likelihood times prior. A
**conjugate** prior is one whose functional form survives this
multiplication: the posterior lands back in the same family as the
prior, with updated parameters, and the (generally hard) normalising
integral never needs to be touched.

**Beta-Binomial.** Prior $\theta\sim\text{Beta}(a,b)$, likelihood of
$k$ successes in $n$ trials $\propto \theta^k(1-\theta)^{n-k}$.
Multiplying gives a posterior $\propto\theta^{a+k-1}(1-\theta)^{b+n-k-1}$,
exactly $\text{Beta}(a+k,\,b+n-k)$: the prior's $(a,b)$ behave like
$a-1$ prior successes and $b-1$ prior failures, simply incremented by
the observed data. The posterior predictive for $y$ successes in $m$
future trials integrates the Binomial likelihood against the Beta
posterior, giving the Beta-Binomial compound distribution
$P(Y=y)=\binom{m}{y}B(y+a,m-y+b)/B(a,b)$.

**Gamma-Poisson.** Prior $\lambda\sim\text{Gamma}(a,\text{rate}=b)$,
likelihood of a count total $S$ over $n$ intervals
$\propto\lambda^S e^{-n\lambda}$. The product gives
$\text{Gamma}(a+S,\,b+n)$: $a$ acts like $a-1$ prior events over $b$
prior intervals of exposure. The posterior predictive integrates the
Poisson likelihood against the Gamma posterior and is a **Negative
Binomial**, the standard way overdispersion relative to a plain
Poisson model gets represented: mixing a Poisson rate over a Gamma
distribution of plausible rates produces more variance than a fixed-
rate Poisson would, exactly the property the EUR/USD application below
exploits directly.

**Normal-Normal.** Prior $\mu\sim N(\mu_0,\tau_0^2)$, likelihood with
known variance $\sigma^2$ and sample mean $\bar x$ over $n$
observations. Completing the square in the product of two Gaussian
kernels gives a third Gaussian with **precision** (inverse variance)
equal to the sum of the prior's and the data's precisions:
$1/\tau_n^2 = 1/\tau_0^2 + n/\sigma^2$, and posterior mean a precision-
weighted average of prior mean and sample mean. More data, or a
smaller $\sigma^2$, pulls the posterior toward the data; a tighter
prior resists that pull more.

### Grid approximation: Bayesian inference with no conjugacy required

For a single (or low-dimensional) parameter, `grid_approximation.py`
evaluates $\log P(\theta) + \log P(D\mid\theta)$ at every point on a
fine grid, then normalises. This needs no closed form and no
assumption about the posterior's shape, at the cost of scaling
exponentially poorly in the number of parameters (a grid fine enough
to resolve one parameter well needs that many points *per dimension*,
compounding multiplicatively). `log_sum_exp` normalises without
overflow by factoring out the largest log-density before
exponentiating the rest, which matters here because a log-posterior
evaluated across thousands of grid points routinely produces values
whose direct exponentials would overflow or underflow a float. Grid
approximation is used in this project both as a method in its own
right and as an independent ground-truth check on the conjugate
closed forms (verified to agree with `conjugate_models.BetaBinomial`
to three decimal places on a shared example).

### Metropolis-Hastings: sampling when grids don't scale

For a symmetric (Gaussian random-walk) proposal, the Metropolis-
Hastings acceptance ratio is just the target density ratio
$p(\theta')/p(\theta)$, since the proposal densities cancel out of the
general Metropolis-Hastings ratio exactly when the proposal is
symmetric. Crucially, this ratio never needs the target's normalising
constant, since it cancels between numerator and denominator: the
target can be known only up to a proportionality constant, which is
exactly the situation an unnormalised posterior is always in. Working
in log-space throughout (`log_posterior(theta') - log_posterior(theta)`
compared against `log(uniform(0,1))`) avoids ever exponentiating a
potentially extreme log-density directly.

**Two diagnostics matter more than the raw chain itself.** The
**effective sample size** (ESS) accounts for the fact that consecutive
MCMC draws are correlated (a random-walk proposal moves gradually, so
neighbouring samples resemble each other): $\text{ESS} = n / (1 + 2
\sum_k \rho_k)$, using Geyer's initial positive sequence rule to
truncate the lag-$k$ autocorrelation sum as soon as a consecutive pair
sums negative (autocorrelation estimates from a finite chain get
noisy, and can spuriously flip sign, far out in the lag). The
**Gelman-Rubin R-hat** diagnostic runs several chains from different
starting points and compares within-chain to between-chain variance:
converged chains should roughly agree (R-hat near 1); chains stuck in
different regions inflate the between-chain variance and push R-hat
well above 1. Both are checked directly, not just implemented: a
converged multi-chain run gives R-hat under 1.05, and a deliberately
constructed set of non-mixing chains (constant offsets with only tiny
internal noise) gives R-hat well above 2.

**A concrete tuning exercise, not just the theory.** Fitting a
Student-t model's degrees of freedom and scale jointly (see the
application below) with generic, untuned proposal standard deviations
gave a 5.9% acceptance rate and an effective sample size of only about
200 out of 6,000 post-burn-in draws: the random-walk proposal was
badly mismatched to the posterior's actual shape at this sample size.
Retuning the per-parameter proposal standard deviations (smaller steps
for both degrees of freedom and scale) brought acceptance up to 29.2%,
squarely inside the classical 20-40% target range for random-walk
Metropolis, and roughly tripled the effective sample size for the same
number of post-burn-in draws. This is not a fixed setting this project
solves once; a different dataset size, likelihood, or parameterisation
would need its own re-tuning, since no adaptive step-size mechanism is
implemented here.

### Bayesian model comparison: Bayes factors, and a cautionary tale

The **Bayes factor** $P(D\mid M_1)/P(D\mid M_2)$ compares two models'
marginal likelihoods (the likelihood integrated over the prior, not
evaluated at a point estimate), and Jeffreys' descriptive scale
(barely worth mentioning, substantial, strong, decisive) gives a
conventional way to describe its strength in words. For the conjugate
families here, the marginal likelihood is itself closed-form (the same
Beta-function or Gamma-function identity behind the posterior
predictive, evaluated at the *prior's* parameters instead of the
posterior's), so comparing two prior specifications, or checking how
much a confidently-wrong prior costs against an uninformative one, is
exact arithmetic.

**No closed form exists once a model leaves the conjugate families**
(a Student-t likelihood, for instance, has no conjugate prior for
either its degrees of freedom or its scale). One tempting shortcut
that needs nothing but posterior *samples* is the **harmonic mean
estimator**, and this project implements it specifically to
demonstrate, against a known closed-form answer, why it has earned its
reputation as (in Radford Neal's words) "the worst Monte Carlo method
ever devised." Tested against the exact Beta-Binomial evidence with 20
independent replications of 2,000 posterior draws each: the estimator
is systematically biased **above** the true log-evidence (every
replication's mean estimate exceeded the true value, by about 0.1-0.4
nats), and its standard error does not shrink reliably with more
samples the way a well-behaved Monte Carlo estimator's would: going from 500 to 16,000 samples per estimate (a 32-fold increase,
which should shrink the standard error by roughly $\sqrt{32}\approx5.7\times$
for a healthy estimator) produced no such improvement in this
project's own test runs. Both failures trace to the same mechanism:
the estimator is dominated by whichever sampled $\theta_i$ happens to
have the smallest likelihood, since $1/L(\theta_i)$ blows up there, and
posterior samples occasionally land in low-likelihood tail regions
purely by chance, giving the estimator effectively infinite variance
in principle and highly erratic variance in any finite run.

## Application: four methods, one EUR/USD dataset

**Beta-Binomial: is EUR/USD direction a fair coin, tracked in real
time?** Starting from a uniform Beta(1,1) prior and updating
sequentially through the full history, the 95% credible interval
narrows from width 0.34 at $n=30$ to width 0.024 at $n=6{,}930$, and
the posterior mean settles at 0.489 (95% CI: [0.477, 0.500]). This is
the Bayesian counterpart to `stochastic-processes`' frequentist
estimate of the same quantity: instead of a point estimate with a
confidence interval computed after the fact, the full posterior
distribution narrows in step by step as each day's outcome arrives.

**Normal-Normal: is there a drift?** With an uninformative prior, the
posterior mean daily log return is $-1.58\times10^{-6}$ with posterior
standard deviation $6.91\times10^{-5}$: zero sits only 0.023 posterior
standard deviations away from the posterior mean, about as
unambiguous a "no meaningful drift" conclusion as a posterior can give, consistent with every sibling repository's finding.

**Gamma-Poisson: a posterior predictive check reproduces the
volatility-clustering finding, from a genuinely Bayesian angle.**
Fitting a Gamma-Poisson model to the rate of large moves (top 5% by
absolute return, the identical definition used in
`stochastic-processes/project3_poisson_processes`) per 20-day window
gives a posterior rate of $1.003\pm0.054$ events per window. But the
posterior predictive distribution of window counts does not match what
actually happened: the empirical variance across windows (2.107) runs
more than double the empirical mean (1.003), and the observed count
table shows both **more windows with zero large moves** (172 observed
vs. 127.1 predicted) **and more windows with several** (a combined 14
windows with 5 or more large moves, against a Poisson-implied
probability of essentially zero for most of that range) than a fitted
Poisson model's own posterior predictive distribution says should
happen. This is a posterior predictive check failing in the most
literal sense the term is used for, and it is the same conclusion
`stochastic-processes`' frequentist Fisher dispersion test reached
(index of dispersion dramatically above 1), now reached by asking a
directly Bayesian question: does a fitted model's own predictions
match reality?

**Metropolis-Hastings: how fat are the tails, really?** Jointly
estimating a zero-mean Student-t distribution's degrees of freedom and
scale via MCMC (no conjugate prior exists for either, so this is a
genuine use of the sampler rather than a demonstration that could have
used a closed form instead) gives a posterior degrees-of-freedom
estimate of $4.75\pm0.32$: comfortably, precisely bounded away from
the Gaussian limit ($\text{df}\to\infty$), confirming the well-known
"financial returns are leptokurtic" stylised fact directly from this
data. The fitted Student-t model's log-likelihood (26,164.8) exceeds
the best-fitting Normal model's own log-likelihood at its maximum-
likelihood estimate (25,916.2) by nearly 250 nats, an
overwhelming preference for the fat-tailed model by any standard.
Worth stating precisely rather than over-interpreting: this finding is
entirely consistent with, and does not contradict, `stochastic-
processes`' conclusion that daily returns are close to a fair,
memoryless coin in their *sign*. Volatility clustering (which
`stochastic-processes`, `black-scholes`, and the Gamma-Poisson result
above all establish independently) is well known to inflate the
*unconditional* kurtosis of a return series even when each day's
return, conditional on the currently prevailing volatility regime, is
close to Gaussian: a mixture of Normals with different, time-varying
variances is itself fat-tailed in aggregate, without requiring any
single day's conditional distribution to be fat-tailed on its own.

## Limitations

- `BetaBinomial.credible_interval` inverts the Beta CDF by numerical
  integration over a fixed grid (`n_grid=20000` by default) rather
  than a proper root-finder; this is accurate to within about
  $1/n_{\text{grid}}$ in $\theta$, which was sufficient for every
  credible interval reported above but would need a finer grid (at
  proportionally higher cost) for a posterior concentrated in a much
  narrower region of $[0,1]$ than any case used here.
- The Student-t application's proposal standard deviations were tuned
  once, by hand, against the full 6,930-observation dataset at one
  particular sample size; they are not guaranteed to still sit in the
  20-40% acceptance sweet spot for a meaningfully different dataset
  size or a different likelihood, since no adaptive proposal
  mechanism is implemented.
- A full Student-t MCMC fit on the complete real dataset (thousands of
  post-burn-in steps, each evaluating a Student-t log-density at all
  6,930 observations) takes on the order of several minutes in pure
  Python; the CLI's default step counts are set lower for interactive
  use, and the exact figures quoted above come from a longer run
  recorded once rather than reproduced on every invocation.
- The harmonic mean estimator's bias and variance instability are
  demonstrated on one specific conjugate example (chosen precisely
  because its true evidence is known in closed form); the *direction*
  of the bias and the qualitative unreliability of its variance are
  general, well-documented properties of the estimator, but the exact
  magnitudes reported here are specific to this example's posterior
  concentration and would differ for a more diffuse or more
  concentrated posterior.

## CLI examples

```bash
# Beta-Binomial conjugate update
python3 cli.py betabinomial --a 1 --b 1 --successes 7 --trials 10

# Gamma-Poisson conjugate update
python3 cli.py gammapoisson --a 1 --b 1 --count 15 --intervals 5

# Normal-Normal conjugate update
python3 cli.py normalnormal --sample-mean 2.5 --sigma-sq 4 --n 20

# Metropolis-Hastings on a known Normal target (sanity check)
python3 cli.py mcmc --target-mean 3 --target-sd 1.5

# Sequential Bayesian up-probability estimation on real EUR/USD data
python3 cli.py fxbeta

# Bayesian drift estimation
python3 cli.py fxdrift

# Gamma-Poisson posterior predictive check on large moves
python3 cli.py fxpoisson

# Student-t fit via MCMC (several minutes at full sample size/steps)
python3 cli.py fxstudentt --n-samples 2000 --burn-in 500

# Cross-check every formula against scipy/numpy oracles, calibration
# checks, and synthetic ground-truth recovery
python3 cli.py verify
```
