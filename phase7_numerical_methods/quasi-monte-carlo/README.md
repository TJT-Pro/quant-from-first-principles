# quasi-monte-carlo

A quasi-Monte Carlo engine built from scratch: the van der Corput and
Halton low-discrepancy sequences, the L2-star discrepancy measure (in
exact closed form, via Warnock's formula), Monte Carlo and
Quasi-Monte Carlo numerical integration with empirically measured
convergence orders, an inverse-normal-CDF Gaussian transform built
specifically for QMC use, and European option pricing calibrated to
real EUR/USD volatility.

This repository is independent of the sibling repositories and reuses
none of their code, including its own from-scratch closed-form
Black-Scholes formula.

## Project structure

```
quasi-monte-carlo/
    README.md
    data/DEXUSEU_returns.csv
    low_discrepancy.py     van der Corput and Halton sequences
    discrepancy.py         L2-star discrepancy, closed form and Monte Carlo cross-check
    integration.py         Monte Carlo and QMC integration, convergence order estimation
    gaussian_transform.py  inverse normal CDF (Beasley-Springer-Moro)
    fx_application.py      MC vs QMC pricing of a real EUR/USD option
    test_*.py
    cli.py
```

## Status

56 tests, 10/10 verify checks, all passing.

## Setup

Python 3.11+. All sequence, discrepancy, and integration code uses
only the standard library. `cli.py verify` uses numpy and scipy as
independent oracles.

```bash
pip install --break-system-packages numpy scipy
```

## Running tests

```bash
python3 -m pytest test_low_discrepancy.py test_discrepancy.py test_integration.py \
    test_gaussian_transform.py test_fx_application.py -v
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

### Van der Corput and Halton: sequences designed to avoid clustering

The van der Corput sequence reflects the base-b digits of n around the
"decimal" point: writing n = sum_k d_k*b^k gives the n-th term as
sum_k d_k*b^{-k-1}, the same digits read in reverse and reinterpreted
as a fraction. Incrementing n changes mostly the least-significant
digit, which after reflection becomes the most significant digit of
the resulting value, so consecutive points jump by roughly 1/b and
systematically fill gaps left by earlier points, rather than
clustering near previous points purely by chance the way pseudo-random
draws occasionally do. The Halton sequence extends this to multiple
dimensions by giving each dimension its own coprime (in practice,
prime) base; matched against the standard sequence in the literature,
base-2 van der Corput reproduces the textbook values
0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875, 0.0625 exactly.

### L2-star discrepancy: an exact closed form, and a real formula bug caught by it being impossible

The L2-star discrepancy squared is
(D_N^*)^2 = integral_{[0,1]^d} (F_N(t)-V(t))^2 dt, where F_N(t) is a
point set's empirical fraction inside the box [0,t) and V(t) is that
box's true volume: a point set fills space evenly exactly when these
match everywhere. Expanding the square and integrating each of the
three resulting terms directly (Warnock 1972) gives a closed form
needing no supremum search or numerical integration:

    (D_N^*)^2 = (1/3)^d - (2/N)*sum_i prod_j[(1-x_ij^2)/2]
                + (1/N^2)*sum_{i,k} prod_j[1-max(x_ij,x_kj)]

A genuine bug was caught here directly by testing, and it announced
itself impossibly. An earlier version divided by 2 once per point
rather than once per dimension inside the product (treating the whole
d-dimensional term as if a single division by 2 were equivalent to
dividing each of d factors by 2). For a single point at (0.5,0.5) in
2D, this produced -0.2014: a negative squared discrepancy,
mathematically impossible for a quantity defined as an integral of a
squared quantity, which is exactly what made the error obvious rather
than silently wrong. Direct grid-integration of the defining formula
confirmed the correct value is +0.0799; fixing the missing per-
dimension division recovered it exactly. The corrected formula is now
checked against an independent Monte Carlo estimate of the same
defining integral (agreeing within 15% at N=150, d=2 and d=3) as a
standing safeguard against this exact class of error recurring.

Halton's advantage over random points is not just present, it grows
with N. At N=50, random points have 1.75x Halton's discrepancy; by
N=1000, that ratio grows to 7.65x, a direct, measured confirmation
that low-discrepancy sequences fill space increasingly more evenly
than random sampling as more points are added, precisely the property
that gives Quasi-Monte Carlo its edge.

### Monte Carlo vs Quasi-Monte Carlo integration

Plain Monte Carlo integration is unbiased with standard error scaling
as O(N^-0.5) regardless of dimension, by the ordinary central limit
theorem. Quasi-Monte Carlo instead averages the integrand over a
low-discrepancy point set; the Koksma-Hlawka inequality bounds its
error by (the integrand's total variation) times (the point set's
discrepancy), which for Halton decays roughly like (log N)^d/N for
smooth, low-dimensional integrands, asymptotically much faster than
N^-0.5. Measured, not assumed: on f(x)=exp(sum x_j) (exact integral
(e-1)^d), Monte Carlo's empirically fitted order sits at 0.50-0.56
across dimensions 1, 2, and 4 (repeated over many independent runs to
average out a single run's own randomness), matching theory almost
exactly, while QMC's fitted order runs from 0.98 (1D) down to 0.87
(4D), degrading with dimension exactly as the (log N)^d term predicts,
but remaining decisively better than Monte Carlo at every dimension
tested.

### The inverse normal CDF, and why Box-Muller is deliberately avoided

Converting a low-discrepancy sequence's uniform points into Gaussian
variates is done here via the inverse CDF (the Beasley-Springer-Moro
rational approximation, matching Python's own statistics.NormalDist to
within 2.8e-9 across the bulk of the distribution and about 7e-8 even
at extreme tail arguments like u=1e-6), applied independently to each
coordinate. Box-Muller, a common choice for ordinary Monte Carlo, is
deliberately not used here: it pairs two uniforms into two
correlated-by-construction Gaussians via a trigonometric transform,
which would entangle two dimensions of a Halton sequence that were
specifically constructed to be low-discrepancy independently of each
other, undermining the entire point of using a low-discrepancy
sequence in the first place. Tail accuracy matters more here than for
ordinary Monte Carlo, too: a low-discrepancy sequence's earliest
points are, by construction, disproportionately likely to sit near
the extremes of [0,1] (van der Corput's first few terms systematically
explore 0.5, 0.25, 0.75, 0.125, ... before filling in the middle), so
the inverse-CDF transform is evaluated at genuinely extreme arguments
more often, in relative terms, than a uniform pseudo-random sequence
would ever produce.

## Application: pricing a real EUR/USD option, and a real, well-documented limit to QMC's advantage

Volatility calibrated directly from real data: sigma_hat=0.0913 (9.1%
annualized) from the sample standard deviation of daily log returns
across the full 6,930-observation EUR/USD series, pricing a 3-month
at-the-money call (spot 1.1684, matching strike, r=3%; closed-form
price 0.024945).

QMC still wins decisively in absolute terms. At every sample size
tested (200 to 12,800 paths), QMC's error is smaller than Monte
Carlo's root-mean-square error (averaged over 20 independent runs at
each size, since a single MC run's error is itself random): 5-16x
smaller at N=200 growing to roughly 12x smaller at N=12,800.

But the fitted convergence order is genuinely, measurably lower than
the smooth-integrand case, and the reason is precise, not vague. QMC's
empirical order on this option-pricing problem comes out at 0.63,
well below the 0.87-0.98 range measured on the smooth exponential test
integrand in integration.py. The reason is the option payoff itself:
max(S_T-K,0) has a kink exactly at S_T=K, where it is continuous but
not differentiable. The Koksma-Hlawka bound QMC's advantage rests on
scales with the integrand's total variation, and a kinked function's
total variation properties are markedly worse than a smooth
exponential's, degrading the achievable convergence rate even though
QMC's absolute error still comfortably beats Monte Carlo's throughout.
This is a well-documented phenomenon in the QMC finance literature (a
standard motivation for smoothing techniques such as conditional Monte
Carlo or Brownian bridge constructions specifically for kinked
payoffs, not implemented here), confirmed here directly on real,
calibrated market data rather than assumed from the literature.

## Limitations

- `low_discrepancy.halton_sequence` uses plain Halton with no
  scrambling or leaping; in higher dimensions (beyond roughly 6-8),
  plain Halton is known to develop visible correlation between
  dimensions using large prime bases, a well-documented weakness
  addressed in practice by scrambled variants (Owen scrambling, for
  instance) not implemented here.
- `discrepancy.l2_star_discrepancy_squared` costs O(N^2 * d) per
  evaluation (the double sum over all point pairs), fine for the point
  sets (a few hundred to a couple thousand points) this project
  actually evaluates it on, but impractical for the tens of thousands
  of points a realistic QMC simulation might use in production; a
  practical discrepancy monitor at that scale would need either a
  faster discrepancy measure or a subsampled estimate.
- The option-pricing application above measures, but does not correct
  for, the kinked-payoff degradation in QMC's convergence order;
  smoothing techniques that restore closer-to-smooth-integrand
  convergence rates for exactly this kind of payoff exist in the
  literature and are noted but not implemented here.

## CLI examples

```bash
# Generate a Halton sequence
python3 cli.py sequence --n 10 --dim 3

# Compare random vs Halton point-set discrepancy
python3 cli.py discrepancy --dim 2

# Compare MC vs QMC integration of a function with a known exact integral
python3 cli.py integrate --f "math.exp(x[0])" --dim 1 --exact "math.e-1"

# Compare MC vs QMC European option pricing
python3 cli.py price --s0 100 --k 100 --r 0.05 --sigma 0.2 --t 1.0

# Price a real EUR/USD option via MC vs QMC, with a convergence study
python3 cli.py fxprice --r 0.03 --t 0.25

# Cross-check every sequence, discrepancy formula, and pricing result
# against scipy/numpy oracles and known theory
python3 cli.py verify
```
