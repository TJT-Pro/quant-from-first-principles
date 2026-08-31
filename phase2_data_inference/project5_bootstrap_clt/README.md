# Project 5 - Bootstrap Sampler & CLT Demonstrator

A resampling library built entirely by hand.  No numpy, scipy or pandas
in the code; resampling is done with the standard library's random
module, seeded so every run can be reproduced exactly.

| Tool | What it does |
|---|---|
| Bootstrap | resamples with replacement, confidence intervals four ways |
| Jackknife | leave-one-out bias and standard error |
| Permutation test | p-value by shuffling, no parametric assumptions |
| CLT demonstrator | shows the distribution of sample means converging |

## The three resampling ideas

**Bootstrap.**  Draw many samples with replacement from the data, each
the same size as the data, and recalculate the statistic on every one.
The spread of those bootstrap replicates is an estimate of the
statistic's sampling error, and the confidence interval comes straight
from the empirical distribution:

- `percentile`: the alpha/2 and 1-alpha/2 quantiles of the replicates.
- `basic`: the same quantiles reflected around the observed estimate.
- `normal`: estimate +- z times the bootstrap standard error.
- `BCa`: bias-corrected and accelerated, with a bias term z0 from the
  fraction of replicates below the estimate and an acceleration from a
  jackknife.  This is the interval of choice when the statistic is
  skewed, which is when the plain percentile interval is least honest.

**Jackknife.**  Leave out one observation at a time and recalculate.
The spread of the leave-one-out estimates gives a standard error, and
how far their mean sits from the full-data estimate gives a bias
correction.  For the mean it is exact: the jackknife standard error is
the sample standard deviation over sqrt(n), and the bias is zero.

**Permutation test.**  Pool both groups, shuffle, and recompute the
statistic.  Repeat.  The p-value is the fraction of shuffles that reach
or pass the observed statistic, with the 1/(N+1) convention so it is
never zero.  There is no parametric model anywhere in it; the data
defines the null distribution.

## The CLT demonstrator

The `clt` command draws many samples of size k from a population and
reports the shape of their means.  The central limit theorem says the
sampling distribution of the mean should:

- center on the population mean,
- have standard deviation population_sd / sqrt(k),
- become normal, meaning skewness and excess kurtosis shrink toward
  zero as k grows.

On the real EUR/USD daily returns the convergence is visible in one
table.  At k = 1 the mean-distribution has the raw skew of the returns;
by k = 30 the skewness is near zero and the empirical standard
deviation matches the theoretical one to four digits.

## Files

- `bootstrap_sampler.py` - the library
- `cli.py` - command-line interface
- `test_bootstrap_sampler.py` - the tests
- `data/DEXUSEU_returns.csv` - the real EUR/USD daily returns from FRED,
  same series used in Projects 3 and 4

## Setup

Python 3.11.  The library itself needs only the standard library.  The
`verify` command and the cross-check tests need numpy and scipy:

```bash
pip install --break-system-packages numpy scipy
```

## Usage

```bash
python3 cli.py bootstrap data/DEXUSEU_returns.csv --col return --n 999 --seed 7
python3 cli.py bootstrap data/DEXUSEU_returns.csv --col return --n 999 --method BCa --seed 7
python3 cli.py jackknife data/DEXUSEU_returns.csv --col return
python3 cli.py permtest data/DEXUSEU_returns.csv --col return --split 500 --seed 3
python3 cli.py permtest data/DEXUSEU_returns.csv --a colA --b colB --n 10000
python3 cli.py clt data/DEXUSEU_returns.csv --col return --sample-sizes 1,5,30,100 --seed 5
python3 cli.py verify
```

Every resampling command takes `--seed` and defaults to the mean
statistic; `--statistic` accepts mean, median, std or variance.  The
permutation command compares two named columns, or one column split
into two blocks with `--split`.  For very large inputs the permutation
command picks an adaptive number of shuffles so the pure-Python run
stays within a few seconds; pass `--n` explicitly for more precision.

## Tests

```bash
python3 -m unittest -v
```

42 tests.  Seeded determinism is checked throughout.  The jackknife is
verified against its exact properties (standard error of the mean is
s/sqrt(n), bias of the mean and of the sample variance are zero).  The
permutation test is checked against the exact p-value on a tiny example
enumerated by hand, and the bootstrap percentile interval against scipy.
Coverage simulations verify the percentile interval and the BCa interval
are honest on symmetric and skewed data respectively.

## Self-verification

```bash
python3 cli.py verify
```

19 checks.  The bootstrap percentile and BCa intervals are compared
with scipy on the real returns, the jackknife against the analytic
standard error and a manual leave-one-out computation, the permutation
test against scipy's within Monte Carlo tolerance, and the CLT
functions against the population standard deviation formula and the
empirical 95% normal coverage at k = 100.

## Choices worth knowing about

- The bootstrap percentile interval uses the same linear-interpolation
  quantile convention as numpy, so the cross-check with scipy is on the
  same definition.
- BCa is checked with property tests rather than against scipy directly.
  The bias-correction term is so sensitive to the exact resample drawn
  that two BCa runs with different random streams differ by more than
  Monte Carlo noise; comparing them is not a meaningful test.  Instead,
  BCa must agree with the percentile interval on symmetric data and
  hold nominal coverage on skewed data.
- Sampling for the CLT is with replacement, which keeps the theory
  exact (variance of the mean is population variance over k) rather
  than introducing a finite-population correction.
- The permutation shuffle randomizes only the first len(a) positions.
  This is a partial Fisher-Yates, still a uniform random subset, and it
  halves the work; the mean-difference statistic has an inlined fast
  path.  Even so, permutation tests on tens of thousands of points are
  slow in pure Python, which is why the CLI picks fewer shuffles on big
  inputs rather than hanging.
