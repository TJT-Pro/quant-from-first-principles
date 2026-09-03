# Project 4 - Hypothesis Testing Engine

A hypothesis testing library built entirely by hand.  The normal, t and
chi-squared distributions are implemented from their definitions, the t
and chi-squared CDFs through the incomplete beta and incomplete gamma
functions evaluated with the classic continued fraction and series
expansions from Numerical Recipes.  No numpy, scipy or pandas in the
code; every p-value can be traced back to the formula it came from.

| Test | What the null says |
|---|---|
| `z_test` | population mean equals mu0, sigma known |
| `t_test_one_sample` | population mean equals mu0, sigma unknown |
| `t_test_two_sample` | two group means are equal, pooled or Welch |
| `t_test_paired` | mean of the paired differences is zero |
| `chi2_goodness_of_fit` | observed counts follow the stated distribution |
| `chi2_independence` | rows and columns of a table are independent |
| `binomial_proportion_test` | the true proportion equals p0 |

Also included: t and Wilson confidence intervals for means and
proportions, Cohen's d for effect size, and power and sample-size
calculations for the z test.

## The distributions, from first principles

- The normal CDF uses the error function; the quantile starts from
  Acklam's rational approximation and is refined with Newton steps
  against the exact CDF, so it is accurate to machine precision.
- The t CDF is expressed through the regularized incomplete beta
  function.  The chi-squared CDF goes through the regularized lower
  incomplete gamma.  Both are evaluated with the standard series and
  continued fraction methods, which converge to full double precision
  everywhere in their domains.
- Quantiles for t and chi-squared are found by bisection on their own
  CDFs, so quantile and CDF are mutually consistent by construction.

## What a p-value is, and is not

A p-value answers one question only: how often would random data
produce a test statistic this extreme or more, if the null hypothesis
were true?  A small p-value is evidence against the null.  It is not the
probability the null is true, and it says nothing about whether the
effect is practically important.  The tests here therefore report the
statistic, the p-value and a plain-language verdict at the chosen alpha,
and it is always worth looking at the effect size (Cohen's d) before
concluding anything.

## Files

- `hypothesis_tests.py` - the library
- `cli.py` - command-line interface
- `test_hypothesis_tests.py` - the tests
- `data/DEXUSEU_returns.csv` - the real EUR/USD daily returns from FRED,
  same series used in Project 3

## Setup

Python 3.11.  The library itself needs only the standard library.  The
`verify` command and the cross-check tests need numpy and scipy:

```bash
pip install --break-system-packages numpy scipy
```

## Usage

```bash
python3 cli.py ztest --mean 0.5 --mu0 0 --sigma 1 --n 4
python3 cli.py ttest-one data/DEXUSEU_returns.csv --col return --mu0 0
python3 cli.py ttest-two data/DEXUSEU_returns.csv --a return --b return --equal-var
python3 cli.py ttest-paired data/DEXUSEU_returns.csv --a return --b return
python3 cli.py chisq-fit --observed 100,95,105,98,102,100 --expected 100,100,100,100,100,100
python3 cli.py chisq-ind --matrix "30,10;20,40"
python3 cli.py prop-test --k 60 --n 100 --p0 0.5
python3 cli.py ci-mean data/DEXUSEU_returns.csv --col return
python3 cli.py ci-proportion --k 3 --n 10
python3 cli.py power --effect-size 0.5 --sigma 1 --n 25
python3 cli.py sample-size --effect-size 0.5 --sigma 1
python3 cli.py verify
```

Every test accepts `--alternative two-sided|greater|less` and
`--alpha`, so a one-sided pre-registered hypothesis is as easy to run
as the default two-sided one.

## Tests

```bash
python3 -m unittest -v
```

68 tests.  Distribution functions are checked against hand-computed
known values (t quantiles, chi-squared quantiles) and against scipy over
dense grids.  Every test with a scipy equivalent is cross-checked on
randomized data (one-sample, two-sample with pooled and Welch variance,
paired, goodness of fit, independence).  Textbook examples anchor the
tests: Mendel's pea data gives chi2 = 0.47 on 3 degrees of freedom, the
Wilson interval for 0 successes in 10 trials tops out at 0.2775.

The mean confidence interval is tested for honesty with a 1000-sample
coverage simulation, and the Wilson interval is verified to be exactly
the inversion of the z test it claims to be, by checking that testing
against either boundary returns p = alpha.

## Self-verification

```bash
python3 cli.py verify
```

3729 checks.  The distributions are compared against scipy over dense
grids of points, every test is compared against its scipy equivalent
using the real 6930 EUR/USD returns (one-sample and paired tests on the
returns, two-sample on the first and second halves, plus randomized
goodness-of-fit and independence tables), and the Wilson and power
formulas are checked against the manual calculations.

## Choices worth knowing about

- Welch's t test is the default for two groups because it does not
  assume the two populations share a variance; the pooled version is
  available with `--equal-var`.
- The chi-squared independence test reports the un-corrected Pearson
  statistic.  Some packages apply Yates's correction automatically to
  2x2 tables; this library does not, on the principle that the test
  should be exactly what its formula says.
- Degenerate data is handled explicitly rather than silently.  A
  constant sample equal to the null mean gives t = 0 and p = 1; a
  constant sample away from it gives p = 0.  Two constant groups make
  the two-sample test undefined, and that raises an error. The
  "constant" check compares `min(data) == max(data)` directly rather
  than testing a computed standard deviation against `0.0` -- an
  earlier version did the latter and passed on Python 3.12 while
  silently failing on Python 3.11, since the two versions round
  identical floats differently when summed. Full account in
  `project8_strategy_backtester`'s README.
- The Wilson interval replaces the Wald interval for proportions,
  because the Wald interval has poor coverage near 0 and 1.
- Power calculations are for the known-sigma z test, which is the
  honest baseline for planning a study.  The two-sided power includes
  both tails and is a standard approximation.
