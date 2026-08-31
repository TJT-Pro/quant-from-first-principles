# Project 3 - Descriptive Statistics Library

A statistics module written entirely by hand.  No numpy, pandas or scipy
anywhere in the code; the formulas are implemented from their definitions
so each one can be traced step by step.  The library is then run against a
real dataset and every output is verified against numpy equivalents.

| Statistic | Notes |
|---|---|
| mean, median, mode | mode returns all tied modes, median handles even length |
| range, q1, q3, iqr | quantiles use linear interpolation, matching numpy |
| variance, stddev | both population (n) and sample (n-1) versions |
| skewness | moment formula, biased and unbiased |
| kurtosis | excess kurtosis by default, biased and unbiased |
| covariance, correlation | population and sample covariance, Pearson r |

## The dataset

The real dataset is the daily EUR/USD exchange rate from the Federal
Reserve (FRED series DEXUSEU), downloaded as a CSV.  It runs from
1999-01-04 to 2026-08-21.

- `data/DEXUSEU.csv` is the raw price series, 7210 rows including US
  holidays, which FRED lists with an empty price.
- Those 279 holiday rows carry no price (the market was closed), so they
  are dropped, leaving 6931 trading-day prices.
- `data/DEXUSEU_returns.csv` holds the daily returns derived from those
  prices: `r_t = P_t / P_{t-1} - 1`, giving 6930 returns.

The derivation is deterministic and documented in the README, and the
same `returns` command can regenerate it from the raw prices.

## The formulas

Sample statistics use the n-1 denominator (Bessel's correction):

    variance (sample)      sum((x - mean)^2) / (n - 1)
    variance (population)  sum((x - mean)^2) / n
    stddev                 sqrt(variance)

Skewness is the third standardized moment:

    g1 = m3 / m2^(3/2),  with m_k = sum((x - mean)^k) / n

The unbiased version multiplies by `sqrt(n(n-1)) / (n-2)`.

Excess kurtosis is the fourth standardized moment minus 3:

    g2 = m4 / m2^2 - 3

The unbiased version multiplies by `(n+1)(n-1) / ((n-2)(n-3))` after
adding `6/(n+1)` inside, matching scipy.stats.kurtosis.  With
`fisher=False` the raw m4/m2^2 is returned instead.

Covariance and Pearson correlation:

    cov(x, y)   = sum((x-mx)(y-my)) / (n or n-1)
    r(x, y)     = sum((x-mx)(y-my)) / sqrt(sum((x-mx)^2) sum((y-my)^2))

The n cancels in r, so population and sample correlation are the same
number.  Correlation is undefined if either series is constant.

## Files

- `descriptive_stats.py` - the library
- `cli.py` - command-line interface
- `test_descriptive_stats.py` - the tests
- `data/DEXUSEU.csv` - real prices from FRED
- `data/DEXUSEU_returns.csv` - derived daily returns

## Setup

Python 3.11.  The library itself needs only the standard library.  The
`verify` command and the cross-check tests need numpy and scipy:

```bash
pip install --break-system-packages numpy scipy
```

## Usage

```bash
python3 cli.py stats data/DEXUSEU_returns.csv
python3 cli.py stats data/DEXUSEU.csv --col DEXUSEU
python3 cli.py cov data/DEXUSEU_returns.csv return --lag 1
python3 cli.py returns data/DEXUSEU.csv
python3 cli.py returns data/DEXUSEU.csv --out data/DEXUSEU_returns.csv
python3 cli.py verify
```

`stats` prints the full summary for every numeric column.  `cov` takes
two columns, or one column with `--lag` to measure autocovariance against
its own past.  `returns` converts a price column to daily returns and can
write them to a CSV.

## Tests

```bash
python3 -m unittest -v
```

55 tests cover the edge cases (empty input, single values, even-length
medians, ties in mode, constant series) and hand-computed known values
for every statistic, plus a randomized cross-check against numpy and
scipy and an integration test on the real dataset.

## Self-verification

```bash
python3 cli.py verify
```

20 checks.  Every function is compared against the numpy/scipy equivalent
on the 6930 real returns: mean, median, percentiles, range, both variance
and stddev versions, biased and unbiased skewness, all four kurtosis
conventions, population and sample covariance, Pearson correlation (on
returns vs their lag), and the mode conventions.

## Choices worth knowing about

- Quantiles use linear interpolation between sorted values, which is the
  numpy default, so cross-checking is exact rather than approximate.
- The library raises ValueError on inputs where a statistic is undefined
  (empty data, sample variance from one value, skewness of a constant
  series).  Returning a made-up number would be worse than refusing.
- All computations are plain Python floats; the only library imports in
  the module are math and the csv/stdlib used by the CLI.
- Skewness and kurtosis follow the scipy conventions so the biased and
  unbiased versions can be compared directly.
