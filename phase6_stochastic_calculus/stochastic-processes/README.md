# stochastic-processes

A standalone stochastic processes laboratory: from-scratch Python
implementations of random walks, Markov chains, Poisson processes,
Brownian motion, martingales, and first-passage problems, each with
its own tests, a verification CLI checked against numpy/scipy
oracles, and an application to real EUR/USD exchange-rate data.

This repository is independent of `math-implementations` but follows
the same standards.

## Project structure

```
stochastic-processes/
    README.md
    data/DEXUSEU_returns.csv
    project1_random_walks/
        random_walks.py
        test_random_walks.py
        cli.py
        README.md
    project2_markov_chains/
        markov_chains.py
        test_markov_chains.py
        cli.py
        README.md
    project3_poisson_processes/
        poisson_processes.py
        test_poisson_processes.py
        cli.py
        README.md
    project4_brownian_motion/
        brownian_motion.py
        test_brownian_motion.py
        cli.py
        README.md
    project5_martingales/
        martingales.py
        test_martingales.py
        cli.py
        README.md
    project6_first_passage/
        first_passage.py
        test_first_passage.py
        cli.py
        README.md
```

## Status

| # | Project | Status |
|---|---------|--------|
| 1 | Random walks | Complete (58 tests, 14/14 verify checks) |
| 2 | Markov chains | Complete (55 tests, 15/15 verify checks) |
| 3 | Poisson processes | Complete (47 tests, 11/11 verify checks) |
| 4 | Brownian motion | Complete (42 tests, 16/16 verify checks) |
| 5 | Martingales | Complete (56 tests, 16/16 verify checks) |
| 6 | First-passage problems | Complete (27 tests, 12/12 verify checks) |

**All 6 projects complete: 285 tests, 84 verify checks, all passing.**
A repository-wide synthesis of what the EUR/USD experiments across all
six projects add up to together is in
`project6_first_passage/README.md`, at the bottom.

## Setup

Python 3.11+. Each project's implementation needs only the standard
library. The `verify` command in every project's CLI needs the
scientific stack as an independent oracle:

```bash
pip install --break-system-packages numpy scipy
```

## Running tests

```bash
python3 -m pytest project1_random_walks/test_random_walks.py -v
```

## Running the oracle

```bash
cd project1_random_walks && python3 cli.py verify
```

## The data

`data/DEXUSEU_returns.csv` holds daily EUR/USD exchange rates sourced
from the Federal Reserve (FRED series DEXUSEU), 1999-01-04 to
2026-08-21 (6,931 price levels, 6,930 daily returns). Columns:

- `date`: observation date
- `price`: U.S. dollars per one euro
- `return`: simple daily return, $(P_t - P_{t-1}) / P_{t-1}$
- `log_return`: daily log return, $\ln(P_t / P_{t-1})$

The file was retrieved once and is committed as-is; no external fetch
is needed to run any test or CLI command.

## Design rules

- **Standard library only in implementations.** numpy and scipy are
  used exclusively inside each `cli.py verify` command as an
  independent oracle, never in the core mathematics.
- **No hardcoded values.** Every number in CLI output is computed from
  the data or from a formula at runtime.
- **Every formula is verified** against an exact analytical value, a
  brute-force enumeration, or a numpy/scipy oracle, with the `verify`
  command printing PASS/FAIL per check.
- **One module fully complete before the next.** Each project ships
  its implementation, tests, CLI, verify command, and README together
  before work starts on the following project.

## What each project README covers

Every project folder documents the same things, in the same order: the
underlying math and where it comes from, how the algorithm follows
from that math, notes on the from-scratch implementation, what the
tests check, how the verify command establishes correctness
independently, what the EUR/USD experiment showed, and where the
method breaks down or falls short. Derivation comes before code, and
code is checked against an independent source before it gets applied
to real data.
