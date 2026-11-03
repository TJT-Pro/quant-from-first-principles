# fourier-analysis

A Fourier analysis engine built from scratch: the naive Discrete
Fourier Transform, the Cooley-Tukey Fast Fourier Transform, linear and
circular convolution (with a deliberately documented pitfall between
them), periodograms, and frequency-domain filtering, applied to real
EUR/USD data.

This repository is independent of the sibling repositories and reuses
none of their code, though the application extends the volatility-
clustering theme those repositories established, now viewed through
the frequency domain rather than the time domain.

## Project structure

```
fourier-analysis/
    README.md
    data/DEXUSEU_returns.csv
    dft.py                naive DFT/IDFT, magnitude/phase spectra
    fft.py                Cooley-Tukey radix-2 FFT and inverse
    convolution.py        linear convolution, FFT convolution, the circular pitfall
    spectral_analysis.py  periodogram, dominant frequencies, spectral filtering
    fx_application.py     spectral analysis of real EUR/USD returns
    test_*.py
    cli.py
```

## Status

58 tests, 11/11 verify checks, all passing.

## Setup

Python 3.11+. All transform and filtering code uses only the standard
library (`cmath` for complex arithmetic). `cli.py verify` uses numpy
and scipy as independent oracles.

```bash
pip install --break-system-packages numpy scipy
```

## Running tests

```bash
python3 -m pytest test_dft.py test_fft.py test_convolution.py test_spectral_analysis.py test_fx_application.py -v
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

### The Discrete Fourier Transform

$$X_k = \sum_{n=0}^{N-1} x_n \, e^{-2\pi i k n / N}, \qquad k=0,\dots,N-1.$$

Each $X_k$ correlates the signal against a complex exponential
completing exactly $k$ full cycles over the $N$-sample window. The
exponentials for different $k$ are pairwise orthogonal over that
window ($\sum_n e^{2\pi i(k-k')n/N}$ is a geometric series that sums
to $N$ when $k=k'$ and telescopes to exactly 0 otherwise), which is
precisely what lets $X_k$ measure "how much of frequency $k$" without
interference from any other frequency. The **inverse** transform,

$$x_n = \frac{1}{N}\sum_{k=0}^{N-1} X_k\, e^{2\pi i k n/N},$$

follows from the same orthogonality relation the other direction:
substituting the forward definition into the inverse formula and
swapping summation order collapses the inner sum to exactly $N$ when
indices match and 0 otherwise, recovering $x_n$ exactly. This is
checked as a round-trip property (`dft` then `idft` returns the
original signal) and independently via **Parseval's theorem**,
$\sum_n|x_n|^2 = \frac{1}{N}\sum_k|X_k|^2$: a strong check on the
normalisation convention specifically, since a wrong factor of $N$
anywhere would break exact energy conservation between domains.

### The Cooley-Tukey FFT

Direct evaluation of the DFT costs $O(N^2)$. Splitting the sum into
even- and odd-indexed terms and substituting $n=2m$, $n=2m+1$
respectively turns each half into an $N/2$-point DFT of the even or
odd subsequence, related to the full transform by

$$X_k = E_k + w^k O_k, \qquad X_{k+N/2} = E_k - w^k O_k, \qquad w=e^{-2\pi i/N},$$

using the periodicity of $E$ and $O$ (period $N/2$) and $w^{N/2}=-1$
to reuse each $E_k$, $O_k$ pair exactly twice. Applied recursively down
to length-1 base cases, this gives $\log_2 N$ levels of $O(N)$
combination work, for $O(N\log N)$ total. Verified directly against
the naive DFT (`test_fft.py`), and the practical payoff is measured,
not assumed: at $N=1024$, the FFT here runs about 141x faster than the
naive DFT in the same interpreter, growing further at larger $N$.

### Convolution, and a pitfall worth building deliberately

**Linear convolution**, $y[n] = \sum_k x[k]h[n-k]$, has output length
$\text{len}(x)+\text{len}(h)-1$. The **convolution theorem** says
convolution in time corresponds to multiplication in frequency:
$\text{DFT}(x * h)_k = X_k H_k$, suggesting an $O(N\log N)$ route to
convolution via FFT instead of the $O(NM)$ direct definition.

**The theorem, applied naively, computes the wrong thing.** Multiplying
two same-length transforms and inverting gives **circular**
convolution: output index $i+j$ wraps modulo the transform length $N$
whenever $i+j \ge N$, folding a contribution that belongs at a later
index back onto an earlier one and corrupting it. This project
demonstrates the distinction directly rather than only stating it:
`circular_convolution_via_fft([1,2,3], [1,1,0])` gives `[4, 3, 5]`,
while the correct linear convolution of `[1,2,3]` with `[1,1]` is
`[1, 3, 5, 3]`, index 0 of the circular result is $1+3=4$, the true
linear index-3 contribution (value 3) having wrapped back onto index
$3 \bmod 3 = 0$ and added itself there. `fft_convolution` fixes this
the standard way: zero-pad both signals to at least
$\text{len}(x)+\text{len}(h)-1$ (rounded up to a power of two) *before*
transforming, so every true linear index falls strictly inside
$[0, N)$ and nothing ever wraps. Both the failure and the fix are
checked directly in the test suite and in `verify` (the padded version
also matches `scipy.signal.fftconvolve` to six decimal places), rather
than the pitfall being mentioned only in prose.

### Periodograms and spectral filtering

The **periodogram**, $|X_k|^2/N$ at each bin, is a classical power
spectral density estimate: by Parseval's theorem its values literally
sum to the signal's total time-domain energy divided by $N$, so each
bin's periodogram value is that frequency's share of the total power.
Only the first $N/2+1$ bins are reported for a real signal, since the
remaining bins are the Hermitian-symmetric mirror ($X_{N-k} =
\overline{X_k}$) and carry no additional information.

**Spectral filtering** zeroes out every frequency bin outside a
requested band, then inverts. A subtlety worth stating precisely: a
bin at frequency $f$ and its mirror bin at $N-k$ (frequency $-f$ by
the DFT's periodicity) must be filtered identically, or Hermitian
symmetry breaks and the inverse transform's imaginary part is no
longer guaranteed to vanish. `spectral_filter` maps every bin index to
its physical frequency in $[0, \text{Nyquist}]$ before applying the
band test specifically to preserve this.

## Application: EUR/USD's spectrum, and what it reveals that the time domain doesn't add on its own

**Raw returns: a close-to-flat spectrum.** The ratio of average
periodogram power in a low-frequency band to a high-frequency band, on
the first 4,096 daily returns, is 1.09: essentially flat, the
signature of an uncorrelated process, exactly consistent with every
sibling repository's finding of no linear predictability in EUR/USD
returns. A specific search for a weekly (5-trading-day) calendar
effect at 0.2 cycles/day finds no spike at all: its power is actually
*below* the average of its neighbouring bins (ratio 0.31), a clean
null result.

**Squared returns: a genuinely red spectrum.** The same low/high
frequency power ratio computed on squared returns is 3.52: low
frequencies carry more than three times the power of high frequencies,
the frequency-domain signature of a persistent, slowly-varying process,
the direct spectral counterpart of every ACF-decay, dispersion-test,
and regime-switching finding elsewhere in this project family, now
confirmed a fifth independent way, through the energy distribution of
the Fourier transform rather than any time-domain statistic.
`low_vs_high_frequency_ratio` was validated first on synthetic data
with a known answer (white noise gives a ratio near 1; a strongly
autocorrelated synthetic AR(1) process gives a ratio well above 3)
before being trusted on the real, ambiguous EUR/USD series.

**A genuinely useful, and genuinely imperfect, practical tool.**
`extract_volatility_trend` applies an ideal low-pass filter to squared
returns to pull out a slowly-varying volatility trend, a direct,
principled alternative to a moving average (which is itself just a
particular, comparatively blunt low-pass filter). Applying it reveals
a real, worth-stating limitation: an ideal brick-wall filter has no
built-in non-negativity constraint, so smoothing an inherently
non-negative series (squared returns can never be negative) can still
produce a trend that dips slightly negative near sharp transitions,
confirmed directly on both a synthetic isolated spike and the real
EUR/USD data (the extracted 20-day trend on the first 4,096
observations dips to about $-1.3\times10^{-5}$ even though every input
value was non-negative by construction). A moving average or another
explicitly non-negative-weighted smoother would not have this
particular failure mode, at the cost of a less sharp frequency
response than an ideal filter provides.

## Limitations

- `fft` and the FFT-based routines built on it (`fft_convolution`,
  the periodogram and spectral filter when the input length happens to
  be a power of two) require a power-of-two length; non-power-of-two
  inputs elsewhere in this project fall back to the $O(N^2)$ naive DFT
  rather than Bluestein's algorithm, which would restore $O(N\log N)$
  performance for arbitrary lengths at the cost of a meaningfully more
  involved implementation.
- The weekly-cycle check tests specifically for a spike at 0.2
  cycles/day; it would not detect a calendar effect at a different,
  non-integer-day period (a monthly or quarterly effect, for instance)
  without being pointed at that frequency explicitly.
- `extract_volatility_trend`'s ideal brick-wall filter can ring
  (oscillate, including below zero) near sharp transitions in the
  input, as demonstrated directly above; a windowed filter design
  (tapering the frequency-domain cutoff rather than cutting it off
  sharply) trades some frequency selectivity for reduced ringing and
  is the standard practical fix, not implemented here.

## CLI examples

```bash
# Compute the DFT or FFT of a small signal
python3 cli.py transform --signal "1,2,3,4" --method fft

# Compare naive DFT vs FFT timing
python3 cli.py speedtest --n 1024

# Linear vs FFT convolution, with the circular-convolution pitfall shown directly
python3 cli.py convolve --x "1,2,3" --h "1,1,0" --show-pitfall

# Periodogram of an arbitrary signal
python3 cli.py periodogram --signal "1,2,3,4,3,2,1,0"

# Spectral analysis of real EUR/USD returns: raw vs squared, weekly cycle check
python3 cli.py fxspectrum --n 4096 --top 5

# Extract a smoothed volatility trend from squared returns
python3 cli.py fxtrend --n 4096 --cutoff 20

# Cross-check every transform, convolution, and filter against
# numpy/scipy oracles and known analytical results
python3 cli.py verify
```
