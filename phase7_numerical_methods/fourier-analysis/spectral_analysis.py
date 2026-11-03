from typing import List, Dict

import fft as fft_module
import dft as dft_module


def periodogram(x: List[float], sample_rate: float = 1.0) -> Dict[str, List[float]]:
    """The periodogram: a simple, classical estimate of a signal's
    power spectral density, |X_k|^2 / N at each frequency bin. By
    Parseval's theorem, sum_n |x_n|^2 = (1/N) sum_k |X_k|^2, so the
    periodogram values sum to the signal's total time-domain energy
    divided by N: each bin's periodogram value is literally that
    frequency's share of the signal's total power. Only the first
    N/2 + 1 bins are returned for a real-valued input, since bins
    beyond the Nyquist frequency (sample_rate/2) are the complex-
    conjugate mirror of the bins already returned and carry no
    additional information for a real signal.
    """
    n = len(x)
    x_freq = fft_module.fft(x) if _is_power_of_two(n) else dft_module.dft(x)
    power = [abs(v) ** 2 / n for v in x_freq]
    freqs = [k * sample_rate / n for k in range(n)]

    n_useful = n // 2 + 1
    return {"frequencies": freqs[:n_useful], "power": power[:n_useful]}


def dominant_frequencies(x: List[float], sample_rate: float = 1.0, n_top: int = 5) -> List[Dict]:
    """The `n_top` frequency bins (excluding the DC bin, which measures
    the signal's mean rather than any oscillation) with the largest
    periodogram power, sorted from strongest to weakest, each reported
    with its frequency, its power, and the corresponding period
    (1/frequency, in the same time units as the sampling interval).
    """
    result = periodogram(x, sample_rate)
    freqs, power = result["frequencies"], result["power"]
    candidates = [(f, p) for f, p in zip(freqs, power) if f > 0]
    candidates.sort(key=lambda item: item[1], reverse=True)
    top = candidates[:n_top]
    return [{"frequency": f, "power": p, "period": 1.0 / f} for f, p in top]


def spectral_filter(x: List[float], sample_rate: float = 1.0, low: float = None,
                     high: float = None) -> List[float]:
    """An ideal ("brick wall") frequency-domain filter: transform to
    the frequency domain, zero out every bin whose frequency falls
    outside [low, high] (treating a missing `low` as 0 and a missing
    `high` as the Nyquist frequency, giving low-pass or high-pass
    behaviour respectively when only one bound is given), then
    transform back. Both a bin's frequency *and* its mirror-image
    negative-frequency bin (at index N-k, corresponding to frequency
    -f by the DFT's periodicity) must be filtered identically, or the
    inverse transform of an asymmetrically-filtered spectrum would
    generally come back complex rather than real, since a real-valued
    time-domain signal always has Hermitian-symmetric spectrum
    (X_{N-k} = conj(X_k)) and breaking that symmetry breaks the
    guarantee that the inverse transform's imaginary part is exactly
    zero.
    """
    n = len(x)
    x_freq = fft_module.fft(x) if _is_power_of_two(n) else dft_module.dft(x)
    nyquist = sample_rate / 2
    low_bound = 0.0 if low is None else low
    high_bound = nyquist if high is None else high

    filtered = list(x_freq)
    for k in range(n):
        freq = k * sample_rate / n
        effective_freq = freq if freq <= nyquist else sample_rate - freq
        if not (low_bound <= effective_freq <= high_bound):
            filtered[k] = 0j

    result = fft_module.ifft(filtered) if _is_power_of_two(n) else dft_module.idft(filtered)
    return [v.real for v in result]


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0
