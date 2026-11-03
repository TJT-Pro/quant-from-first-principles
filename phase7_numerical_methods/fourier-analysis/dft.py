import cmath
import math
from typing import List


def dft(x: List[complex]) -> List[complex]:
    """The Discrete Fourier Transform, by direct evaluation of its
    definition:

        X_k = sum_{n=0}^{N-1} x_n * exp(-2*pi*i*k*n/N),  k = 0,...,N-1.

    Each X_k is the correlation of the signal x with the complex
    exponential e^{2*pi*i*k*n/N}: a pure oscillation completing exactly
    k cycles over the N samples. The complex exponentials for
    different k are pairwise orthogonal over a length-N window
    (sum_n e^{2*pi*i*(k-k')*n/N} equals N if k=k' and 0 otherwise,
    a geometric series that telescopes to zero whenever k != k'), which
    is exactly what makes X_k measure "how much of frequency k is
    present" without interference from any other frequency. This
    direct evaluation costs O(N^2) (N output values, N terms summed
    for each); `fft.py` implements the O(N log N) Cooley-Tukey
    algorithm that computes the identical result far faster, and is
    checked directly against this function rather than trusted on its
    own.
    """
    n = len(x)
    result = []
    for k in range(n):
        total = 0j
        for t in range(n):
            angle = -2j * cmath.pi * k * t / n
            total += x[t] * cmath.exp(angle)
        result.append(total)
    return result


def idft(x_freq: List[complex]) -> List[complex]:
    """The inverse DFT:

        x_n = (1/N) * sum_{k=0}^{N-1} X_k * exp(+2*pi*i*k*n/N).

    The sign flip in the exponent and the 1/N normalisation both follow
    directly from the same orthogonality relation used to derive the
    forward transform: substituting the forward transform's definition
    of X_k into this formula and swapping the order of summation
    collapses the inner sum over k to N (via orthogonality) exactly
    when the remaining index matches n, and to 0 otherwise, recovering
    x_n exactly. This is checked directly in the test suite: dft
    followed by idft must return the original signal (up to floating-
    point rounding) for arbitrary input, which is the defining
    round-trip property of any correctly normalised transform pair.
    """
    n = len(x_freq)
    result = []
    for t in range(n):
        total = 0j
        for k in range(n):
            angle = 2j * cmath.pi * k * t / n
            total += x_freq[k] * cmath.exp(angle)
        result.append(total / n)
    return result


def magnitude_spectrum(x_freq: List[complex]) -> List[float]:
    return [abs(v) for v in x_freq]


def phase_spectrum(x_freq: List[complex]) -> List[float]:
    return [cmath.phase(v) for v in x_freq]


def frequency_bins(n: int, sample_rate: float = 1.0) -> List[float]:
    """The physical frequency (in cycles per unit time, given samples
    taken at `sample_rate` samples per unit time) each DFT output bin
    k corresponds to: bin k represents k complete cycles over the
    N-sample window, i.e. a frequency of k*sample_rate/N. Bins from
    N/2 (or (N-1)/2 for odd N) onward correspond to negative
    frequencies by the standard convention (aliased back from
    exp(-2*pi*i*k*n/N) with k > N/2 being indistinguishable from a
    negative-frequency oscillation, a direct consequence of the
    transform's periodicity in k with period N), which is why spectral
    analysis conventionally only inspects the first N/2 + 1 bins for a
    real-valued input signal.
    """
    return [k * sample_rate / n for k in range(n)]
