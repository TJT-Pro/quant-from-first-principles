from typing import List

import fft as fft_module


def linear_convolution(x: List[float], h: List[float]) -> List[float]:
    """Linear convolution by direct evaluation of its definition:

        y[n] = sum_k x[k] * h[n-k],

    summed over every k for which both x[k] and h[n-k] are defined
    (i.e. treating both signals as zero outside their given range).
    The output has length len(x) + len(h) - 1: the earliest nonzero
    output index is where the two signals first overlap by one sample,
    and the latest is where they last overlap by one sample.
    """
    n, m = len(x), len(h)
    result = [0.0] * (n + m - 1)
    for i in range(n):
        for j in range(m):
            result[i + j] += x[i] * h[j]
    return result


def circular_convolution_via_fft(x: List[complex], h: List[complex]) -> List[complex]:
    """Circular convolution of two length-N sequences, computed via the
    convolution theorem: the DFT of a circular convolution equals the
    elementwise product of the two DFTs,

        DFT(x CircConv h)_k = X_k * H_k,

    so circular convolution can be computed as IFFT(FFT(x) * FFT(h))
    directly, requiring x and h to already be the same length N (the
    length the "circle" wraps around). This is deliberately named and
    documented separately from `fft_convolution` below: calling this
    function on two signals of different lengths, or expecting it to
    equal `linear_convolution`, is a common, genuine mistake, not a
    hypothetical one (found directly while testing this module: an
    early draft used exactly this function, unpadded, to try to
    reproduce linear convolution, and the results silently disagreed
    with `linear_convolution` at several output positions rather than
    raising any error, since circular convolution is a perfectly
    well-defined operation in its own right that simply is not the
    same operation).
    """
    if len(x) != len(h):
        raise ValueError("circular convolution requires equal-length inputs")
    n = len(x)
    x_freq = fft_module.fft(x) if _is_power_of_two(n) else _dft_fallback(x)
    h_freq = fft_module.fft(h) if _is_power_of_two(n) else _dft_fallback(h)
    product = [a * b for a, b in zip(x_freq, h_freq)]
    return fft_module.ifft(product) if _is_power_of_two(n) else _idft_fallback(product)


def fft_convolution(x: List[float], h: List[float]) -> List[float]:
    """Linear convolution computed via the FFT, done correctly: zero-
    pad both x and h to a common length of at least
    len(x) + len(h) - 1 (rounded up to the next power of two, so the
    fast radix-2 FFT can be used) *before* taking the circular
    convolution, then keep only the first len(x)+len(h)-1 samples of
    the result.

    The padding is not a minor implementation detail; it is the whole
    reason this function's result matches `linear_convolution` while
    `circular_convolution_via_fft` on the unpadded signals does not.
    Circular convolution wraps output index n+m around modulo the
    transform length N: whenever the true linear output index i+j
    would exceed N-1, circular convolution folds that contribution
    back to index (i+j) mod N and adds it there, contaminating an
    output sample that should have come from a completely different
    pair of input samples. Padding both signals so that N is at least
    len(x)+len(h)-1 makes every true linear-convolution index i+j fall
    strictly within [0, N-1], so nothing ever wraps and the two
    computations coincide exactly (up to floating-point rounding).
    """
    n, m = len(x), len(h)
    full_length = n + m - 1
    padded_length = fft_module.next_power_of_two(full_length)

    x_padded = list(x) + [0.0] * (padded_length - n)
    h_padded = list(h) + [0.0] * (padded_length - m)

    x_freq = fft_module.fft(x_padded)
    h_freq = fft_module.fft(h_padded)
    product = [a * b for a, b in zip(x_freq, h_freq)]
    result = fft_module.ifft(product)

    return [v.real for v in result[:full_length]]


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _dft_fallback(x):
    import dft as dft_module
    return dft_module.dft(x)


def _idft_fallback(x):
    import dft as dft_module
    return dft_module.idft(x)
