import cmath
from typing import List


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def fft(x: List[complex]) -> List[complex]:
    """The Cooley-Tukey radix-2 FFT: a divide-and-conquer algorithm
    computing the same result as `dft.dft` in O(N log N) instead of
    O(N^2) operations, for N a power of two.

    Split the N-point DFT sum into its even- and odd-indexed terms:

        X_k = sum_{n even} x_n*w^{kn} + sum_{n odd} x_n*w^{kn},  w = e^{-2*pi*i/N}

    Substituting n=2m for the even terms and n=2m+1 for the odd terms,
    and factoring w^{2mk} = (w^2)^{mk} = e^{-2*pi*i*mk/(N/2)} (a
    twiddle factor for an N/2-point transform), each half becomes
    exactly an N/2-point DFT of the even- or odd-indexed samples:

        X_k = E_k + w^k * O_k

    where E and O are the N/2-point DFTs of the even and odd subarrays.
    Because E and O are themselves periodic with period N/2, this same
    pair of equations, evaluated at k and at k+N/2, reuses each E_k and
    O_k exactly once more (with w^{k+N/2} = -w^k, since w^{N/2} = -1),
    giving the classic "butterfly" structure:

        X_k       = E_k + w^k * O_k
        X_{k+N/2} = E_k - w^k * O_k

    Applying this split recursively down to length-1 base cases (a
    length-1 "transform" is just the single sample itself) gives
    log2(N) levels of recursion, each doing O(N) combination work,
    for O(N log N) total. This implementation requires N to be a
    power of two; `fft.py` does not implement Bluestein's algorithm or
    zero-padding to the next power of two automatically, so calling it
    on another length raises an error rather than silently doing
    something else (padding changes the meaning of the result: a
    zero-padded FFT computes samples of a different, denser frequency
    grid, not the same DFT with different bookkeeping, which is
    exactly the kind of silent correctness trap this project's
    `convolution.py` runs into deliberately in order to document it).
    """
    n = len(x)
    if not _is_power_of_two(n):
        raise ValueError(f"fft requires a power-of-two length, got {n}")
    return _fft_recursive(list(x))


def _fft_recursive(x: List[complex]) -> List[complex]:
    n = len(x)
    if n == 1:
        return x
    even = _fft_recursive(x[0::2])
    odd = _fft_recursive(x[1::2])
    combined = [0j] * n
    for k in range(n // 2):
        twiddle = cmath.exp(-2j * cmath.pi * k / n) * odd[k]
        combined[k] = even[k] + twiddle
        combined[k + n // 2] = even[k] - twiddle
    return combined


def ifft(x_freq: List[complex]) -> List[complex]:
    """The inverse FFT, via the standard trick of reusing the forward
    FFT itself: conjugating the input, running the forward FFT, then
    conjugating and rescaling the output. This works because

        IDFT(X) = (1/N) * conj(DFT(conj(X)))

    which follows directly from comparing the forward and inverse DFT
    definitions: conjugating X_k flips the sign inside its own
    definition's exponent implicitly (conj(e^{-i*theta}) = e^{i*theta}),
    so running the forward transform on the conjugated input computes
    the same sum structure the inverse transform needs, up to the
    final conjugation and the 1/N normalisation the inverse transform
    requires.
    """
    n = len(x_freq)
    conjugated_input = [v.conjugate() for v in x_freq]
    transformed = fft(conjugated_input)
    return [v.conjugate() / n for v in transformed]


def next_power_of_two(n: int) -> int:
    power = 1
    while power < n:
        power *= 2
    return power


def fft_auto_pad(x: List[complex]) -> List[complex]:
    """Zero-pad x up to the next power of two and run the FFT. This is
    explicitly a *different computation* from the DFT of the original
    (unpadded) x, not merely a faster way to get the same numbers: a
    zero-padded FFT samples the same underlying continuous spectrum
    more densely (interpolating between what the unpadded transform
    would have shown) rather than computing extra information about
    the original finite sequence. It is provided here as a convenience
    for inputs of an inconvenient length, with that distinction stated
    explicitly rather than left for the caller to discover by comparing
    numbers that look similar but are not the same quantity.
    """
    n = len(x)
    padded_length = next_power_of_two(n)
    padded = list(x) + [0j] * (padded_length - n)
    return fft(padded)
