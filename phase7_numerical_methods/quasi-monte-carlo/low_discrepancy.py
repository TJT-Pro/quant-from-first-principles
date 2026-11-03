from typing import List


def first_n_primes(n: int) -> List[int]:
    """The first n primes, via simple trial division. Used to assign
    each dimension of a Halton sequence its own base: pairwise
    coprime bases across dimensions are what keep each dimension's
    van der Corput sequence's periodic patterns from lining up with
    another dimension's, avoiding the visible correlation between
    dimensions that using the same base twice would otherwise cause.
    """
    primes = []
    candidate = 2
    while len(primes) < n:
        if all(candidate % p != 0 for p in primes):
            primes.append(candidate)
        candidate += 1
    return primes


def van_der_corput(n: int, base: int) -> float:
    """The van der Corput sequence: reflect the base-`base` digits of
    n around the "decimal" point. Writing n in base b as
    n = sum_k d_k*b^k, the n-th van der Corput number is
    sum_k d_k*b^{-k-1}: the same digits, read in reverse order and
    reinterpreted as a fraction. This construction is what makes the
    sequence low-discrepancy: incrementing n by 1 changes only the
    least-significant digit most of the time, which after reflection
    becomes the most significant digit of the fractional value, so
    consecutive points jump by roughly 1/b each step and
    systematically fill in the gaps left by earlier points, rather
    than landing near previously-visited points the way successive
    pseudo-random draws occasionally do purely by chance.
    """
    result = 0.0
    denom = 1.0
    while n > 0:
        n, remainder = divmod(n, base)
        denom *= base
        result += remainder / denom
    return result


def halton_sequence(n_points: int, dim: int, skip: int = 0) -> List[List[float]]:
    """The Halton sequence: a direct multi-dimensional extension of
    van der Corput, using a different (coprime, in practice the first
    dim primes) base for each dimension. Because each dimension is its
    own independently low-discrepancy 1D sequence, and the bases are
    pairwise coprime, the resulting dim-dimensional point set is
    itself low-discrepancy, filling the unit hypercube far more evenly
    than the same number of independent pseudo-random draws would
    (see discrepancy.py for a direct, quantitative comparison). The
    first several points of a Halton sequence in higher dimensions can
    show visible correlation between dimensions using large prime
    bases (a well-documented weakness of plain Halton, addressed in
    practice by leaped/scrambled variants not implemented here); this
    implementation optionally skips the first `skip` indices, a
    simple, standard mitigation.
    """
    primes = first_n_primes(dim)
    return [[van_der_corput(n + skip, p) for p in primes] for n in range(1, n_points + 1)]
