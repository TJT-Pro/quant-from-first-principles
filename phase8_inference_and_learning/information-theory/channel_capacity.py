import math
import random
from typing import Dict


def binary_entropy(p: float) -> float:
    """H_b(p) = -p*log2(p) - (1-p)*log2(1-p), the entropy of a single
    Bernoulli(p) coin flip, used throughout channel-capacity formulas
    involving binary inputs or outputs.
    """
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def bsc_capacity(crossover_prob: float) -> float:
    """The capacity of a Binary Symmetric Channel with crossover
    (bit-flip) probability p, C = 1 - H_b(p): the channel's output Y
    is the input X corrupted by an independent Bernoulli(p) flip, so
    I(X;Y) = H(Y) - H(Y|X). H(Y|X) = H_b(p) regardless of X's own
    distribution (given any specific input bit, the flip noise alone
    determines Y's remaining uncertainty), and H(Y) is maximised at
    exactly 1 bit when the input distribution is uniform (by symmetry,
    a uniform input keeps the output uniform too). Maximising I(X;Y)
    over input distributions therefore reduces to maximising H(Y)
    alone, giving C = 1 - H_b(p): the channel loses exactly H_b(p)
    bits of capacity to the noise, out of the 1 bit a noiseless binary
    channel could carry. C=1 at p=0 (perfect channel) or p=1
    (perfectly invertible, still noiseless in effect), and C=0 at
    p=0.5 (output completely independent of input: pure noise).
    """
    return 1 - binary_entropy(crossover_prob)


def gaussian_channel_capacity(snr: float) -> float:
    """The Shannon-Hartley theorem: the capacity of an additive white
    Gaussian noise channel with signal-to-noise ratio `snr` (a power
    ratio, not in decibels) is C = 0.5*log2(1+snr) bits per channel
    use. This is the continuous-alphabet analogue of the BSC capacity
    formula above: among all input distributions with a fixed power
    constraint, a Gaussian input distribution maximises the mutual
    information with a Gaussian-noise channel (differential entropy is
    maximised, for a fixed variance, by the Gaussian distribution
    itself), and the resulting capacity works out to this closed form.
    """
    return 0.5 * math.log2(1 + snr)


def simulate_repetition_code(crossover_prob: float, code_length: int, n_trials: int,
                              seed: int = 0) -> Dict:
    """Simulate a simple (length-n, majority-vote) repetition code
    over a Binary Symmetric Channel: encode each bit as n identical
    copies, transmit each copy independently through the noisy
    channel, and decode by majority vote. This trades rate (1/n
    information bits transmitted per channel use) for reliability
    (the decoded bit is wrong only if a majority of the n copies were
    flipped, an event whose probability shrinks rapidly as n grows,
    for any crossover_prob < 0.5). This demonstrates channel coding's
    basic promise concretely: reliability can be traded for rate, at a
    cost measured directly against the channel's own capacity, rather
    than being an abstract claim.
    """
    rng = random.Random(seed)
    errors = 0
    for _ in range(n_trials):
        true_bit = rng.choice([0, 1])
        received = [true_bit if rng.random() > crossover_prob else 1 - true_bit
                    for _ in range(code_length)]
        decoded = 1 if sum(received) > code_length / 2 else 0
        if decoded != true_bit:
            errors += 1
    error_rate = errors / n_trials
    rate = 1.0 / code_length
    return {"code_length": code_length, "rate": rate, "empirical_error_rate": error_rate,
            "channel_capacity": bsc_capacity(crossover_prob)}
