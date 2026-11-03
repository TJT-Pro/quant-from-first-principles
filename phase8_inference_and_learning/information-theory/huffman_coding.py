import heapq
import itertools
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import entropy as ent


class _Node:
    __slots__ = ("freq", "symbol", "left", "right")

    def __init__(self, freq, symbol=None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.left is None and self.right is None


def build_huffman_tree(frequencies: Dict) -> _Node:
    """Build the Huffman tree via the standard greedy algorithm:
    repeatedly merge the two least-frequent remaining nodes into a new
    parent node whose frequency is their sum, until a single tree
    remains. Greedily combining the two rarest symbols first is
    optimal (not merely a reasonable heuristic) because any optimal
    prefix code's tree must have its two least-frequent symbols as
    siblings at the deepest level (if they were not, swapping them
    with whichever symbols ARE at the deepest level could only
    decrease or maintain the tree's weighted path length, since the
    two rarest symbols cost the least to place deeper); having fixed
    that, the rest of the problem reduces to building an optimal code
    for a smaller alphabet where those two symbols have been replaced
    by one combined symbol of their summed frequency, exactly the
    recursive structure this greedy merge implements.

    A tie-breaking counter is used alongside frequency in the priority
    queue so that Node objects (which have no natural ordering of
    their own) never need to be compared directly when two entries
    share the same frequency.
    """
    counter = itertools.count()
    heap = [(freq, next(counter), _Node(freq, symbol=sym)) for sym, freq in frequencies.items()]
    heapq.heapify(heap)

    if len(heap) == 1:
        # A single-symbol alphabet needs a trivial one-node tree; the
        # encoding logic below assigns it the single-bit code "0".
        only_freq, _, only_node = heap[0]
        return _Node(only_freq, left=only_node)

    while len(heap) > 1:
        freq1, _, node1 = heapq.heappop(heap)
        freq2, _, node2 = heapq.heappop(heap)
        merged = _Node(freq1 + freq2, left=node1, right=node2)
        heapq.heappush(heap, (merged.freq, next(counter), merged))

    return heap[0][2]


def build_codes(tree: _Node) -> Dict:
    """Walk the Huffman tree to assign each symbol its binary
    codeword: left branches append '0', right branches append '1'.
    Because every symbol sits at a *leaf* of the tree, and every leaf
    is reached by a unique root-to-leaf path, no codeword can ever be
    a prefix of another (following one codeword's bits to its leaf
    necessarily leaves the tree, so no continuation of those bits can
    lead to a different leaf): this is exactly what makes a Huffman
    code prefix-free, and hence uniquely and instantaneously
    decodable without any separator between codewords.
    """
    codes = {}

    def walk(node, prefix):
        if node.is_leaf():
            codes[node.symbol] = prefix or "0"
            return
        if node.left is not None:
            walk(node.left, prefix + "0")
        if node.right is not None:
            walk(node.right, prefix + "1")

    walk(tree, "")
    return codes


def huffman_encode(symbols: Sequence) -> Tuple[str, Dict]:
    """Build a Huffman code for the empirical frequency distribution
    of `symbols` and encode the sequence with it, returning the
    encoded bitstring and the codebook (needed by the decoder, and
    would in practice be transmitted once alongside the encoded data,
    or fixed in advance if both sides already agree on the symbol
    frequencies).
    """
    frequencies = Counter(symbols)
    tree = build_huffman_tree(dict(frequencies))
    codes = build_codes(tree)
    encoded = "".join(codes[s] for s in symbols)
    return encoded, codes


def huffman_decode(encoded: str, codes: Dict) -> List:
    """Decode a Huffman-encoded bitstring using the same codebook it
    was encoded with, by greedily matching the longest valid prefix at
    each position: prefix-freedom guarantees this greedy approach
    never needs to backtrack, since at most one codeword can ever
    match the bits read so far.
    """
    reverse_codes = {v: k for k, v in codes.items()}
    result = []
    buffer = ""
    for bit in encoded:
        buffer += bit
        if buffer in reverse_codes:
            result.append(reverse_codes[buffer])
            buffer = ""
    return result


def average_code_length(symbols: Sequence, codes: Dict) -> float:
    n = len(symbols)
    counts = Counter(symbols)
    return sum(counts[s] * len(codes[s]) for s in counts) / n


def verify_source_coding_bound(symbols: Sequence) -> Dict:
    """Shannon's source coding theorem: for a source with entropy
    H(X) (in bits), any uniquely-decodable code's average codeword
    length L satisfies L >= H(X) (no code can beat the entropy bound
    on average, since that would imply compressing genuinely random
    information below its true information content), and there always
    exists a prefix code achieving L < H(X) + 1 bit (Huffman's
    algorithm being the standard constructive proof of this upper
    bound: it is provably optimal among prefix codes, so if any prefix
    code could get within 1 bit of the entropy bound, Huffman's does
    too). Both halves of this squeeze are checked directly here on the
    actual Huffman code built for the given data, not merely quoted
    from theory.
    """
    h = ent.entropy_from_counts(symbols)
    encoded, codes = huffman_encode(symbols)
    l_avg = average_code_length(symbols, codes)
    return {"entropy": h, "average_code_length": l_avg,
            "satisfies_lower_bound": l_avg >= h - 1e-9,
            "satisfies_upper_bound": l_avg < h + 1.0 + 1e-9}
