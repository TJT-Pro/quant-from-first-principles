from typing import Callable, List

import dual_numbers as dn
from autodiff_engine import Value, jacobian_reverse_mode


def jacobian_numerical(f: Callable[[List[float]], List[float]], x: List[float],
                        h: float = 1e-6) -> List[List[float]]:
    """Central-difference Jacobian, used as a third, independent
    reference point alongside forward-mode and reverse-mode automatic
    differentiation. Unlike both of those, this needs 2n evaluations
    of f (one pair per input dimension) and only ever approximates the
    true derivative, with truncation error of order h^2 and roundoff
    error that grows as h shrinks, whereas both AD methods above are
    exact to floating-point precision regardless of h, because they
    never take a difference of two nearby function values at all.
    """
    n = len(x)
    m = len(f(list(x)))
    jacobian = [[0.0] * n for _ in range(m)]
    for j in range(n):
        x_up = list(x)
        x_down = list(x)
        x_up[j] += h
        x_down[j] -= h
        out_up = f(x_up)
        out_down = f(x_down)
        for row in range(m):
            jacobian[row][j] = (out_up[row] - out_down[row]) / (2 * h)
    return jacobian


def count_evaluations_forward_mode(n_inputs: int) -> int:
    """Forward mode needs exactly one full evaluation of f per input
    dimension, regardless of how many outputs f has: `n_inputs`.
    """
    return n_inputs


def count_evaluations_reverse_mode(n_outputs: int) -> int:
    """Reverse mode needs one forward pass to build the graph plus one
    backward pass per output dimension, regardless of how many inputs
    f has: effectively `n_outputs` backward passes (the one shared
    forward pass is the same asymptotic cost as a single ordinary
    function evaluation, so it does not change which method wins).
    """
    return n_outputs


def compare_costs(n_inputs: int, n_outputs: int) -> dict:
    """The concrete efficiency asymmetry that decides which mode to
    use: forward mode costs one pass per input, reverse mode costs one
    pass per output. A scalar loss function of many parameters
    (n_outputs=1, n_inputs potentially in the millions, exactly the
    situation training a neural network is in) is the textbook case
    for reverse mode; a vector-valued function of very few inputs is
    the textbook case for forward mode.
    """
    forward_cost = count_evaluations_forward_mode(n_inputs)
    reverse_cost = count_evaluations_reverse_mode(n_outputs)
    if forward_cost < reverse_cost:
        cheaper = "forward"
    elif reverse_cost < forward_cost:
        cheaper = "reverse"
    else:
        cheaper = "tie"
    return {
        "n_inputs": n_inputs,
        "n_outputs": n_outputs,
        "forward_mode_passes": forward_cost,
        "reverse_mode_passes": reverse_cost,
        "cheaper_method": cheaper,
    }


def jacobian_forward(f_dual: Callable[[List[dn.Dual]], List[dn.Dual]], x: List[float]) -> List[List[float]]:
    """Thin wrapper naming this project's forward-mode Jacobian
    alongside its reverse-mode and numerical counterparts for direct
    side-by-side comparison in the CLI and tests.
    """
    return dn.jacobian_forward_mode(f_dual, x)


def jacobian_reverse(f_value: Callable[[List[Value]], List[Value]], x: List[float]) -> List[List[float]]:
    return jacobian_reverse_mode(f_value, x)
