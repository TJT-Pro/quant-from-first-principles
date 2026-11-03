import math
from typing import Callable, List, Set, Tuple, Union

Number = Union[float, int]


class Value:
    """A node in a computational graph, wrapping a scalar and
    recording enough information to run reverse-mode automatic
    differentiation (backpropagation) afterward.

    Forward-mode AD (dual_numbers.py) propagates a derivative forward
    alongside the computation, one input direction at a time; it costs
    one pass per input. Reverse-mode AD does the opposite: it first
    builds the computational graph forward (every operation records
    its inputs and how to route a gradient back through itself), then
    walks the graph backward exactly once, accumulating dL/dv at every
    node v using the multivariable chain rule. This costs one pass
    total regardless of how many inputs there are, provided there is
    only one output (a loss, say) -- which is exactly the situation a
    trained neural network is in: many parameters, one scalar loss.
    This asymmetry, one pass through the whole graph gives every
    parameter's gradient at once, is the entire reason backpropagation
    is computationally feasible for large models; `jacobian_reverse_mode`
    below makes the cost explicit by showing it needs one backward pass
    per *output*, not per input.
    """

    def __init__(self, data: Number, _children: Tuple["Value", ...] = (), _op: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set[Value] = set(_children)
        self._op = _op

    @staticmethod
    def _coerce(other) -> "Value":
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other) -> "Value":
        other = Value._coerce(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(out)/d(self) = 1, d(out)/d(other) = 1; the incoming
            # gradient with respect to `out` is simply passed through
            # to both operands unchanged and accumulated (not
            # overwritten, since a node can feed into more than one
            # downstream computation, and the total gradient is the
            # sum of contributions from every path, by the
            # multivariable chain rule).
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    __radd__ = __add__

    def __neg__(self) -> "Value":
        return self * -1.0

    def __sub__(self, other) -> "Value":
        return self + (-Value._coerce(other))

    def __rsub__(self, other) -> "Value":
        return Value._coerce(other) + (-self)

    def __mul__(self, other) -> "Value":
        other = Value._coerce(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # Product rule: d(out)/d(self) = other.data, and vice versa.
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __pow__(self, power: Number) -> "Value":
        if isinstance(power, Value):
            raise NotImplementedError("Value-valued exponents are not supported")
        out = Value(self.data ** power, (self,), f"**{power}")

        def _backward():
            # d(x^n)/dx = n*x^(n-1).
            self.grad += (power * self.data ** (power - 1)) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other) -> "Value":
        return self * Value._coerce(other) ** -1

    def __rtruediv__(self, other) -> "Value":
        return Value._coerce(other) * self ** -1

    def exp(self) -> "Value":
        value = math.exp(self.data)
        out = Value(value, (self,), "exp")

        def _backward():
            self.grad += value * out.grad
        out._backward = _backward
        return out

    def log(self) -> "Value":
        if self.data <= 0:
            raise ValueError("log requires a positive value")
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def relu(self) -> "Value":
        value = self.data if self.data > 0 else 0.0
        out = Value(value, (self,), "relu")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def backward(self) -> None:
        """Run reverse-mode automatic differentiation from this node.

        First, a topological sort of the graph guarantees every node
        is processed after everything that depends on it has already
        contributed its share of the gradient. Then, walking that
        order in reverse, starting from d(self)/d(self)=1, each node's
        recorded `_backward` closure pushes its accumulated gradient
        onto its own inputs. By the time a node is reached in this
        reverse walk, every path from it to the root has already been
        accounted for.

        The topological sort is implemented as an explicit,
        stack-based DFS rather than the more natural-looking recursive
        version. This is not a style preference: a recursive DFS
        blows Python's default recursion limit (1000 frames) on any
        computational graph built from a training loop over a
        realistically sized dataset, since the graph's depth grows
        with the number of training examples folded into a single
        loss expression (this project's full-batch MLP training loop
        builds exactly such a graph, and hit this limit directly
        during development on 5,540 real EUR/USD training examples).
        The iterative version below is bounded only by available
        memory, not by the interpreter's call-stack depth.
        """
        topo: List[Value] = []
        visited: Set[Value] = set()
        stack: List[Tuple[Value, iter]] = [(self, iter(self._prev))]
        visited.add(self)

        while stack:
            node, children_iter = stack[-1]
            advanced = False
            for child in children_iter:
                if child not in visited:
                    visited.add(child)
                    stack.append((child, iter(child._prev)))
                    advanced = True
                    break
            if not advanced:
                topo.append(node)
                stack.pop()

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def zero_grad(self) -> None:
        """Reset this node's gradient. Training loops must zero every
        parameter's gradient before each backward pass, since
        `backward` accumulates (+=) rather than overwrites, which is
        correct within a single pass (a node used twice needs both
        contributions summed) but wrong across passes if not reset.
        """
        self.grad = 0.0

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"


def jacobian_reverse_mode(f: Callable[[List[Value]], List[Value]], x: List[Number]) -> List[List[float]]:
    """The full Jacobian of f: R^n -> R^m at point x, by reverse mode:
    m backward passes (one per output), each producing one *row* of
    the Jacobian (the derivative of that one output with respect to
    every input). The graph is rebuilt fresh for each output because
    `Value.backward` accumulates gradients destructively into the leaf
    nodes; sharing one graph across multiple backward passes would
    require zeroing every node's gradient between passes and is
    avoided here for clarity.
    """
    n = len(x)
    inputs_probe = [Value(xi) for xi in x]
    m = len(f(inputs_probe))
    jacobian = [[0.0] * n for _ in range(m)]

    for row in range(m):
        inputs = [Value(xi) for xi in x]
        outputs = f(inputs)
        outputs[row].backward()
        for col in range(n):
            jacobian[row][col] = inputs[col].grad
    return jacobian
