import math
from typing import Callable, List, Union

Number = Union[float, int]


class Dual:
    """A dual number a + b*eps, with eps^2 = 0 by definition (eps is
    not a real number; it is a formal symbol adjoined to the reals,
    the same way i is adjoined to form complex numbers, except here
    the defining relation kills the square instead of making it -1).

    The point of this algebra is that evaluating an ordinary function
    f at the dual number x + 1*eps, using only the arithmetic rules
    below, produces f(x) + f'(x)*eps automatically. This follows
    directly from a Taylor expansion: f(x + eps) = f(x) + f'(x)*eps +
    f''(x)*eps^2/2 + ... and every term from the second one onward
    vanishes because eps^2 = 0. No symbolic differentiation and no
    finite-difference approximation is involved; the derivative falls
    out of ordinary function evaluation performed in the dual algebra.
    """

    __slots__ = ("real", "dual")

    def __init__(self, real: Number, dual: Number = 0.0):
        self.real = float(real)
        self.dual = float(dual)

    @staticmethod
    def _coerce(other) -> "Dual":
        return other if isinstance(other, Dual) else Dual(other, 0.0)

    def __add__(self, other) -> "Dual":
        other = Dual._coerce(other)
        return Dual(self.real + other.real, self.dual + other.dual)

    __radd__ = __add__

    def __sub__(self, other) -> "Dual":
        other = Dual._coerce(other)
        return Dual(self.real - other.real, self.dual - other.dual)

    def __rsub__(self, other) -> "Dual":
        return Dual._coerce(other) - self

    def __mul__(self, other) -> "Dual":
        """The product rule falls out of eps^2 = 0 directly:
        (a + b*eps)(c + d*eps) = ac + (ad + bc)*eps + bd*eps^2
                                = ac + (ad + bc)*eps.
        """
        other = Dual._coerce(other)
        return Dual(self.real * other.real, self.real * other.dual + self.dual * other.real)

    __rmul__ = __mul__

    def __truediv__(self, other) -> "Dual":
        """The quotient rule, by writing 1/(c+d*eps) as its first-order
        expansion (c - d*eps)/c^2 (valid since eps^2=0 makes this an
        exact inverse, not merely an approximation: (c+d*eps)(c-d*eps)/c^2
        = (c^2 - d^2*eps^2)/c^2 = 1 exactly), then multiplying through.
        """
        other = Dual._coerce(other)
        if other.real == 0:
            raise ZeroDivisionError("dual number division by zero real part")
        inv_real = 1.0 / other.real
        inv_dual = -other.dual / (other.real ** 2)
        return self * Dual(inv_real, inv_dual)

    def __rtruediv__(self, other) -> "Dual":
        return Dual._coerce(other) / self

    def __neg__(self) -> "Dual":
        return Dual(-self.real, -self.dual)

    def __pow__(self, power: Number) -> "Dual":
        """d/dx[x^n] = n*x^(n-1), applied at the dual number's real
        part, then the dual (derivative) component is scaled by the
        chain rule: (x+b*eps)^n = x^n + n*x^(n-1)*b*eps + O(eps^2).
        Only constant real-number exponents are supported; a
        dual-valued exponent would need the general a^b = exp(b*ln a)
        rule, not implemented here since it never arises in this
        project's usage.
        """
        if isinstance(power, Dual):
            raise NotImplementedError("dual-valued exponents are not supported")
        return Dual(self.real ** power, power * (self.real ** (power - 1)) * self.dual)

    def __repr__(self) -> str:
        return f"Dual({self.real}, {self.dual})"

    def __eq__(self, other) -> bool:
        other = Dual._coerce(other)
        return self.real == other.real and self.dual == other.dual


def sin(x: Union[Dual, Number]) -> Dual:
    """d/dx[sin x] = cos x, by the chain rule on the dual part."""
    x = Dual._coerce(x)
    return Dual(math.sin(x.real), math.cos(x.real) * x.dual)


def cos(x: Union[Dual, Number]) -> Dual:
    """d/dx[cos x] = -sin x."""
    x = Dual._coerce(x)
    return Dual(math.cos(x.real), -math.sin(x.real) * x.dual)


def exp(x: Union[Dual, Number]) -> Dual:
    """d/dx[e^x] = e^x, the function that is its own derivative."""
    x = Dual._coerce(x)
    value = math.exp(x.real)
    return Dual(value, value * x.dual)


def log(x: Union[Dual, Number]) -> Dual:
    """d/dx[ln x] = 1/x."""
    x = Dual._coerce(x)
    if x.real <= 0:
        raise ValueError("log requires a positive real part")
    return Dual(math.log(x.real), x.dual / x.real)


def sqrt(x: Union[Dual, Number]) -> Dual:
    """d/dx[sqrt(x)] = 1/(2*sqrt(x))."""
    x = Dual._coerce(x)
    if x.real < 0:
        raise ValueError("sqrt requires a non-negative real part")
    value = math.sqrt(x.real)
    return Dual(value, x.dual / (2 * value))


def tanh(x: Union[Dual, Number]) -> Dual:
    """d/dx[tanh x] = 1 - tanh(x)^2."""
    x = Dual._coerce(x)
    value = math.tanh(x.real)
    return Dual(value, (1 - value ** 2) * x.dual)


def derivative(f: Callable[[Dual], Dual], x: Number) -> float:
    """The derivative of a scalar function f at x, read directly off
    the dual component of f(x + 1*eps).
    """
    return f(Dual(x, 1.0)).dual


def gradient_forward_mode(f: Callable[[List[Dual]], Dual], x: List[Number]) -> List[float]:
    """The gradient of a scalar function f: R^n -> R at point x, by
    forward-mode automatic differentiation: one full evaluation of f
    per input dimension, seeding that dimension's dual part to 1 and
    every other dimension's dual part to 0. This costs n evaluations
    of f for n inputs, regardless of how many outputs f has (here,
    always one, since the gradient is defined for a scalar-valued
    function) -- the cost structure that makes forward mode efficient
    for functions with few inputs and inefficient for functions with
    many inputs, exactly the opposite of reverse mode.
    """
    n = len(x)
    grad = []
    for i in range(n):
        dual_inputs = [Dual(x[j], 1.0 if j == i else 0.0) for j in range(n)]
        grad.append(f(dual_inputs).dual)
    return grad


def jacobian_forward_mode(f: Callable[[List[Dual]], List[Dual]], x: List[Number]) -> List[List[float]]:
    """The full Jacobian of f: R^n -> R^m at point x, by forward mode:
    n evaluations of f (one per input dimension), each producing one
    *column* of the Jacobian (the derivative of every output with
    respect to that one input). Returned as a list of m rows (one per
    output) for consistency with the reverse-mode Jacobian in
    `autodiff_engine.py`, but the underlying computation genuinely
    produces the matrix a column at a time.
    """
    n = len(x)
    outputs_at_x = f([Dual(xi, 0.0) for xi in x])
    m = len(outputs_at_x)
    jacobian = [[0.0] * n for _ in range(m)]
    for j in range(n):
        dual_inputs = [Dual(x[i], 1.0 if i == j else 0.0) for i in range(n)]
        outputs = f(dual_inputs)
        for row in range(m):
            jacobian[row][j] = outputs[row].dual
    return jacobian
