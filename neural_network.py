import csv
import math
import random
import statistics
from typing import List, Optional

from autodiff_engine import Value


class Neuron:
    """A single neuron: a weighted sum of inputs plus a bias, passed
    through an activation function. Every weight and the bias are
    Value nodes, so the entire forward computation is automatically
    recorded onto the computational graph, and a single call to
    `backward()` on the network's loss will populate every neuron's
    weight and bias gradients without any additional bookkeeping.
    """

    def __init__(self, n_inputs: int, activation: str = "tanh", seed: Optional[int] = None):
        rng = random.Random(seed)
        # Small random initialisation breaks the symmetry that would
        # otherwise make every neuron in a layer compute the same
        # function and receive the same gradient forever.
        self.weights = [Value(rng.uniform(-1, 1) / math.sqrt(n_inputs)) for _ in range(n_inputs)]
        self.bias = Value(0.0)
        self.activation = activation

    def __call__(self, inputs: List[Value]) -> Value:
        total = self.bias
        for w, x in zip(self.weights, inputs):
            total = total + w * x
        if self.activation == "tanh":
            return total.tanh()
        if self.activation == "relu":
            return total.relu()
        if self.activation == "linear":
            return total
        raise ValueError(f"unknown activation: {self.activation}")

    def parameters(self) -> List[Value]:
        return self.weights + [self.bias]


class Layer:
    def __init__(self, n_inputs: int, n_outputs: int, activation: str = "tanh", seed: Optional[int] = None):
        rng = random.Random(seed)
        self.neurons = [Neuron(n_inputs, activation, seed=rng.randrange(2 ** 31)) for _ in range(n_outputs)]

    def __call__(self, inputs: List[Value]) -> List[Value]:
        return [neuron(inputs) for neuron in self.neurons]

    def parameters(self) -> List[Value]:
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params


class MLP:
    """A feedforward network: a sequence of Layers, each fully
    connected to the previous one's outputs. The hidden layers use
    tanh (a smooth, bounded nonlinearity, avoiding the "dead neuron"
    failure mode of relu on a network this small); the output layer is
    linear, appropriate for a regression target rather than a bounded
    or probabilistic one.
    """

    def __init__(self, n_inputs: int, layer_sizes: List[int], seed: Optional[int] = None):
        rng = random.Random(seed)
        sizes = [n_inputs] + layer_sizes
        self.layers = []
        for i in range(len(layer_sizes)):
            activation = "tanh" if i < len(layer_sizes) - 1 else "linear"
            self.layers.append(Layer(sizes[i], sizes[i + 1], activation, seed=rng.randrange(2 ** 31)))

    def __call__(self, inputs: List[Value]) -> List[Value]:
        out = inputs
        for layer in self.layers:
            out = layer(out)
        return out

    def parameters(self) -> List[Value]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


def mse_loss(predictions: List[Value], targets: List[float]) -> Value:
    n = len(predictions)
    total = Value(0.0)
    for pred, target in zip(predictions, targets):
        diff = pred - target
        total = total + diff * diff
    return total * (1.0 / n)


def train(model: MLP, x_train: List[List[float]], y_train: List[float],
          learning_rate: float = 0.05, epochs: int = 200) -> List[float]:
    """Full-batch gradient descent: for a dataset small enough to fit
    the whole computational graph in memory at once (as every
    application in this project is), there is no need for the
    mini-batch machinery a large-scale training loop would use. Each
    epoch: zero every parameter's accumulated gradient (backward()
    accumulates, so stale gradients from the previous epoch must be
    cleared first), rebuild the forward graph for the whole dataset,
    compute the average squared-error loss, call backward() once to
    populate every parameter's gradient via the chain rule, then step
    every parameter by -learning_rate * gradient.
    """
    params = model.parameters()
    loss_history = []
    for epoch in range(epochs):
        for p in params:
            p.zero_grad()

        predictions = [model([Value(xi) for xi in row])[0] for row in x_train]
        loss = mse_loss(predictions, y_train)
        loss.backward()

        for p in params:
            p.data -= learning_rate * p.grad

        loss_history.append(loss.data)
    return loss_history


def predict(model: MLP, x: List[List[float]]) -> List[float]:
    return [model([Value(xi) for xi in row])[0].data for row in x]


def load_log_returns(csv_path: str, column: str = "log_return") -> List[float]:
    values = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row[column]))
    return values


def build_volatility_features(log_returns: List[float], n_lags: int, forward_window: int = 1) -> tuple:
    """Build a supervised dataset for volatility prediction: features
    are the previous n_lags squared returns, and the target is the
    average squared return over the following `forward_window` days.

    forward_window=1 (the default) targets a single day's squared
    return directly, the most literal test of the volatility-
    clustering hypothesis but also, as the accompanying README
    discusses, a famously noisy one: a single day's squared return is
    a highly volatile, unbiased estimator of the *underlying* latent
    variance, so even a perfectly specified model of that latent
    process will show low R^2 against the raw single-day realization.
    Averaging over a wider forward_window reduces that idiosyncratic
    noise (the same law-of-large-numbers argument behind every
    variance-of-the-mean calculation elsewhere in this project's
    sibling repositories), at the cost of predicting a smoother,
    coarser-grained target.
    """
    squared = [r ** 2 for r in log_returns]
    x, y = [], []
    for i in range(n_lags, len(squared) - forward_window + 1):
        x.append(squared[i - n_lags:i])
        y.append(statistics.mean(squared[i:i + forward_window]))
    return x, y


def train_test_split_time_ordered(x: List[List[float]], y: List[float], train_fraction: float = 0.8):
    """A time-ordered split, not a random shuffle: shuffling before
    splitting a time series would let the model train on data that
    comes chronologically after some of its test points, a direct
    form of look-ahead bias that would make the reported test
    performance meaningless for anything resembling real forecasting.
    """
    n = len(x)
    split = int(n * train_fraction)
    return x[:split], y[:split], x[split:], y[split:]


def naive_baseline_mse(y_train: List[float], y_test: List[float]) -> float:
    """The MSE of always predicting the training-set mean target,
    computed on the test set. Any model claiming predictive skill for
    volatility must beat this baseline out of sample, not merely fit
    the training data well.
    """
    mean_prediction = statistics.mean(y_train)
    return statistics.mean([(mean_prediction - actual) ** 2 for actual in y_test])


def model_mse(predictions: List[float], targets: List[float]) -> float:
    return statistics.mean([(p - t) ** 2 for p, t in zip(predictions, targets)])
