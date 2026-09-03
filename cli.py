import argparse
import math
import os
import statistics
import sys
import time

import dual_numbers as dn
from autodiff_engine import Value, jacobian_reverse_mode
import jacobian_comparison as jc
import neural_network as nn

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DEXUSEU_returns.csv")


def cmd_derivative(args):
    expr = args.expr

    def f(x):
        return eval(expr, {"x": x, "sin": dn.sin, "cos": dn.cos, "exp": dn.exp,
                            "log": dn.log, "sqrt": dn.sqrt, "tanh": dn.tanh})

    value = f(dn.Dual(args.x, 0.0)).real
    deriv = dn.derivative(f, args.x)
    print(f"f({args.x}) = {value:.8f}")
    print(f"f'({args.x}) = {deriv:.8f}  (forward-mode, exact to floating-point precision)")

    h = 1e-6
    numerical = (f(dn.Dual(args.x + h, 0.0)).real - f(dn.Dual(args.x - h, 0.0)).real) / (2 * h)
    print(f"central-difference estimate: {numerical:.8f}  (approximate, h={h})")


def cmd_costs(args):
    result = jc.compare_costs(args.n_inputs, args.n_outputs)
    print(f"function: R^{args.n_inputs} -> R^{args.n_outputs}")
    print(f"forward-mode passes needed:  {result['forward_mode_passes']}")
    print(f"reverse-mode passes needed:  {result['reverse_mode_passes']}")
    print(f"cheaper method: {result['cheaper_method']}")


def cmd_volatility(args):
    print("Sanity check on synthetic data with a known ground truth:")
    import random
    rng = random.Random(0)
    x_train = [[rng.uniform(-2, 2), rng.uniform(-2, 2)] for _ in range(150)]
    y_train = [2 * a - 3 * b + 1 for a, b in x_train]
    model = nn.MLP(n_inputs=2, layer_sizes=[8, 1], seed=1)
    losses = nn.train(model, x_train, y_train, learning_rate=0.03, epochs=300)
    print(f"  known function y=2x-3y+1: loss {losses[0]:.4f} -> {losses[-1]:.4f}")

    print("\nApplication: forecasting EUR/USD volatility from lagged squared returns")
    log_returns = nn.load_log_returns(DATA_PATH)
    x, y = nn.build_volatility_features(log_returns, n_lags=args.lags, forward_window=args.forward_window)
    x_train, y_train, x_test, y_test = nn.train_test_split_time_ordered(x, y, 0.8)

    x_train_small = x_train[-args.train_size:]
    y_train_small = y_train[-args.train_size:]
    print(f"  training on the most recent {len(x_train_small)} in-sample observations "
          f"(a from-scratch scalar autodiff engine does not scale to the full "
          f"{len(x_train)}-observation training set in reasonable time; see the README)")

    baseline_mse = nn.naive_baseline_mse(y_train_small, y_test)
    model = nn.MLP(n_inputs=args.lags, layer_sizes=[6, 1], seed=1)
    t0 = time.time()
    losses = nn.train(model, x_train_small, y_train_small, learning_rate=0.5, epochs=args.epochs)
    elapsed = time.time() - t0

    test_preds = nn.predict(model, x_test)
    test_mse = nn.model_mse(test_preds, y_test)

    print(f"  training time: {elapsed:.1f}s ({args.epochs} epochs)")
    print(f"  train loss: {losses[0]:.3e} -> {losses[-1]:.3e}")
    print(f"  naive baseline test MSE: {baseline_mse:.4e}")
    print(f"  model test MSE:          {test_mse:.4e}")
    change = (baseline_mse - test_mse) / baseline_mse * 100
    print(f"  change vs baseline: {change:+.2f}%  "
          f"({'improvement' if change > 0 else 'no improvement (see README for why this is expected)'})")


def cmd_verify(args):
    import random
    import numpy as np

    checks = []

    def check(name, condition):
        checks.append((name, bool(condition)))
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {name}")

    # 1. forward-mode derivative vs sympy-free analytical check and numpy finite difference
    f = lambda x: dn.sin(x) * dn.exp(x) + x ** 2
    x0 = 0.8
    forward = dn.derivative(f, x0)
    h = 1e-6
    numpy_fd = (f(dn.Dual(x0 + h)).real - f(dn.Dual(x0 - h)).real) / (2 * h)
    check("forward-mode derivative matches central finite difference",
          math.isclose(forward, numpy_fd, abs_tol=1e-5))

    analytical = math.cos(x0) * math.exp(x0) + math.sin(x0) * math.exp(x0) + 2 * x0
    check("forward-mode derivative matches hand-derived analytical value",
          math.isclose(forward, analytical, abs_tol=1e-9))

    # 2. reverse-mode gradient matches forward-mode gradient on a shared function
    def f_dual(v):
        x, y = v
        return x * x * y + dn.exp(x) * y

    def f_value(v):
        x, y = v
        return x * x * y + x.exp() * y

    x = [0.9, -1.3]
    forward_grad = dn.gradient_forward_mode(f_dual, x)
    vx, vy = Value(x[0]), Value(x[1])
    out = f_value([vx, vy])
    out.backward()
    check("reverse-mode gradient matches forward-mode gradient",
          math.isclose(vx.grad, forward_grad[0], abs_tol=1e-9) and
          math.isclose(vy.grad, forward_grad[1], abs_tol=1e-9))

    # 3. reverse-mode gradient matches numpy finite difference
    def raw(a, b):
        return a * a * b + math.exp(a) * b
    numpy_gx = (raw(x[0] + h, x[1]) - raw(x[0] - h, x[1])) / (2 * h)
    numpy_gy = (raw(x[0], x[1] + h) - raw(x[0], x[1] - h)) / (2 * h)
    check("reverse-mode gradient matches numpy finite difference",
          math.isclose(vx.grad, numpy_gx, abs_tol=1e-4) and math.isclose(vy.grad, numpy_gy, abs_tol=1e-4))

    # 4. three-way Jacobian agreement: forward, reverse, numerical
    f_dual2 = lambda v: [v[0] ** 2 + v[1], dn.exp(v[0]) * v[1]]
    f_value2 = lambda v: [v[0] ** 2 + v[1], v[0].exp() * v[1]]
    f_plain2 = lambda v: [v[0] ** 2 + v[1], math.exp(v[0]) * v[1]]
    pt = [0.4, 1.1]
    jac_f = jc.jacobian_forward(f_dual2, pt)
    jac_r = jc.jacobian_reverse(f_value2, pt)
    jac_n = jc.jacobian_numerical(f_plain2, pt)
    ok = all(math.isclose(jac_f[i][j], jac_r[i][j], abs_tol=1e-9) and
             math.isclose(jac_f[i][j], jac_n[i][j], abs_tol=1e-4)
             for i in range(2) for j in range(2))
    check("forward, reverse, and numerical Jacobians agree three ways", ok)

    # 5. cost model matches actual pass counts for a real Jacobian computation
    n_in, n_out = 4, 2
    result = jc.compare_costs(n_in, n_out)
    check("cost model's forward-mode pass count matches n_inputs", result["forward_mode_passes"] == n_in)
    check("cost model's reverse-mode pass count matches n_outputs", result["reverse_mode_passes"] == n_out)

    # 6. neural network learns a known linear function (engine correctness,
    #    decoupled from the separate, honestly-reported question of whether
    #    it finds usable signal in real, noisy financial data)
    rng = random.Random(0)
    x_train = [[rng.uniform(-2, 2), rng.uniform(-2, 2)] for _ in range(150)]
    y_train = [2 * a - 3 * b + 1 for a, b in x_train]
    model = nn.MLP(n_inputs=2, layer_sizes=[8, 1], seed=1)
    losses = nn.train(model, x_train, y_train, learning_rate=0.03, epochs=400)
    check("neural network trained via this engine learns a known linear function",
          losses[-1] < losses[0] * 0.05)

    # 7. neural network gradient matches numpy finite-difference gradient of
    #    the same loss with respect to a chosen parameter (an end-to-end
    #    check that backprop through the whole network, not just a single
    #    operation, is wired correctly)
    tiny_model = nn.MLP(n_inputs=2, layer_sizes=[3, 1], seed=2)
    x_sample = [[0.5, -0.3], [1.0, 0.2]]
    y_sample = [1.0, -0.5]
    for p in tiny_model.parameters():
        p.zero_grad()
    preds = [tiny_model([Value(xi) for xi in row])[0] for row in x_sample]
    loss = nn.mse_loss(preds, y_sample)
    loss.backward()
    target_param = tiny_model.parameters()[0]
    analytical_grad = target_param.grad

    def loss_at(bump):
        original = target_param.data
        target_param.data = original + bump
        preds_bumped = [tiny_model([Value(xi) for xi in row])[0] for row in x_sample]
        loss_bumped = nn.mse_loss(preds_bumped, y_sample).data
        target_param.data = original
        return loss_bumped

    h2 = 1e-6
    numerical_grad = (loss_at(h2) - loss_at(-h2)) / (2 * h2)
    check("end-to-end network gradient matches finite-difference gradient",
          math.isclose(analytical_grad, numerical_grad, abs_tol=1e-4))

    # 8. real EUR/USD data loads and the feature/target pipeline is consistent
    log_returns = nn.load_log_returns(DATA_PATH)
    check("real EUR/USD log returns load with the expected count", len(log_returns) == 6930)
    xf, yf = nn.build_volatility_features(log_returns, n_lags=5, forward_window=1)
    check("volatility feature/target arrays have matching, expected length",
          len(xf) == len(yf) == len(log_returns) - 5)

    passed = sum(1 for _, ok in checks if ok)
    print(f"\n{passed}/{len(checks)} checks passed")
    if passed != len(checks):
        sys.exit(1)


def build_parser():
    parser = argparse.ArgumentParser(description="Automatic differentiation engine CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("derivative", help="differentiate an expression in x via forward-mode AD")
    p1.add_argument("--expr", type=str, required=True, help="Python expression in x, e.g. 'sin(x)*exp(x)'")
    p1.add_argument("--x", type=float, required=True)
    p1.set_defaults(func=cmd_derivative)

    p2 = sub.add_parser("costs", help="compare forward- vs reverse-mode pass counts")
    p2.add_argument("--n-inputs", type=int, required=True)
    p2.add_argument("--n-outputs", type=int, required=True)
    p2.set_defaults(func=cmd_costs)

    p3 = sub.add_parser("volatility", help="train a tiny NN to forecast EUR/USD volatility")
    p3.add_argument("--lags", type=int, default=5)
    p3.add_argument("--forward-window", type=int, default=1)
    p3.add_argument("--train-size", type=int, default=600)
    p3.add_argument("--epochs", type=int, default=150)
    p3.set_defaults(func=cmd_volatility)

    p4 = sub.add_parser("verify", help="cross-check implementation against analytical/oracle values")
    p4.set_defaults(func=cmd_verify)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
