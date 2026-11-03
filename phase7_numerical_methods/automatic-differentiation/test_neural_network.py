import os
import random
import statistics
import unittest

import neural_network as nn

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DEXUSEU_returns.csv")


class TestNeuronAndLayer(unittest.TestCase):
    def test_neuron_output_is_bounded_for_tanh(self):
        neuron = nn.Neuron(3, activation="tanh", seed=1)
        from autodiff_engine import Value
        out = neuron([Value(5.0), Value(-3.0), Value(2.0)])
        self.assertGreaterEqual(out.data, -1.0)
        self.assertLessEqual(out.data, 1.0)

    def test_neuron_parameter_count(self):
        neuron = nn.Neuron(4, seed=1)
        self.assertEqual(len(neuron.parameters()), 5)  # 4 weights + 1 bias

    def test_unknown_activation_raises(self):
        from autodiff_engine import Value
        neuron = nn.Neuron(2, activation="sigmoid", seed=1)
        with self.assertRaises(ValueError):
            neuron([Value(1.0), Value(1.0)])

    def test_layer_output_count(self):
        layer = nn.Layer(3, 5, seed=1)
        from autodiff_engine import Value
        out = layer([Value(1.0), Value(2.0), Value(3.0)])
        self.assertEqual(len(out), 5)


class TestMLPStructure(unittest.TestCase):
    def test_output_dimension(self):
        model = nn.MLP(n_inputs=4, layer_sizes=[8, 3], seed=1)
        preds = nn.predict(model, [[1.0, 2.0, 3.0, 4.0]])
        self.assertEqual(len(preds), 1)

    def test_final_layer_is_linear(self):
        model = nn.MLP(n_inputs=2, layer_sizes=[4, 1], seed=1)
        self.assertEqual(model.layers[-1].neurons[0].activation, "linear")

    def test_hidden_layers_are_tanh(self):
        model = nn.MLP(n_inputs=2, layer_sizes=[4, 4, 1], seed=1)
        self.assertEqual(model.layers[0].neurons[0].activation, "tanh")
        self.assertEqual(model.layers[1].neurons[0].activation, "tanh")

    def test_parameter_count(self):
        model = nn.MLP(n_inputs=2, layer_sizes=[3, 1], seed=1)
        # layer 1: 3 neurons * (2 weights + 1 bias) = 9
        # layer 2: 1 neuron * (3 weights + 1 bias) = 4
        self.assertEqual(len(model.parameters()), 13)


class TestTrainingOnSyntheticGroundTruth(unittest.TestCase):
    def test_learns_a_known_linear_function(self):
        # Validate the engine and training loop on a problem with a
        # known answer before trusting either on ambiguous real data.
        rng = random.Random(0)
        x_train = [[rng.uniform(-2, 2), rng.uniform(-2, 2)] for _ in range(80)]
        y_train = [2 * a - 3 * b + 1 for a, b in x_train]

        model = nn.MLP(n_inputs=2, layer_sizes=[8, 1], seed=1)
        losses = nn.train(model, x_train, y_train, learning_rate=0.03, epochs=150)

        self.assertLess(losses[-1], losses[0] * 0.1)

    def test_loss_decreases_monotonically_on_average(self):
        rng = random.Random(1)
        x_train = [[rng.uniform(-1, 1)] for _ in range(40)]
        y_train = [3 * a[0] ** 2 - 1 for a in x_train]

        model = nn.MLP(n_inputs=1, layer_sizes=[6, 1], seed=2)
        losses = nn.train(model, x_train, y_train, learning_rate=0.05, epochs=100)

        first_half = statistics.mean(losses[:20])
        second_half = statistics.mean(losses[-20:])
        self.assertLess(second_half, first_half)

    def test_zero_grad_between_epochs_prevents_gradient_accumulation_bug(self):
        # If zero_grad were skipped, gradients would accumulate across
        # epochs and the effective step size would grow without bound,
        # eventually diverging the loss to something huge or non-finite.
        rng = random.Random(3)
        x_train = [[rng.uniform(-1, 1)] for _ in range(30)]
        y_train = [a[0] for a in x_train]
        model = nn.MLP(n_inputs=1, layer_sizes=[4, 1], seed=4)
        losses = nn.train(model, x_train, y_train, learning_rate=0.05, epochs=80)
        self.assertTrue(all(l == l for l in losses))  # no NaNs
        self.assertLess(losses[-1], 10.0)


class TestFeatureConstruction(unittest.TestCase):
    def test_feature_target_alignment(self):
        returns = [0.01, -0.02, 0.03, -0.01, 0.02, 0.005]
        x, y = nn.build_volatility_features(returns, n_lags=2, forward_window=1)
        self.assertEqual(x[0], [0.01 ** 2, 0.02 ** 2])
        self.assertAlmostEqual(y[0], 0.03 ** 2)

    def test_forward_window_averages_correctly(self):
        returns = [0.1, 0.1, 0.2, 0.3, 0.1]
        x, y = nn.build_volatility_features(returns, n_lags=1, forward_window=2)
        # First sample: feature = returns[0]^2, target = mean(returns[1]^2, returns[2]^2)
        self.assertAlmostEqual(y[0], statistics.mean([0.1 ** 2, 0.2 ** 2]))

    def test_output_lengths_match(self):
        returns = [0.01] * 50
        x, y = nn.build_volatility_features(returns, n_lags=5, forward_window=3)
        self.assertEqual(len(x), len(y))


class TestTrainTestSplit(unittest.TestCase):
    def test_split_preserves_order(self):
        x = [[i] for i in range(100)]
        y = list(range(100))
        x_train, y_train, x_test, y_test = nn.train_test_split_time_ordered(x, y, 0.7)
        self.assertEqual(len(x_train), 70)
        self.assertEqual(x_train[-1][0], 69)
        self.assertEqual(x_test[0][0], 70)

    def test_no_overlap_between_train_and_test(self):
        x = [[i] for i in range(50)]
        y = list(range(50))
        x_train, y_train, x_test, y_test = nn.train_test_split_time_ordered(x, y, 0.6)
        train_indices = {row[0] for row in x_train}
        test_indices = {row[0] for row in x_test}
        self.assertEqual(train_indices & test_indices, set())


class TestBaselineAndModelMSE(unittest.TestCase):
    def test_naive_baseline_matches_manual_calculation(self):
        y_train = [1.0, 2.0, 3.0]
        y_test = [4.0, 5.0]
        mean_pred = statistics.mean(y_train)
        expected = statistics.mean([(mean_pred - 4.0) ** 2, (mean_pred - 5.0) ** 2])
        self.assertAlmostEqual(nn.naive_baseline_mse(y_train, y_test), expected)

    def test_model_mse_zero_for_perfect_predictions(self):
        self.assertAlmostEqual(nn.model_mse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 0.0)


class TestFXDataLoading(unittest.TestCase):
    def test_load_log_returns_reads_real_data(self):
        returns = nn.load_log_returns(DATA_PATH)
        self.assertEqual(len(returns), 6930)

    def test_volatility_features_from_real_data_run_end_to_end(self):
        # A fast smoke test of the full real-data pipeline (small
        # subsample, few epochs): checks the plumbing works end to
        # end and returns finite, sane values. It deliberately does
        # NOT assert the model beats the naive baseline, since the
        # README documents, honestly, that it typically does not at
        # this scale; that is a reported finding, not a test
        # assumption to bake in as a pass/fail criterion.
        returns = nn.load_log_returns(DATA_PATH)
        x, y = nn.build_volatility_features(returns, n_lags=5, forward_window=1)
        x_train, y_train, x_test, y_test = nn.train_test_split_time_ordered(x, y, 0.8)

        x_small = x_train[-60:]
        y_small = y_train[-60:]
        model = nn.MLP(n_inputs=5, layer_sizes=[4, 1], seed=1)
        losses = nn.train(model, x_small, y_small, learning_rate=0.3, epochs=20)

        self.assertTrue(all(l == l for l in losses))
        preds = nn.predict(model, x_test[:20])
        self.assertEqual(len(preds), 20)
        self.assertTrue(all(p == p for p in preds))


if __name__ == "__main__":
    unittest.main()
