import math
import random
import unittest

import overfitting_demo as od


class TestSharpeRatio(unittest.TestCase):
    def test_matches_manual_computation(self):
        import statistics
        returns = [0.01, -0.02, 0.015, 0.005, -0.01]
        expected = (statistics.mean(returns) / statistics.pstdev(returns)) * math.sqrt(252)
        self.assertAlmostEqual(od.sharpe_ratio(returns), expected)

    def test_zero_std_returns_zero(self):
        self.assertEqual(od.sharpe_ratio([0.01, 0.01, 0.01]), 0.0)

    def test_too_short_returns_zero(self):
        self.assertEqual(od.sharpe_ratio([0.01]), 0.0)
        self.assertEqual(od.sharpe_ratio([]), 0.0)


class TestTrainTestSplit(unittest.TestCase):
    def test_correct_split_sizes(self):
        data = list(range(100))
        train, test = od.train_test_split(data, 0.7)
        self.assertEqual(len(train), 70)
        self.assertEqual(len(test), 30)

    def test_no_overlap_and_full_coverage(self):
        data = list(range(100))
        train, test = od.train_test_split(data, 0.6)
        self.assertEqual(train + test, data)


class TestMovingAverageCrossoverReturns(unittest.TestCase):
    def test_output_length(self):
        returns = [0.01] * 100
        result = od.moving_average_crossover_returns(returns, (5, 20))
        self.assertEqual(len(result), 80)

    def test_long_window_exceeding_data_gives_empty(self):
        returns = [0.01] * 10
        result = od.moving_average_crossover_returns(returns, (5, 20))
        self.assertEqual(result, [])

    def test_deterministic_and_finite(self):
        rng = random.Random(1)
        returns = [rng.gauss(0, 0.01) for _ in range(200)]
        result = od.moving_average_crossover_returns(returns, (5, 20))
        self.assertTrue(all(math.isfinite(v) for v in result))


class TestGridSearchOverfittingDemo(unittest.TestCase):
    def test_best_in_sample_sharpe_is_the_maximum_among_tested_combos(self):
        rng = random.Random(1)
        returns = [rng.gauss(0, 0.01) for _ in range(2000)]
        train, test = od.train_test_split(returns, 0.6)
        param_grid = [(s, l) for s in (5, 10, 15) for l in (30, 50, 80) if s < l]
        result = od.grid_search_overfitting_demo(train, test, od.moving_average_crossover_returns, param_grid)
        max_in_sample = max(r["in_sample_sharpe"] for r in result["all_results"])
        self.assertAlmostEqual(result["best_in_sample_sharpe"], max_in_sample)

    def test_reports_one_result_per_grid_combination(self):
        rng = random.Random(2)
        returns = [rng.gauss(0, 0.01) for _ in range(2000)]
        train, test = od.train_test_split(returns, 0.6)
        param_grid = [(5, 30), (10, 50), (15, 80)]
        result = od.grid_search_overfitting_demo(train, test, od.moving_average_crossover_returns, param_grid)
        self.assertEqual(result["n_combinations_tested"], 3)
        self.assertEqual(len(result["all_results"]), 3)

    def test_overfitting_demonstrated_on_genuinely_random_walk_data(self):
        # On data with no genuine exploitable structure at all (pure
        # i.i.d. noise, by construction), a sufficiently rich
        # parameter grid should typically find an in-sample "winner"
        # whose out-of-sample performance is meaningfully worse than
        # its own in-sample performance: the central overfitting
        # mechanism, demonstrated on data where any apparent structure
        # is known, by construction, to be pure noise.
        rng = random.Random(3)
        returns = [rng.gauss(0, 0.01) for _ in range(6000)]
        train, test = od.train_test_split(returns, 0.6)
        param_grid = [(s, l) for s in (3, 5, 8, 10, 15, 20) for l in (20, 30, 40, 50, 60, 80, 100, 120) if s < l]
        result = od.grid_search_overfitting_demo(train, test, od.moving_average_crossover_returns, param_grid)
        self.assertGreater(result["best_in_sample_sharpe"], result["best_combo_out_of_sample_sharpe"])

    def test_real_eurusd_data_shows_the_same_overfitting_pattern(self):
        import os
        import csv

        def load_log_returns(path):
            vals = []
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    vals.append(float(row["log_return"]))
            return vals

        data_path = os.path.join(os.path.dirname(__file__), "data", "DEXUSEU_returns.csv")
        returns = load_log_returns(data_path)
        train, test = od.train_test_split(returns, 0.6)
        param_grid = [(s, l) for s in (3, 5, 8, 10, 15, 20) for l in (20, 30, 40, 50, 60, 80, 100, 120) if s < l]
        result = od.grid_search_overfitting_demo(train, test, od.moving_average_crossover_returns, param_grid)
        self.assertGreater(result["best_in_sample_sharpe"], result["best_combo_out_of_sample_sharpe"])


if __name__ == "__main__":
    unittest.main()
