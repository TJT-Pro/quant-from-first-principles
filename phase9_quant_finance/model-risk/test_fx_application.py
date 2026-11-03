import os
import unittest

import fx_application as fx

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DEXUSEU_returns.csv")


def load_prices(path):
    import csv
    prices = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            prices.append(float(row["price"]))
    return prices


class TestDataLoading(unittest.TestCase):
    def test_loads_expected_count(self):
        returns = fx.load_log_returns(DATA_PATH)
        self.assertEqual(len(returns), 6930)


class TestVarModelComparison(unittest.TestCase):
    def setUp(self):
        self.returns = fx.load_log_returns(DATA_PATH)

    def test_both_models_pass_at_95_percent_confidence(self):
        result = fx.var_model_comparison(self.returns, confidences=[0.95])
        self.assertFalse(result[0.95]["gaussian"]["kupiec"]["rejected_at_5_percent"])
        self.assertFalse(result[0.95]["historical"]["kupiec"]["rejected_at_5_percent"])

    def test_gaussian_model_fails_at_99_percent_confidence(self):
        result = fx.var_model_comparison(self.returns, confidences=[0.99])
        self.assertTrue(result[0.99]["gaussian"]["kupiec"]["rejected_at_5_percent"])

    def test_historical_model_passes_at_99_percent_confidence(self):
        result = fx.var_model_comparison(self.returns, confidences=[0.99])
        self.assertFalse(result[0.99]["historical"]["kupiec"]["rejected_at_5_percent"])

    def test_gaussian_underestimates_tail_risk_more_severely_than_historical(self):
        result = fx.var_model_comparison(self.returns, confidences=[0.99])
        gaussian_excess = (result[0.99]["gaussian"]["exceptions"]
                            - result[0.99]["gaussian"]["kupiec"]["expected_exceptions"])
        historical_excess = abs(result[0.99]["historical"]["exceptions"]
                                 - result[0.99]["historical"]["kupiec"]["expected_exceptions"])
        self.assertGreater(gaussian_excess, historical_excess * 5)


class TestOverfittingAnalysis(unittest.TestCase):
    def test_best_in_sample_beats_its_own_out_of_sample_performance(self):
        returns = fx.load_log_returns(DATA_PATH)
        result = fx.overfitting_analysis(returns)
        self.assertGreater(result["best_in_sample_sharpe"], result["best_combo_out_of_sample_sharpe"])

    def test_returns_a_reasonable_number_of_combinations(self):
        returns = fx.load_log_returns(DATA_PATH)
        result = fx.overfitting_analysis(returns)
        self.assertGreater(result["n_combinations_tested"], 20)


class TestOptionPriceUncertainty(unittest.TestCase):
    def test_price_uncertainty_is_positive_and_plausible(self):
        returns = fx.load_log_returns(DATA_PATH)
        prices = load_prices(DATA_PATH)
        s0 = prices[-1]
        k = round(s0, 2)
        result = fx.option_price_uncertainty(returns, s0, k, 0.03, 0.25, n_bootstrap=500)
        self.assertGreater(result["output_std"], 0.0)
        relative_uncertainty = result["output_std"] / result["output_point_estimate"]
        self.assertGreater(relative_uncertainty, 0.001)
        self.assertLess(relative_uncertainty, 0.5)

    def test_confidence_interval_contains_point_estimate(self):
        returns = fx.load_log_returns(DATA_PATH)
        prices = load_prices(DATA_PATH)
        s0 = prices[-1]
        k = round(s0, 2)
        result = fx.option_price_uncertainty(returns, s0, k, 0.03, 0.25, n_bootstrap=500)
        self.assertLessEqual(result["output_5th_percentile"], result["output_point_estimate"])
        self.assertGreaterEqual(result["output_95th_percentile"], result["output_point_estimate"])


if __name__ == "__main__":
    unittest.main()
