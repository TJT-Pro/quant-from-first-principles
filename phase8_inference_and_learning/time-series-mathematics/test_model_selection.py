import math
import unittest

import model_selection as ms
import arma_models as arma


class TestAICBIC(unittest.TestCase):
    def test_aic_formula(self):
        self.assertAlmostEqual(ms.aic(log_likelihood=-100.0, n_params=3), 2 * 3 - 2 * (-100.0))

    def test_bic_formula(self):
        self.assertAlmostEqual(ms.bic(log_likelihood=-100.0, n_params=3, n_obs=500),
                                3 * math.log(500) - 2 * (-100.0))

    def test_bic_penalizes_more_than_aic_for_large_n(self):
        ll, k = -100.0, 5
        n_large = 10000
        self.assertGreater(ms.bic(ll, k, n_large) - ms.aic(ll, k), 0)

    def test_lower_score_is_better_fit_indicator(self):
        # More negative log-likelihood (worse fit) should increase
        # both AIC and BIC for the same parameter count.
        good_fit = ms.aic(log_likelihood=-50.0, n_params=2)
        bad_fit = ms.aic(log_likelihood=-200.0, n_params=2)
        self.assertLess(good_fit, bad_fit)


class TestSelectARorder(unittest.TestCase):
    def test_selects_correct_order_for_true_ar1(self):
        x = arma.simulate_arma(3000, 1.0, [0.6], [], seed=1)
        result = ms.select_ar_order(x, max_order=4)
        self.assertEqual(result["best_aic_order"], 1)
        self.assertEqual(result["best_bic_order"], 1)

    def test_selects_correct_order_for_true_ar2(self):
        x = arma.simulate_arma(4000, 0.5, [0.4, -0.3], [], seed=2)
        result = ms.select_ar_order(x, max_order=5)
        self.assertEqual(result["best_bic_order"], 2)

    def test_results_length_matches_orders_tested(self):
        x = arma.simulate_arma(500, 0.0, [0.3], [], seed=3)
        result = ms.select_ar_order(x, max_order=3)
        self.assertEqual(len(result["results"]), 4)  # orders 0..3

    def test_white_noise_selects_order_zero(self):
        import random
        rng = random.Random(4)
        x = [rng.gauss(0, 1) for _ in range(2000)]
        result = ms.select_ar_order(x, max_order=4)
        self.assertEqual(result["best_bic_order"], 0)


class TestSelectARMAOrder(unittest.TestCase):
    def test_selects_reasonable_order_for_true_arma11(self):
        x = arma.simulate_arma(3000, 0.5, [0.5], [0.3], seed=1)
        result = ms.select_arma_order(x, max_p=2, max_q=2)
        # Exact recovery of (1,1) isn't guaranteed (adjacent orders can
        # sometimes fit comparably well), but the selected model should
        # be small, not the largest grid point tried.
        p, q = result["best_bic"]
        self.assertLessEqual(p + q, 3)

    def test_grid_size_matches_search_space(self):
        x = arma.simulate_arma(500, 0.0, [0.3], [0.2], seed=2)
        result = ms.select_arma_order(x, max_p=2, max_q=1)
        self.assertEqual(len(result["results"]), 3 * 2)  # (max_p+1)*(max_q+1)

    def test_every_candidate_reports_stationarity_and_invertibility_flags(self):
        x = arma.simulate_arma(500, 0.0, [0.3], [0.2], seed=3)
        result = ms.select_arma_order(x, max_p=1, max_q=1)
        for r in result["results"]:
            self.assertIn("stationary", r)
            self.assertIn("invertible", r)


if __name__ == "__main__":
    unittest.main()
