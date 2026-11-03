import unittest

import mean_variance as mv
import linear_algebra as la


MU = [0.05, 0.10, 0.15]
SIGMA = [[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]]


class TestEfficientFrontierPortfolio(unittest.TestCase):
    def test_achieves_the_target_return_exactly(self):
        result = mv.efficient_frontier_portfolio(MU, SIGMA, 0.08)
        self.assertAlmostEqual(mv.portfolio_return(result["weights"], MU), 0.08, places=10)

    def test_weights_sum_to_one(self):
        result = mv.efficient_frontier_portfolio(MU, SIGMA, 0.08)
        self.assertAlmostEqual(sum(result["weights"]), 1.0, places=10)

    def test_reported_variance_matches_direct_computation_from_weights(self):
        result = mv.efficient_frontier_portfolio(MU, SIGMA, 0.08)
        direct = mv.portfolio_variance(result["weights"], SIGMA)
        self.assertAlmostEqual(result["variance"], direct, places=10)

    def test_reported_variance_matches_independent_parametric_formula(self):
        for target in (0.06, 0.08, 0.12, 0.14):
            result = mv.efficient_frontier_portfolio(MU, SIGMA, target)
            formula_variance = mv.frontier_variance_formula(MU, SIGMA, target)
            self.assertAlmostEqual(result["variance"], formula_variance, places=10)

    def test_different_targets_give_different_portfolios(self):
        low = mv.efficient_frontier_portfolio(MU, SIGMA, 0.06)
        high = mv.efficient_frontier_portfolio(MU, SIGMA, 0.14)
        self.assertNotAlmostEqual(low["weights"][0], high["weights"][0], places=4)


class TestGlobalMinimumVariancePortfolio(unittest.TestCase):
    def test_matches_direct_closed_form_sigma_inv_ones_over_a(self):
        gmv = mv.global_minimum_variance_portfolio(MU, SIGMA)
        sigma_inv = la.invert(SIGMA)
        ones = [1.0] * len(MU)
        numer = la.mat_vec(sigma_inv, ones)
        denom = la.dot(ones, numer)
        closed_form = [x / denom for x in numer]
        for a, b in zip(gmv["weights"], closed_form):
            self.assertAlmostEqual(a, b, places=8)

    def test_has_lower_variance_than_any_other_frontier_point_tested(self):
        gmv = mv.global_minimum_variance_portfolio(MU, SIGMA)
        for target in (0.03, 0.06, 0.09, 0.12, 0.18, 0.25):
            other = mv.efficient_frontier_portfolio(MU, SIGMA, target)
            self.assertLessEqual(gmv["variance"], other["variance"] + 1e-10)

    def test_weights_sum_to_one(self):
        gmv = mv.global_minimum_variance_portfolio(MU, SIGMA)
        self.assertAlmostEqual(sum(gmv["weights"]), 1.0, places=8)


class TestFrontierVarianceFormula(unittest.TestCase):
    def test_is_a_parabola_convex_in_target_return(self):
        targets = [0.02, 0.05, 0.08, 0.11, 0.14, 0.20]
        variances = [mv.frontier_variance_formula(MU, SIGMA, t) for t in targets]
        slopes = [(variances[i + 1] - variances[i]) / (targets[i + 1] - targets[i])
                  for i in range(len(targets) - 1)]
        for earlier, later in zip(slopes, slopes[1:]):
            self.assertLess(earlier, later)

    def test_minimum_of_parabola_matches_gmv_variance(self):
        gmv = mv.global_minimum_variance_portfolio(MU, SIGMA)
        for target in [t / 1000 for t in range(-50, 300, 5)]:
            v = mv.frontier_variance_formula(MU, SIGMA, target)
            self.assertGreaterEqual(v, gmv["variance"] - 1e-9)


class TestEfficientFrontierCurve(unittest.TestCase):
    def test_returns_one_result_per_target(self):
        targets = [0.05, 0.08, 0.11, 0.14]
        curve = mv.efficient_frontier_curve(MU, SIGMA, targets)
        self.assertEqual(len(curve), 4)
        for r, t in zip(curve, targets):
            self.assertAlmostEqual(r["target_return"], t)


class TestPortfolioReturnAndVariance(unittest.TestCase):
    def test_portfolio_return_matches_manual_computation(self):
        w = [0.5, 0.3, 0.2]
        expected = 0.5 * 0.05 + 0.3 * 0.10 + 0.2 * 0.15
        self.assertAlmostEqual(mv.portfolio_return(w, MU), expected)

    def test_portfolio_variance_matches_manual_computation(self):
        w = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(mv.portfolio_variance(w, SIGMA), SIGMA[0][0])


if __name__ == "__main__":
    unittest.main()
