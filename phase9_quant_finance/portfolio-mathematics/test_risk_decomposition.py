import unittest

import risk_decomposition as rd

W = [0.5, 0.3, 0.2]
SIGMA = [[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]]


class TestPortfolioVolatility(unittest.TestCase):
    def test_single_asset_matches_its_own_std(self):
        result = rd.portfolio_volatility([1.0, 0.0, 0.0], SIGMA)
        self.assertAlmostEqual(result, SIGMA[0][0] ** 0.5)


class TestMarginalContributionToRisk(unittest.TestCase):
    def test_matches_finite_difference_derivative(self):
        mctr = rd.marginal_contribution_to_risk(W, SIGMA)
        h = 1e-6
        base_vol = rd.portfolio_volatility(W, SIGMA)
        for i in range(3):
            w_plus = list(W)
            w_plus[i] += h
            fd = (rd.portfolio_volatility(w_plus, SIGMA) - base_vol) / h
            self.assertAlmostEqual(mctr[i], fd, places=4)


class TestComponentContributionToRisk(unittest.TestCase):
    def test_equals_weight_times_mctr(self):
        cctr = rd.component_contribution_to_risk(W, SIGMA)
        mctr = rd.marginal_contribution_to_risk(W, SIGMA)
        for i in range(3):
            self.assertAlmostEqual(cctr[i], W[i] * mctr[i])


class TestEulerDecomposition(unittest.TestCase):
    def test_sum_of_components_equals_portfolio_volatility_exactly(self):
        result = rd.verify_euler_decomposition(W, SIGMA)
        self.assertTrue(result["matches"])
        self.assertAlmostEqual(result["sum_of_components"], result["portfolio_volatility"], places=10)

    def test_holds_for_several_different_weight_vectors(self):
        for w in ([0.6, 0.2, 0.2], [0.1, 0.1, 0.8], [0.4, 0.4, 0.2], [1.0, 0.0, 0.0]):
            result = rd.verify_euler_decomposition(w, SIGMA)
            self.assertTrue(result["matches"], msg=f"failed for weights {w}")

    def test_scaling_weights_scales_volatility_and_components_identically(self):
        w_scaled = [2 * x for x in W]
        vol_base = rd.portfolio_volatility(W, SIGMA)
        vol_scaled = rd.portfolio_volatility(w_scaled, SIGMA)
        self.assertAlmostEqual(vol_scaled, 2 * vol_base, places=8)

        cctr_base = rd.component_contribution_to_risk(W, SIGMA)
        cctr_scaled = rd.component_contribution_to_risk(w_scaled, SIGMA)
        for a, b in zip(cctr_base, cctr_scaled):
            self.assertAlmostEqual(b, 2 * a, places=8)


class TestRiskContributionPercentages(unittest.TestCase):
    def test_sums_to_one(self):
        pct = rd.risk_contribution_percentages(W, SIGMA)
        self.assertAlmostEqual(sum(pct), 1.0, places=10)

    def test_reveals_risk_concentration_hidden_by_capital_weights(self):
        w = [0.9, 0.05, 0.05]
        sigma = [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 1.0]]
        pct = rd.risk_contribution_percentages(w, sigma)
        self.assertGreater(pct[2], w[2] * 3)

    def test_equal_weight_uncorrelated_equal_variance_assets_split_risk_equally(self):
        w = [1 / 3, 1 / 3, 1 / 3]
        sigma = [[0.04, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.04]]
        pct = rd.risk_contribution_percentages(w, sigma)
        for p in pct:
            self.assertAlmostEqual(p, 1 / 3, places=8)


if __name__ == "__main__":
    unittest.main()
