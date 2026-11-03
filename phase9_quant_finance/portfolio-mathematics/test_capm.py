import random
import statistics
import unittest

import capm


class TestEstimateBetaAlpha(unittest.TestCase):
    def test_recovers_known_beta_from_synthetic_data(self):
        rng = random.Random(1)
        true_beta, true_alpha = 1.5, 0.0005
        market = [rng.gauss(0.001, 0.01) for _ in range(5000)]
        asset = [true_alpha + true_beta * m + rng.gauss(0, 0.002) for m in market]
        result = capm.estimate_beta_alpha(asset, market)
        self.assertAlmostEqual(result["beta"], true_beta, delta=0.05)
        self.assertAlmostEqual(result["alpha"], true_alpha, delta=0.0005)

    def test_beta_one_when_asset_equals_market(self):
        rng = random.Random(2)
        market = [rng.gauss(0.001, 0.01) for _ in range(1000)]
        result = capm.estimate_beta_alpha(market, market)
        self.assertAlmostEqual(result["beta"], 1.0, places=8)
        self.assertAlmostEqual(result["alpha"], 0.0, places=10)

    def test_beta_zero_for_uncorrelated_asset(self):
        rng = random.Random(3)
        market = [rng.gauss(0.001, 0.01) for _ in range(20000)]
        asset = [rng.gauss(0.0005, 0.005) for _ in range(20000)]
        result = capm.estimate_beta_alpha(asset, market)
        self.assertAlmostEqual(result["beta"], 0.0, delta=0.05)


class TestRiskDecomposition(unittest.TestCase):
    def test_variance_decomposition_holds_exactly(self):
        rng = random.Random(1)
        market = [rng.gauss(0.001, 0.01) for _ in range(2000)]
        asset = [1.5 * m + rng.gauss(0.0005, 0.005) for m in market]
        result = capm.risk_decomposition(asset, market)
        self.assertAlmostEqual(result["total_variance"],
                                result["systematic_variance"] + result["idiosyncratic_variance"], places=12)

    def test_residuals_are_exactly_uncorrelated_with_market_ols_orthogonality(self):
        rng = random.Random(2)
        market = [rng.gauss(0.001, 0.01) for _ in range(2000)]
        asset = [0.8 * m + rng.gauss(0.0003, 0.004) for m in market]
        result = capm.risk_decomposition(asset, market)
        residuals = result["residuals"]
        mean_m = statistics.mean(market)
        mean_res = statistics.mean(residuals)
        cov = sum((market[i] - mean_m) * (residuals[i] - mean_res) for i in range(len(market))) / len(market)
        self.assertAlmostEqual(cov, 0.0, places=10)

    def test_r_squared_between_zero_and_one(self):
        rng = random.Random(3)
        market = [rng.gauss(0.001, 0.01) for _ in range(2000)]
        asset = [1.2 * m + rng.gauss(0.0005, 0.006) for m in market]
        result = capm.risk_decomposition(asset, market)
        self.assertGreaterEqual(result["r_squared"], 0.0)
        self.assertLessEqual(result["r_squared"], 1.0 + 1e-9)

    def test_perfectly_market_tracking_asset_has_r_squared_near_one(self):
        rng = random.Random(4)
        market = [rng.gauss(0.001, 0.01) for _ in range(2000)]
        asset = [1.0 * m for m in market]
        result = capm.risk_decomposition(asset, market)
        self.assertAlmostEqual(result["r_squared"], 1.0, places=8)

    def test_pure_noise_asset_has_r_squared_near_zero(self):
        rng = random.Random(5)
        market = [rng.gauss(0.001, 0.01) for _ in range(20000)]
        asset = [rng.gauss(0.0005, 0.005) for _ in range(20000)]
        result = capm.risk_decomposition(asset, market)
        self.assertLess(result["r_squared"], 0.02)


class TestSecurityMarketLine(unittest.TestCase):
    def test_zero_beta_gives_risk_free_rate(self):
        self.assertAlmostEqual(capm.security_market_line(0.0, 0.02, 0.08), 0.02)

    def test_beta_one_gives_market_return(self):
        self.assertAlmostEqual(capm.security_market_line(1.0, 0.02, 0.08), 0.08)

    def test_matches_manual_formula(self):
        beta, rf, market_return = 1.5, 0.03, 0.10
        expected = rf + beta * (market_return - rf)
        self.assertAlmostEqual(capm.security_market_line(beta, rf, market_return), expected)


if __name__ == "__main__":
    unittest.main()
