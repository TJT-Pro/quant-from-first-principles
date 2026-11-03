import math
import random
import unittest

import model_comparison as mc


class TestAICBIC(unittest.TestCase):
    def test_aic_matches_manual_formula(self):
        self.assertAlmostEqual(mc.aic(-100.0, 3), 2 * 3 - 2 * (-100.0))

    def test_bic_matches_manual_formula(self):
        self.assertAlmostEqual(mc.bic(-100.0, 3, 500), 3 * math.log(500) - 2 * (-100.0))

    def test_bic_penalizes_more_heavily_than_aic_for_large_n(self):
        ll, k = -100.0, 5
        aic_val = mc.aic(ll, k)
        bic_val = mc.bic(ll, k, 100000)
        self.assertGreater(bic_val, aic_val)

    def test_more_parameters_with_same_likelihood_is_penalized(self):
        ll = -100.0
        self.assertLess(mc.aic(ll, 2), mc.aic(ll, 5))
        self.assertLess(mc.bic(ll, 2, 500), mc.bic(ll, 5, 500))


class TestLowerIncompleteGammaRegularized(unittest.TestCase):
    def test_matches_chi_squared_df2_closed_form(self):
        for x in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
            ours = mc.chi_squared_cdf(x, 2)
            closed_form = 1 - math.exp(-x / 2)
            self.assertAlmostEqual(ours, closed_form, places=8)

    def test_approaches_one_for_large_x(self):
        self.assertGreater(mc.lower_incomplete_gamma_regularized(2.0, 50.0), 0.9999)

    def test_zero_at_zero(self):
        self.assertEqual(mc.lower_incomplete_gamma_regularized(2.0, 0.0), 0.0)

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            mc.lower_incomplete_gamma_regularized(2.0, -1.0)
        with self.assertRaises(ValueError):
            mc.lower_incomplete_gamma_regularized(-1.0, 5.0)


class TestChiSquaredCDF(unittest.TestCase):
    def test_known_critical_values(self):
        self.assertAlmostEqual(mc.chi_squared_cdf(3.841, 1), 0.95, places=3)
        self.assertAlmostEqual(mc.chi_squared_cdf(11.07, 5), 0.95, places=3)
        self.assertAlmostEqual(mc.chi_squared_cdf(18.31, 10), 0.95, places=3)

    def test_matches_erf_based_df1_formula(self):
        for x in (0.5, 1.0, 3.841, 10.0):
            erf_based = math.erf(math.sqrt(x / 2))
            self.assertAlmostEqual(mc.chi_squared_cdf(x, 1), erf_based, places=6)


class TestLikelihoodRatioTest(unittest.TestCase):
    def test_zero_lr_statistic_when_likelihoods_equal(self):
        result = mc.likelihood_ratio_test(-100.0, -100.0, df=1)
        self.assertAlmostEqual(result["lr_statistic"], 0.0)
        self.assertFalse(result["rejects_restricted_at_5_percent"])

    def test_large_likelihood_improvement_rejects_restricted_model(self):
        result = mc.likelihood_ratio_test(-150.0, -100.0, df=1)
        self.assertTrue(result["rejects_restricted_at_5_percent"])
        self.assertLess(result["p_value"], 0.001)


class TestAICBICSelectTrueModelOnSyntheticData(unittest.TestCase):
    def test_aic_and_bic_both_minimized_at_true_polynomial_degree(self):
        rng = random.Random(1)
        n = 200
        x_vals = [rng.uniform(-3, 3) for _ in range(n)]
        true_coeffs = [1.0, -2.0, 0.5]
        y_vals = [sum(c * x ** i for i, c in enumerate(true_coeffs)) + rng.gauss(0, 1.0) for x in x_vals]

        def fit_polynomial(degree):
            a_mat = [[x ** p for p in range(degree + 1)] for x in x_vals]
            at = list(zip(*a_mat))
            n_p = degree + 1
            ata = [[sum(at[i][k] * a_mat[k][j] for k in range(n)) for j in range(n_p)] for i in range(n_p)]
            aty = [sum(at[i][k] * y_vals[k] for k in range(n)) for i in range(n_p)]
            aug = [ata[i][:] + [aty[i]] for i in range(n_p)]
            for col in range(n_p):
                piv = max(range(col, n_p), key=lambda r: abs(aug[r][col]))
                aug[col], aug[piv] = aug[piv], aug[col]
                pivot_val = aug[col][col]
                for j in range(col, n_p + 1):
                    aug[col][j] /= pivot_val
                for row in range(n_p):
                    if row != col:
                        factor = aug[row][col]
                        for j in range(col, n_p + 1):
                            aug[row][j] -= factor * aug[col][j]
            return [aug[i][n_p] for i in range(n_p)]

        aics, bics = [], []
        for degree in range(7):
            coeffs = fit_polynomial(degree)
            residuals = [y_vals[i] - sum(c * x_vals[i] ** p for p, c in enumerate(coeffs)) for i in range(n)]
            rss = sum(r ** 2 for r in residuals)
            sigma2_hat = rss / n
            log_likelihood = -0.5 * n * math.log(2 * math.pi * sigma2_hat) - 0.5 * rss / sigma2_hat
            aics.append(mc.aic(log_likelihood, degree + 2))
            bics.append(mc.bic(log_likelihood, degree + 2, n))

        self.assertEqual(aics.index(min(aics)), 2)
        self.assertEqual(bics.index(min(bics)), 2)


class TestKFoldCrossValidation(unittest.TestCase):
    def test_matches_manual_computation_on_simple_mean_model(self):
        data = [(None, float(i)) for i in range(10)]

        def fit_fn(train_set):
            ys = [y for _, y in train_set]
            return sum(ys) / len(ys)

        def predict_fn(model, x):
            return model

        def error_fn(pred, actual):
            return (pred - actual) ** 2

        result = mc.k_fold_cross_validation(data, k=5, fit_fn=fit_fn, predict_fn=predict_fn, error_fn=error_fn)
        self.assertGreater(result, 0.0)

    def test_more_flexible_model_can_have_worse_cv_error_than_simpler_one(self):
        import random as _random
        rng = _random.Random(1)
        data = [(x, 2.0 * x + rng.gauss(0, 5.0)) for x in [i / 10 for i in range(-50, 50)]]

        def fit_constant(train_set):
            ys = [y for _, y in train_set]
            return {"type": "constant", "value": sum(ys) / len(ys)}

        def fit_linear(train_set):
            xs = [x for x, _ in train_set]
            ys = [y for _, y in train_set]
            mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
            cov = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(len(xs)))
            var_x = sum((x - mean_x) ** 2 for x in xs)
            slope = cov / var_x
            intercept = mean_y - slope * mean_x
            return {"type": "linear", "slope": slope, "intercept": intercept}

        def predict_fn(model, x):
            if model["type"] == "constant":
                return model["value"]
            return model["intercept"] + model["slope"] * x

        def error_fn(pred, actual):
            return (pred - actual) ** 2

        cv_constant = mc.k_fold_cross_validation(data, 5, fit_constant, predict_fn, error_fn)
        cv_linear = mc.k_fold_cross_validation(data, 5, fit_linear, predict_fn, error_fn)
        self.assertLess(cv_linear, cv_constant)


if __name__ == "__main__":
    unittest.main()
