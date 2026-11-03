import math
import unittest

import spectral_analysis as sa


class TestPeriodogram(unittest.TestCase):
    def test_pure_sinusoid_peak_matches_theory(self):
        n = 256
        freq_bin = 10
        x = [math.sin(2 * math.pi * freq_bin * t / n) for t in range(n)]
        result = sa.periodogram(x, sample_rate=1.0)
        expected_power = (n / 2) ** 2 / n
        idx = freq_bin
        self.assertAlmostEqual(result["power"][idx], expected_power, places=4)

    def test_length_is_half_plus_one(self):
        x = [1.0] * 64
        result = sa.periodogram(x)
        self.assertEqual(len(result["power"]), 33)
        self.assertEqual(len(result["frequencies"]), 33)

    def test_parseval_energy_matches_time_domain(self):
        x = [1.0, -2.0, 3.0, 0.5, -1.5, 2.2, 4.0, -3.3]
        n = len(x)
        result = sa.periodogram(x)
        time_energy = sum(v ** 2 for v in x)
        freq_energy = result["power"][0] + result["power"][-1] + 2 * sum(result["power"][1:-1])
        self.assertAlmostEqual(time_energy, freq_energy, places=6)


class TestDominantFrequencies(unittest.TestCase):
    def test_identifies_known_frequency(self):
        n = 128
        freq_bin = 8
        x = [math.cos(2 * math.pi * freq_bin * t / n) for t in range(n)]
        result = sa.dominant_frequencies(x, sample_rate=1.0, n_top=1)
        self.assertAlmostEqual(result[0]["frequency"], freq_bin / n, places=8)

    def test_two_tone_signal_identifies_both(self):
        n = 256
        x = [math.sin(2 * math.pi * 5 * t / n) + 0.5 * math.sin(2 * math.pi * 50 * t / n)
             for t in range(n)]
        result = sa.dominant_frequencies(x, sample_rate=1.0, n_top=2)
        frequencies = sorted(r["frequency"] for r in result)
        self.assertAlmostEqual(frequencies[0], 5 / n, places=4)
        self.assertAlmostEqual(frequencies[1], 50 / n, places=4)

    def test_excludes_dc_component(self):
        x = [5.0 + math.sin(2 * math.pi * 3 * t / 64) for t in range(64)]
        result = sa.dominant_frequencies(x, n_top=5)
        for r in result:
            self.assertGreater(r["frequency"], 0.0)

    def test_period_is_inverse_of_frequency(self):
        n = 100
        freq_bin = 4
        x = [math.sin(2 * math.pi * freq_bin * t / n) for t in range(n)]
        result = sa.dominant_frequencies(x, sample_rate=1.0, n_top=1)
        self.assertAlmostEqual(result[0]["period"], 1.0 / result[0]["frequency"])


class TestSpectralFilter(unittest.TestCase):
    def test_low_pass_removes_high_frequency_component(self):
        n = 256
        x = [math.sin(2 * math.pi * 5 * t / n) + 0.5 * math.sin(2 * math.pi * 50 * t / n)
             for t in range(n)]
        filtered = sa.spectral_filter(x, sample_rate=1.0, low=0.0, high=20 / n)
        result = sa.periodogram(filtered)
        idx_low = 5
        idx_high = 50
        self.assertGreater(result["power"][idx_low], 1.0)
        self.assertLess(result["power"][idx_high], 1e-10)

    def test_high_pass_removes_low_frequency_component(self):
        n = 256
        x = [math.sin(2 * math.pi * 5 * t / n) + 0.5 * math.sin(2 * math.pi * 50 * t / n)
             for t in range(n)]
        filtered = sa.spectral_filter(x, sample_rate=1.0, low=20 / n, high=0.5)
        result = sa.periodogram(filtered)
        idx_low = 5
        idx_high = 50
        self.assertLess(result["power"][idx_low], 1e-10)
        self.assertGreater(result["power"][idx_high], 1.0)

    def test_filtered_output_is_real_valued(self):
        n = 128
        x = [math.sin(2 * math.pi * 7 * t / n) + 0.3 * (t % 3) for t in range(n)]
        filtered = sa.spectral_filter(x, sample_rate=1.0, low=0.0, high=0.1)
        self.assertTrue(all(isinstance(v, float) for v in filtered))

    def test_full_band_filter_is_near_identity(self):
        x = [1.0, -2.0, 3.0, 0.5, -1.5, 2.2, 4.0, -3.3]
        filtered = sa.spectral_filter(x, sample_rate=1.0, low=0.0, high=0.5)
        for a, b in zip(x, filtered):
            self.assertAlmostEqual(a, b, places=6)


if __name__ == "__main__":
    unittest.main()
