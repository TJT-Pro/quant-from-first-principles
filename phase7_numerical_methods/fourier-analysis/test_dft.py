import cmath
import math
import unittest

import dft


class TestDFTKnownCases(unittest.TestCase):
    def test_constant_signal_has_only_dc_component(self):
        x = [3.0] * 8
        result = dft.dft(x)
        self.assertAlmostEqual(result[0].real, 24.0, places=8)
        self.assertAlmostEqual(result[0].imag, 0.0, places=8)
        for k in range(1, 8):
            self.assertAlmostEqual(abs(result[k]), 0.0, places=8)

    def test_pure_cosine_shows_two_symmetric_peaks(self):
        n = 16
        freq_bin = 2
        x = [math.cos(2 * math.pi * freq_bin * t / n) for t in range(n)]
        mags = dft.magnitude_spectrum(dft.dft(x))
        for k in range(n):
            if k in (freq_bin, n - freq_bin):
                self.assertAlmostEqual(mags[k], n / 2, places=8)
            else:
                self.assertAlmostEqual(mags[k], 0.0, places=6)

    def test_pure_sine_has_correct_phase(self):
        n = 16
        freq_bin = 3
        x = [math.sin(2 * math.pi * freq_bin * t / n) for t in range(n)]
        result = dft.dft(x)
        self.assertAlmostEqual(result[freq_bin].real, 0.0, places=6)
        self.assertGreater(abs(result[freq_bin].imag), 1.0)

    def test_impulse_has_flat_spectrum(self):
        n = 8
        x = [1.0] + [0.0] * (n - 1)
        mags = dft.magnitude_spectrum(dft.dft(x))
        for m in mags:
            self.assertAlmostEqual(m, 1.0, places=8)


class TestRoundTrip(unittest.TestCase):
    def test_dft_then_idft_recovers_signal(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0, -1.0, 2.5, 0.0]
        recovered = dft.idft(dft.dft(x))
        for original, back in zip(x, recovered):
            self.assertAlmostEqual(back.real, original, places=8)
            self.assertAlmostEqual(back.imag, 0.0, places=8)

    def test_round_trip_on_complex_input(self):
        x = [1 + 2j, 3 - 1j, 0 + 0j, -2 + 4j, 5 + 0j]
        recovered = dft.idft(dft.dft(x))
        for original, back in zip(x, recovered):
            self.assertAlmostEqual(back.real, original.real, places=8)
            self.assertAlmostEqual(back.imag, original.imag, places=8)

    def test_round_trip_odd_length(self):
        x = [1.0, -3.0, 2.0, 4.5, 0.5, -1.5, 7.0]
        recovered = dft.idft(dft.dft(x))
        for original, back in zip(x, recovered):
            self.assertAlmostEqual(back.real, original, places=8)


class TestParsevalsTheorem(unittest.TestCase):
    def test_energy_conservation(self):
        x = [1.0, -2.0, 3.5, 0.0, -1.5, 2.2, 4.0, -3.3]
        n = len(x)
        time_energy = sum(abs(v) ** 2 for v in x)
        freq_energy = sum(abs(v) ** 2 for v in dft.dft(x)) / n
        self.assertAlmostEqual(time_energy, freq_energy, places=6)


class TestFrequencyBins(unittest.TestCase):
    def test_bin_zero_is_dc(self):
        bins = dft.frequency_bins(8, sample_rate=1.0)
        self.assertEqual(bins[0], 0.0)

    def test_bins_scale_with_sample_rate(self):
        bins_1hz = dft.frequency_bins(10, sample_rate=1.0)
        bins_2hz = dft.frequency_bins(10, sample_rate=2.0)
        for b1, b2 in zip(bins_1hz, bins_2hz):
            self.assertAlmostEqual(b2, 2 * b1)

    def test_bin_spacing(self):
        bins = dft.frequency_bins(100, sample_rate=1.0)
        spacing = bins[1] - bins[0]
        self.assertAlmostEqual(spacing, 1.0 / 100)


if __name__ == "__main__":
    unittest.main()
