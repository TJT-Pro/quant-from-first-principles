import math
import random
import time
import unittest

import dft
import fft


class TestFFTMatchesNaiveDFT(unittest.TestCase):
    def test_matches_on_random_real_signal(self):
        rng = random.Random(1)
        x = [rng.gauss(0, 1) for _ in range(64)]
        naive = dft.dft(x)
        fast = fft.fft(x)
        for a, b in zip(naive, fast):
            self.assertAlmostEqual(a.real, b.real, places=6)
            self.assertAlmostEqual(a.imag, b.imag, places=6)

    def test_matches_on_complex_signal(self):
        rng = random.Random(2)
        x = [complex(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(32)]
        naive = dft.dft(x)
        fast = fft.fft(x)
        for a, b in zip(naive, fast):
            self.assertAlmostEqual(abs(a - b), 0.0, places=6)

    def test_matches_across_several_power_of_two_sizes(self):
        rng = random.Random(3)
        for n in (2, 4, 8, 16, 128):
            x = [rng.gauss(0, 1) for _ in range(n)]
            naive = dft.dft(x)
            fast = fft.fft(x)
            for a, b in zip(naive, fast):
                self.assertAlmostEqual(abs(a - b), 0.0, places=6)

    def test_non_power_of_two_raises(self):
        with self.assertRaises(ValueError):
            fft.fft([1.0, 2.0, 3.0])


class TestFFTSpeedup(unittest.TestCase):
    def test_fft_is_substantially_faster_at_moderate_size(self):
        rng = random.Random(4)
        x = [rng.gauss(0, 1) for _ in range(512)]

        t0 = time.time()
        dft.dft(x)
        naive_time = time.time() - t0

        t0 = time.time()
        fft.fft(x)
        fft_time = time.time() - t0

        self.assertGreater(naive_time / fft_time, 5.0)


class TestInverseFFT(unittest.TestCase):
    def test_round_trip_real_signal(self):
        rng = random.Random(5)
        x = [rng.gauss(0, 1) for _ in range(64)]
        recovered = fft.ifft(fft.fft(x))
        for original, back in zip(x, recovered):
            self.assertAlmostEqual(back.real, original, places=8)
            self.assertAlmostEqual(back.imag, 0.0, places=8)

    def test_round_trip_complex_signal(self):
        rng = random.Random(6)
        x = [complex(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(32)]
        recovered = fft.ifft(fft.fft(x))
        for original, back in zip(x, recovered):
            self.assertAlmostEqual(back.real, original.real, places=8)
            self.assertAlmostEqual(back.imag, original.imag, places=8)

    def test_ifft_matches_naive_idft(self):
        rng = random.Random(7)
        x = [rng.gauss(0, 1) for _ in range(32)]
        x_freq = fft.fft(x)
        naive = dft.idft(x_freq)
        fast = fft.ifft(x_freq)
        for a, b in zip(naive, fast):
            self.assertAlmostEqual(abs(a - b), 0.0, places=8)


class TestPowerOfTwoHelpers(unittest.TestCase):
    def test_next_power_of_two(self):
        self.assertEqual(fft.next_power_of_two(1), 1)
        self.assertEqual(fft.next_power_of_two(5), 8)
        self.assertEqual(fft.next_power_of_two(16), 16)
        self.assertEqual(fft.next_power_of_two(17), 32)

    def test_auto_pad_produces_power_of_two_length(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = fft.fft_auto_pad(x)
        self.assertEqual(len(result), 8)

    def test_auto_pad_differs_from_naive_dft_of_original_length(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        padded_result = fft.fft_auto_pad(x)
        unpadded_result = dft.dft(x)
        self.assertNotEqual(len(padded_result), len(unpadded_result))


if __name__ == "__main__":
    unittest.main()
