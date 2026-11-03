import random
import unittest

import convolution as conv


class TestLinearConvolution(unittest.TestCase):
    def test_known_small_case(self):
        x = [1.0, 2.0, 3.0]
        h = [1.0, 1.0]
        result = conv.linear_convolution(x, h)
        self.assertEqual(result, [1.0, 3.0, 5.0, 3.0])

    def test_output_length(self):
        x = [1.0] * 5
        h = [1.0] * 3
        result = conv.linear_convolution(x, h)
        self.assertEqual(len(result), 5 + 3 - 1)

    def test_convolution_with_impulse_is_identity(self):
        x = [2.0, -1.0, 3.0, 0.5]
        impulse = [1.0]
        result = conv.linear_convolution(x, impulse)
        self.assertEqual(result, x)

    def test_convolution_is_commutative(self):
        x = [1.0, 2.0, 3.0, 4.0]
        h = [0.5, -0.5, 1.0]
        self.assertEqual(conv.linear_convolution(x, h), conv.linear_convolution(h, x))


class TestFFTConvolutionMatchesLinear(unittest.TestCase):
    def test_matches_on_known_small_case(self):
        x = [1.0, 2.0, 3.0]
        h = [1.0, 1.0]
        linear = conv.linear_convolution(x, h)
        fast = conv.fft_convolution(x, h)
        for a, b in zip(linear, fast):
            self.assertAlmostEqual(a, b, places=8)

    def test_matches_on_random_signals(self):
        rng = random.Random(1)
        x = [rng.gauss(0, 1) for _ in range(37)]
        h = [rng.gauss(0, 1) for _ in range(13)]
        linear = conv.linear_convolution(x, h)
        fast = conv.fft_convolution(x, h)
        for a, b in zip(linear, fast):
            self.assertAlmostEqual(a, b, places=6)

    def test_matches_when_lengths_are_already_powers_of_two(self):
        rng = random.Random(2)
        x = [rng.gauss(0, 1) for _ in range(16)]
        h = [rng.gauss(0, 1) for _ in range(8)]
        linear = conv.linear_convolution(x, h)
        fast = conv.fft_convolution(x, h)
        for a, b in zip(linear, fast):
            self.assertAlmostEqual(a, b, places=6)

    def test_output_length_matches_linear_convolution(self):
        x = list(range(20))
        h = [1.0, -1.0, 0.5]
        linear = conv.linear_convolution([float(v) for v in x], h)
        fast = conv.fft_convolution([float(v) for v in x], h)
        self.assertEqual(len(linear), len(fast))


class TestCircularConvolutionWraparoundPitfall(unittest.TestCase):
    def test_unpadded_circular_convolution_disagrees_with_linear_convolution(self):
        x = [1.0, 2.0, 3.0]
        h = [1.0, 1.0, 0.0]
        linear = conv.linear_convolution(x, h[:2])
        circular = conv.circular_convolution_via_fft(x, h)
        circular_real = [v.real for v in circular]
        self.assertEqual(len(circular_real), 3)
        self.assertNotEqual(len(circular_real), len(linear))
        self.assertNotAlmostEqual(circular_real[0], linear[0], places=6)

    def test_wraparound_mechanism_matches_hand_calculation(self):
        # x=[1,2,3], h=[1,1,0]: linear conv is [1,3,5,3]. Circular
        # convolution of length 3 wraps index 3 (value 3) back to
        # index 0, giving circular[0] = linear[0] + linear[3] = 1+3=4,
        # while indices 1 and 2 are unaffected (nothing wraps onto them).
        x = [1.0, 2.0, 3.0]
        h = [1.0, 1.0, 0.0]
        circular = [v.real for v in conv.circular_convolution_via_fft(x, h)]
        self.assertAlmostEqual(circular[0], 4.0, places=6)
        self.assertAlmostEqual(circular[1], 3.0, places=6)
        self.assertAlmostEqual(circular[2], 5.0, places=6)

    def test_padding_to_avoid_wraparound_recovers_linear_convolution(self):
        x = [1.0, 2.0, 3.0]
        h = [1.0, 1.0]
        full_length = len(x) + len(h) - 1
        x_padded = x + [0.0] * (full_length - len(x))
        h_padded = h + [0.0] * (full_length - len(h))
        circular_padded = [v.real for v in conv.circular_convolution_via_fft(x_padded, h_padded)]
        linear = conv.linear_convolution(x, h)
        for a, b in zip(circular_padded, linear):
            self.assertAlmostEqual(a, b, places=6)

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            conv.circular_convolution_via_fft([1.0, 2.0], [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
