import unittest

import low_discrepancy as ld


class TestFirstNPrimes(unittest.TestCase):
    def test_known_first_primes(self):
        self.assertEqual(ld.first_n_primes(5), [2, 3, 5, 7, 11])

    def test_correct_count(self):
        self.assertEqual(len(ld.first_n_primes(20)), 20)

    def test_all_actually_prime(self):
        primes = ld.first_n_primes(15)
        for p in primes:
            self.assertTrue(all(p % d != 0 for d in range(2, p)))


class TestVanDerCorput(unittest.TestCase):
    def test_known_base_2_sequence(self):
        expected = [0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875, 0.0625]
        result = [ld.van_der_corput(n, 2) for n in range(1, 9)]
        for a, b in zip(result, expected):
            self.assertAlmostEqual(a, b)

    def test_known_base_3_sequence(self):
        expected = [1 / 3, 2 / 3, 1 / 9, 4 / 9, 7 / 9, 2 / 9, 5 / 9, 8 / 9]
        result = [ld.van_der_corput(n, 3) for n in range(1, 9)]
        for a, b in zip(result, expected):
            self.assertAlmostEqual(a, b)

    def test_always_in_unit_interval(self):
        for n in range(1, 500):
            v = ld.van_der_corput(n, 2)
            self.assertGreaterEqual(v, 0.0)
            self.assertLess(v, 1.0)

    def test_zero_gives_zero(self):
        self.assertEqual(ld.van_der_corput(0, 2), 0.0)

    def test_no_two_of_first_n_values_repeat(self):
        # A defining low-discrepancy property: the first b^k values (for
        # base b) are a permutation of {0, 1/b^k, 2/b^k, ...}, all distinct.
        values = [ld.van_der_corput(n, 2) for n in range(1, 17)]
        self.assertEqual(len(set(values)), len(values))


class TestHaltonSequence(unittest.TestCase):
    def test_shape(self):
        points = ld.halton_sequence(10, 3)
        self.assertEqual(len(points), 10)
        for p in points:
            self.assertEqual(len(p), 3)

    def test_first_dimension_matches_base_2_van_der_corput(self):
        points = ld.halton_sequence(8, 2)
        expected = [ld.van_der_corput(n, 2) for n in range(1, 9)]
        for p, e in zip(points, expected):
            self.assertAlmostEqual(p[0], e)

    def test_second_dimension_matches_base_3_van_der_corput(self):
        points = ld.halton_sequence(8, 2)
        expected = [ld.van_der_corput(n, 3) for n in range(1, 9)]
        for p, e in zip(points, expected):
            self.assertAlmostEqual(p[1], e)

    def test_all_coordinates_in_unit_interval(self):
        points = ld.halton_sequence(500, 4)
        for p in points:
            for v in p:
                self.assertGreaterEqual(v, 0.0)
                self.assertLess(v, 1.0)

    def test_skip_shifts_the_sequence(self):
        points_no_skip = ld.halton_sequence(5, 2, skip=0)
        points_skipped = ld.halton_sequence(5, 2, skip=10)
        points_with_offset = ld.halton_sequence(15, 2, skip=0)[10:]
        for a, b in zip(points_skipped, points_with_offset):
            self.assertAlmostEqual(a[0], b[0])
            self.assertAlmostEqual(a[1], b[1])


if __name__ == "__main__":
    unittest.main()
