import unittest

import numpy as np

from experiments.synthesis.trellis import decode_bits
from experiments.synthesis.weighted_trellis import encode, exact_circular_small, refit_scale


class WeightedTrellis(unittest.TestCase):
    def test_exhaustive(self):
        rng = np.random.default_rng(123)
        for state_bits, stride, length in ((3, 1, 5), (4, 2, 4), (5, 2, 4)):
            lookup = rng.normal(size=1 << state_bits)
            for weights in (np.zeros(length), np.array([0, 100, 1, 0.01, 4])[:length], rng.random(length)):
                target = rng.normal(size=length)
                decoded, payload, objective = encode(target, weights, state_bits, stride, lookup)
                all_costs = []
                for candidate in range(1 << (state_bits + stride * (length-1))):
                    row = decode_bits(candidate, length, state_bits, stride, lookup)
                    all_costs.append(np.sum(weights * (target-row)**2))
                self.assertAlmostEqual(objective, min(all_costs), places=11)
                self.assertAlmostEqual(objective, all_costs[payload], places=11)
                np.testing.assert_array_equal(decoded, decode_bits(payload, length, state_bits, stride, lookup))

    def test_scale(self):
        t = np.array([1., 2., -3.])
        c = np.array([2., 1., -1.])
        h = np.array([1., 9., 0.])
        a = refit_scale(t, h, c, 1.)
        self.assertAlmostEqual(a, 20/13)
        self.assertEqual(refit_scale(t, np.zeros(3), c, .5), .5)

    def test_small_circular_boundary(self):
        lookup = np.linspace(-1, 1, 16)
        target = np.array([0.2, -0.8, 0.4, 0.1])
        decoded, payload, objective = exact_circular_small(target, np.ones(4), 4, 2, lookup)
        first = payload & 15
        last = (payload >> 6) & 15
        self.assertEqual(first >> 2, last & 3)
        self.assertAlmostEqual(objective, np.sum((target-decoded)**2))
