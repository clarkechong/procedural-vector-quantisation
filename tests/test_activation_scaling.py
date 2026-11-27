import unittest

import numpy as np

from experiments.synthesis.trellis import linear_map
from experiments.synthesis.weighted_trellis import encode


class ActivationScaling(unittest.TestCase):
    def test_alpha_one_weight_becomes_constant(self):
        rng = np.random.default_rng(88)
        h = np.exp(rng.normal(size=8))
        s = np.sqrt(h)
        s /= np.exp(np.mean(np.log(s)))
        transformed_h = h/s**2
        np.testing.assert_allclose(transformed_h, np.full(8, transformed_h[0]), rtol=1e-14)
        target = rng.normal(size=8)
        lookup = linear_map(5)*0.01
        b = encode(target*s, np.ones(8), 5, 2, lookup)
        d = encode(target*s, transformed_h, 5, 2, lookup)
        self.assertEqual(b[1], d[1])
        np.testing.assert_array_equal(b[0], d[0])
