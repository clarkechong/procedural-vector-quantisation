import itertools
import unittest

import numpy as np

from experiments.encoding.codebooks import lattice_vectors, one_mad, trellis_row
from experiments.encoding.discrete_oracle import exact, integer_codes
from experiments.encoding.seedlm import Config, basis, search, seed_ids


class ComputationalCodebookTests(unittest.TestCase):
    def test_exact_pruning_matches_brute_force_with_rank_deficiencies(self):
        config = Config(4, 2, 3, 2, "binary")
        blocks = np.array([[0.015, -0.10, 0.03, -0.1], [0.002, 0.001, -0.0005, 0.0008]], dtype=np.float32)
        _, _, incumbent = next(search(config, blocks, ("refine",)))
        bases = basis(config, seed_ids(config)).astype(np.float64)
        coefficients = integer_codes(config)
        vectors = bases @ coefficients.T
        all_reconstructions = vectors[..., None] * np.exp2(np.arange(-8, 8))
        for block, row in zip(blocks, exact(config, blocks, incumbent)):
            brute = np.square(all_reconstructions - block[None, :, None, None]).sum(axis=1).min()
            self.assertAlmostEqual(row[0], brute, places=12)
            self.assertGreaterEqual(row[0] + 1e-12, row[5])
            self.assertGreaterEqual(row[5] + 1e-12, row[6])
            self.assertGreaterEqual(row[6] + 1e-12, row[4])

    def test_trellis_path_matches_exhaustive_sequences(self):
        row = np.array([1.6, 0.42, -0.18, 1.2, -1.33], dtype=np.float32)
        lookup = one_mad(5)
        minimum = float("inf")
        for path in itertools.product(range(4), repeat=len(row)):
            state = 0
            trial = []
            for digit in path:
                state = (state * 4 + digit) & 31
                trial.append(lookup[state])
            minimum = min(minimum, np.square(row - trial).sum())
        np.testing.assert_allclose(np.square(row - trellis_row(row, 2, 5)).sum(), minimum, rtol=1e-6)

    def test_lattice_code_has_full_cardinality(self):
        values = lattice_vectors()
        self.assertEqual((65536, 8), values.shape)
        self.assertEqual(len(np.unique(values, axis=0)), len(values))
        self.assertTrue(np.all(np.abs(values * 4) % 2 == 1))
