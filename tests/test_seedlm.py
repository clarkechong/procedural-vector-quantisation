import unittest

import numpy as np

from experiments.encoding.seedlm import (
    Config, PUBLISHED, basis, check_reconstruction, coefficient_codes, lfsr, pack, search, unpack,
)


class SeedLMTests(unittest.TestCase):
    def test_paper_figure_and_lfsr_cycle(self):
        expected = np.array([2, 5, 6, 7, 3, 1, 4, 2])
        generated = lfsr(np.array([4], dtype=np.uint32), len(expected), 3)[0]
        np.testing.assert_allclose(generated, (expected - 4) / 3, atol=1e-7)
        generated = lfsr(np.array([1], dtype=np.uint32), 65535, 16)[0]
        self.assertEqual(len(np.unique(generated)), 65535)
        self.assertAlmostEqual(float(generated[-1]), (1 - 32768) / 32767)

    def test_published_payload_bit_counts_and_signed_fields(self):
        seeds = np.array([1, 65535, 18], dtype=np.uint32)
        exponents = np.array([-8, 7, -1])
        for config, size in zip(PUBLISHED, (14, 12)):
            coefficients = np.tile([-8, -1, 7, 0], (3, 1))[:, :config.latent]
            payload = pack(config, seeds, coefficients, exponents)
            self.assertEqual(len(payload), size)
            for actual, expected in zip(unpack(config, payload, 3), (seeds, coefficients, exponents)):
                np.testing.assert_array_equal(actual, expected)

    def test_paper_quantizer_has_expected_floor_and_clamp(self):
        weights = np.array([[[0.375, 1000, 0], [-0.125, -1000, 0]]], dtype=np.float32)
        values, exponents, _ = coefficient_codes(weights, 4, "paper")
        np.testing.assert_array_equal(exponents, [[[-2, 7, -8]]])
        np.testing.assert_array_equal(values, [[[2, 7, 0], [0, -8, 0]]])

    def test_exhaustive_search_matches_scalar_reference(self):
        config = Config(8, 3, 4)
        rng = np.random.default_rng(5)
        blocks = rng.normal(0, 0.05, size=(5, config.block)).astype(np.float32)
        reference = []
        for seed in range(1, 16):
            matrix = basis(config, np.array([seed], dtype=np.uint32))[0]
            coefficients = blocks @ np.linalg.pinv(matrix).T
            maximum = abs(coefficients).max(axis=1, keepdims=True)
            scale = np.exp2(np.clip(np.floor(np.log2(maximum)), -8, 7))
            rounded = np.clip(np.rint(coefficients / scale), -8, 7) * scale
            reference.append(np.sum((blocks - rounded @ matrix.T) ** 2, axis=1))
        steps = list(search(config, blocks, ("paper", "range", "refine"), prefixes=(1, 5), seed_batch=3))
        for name in ("paper", "range", "refine"):
            losses = [z[0] for _, rule, z in steps if rule == name]
            self.assertTrue(np.all(np.diff(losses, axis=0) <= 0))
        _, _, final = steps[-3]
        np.testing.assert_allclose(final[0], np.min(reference, axis=0), rtol=1e-5)
        check_reconstruction(config, blocks, final)
        np.testing.assert_array_less(steps[-1][2][0] - 1e-7, steps[-2][2][0])
