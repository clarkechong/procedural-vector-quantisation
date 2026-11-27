"""Independent exhaustive checks for the overlapping-window encoder."""

import unittest

import numpy as np

from experiments.generator_algebra import bitvector
from experiments.synthesis.trellis import (decode_bits, dependency_graph, encode,
                                           linear_map, window_dependency)
from experiments.synthesis.simplify import fold_multipliers


class TrellisTests(unittest.TestCase):
    def test_viterbi_matches_exhaustive(self):
        rng = np.random.default_rng(31)
        for bits, stride, length in ((4, 1, 4), (5, 1, 4), (5, 2, 4),
                                     (4, 2, 5), (3, 3, 3)):
            values = rng.normal(size=1 << bits)
            row = rng.normal(size=length)
            reconstructed, payload, error = encode(row, bits, stride, values)
            total = bits + stride * (length - 1)
            candidates = np.asarray([decode_bits(seed, length, bits, stride, values)
                                     for seed in range(1 << total)])
            minimum = np.square(candidates - row).sum(axis=1).min()
            self.assertAlmostEqual(error, minimum, places=11)
            np.testing.assert_allclose(reconstructed,
                                       decode_bits(payload, length, bits, stride, values))
            self.assertAlmostEqual(np.square(reconstructed - row).sum(), minimum)

    def test_folded_program_and_dependency(self):
        original = (bitvector.Instruction("SHR", ("s", "i")),
                    bitvector.Instruction("MUL", ("r0", 15)),
                    bitvector.Instruction("MUL", ("r1", 8)))
        folded = (bitvector.Instruction("SHR", ("s", "i")),
                  bitvector.Instruction("MUL", ("r0", 120)))
        np.testing.assert_array_equal(bitvector.codebook(original, 8, 4),
                                      bitvector.codebook(folded, 8, 4))
        np.testing.assert_array_equal(dependency_graph(original, 8, 4),
                                      window_dependency(5, 1, 4, linear_map(5)))
        simplified, folds = fold_multipliers(original)
        self.assertEqual(folds, 1)
        self.assertEqual(len(simplified), 2)
        np.testing.assert_array_equal(bitvector.codebook(original, 8, 4),
                                      bitvector.codebook(simplified, 8, 4))

    def test_model_payload_decodes_to_reported_weights(self):
        from experiments.evaluation.trellis_model import WindowEncoder

        rng = np.random.default_rng(78)
        matrix = rng.normal(size=(6, 64)).astype(np.float32)
        for stride in (1, 2):
            restored, payload = WindowEncoder(stride=stride).encode_matrix(matrix)
            scale = np.frombuffer(payload[:4], dtype=np.float32)[0]
            bytes_per_row = (5 + stride * 63 + 7) // 8
            self.assertEqual(len(payload), 4 + 6 * bytes_per_row)
            for row in range(6):
                bits = int.from_bytes(payload[4 + row * bytes_per_row:
                                              4 + (row + 1) * bytes_per_row], "little")
                decoded = decode_bits(bits, 64, 5, stride, linear_map(5) * scale)
                np.testing.assert_array_equal(restored[row], decoded.astype(np.float32))


if __name__ == "__main__":
    unittest.main()
