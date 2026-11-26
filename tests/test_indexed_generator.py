import importlib.util
import unittest

import numpy as np

from experiments.encoding import synthesized
from experiments.generator_algebra.bitvector import Instruction


@unittest.skipUnless(importlib.util.find_spec("scipy"), "Requires the model extra")
class IndexedGeneratorTests(unittest.TestCase):
    def test_separate_scale_index_matches_exhaustive_fields(self):
        from scipy.spatial import cKDTree
        from experiments.encoding.indexed_generator import DirectEncoder

        program = (Instruction("MUL", ("s", 0x9E3779B9)),
                   Instruction("SHR", ("r0", "i")),
                   Instruction("SHR", ("r1", "r1")))
        encoder = DirectEncoder(program, 8, 1)
        rng = np.random.default_rng(901)
        blocks = np.vstack((rng.normal(0, .05, (12, 8)), np.zeros((1, 8)),
                            encoder.unscaled[3].astype(float)[None] * 1.5,
                            rng.normal(0, 1000, (2, 8))))
        _, seeds, exponents, restored = synthesized.exact_search(encoder.unscaled, blocks)
        expected = synthesized.pack(seeds, exponents, 8)
        unique, encoder.encodings = np.unique(encoder.unscaled, axis=0, return_index=True)
        encoder.index = cKDTree(unique, leafsize=24)
        encoder.separate_scales = True
        payload = encoder.encode(blocks)
        self.assertEqual(payload, expected)
        np.testing.assert_array_equal(encoder.decode(payload, len(blocks)), restored)


if __name__ == "__main__":
    unittest.main()
