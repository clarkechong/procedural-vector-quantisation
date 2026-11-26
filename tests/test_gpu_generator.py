import importlib.util
import unittest

import numpy as np

from experiments.encoding import synthesized
from experiments.generator_algebra.bitvector import Instruction, codebook


@unittest.skipUnless(importlib.util.find_spec("torch"), "Requires PyTorch")
class GPUGeneratorTests(unittest.TestCase):
    def test_small_domain_matches_float64_reference_fields(self):
        import torch
        if not torch.version.hip or not torch.cuda.is_available():
            self.skipTest("Requires a visible ROCm device")
        from experiments.encoding.gpu_generator import GPUEncoder

        program = (Instruction("MUL", ("s", 0x9E3779B9)),
                   Instruction("SHR", ("r0", "i")),
                   Instruction("SHR", ("r1", "r1")))
        blocks = np.vstack((np.zeros((1, 8)),
                            np.random.default_rng(171).normal(0, .1, (6, 8))))
        _, expected_seeds, expected_exponents, _ = synthesized.exact_search(
            codebook(program, 8, 8), blocks)
        for backend in ("eager", "compiled"):
            with self.subTest(backend=backend):
                encoder = GPUEncoder(program, 8, seed_tile=64, target_tile=3,
                                     score_backend=backend)
                seeds, exponents = encoder.fields(blocks)
                np.testing.assert_array_equal(seeds, expected_seeds)
                np.testing.assert_array_equal(exponents, expected_exponents)
                payload = synthesized.pack(seeds, exponents, 8)
                decoded = np.asarray(synthesized.decode(program, payload, 8, 8, count=len(blocks)))
                np.testing.assert_array_equal(decoded, np.ldexp(codebook(program, 8, 8)[seeds],
                                                               exponents[:, None]))


if __name__ == "__main__":
    unittest.main()
