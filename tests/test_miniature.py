import unittest

import numpy as np

from experiments.generator_algebra.bitvector import Instruction
from experiments.synthesis.miniature import algebra, code, nearest_errors


class MiniatureTests(unittest.TestCase):
    def test_exact_codeword_error_and_planted_recovery(self):
        book = code((Instruction("XOR", ("s", "i")),))
        scale = 0.01
        targets = book[[0, 1, 127, 255]].astype(np.float32) * scale
        errors = nearest_errors(book[None], targets, scale, "cpu")
        self.assertEqual(errors[0], 0)
        shifted = targets.copy()
        shifted[0, 0] += 0.001
        measured = nearest_errors(book[None], shifted, scale, "cpu")[0]
        brute = np.square(book[:, None, :].astype(np.float64) * scale - shifted[None]).sum(2).min(0).sum()
        self.assertAlmostEqual(measured, brute, places=8)

    def test_codebook_algebra(self):
        book = code((Instruction("XOR", ("s", "i")),))
        properties = algebra(book)
        self.assertEqual(properties["polynomial_degree"], 1)
        self.assertEqual(len(np.unique(book, axis=0)), 256)


if __name__ == "__main__":
    unittest.main()
