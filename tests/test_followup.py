import unittest

import numpy as np

from experiments.synthesis.followup import optimize_scale, scalar_program
from experiments.synthesis.guided import parse_program
from experiments.generator_algebra.bitvector import expression


class FollowupTests(unittest.TestCase):
    def test_scalar_program_contains_all_four_level_vectors(self):
        levels = np.array([-3, -1, 1, 4], dtype=np.float32)
        book = scalar_program(np.arange(256), levels)
        self.assertEqual(book.shape, (256, 4))
        self.assertEqual(len(np.unique(book, axis=0)), 256)
        self.assertTrue(np.array_equal(book[0], [-3] * 4))
        self.assertTrue(np.array_equal(book[255], [4] * 4))
        self.assertTrue(np.array_equal(book[0b11100100], levels))

    def test_scale_fit_selects_seed_and_scale(self):
        book = np.array([[[1, 0, 0, 0], [0, 2, 0, 0]]], dtype=np.float32)
        targets = np.array([[3, 0, 0, 0], [0, 6, 0, 0]], dtype=np.float32)
        errors, scales = optimize_scale(book, targets, "cpu", 1.0)
        self.assertAlmostEqual(errors[0], 0, places=7)
        self.assertAlmostEqual(scales[0], 3, places=7)

    def test_program_parser_roundtrip(self):
        source = "r0=SHR(s, i); r1=MUL(r0, 0xf); r2=MUL(r1, 0x8)"
        self.assertEqual(expression(parse_program(source)), source)


if __name__ == "__main__":
    unittest.main()
