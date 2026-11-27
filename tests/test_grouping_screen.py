import importlib.util
import itertools
import unittest
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parents[1] / 'experiments/clustering/grouping_screen.py'
spec = importlib.util.spec_from_file_location('grouping_screen', P)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


def exact_partitions(indices, size):
    if not indices:
        yield []
        return
    first, *rest = indices
    for others in itertools.combinations(rest, size - 1):
        remaining = [x for x in rest if x not in others]
        for tail in exact_partitions(remaining, size):
            yield [[first, *others], *tail]


class GroupingTests(unittest.TestCase):
    def test_controls_and_exact_oracle(self):
        cases = [np.zeros((6, 9), bool), np.ones((6, 9), bool),
                 np.eye(6, 9, dtype=bool),
                 np.array([[1, 0, 0, 0], [1, 0, 0, 0],
                           [0, 0, 1, 0], [0, 0, 1, 0],
                           [1, 0, 0, 0], [0, 0, 1, 0]], bool)]
        for a in cases:
            with self.subTest(shape=a.shape):
                oracle = min(g.cost(a, np.asarray(p)) for p in exact_partitions(list(range(6)), 2))
                start = g.original(6, 2)
                refined = g.refine(a, start, 3)
                self.assertGreaterEqual(g.cost(a, refined), oracle)
                self.assertLessEqual(g.cost(a, refined), g.cost(a, start))
                self.assertTrue(np.array_equal(np.sort(refined.ravel()), np.arange(6)))

    def test_channel_block_mapping(self):
        x = np.zeros((2, 17), dtype=np.float32)
        x[0, 0] = 1
        x[1, 16] = -1
        a = g.block_masks(x, 16)
        np.testing.assert_array_equal(a, [[True, False], [False, True]])
        self.assertEqual(g.cost(a, g.original(2, 2)), 2)


if __name__ == '__main__':
    unittest.main()
