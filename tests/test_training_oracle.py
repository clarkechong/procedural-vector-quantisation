import unittest

import numpy as np

from experiments.training.tiny_oracle import codebooks, full_cost


class TinyOracle(unittest.TestCase):
    def test_full_task_loss(self):
        book = codebooks()["discovered"]
        rng = np.random.default_rng(4)
        x = rng.normal(size=(16, 8))
        target = rng.normal(size=16)
        costs = full_cost(book, x, target)
        for i, j in ((0, 0), (77, 201), (255, 3)):
            direct = np.mean((x @ np.r_[book[i], book[j]] - target)**2)
            self.assertAlmostEqual(costs[i, j], direct, places=12)
