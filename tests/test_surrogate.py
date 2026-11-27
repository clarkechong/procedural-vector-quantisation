import unittest

import numpy as np
import torch

from experiments.training.surrogate import exact_expectation, vertex_jacobian
from experiments.training.tiny_oracle import codebooks


class SurrogateTests(unittest.TestCase):
    def test_vertex_jacobian(self):
        book = codebooks()["discovered"]
        seeds = np.arange(256)
        j = vertex_jacobian(book, seeds)
        for bit in range(8):
            off = seeds & ~(1 << bit)
            np.testing.assert_array_equal(j[:, :, bit], book[off | (1 << bit)]-book[off])
            np.testing.assert_array_equal(book[seeds ^ (1 << bit)]-book[seeds],
                                          j[:, :, bit] * (1-2*((seeds >> bit)&1))[:, None])

    def test_expectation_corner_and_gradient(self):
        book = codebooks()["discovered"]
        seed = 73
        bits = (seed >> np.arange(8)) & 1
        logits = torch.tensor(np.where(bits, 25., -25.), dtype=torch.float64, requires_grad=True)
        mean = exact_expectation(book, logits)
        torch.testing.assert_close(mean, torch.tensor(book[seed]), atol=2e-9, rtol=0)
        logits = torch.tensor([.2,-.1,.4,.6,-.3,.2,.1,-.5], dtype=torch.float64, requires_grad=True)
        direction = torch.tensor([.5, -.2, .3, -.1], dtype=torch.float64)
        scalar = (exact_expectation(book, logits)*direction).sum()
        analytic = torch.autograd.grad(scalar, logits)[0]
        eps = 1e-5
        for bit in range(8):
            moved = logits.detach().clone()
            moved[bit] += eps
            high = (exact_expectation(book, moved)*direction).sum()
            moved[bit] -= 2*eps
            low = (exact_expectation(book, moved)*direction).sum()
            self.assertAlmostEqual(float(analytic[bit]), float((high-low)/(2*eps)), places=8)
