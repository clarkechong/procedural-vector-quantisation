import unittest

import numpy as np
import torch

from experiments.training.latent_qat import HardSeed
from experiments.training.low_rate_qat import make_book, storage, WEIGHTS
from experiments.reference.qtip.tailbite import viterbi, quantize, pack_path, unpack_path


class LowRateTrainingTests(unittest.TestCase):
    def test_random_codebook_storage_counts_table(self):
        self.assertEqual(storage("random_vq_0.5bpw", 16) - storage("hash32_0.5bpw", 16),
                         256 * 16 * 32)
        self.assertGreater(storage("random_vq_0.5bpw", 16) / WEIGHTS, 1.0)

    def test_exact_discrete_jacobian(self):
        book = torch.tensor(make_book("mul120", 4))
        seed = 37
        z = torch.tensor([[0.25 if seed & (1 << bit) else -0.25 for bit in range(8)]],
                         requires_grad=True)
        direction = torch.tensor([[0.7, -0.2, 0.1, 1.3]])
        (HardSeed.apply(z, book, "discrete_jacobian") * direction).sum().backward()
        expected = []
        for bit in range(8):
            off = seed & ~(1 << bit)
            expected.append(float(((book[off | (1 << bit)] - book[off]) * direction[0]).sum()) * 0.75)
        np.testing.assert_allclose(z.grad.numpy()[0], expected, rtol=1e-6, atol=1e-6)

    def test_bernoulli_backward_matches_expectation_finite_difference(self):
        book = torch.tensor(make_book("qtip_1mad", 4), dtype=torch.float64)
        z = torch.linspace(-0.6, 0.8, 8, dtype=torch.float64).reshape(1, 8).requires_grad_()
        direction = torch.tensor([[0.7, -0.2, 0.1, 1.3]], dtype=torch.float64)
        (HardSeed.apply(z, book, "bernoulli_exact") * direction).sum().backward()
        states = torch.arange(256)
        bits = ((states[:, None] >> torch.arange(8)) & 1).bool()

        def expectation(logits):
            p = logits.sigmoid()
            probability = torch.prod(torch.where(bits, p, 1-p), -1)
            return float(((probability @ book) * direction[0]).sum())

        numerical = []
        for bit in range(8):
            plus = z.detach().clone(); plus[0, bit] += 1e-5
            minus = z.detach().clone(); minus[0, bit] -= 1e-5
            numerical.append((expectation(plus[0]) - expectation(minus[0])) / 2e-5)
        np.testing.assert_allclose(z.grad.numpy()[0], numerical, rtol=1e-6, atol=1e-7)

    def test_tailbite_path_and_fixed_overlap_oracle(self):
        rng = np.random.default_rng(13)
        mapping = rng.normal(size=16)
        target = rng.normal(size=8)
        decoded, path = quantize(target, mapping, state_bits=4, stride=1)
        self.assertEqual(int(path[0] >> 1), int(path[-1] & 7))
        np.testing.assert_array_equal(path[1:] >> 1, path[:-1] & 7)
        self.assertEqual(len(pack_path(path,4,1)),1)
        np.testing.assert_array_equal(unpack_path(pack_path(path,4,1),len(path),4,1),path)
        overlap = 3
        selected = viterbi(target, mapping, 4, 1, overlap)
        costs = []
        for bits in range(1 << len(target)):
            state = (overlap << 1) | (bits & 1)
            candidate = [state]
            for position in range(1, len(target)):
                state = ((state << 1) & 15) | ((bits >> position) & 1)
                candidate.append(state)
            if candidate[-1] & 7 == overlap:
                costs.append(np.sum((target.astype(np.float16).astype(np.float32)
                                     - mapping[candidate])**2))
        self.assertAlmostEqual(float(np.sum((target.astype(np.float16).astype(np.float32)
                                              - mapping[selected])**2)), min(costs), places=5)


if __name__ == "__main__":
    unittest.main()
