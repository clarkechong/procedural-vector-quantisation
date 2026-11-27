import unittest

import numpy as np
import torch

from experiments.reference.qtip.batched_trellis import quantize
from experiments.reference.qtip.tailbite import quantize as cpu_quantize


class BatchedTrellisTest(unittest.TestCase):
    def test_two_pass_paths_match_cpu_with_small_state_space(self):
        rng = np.random.default_rng(41)
        rows = rng.standard_normal((6, 20)).astype(np.float32)
        mapping = rng.standard_normal(256).astype(np.float32)
        decoded, paths = quantize(torch.from_numpy(rows), torch.from_numpy(mapping),
                                  state_bits=8, stride=2)
        for index in range(len(rows)):
            reference, path = cpu_quantize(rows[index], mapping, state_bits=8, stride=2)
            np.testing.assert_array_equal(paths[index].numpy(), path)
            np.testing.assert_array_equal(decoded[index].numpy(), reference)

    def test_odd_length_rejected(self):
        with self.assertRaises(ValueError):
            quantize(torch.zeros(2, 5), torch.zeros(256), state_bits=8)


if __name__ == "__main__":
    unittest.main()
