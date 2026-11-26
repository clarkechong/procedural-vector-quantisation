import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from experiments.encoding.pursuit import fp16, pursuit, seeded_basis
from experiments.rate_distortion.pilot import control_datasets
from experiments.generator_algebra.dictionary import dictionary, parity, planted, random_signs
from experiments.manifests.weights import Weights


class ExperimentTests(unittest.TestCase):
    def test_pursuit_matches_explicit_reconstruction(self):
        rng = np.random.default_rng(5)
        atoms, _ = dictionary((8, 8), 4)
        blocks = rng.normal(size=(3, 64)).astype(np.float32)
        residual = blocks.copy()
        rows = np.arange(len(blocks))
        norms = np.sum(atoms ** 2, axis=1)
        for _, energies in pursuit(blocks, atoms, 8):
            dots = residual @ atoms.T
            coefficients = fp16(dots / norms)
            gains = coefficients * (2 * dots - coefficients * norms)
            best = np.argmax(gains, axis=1)
            residual -= coefficients[rows, best, None] * atoms[best]
            np.testing.assert_allclose(energies, np.sum(residual ** 2, axis=1), rtol=1e-5)

    def test_seeded_basis_uses_real_geometry(self):
        rng = np.random.default_rng(6)
        blocks = rng.normal(size=(3, 18)).astype(np.float32)
        positions = np.arange(18, dtype=np.uint32)
        for count, energy in seeded_basis(blocks, seeds=4, counts=(1, 2, 4)):
            walsh = parity(positions & np.arange(count, dtype=np.uint32)[:, None]) / np.sqrt(32)
            results = []
            for seed in range(4):
                atoms = walsh * random_signs(positions, seed)
                coefficients = fp16(blocks @ np.linalg.pinv(atoms))
                results.append(np.sum((blocks - coefficients @ atoms) ** 2, axis=1))
            np.testing.assert_allclose(energy, np.min(results, axis=0), rtol=1e-6)

    def test_controls_preserve_distributions_and_recover_plant(self):
        rng = np.random.default_rng(7)
        weights = np.arange(256, dtype=np.float32).reshape(16, 16)
        tiles = np.stack((weights[:8, :8], weights[8:, 8:]))
        cases = {(name, trial): values for name, trial, values in
                 control_datasets(weights, tiles, 3, 1.0, rng)}
        for trial in range(3):
            local = cases["within_tile_shuffle", trial].reshape(2, -1)
            np.testing.assert_array_equal(np.sort(local), np.sort(tiles.reshape(2, -1)))
            global_draw = cases["global_shuffle", trial]
            self.assertEqual(len(np.unique(global_draw)), tiles.size)
        atoms, _ = dictionary((16, 16), 4)
        signal = planted((16, 16), rng).astype(np.float32).reshape(1, -1)
        _, error = list(pursuit(signal, atoms, 4))[-1]
        self.assertLess(float(error[0] / np.sum(signal ** 2)), 1e-5)

    def test_sharded_checkpoint_with_fp16_and_bf16(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = np.array([[1.0, -2.0], [0.5, 4.0]], dtype=np.float32)
            tensors = {
                "model.layers.0.self_attn.q_proj.weight":
                    ("F16", original.astype("<f2")),
                "model.layers.0.mlp.up_proj.weight":
                    ("BF16", (original.view(np.uint32) >> 16).astype("<u2")),
            }
            index = {}
            for number, (name, (dtype, values)) in enumerate(tensors.items()):
                shard = f"part-{number}.safetensors"
                header = json.dumps({
                    name: {"dtype": dtype, "shape": [2, 2], "data_offsets": [0, values.nbytes]}
                }).encode()
                (root / shard).write_bytes(struct.pack("<Q", len(header)) + header + values.tobytes())
                index[name] = shard
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
            weights = Weights(root)
            self.assertEqual(weights.layers, [0])
            np.testing.assert_array_equal(weights.get(0, "q_proj"), original)
            np.testing.assert_array_equal(weights.get(0, "up_proj"), original)
