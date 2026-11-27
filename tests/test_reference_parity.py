"""Pinned upstream parity tests; set QTIP_SOURCE and SMOOTHQUANT_SOURCE to clones."""

import ast
import os
from pathlib import Path
import unittest

import numpy as np
import torch

from experiments.reference.qtip.codebooks import one_mad, three_inst
from experiments.reference.smoothquant.transform import scales, transform


class ReferenceParity(unittest.TestCase):
    def test_qtip_all_states(self):
        path = Path(os.environ.get("QTIP_SOURCE", "/tmp/qtip-official")) / "lib/codebook/bitshift.py"
        if not path.exists():
            self.skipTest("pinned official QTIP source unavailable")
        tree = ast.parse(path.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("decode_1mad", "decode_3inst")]
        scope = {"torch": torch, "np": np}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), scope)
        states = np.arange(65536, dtype=np.int64)
        for name, local in (("decode_1mad", one_mad), ("decode_3inst", three_inst)):
            expected = scope[name](torch.from_numpy(states)).numpy()
            np.testing.assert_array_equal(local(states), expected)

    def test_smoothquant_shared_input(self):
        root = Path(os.environ.get("SMOOTHQUANT_SOURCE", "/tmp/smoothquant-official"))
        if not root.exists():
            self.skipTest("pinned official SmoothQuant source unavailable")
        import sys
        sys.path.insert(0, str(root))
        from smoothquant.smooth import smooth_ln_fcs_llama_like
        from transformers.models.llama.modeling_llama import LlamaRMSNorm
        torch.manual_seed(7)
        norm = LlamaRMSNorm(8).double()
        first, second = torch.nn.Linear(8, 5, bias=False).double(), torch.nn.Linear(8, 3, bias=False).double()
        act = torch.rand(8, dtype=torch.float64) + 0.01
        weights = [first.weight.detach().clone(), second.weight.detach().clone()]
        expected_weights, expected_scale = transform(weights, act, 0.85)
        expected_norm = norm.weight.detach().clone() / expected_scale
        smooth_ln_fcs_llama_like(norm, [first, second], act, 0.85)
        torch.testing.assert_close(norm.weight, expected_norm, rtol=0, atol=0)
        for actual, expected in zip((first.weight, second.weight), expected_weights):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        x = torch.randn(8, 13, dtype=torch.float64)
        for original, transformed in zip(weights, expected_weights):
            torch.testing.assert_close(original @ x, transformed @ (x / expected_scale[:, None]), rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(scales(weights, act, .85), expected_scale)
