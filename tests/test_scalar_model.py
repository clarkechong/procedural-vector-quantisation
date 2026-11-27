import unittest

import numpy as np

from experiments.evaluation.scalar_model import ScalarTwoBit


class ScalarModel(unittest.TestCase):
    def test_payload_roundtrip(self):
        rng=np.random.default_rng(11)
        matrix=rng.normal(size=(3,13)).astype(np.float32)
        reconstructed,payload=ScalarTwoBit().encode_matrix(matrix)
        scale=np.frombuffer(payload[:4],dtype=np.float32)[0]
        packed=np.frombuffer(payload[4:],dtype=np.uint8).reshape(3,-1)
        bits=np.unpackbits(packed,axis=1,bitorder="little")[:,:26].reshape(3,13,2)
        indices=bits[...,0]+2*bits[...,1]
        decoded=(2*indices.astype(np.int16)-3)*scale
        np.testing.assert_array_equal(reconstructed,decoded)
