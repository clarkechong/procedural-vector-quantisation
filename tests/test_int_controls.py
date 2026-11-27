import unittest

import numpy as np

from experiments.evaluation.int_controls import GroupedScalar


class GroupedScalarTests(unittest.TestCase):
    def test_packed_grouped_int4_reconstructs_exported_values(self):
        values=np.random.default_rng(33).normal(size=(3,128)).astype(np.float32)
        encoder=GroupedScalar(4,64)
        restored,payload=encoder.encode_matrix(values)
        scales=np.frombuffer(payload[:3*2*4],dtype=np.float32).reshape(3,2,1)
        packed=np.frombuffer(payload[3*2*4:],dtype=np.uint8).reshape(3,64)
        codes=np.empty((3,128),dtype=np.uint8)
        codes[:,0::2]=packed&15
        codes[:,1::2]=packed>>4
        decoded=((codes.reshape(3,2,64).astype(np.int16)-7)*scales).reshape(3,128)
        np.testing.assert_array_equal(decoded,restored)

    def test_int8_grouped_error_is_small(self):
        values=np.random.default_rng(34).normal(size=(4,128)).astype(np.float32)
        restored,_=GroupedScalar(8,"channel").encode_matrix(values)
        nmse=np.sum((restored-values)**2)/np.sum(values**2)
        self.assertLess(nmse,0.0002)


if __name__=="__main__":unittest.main()
