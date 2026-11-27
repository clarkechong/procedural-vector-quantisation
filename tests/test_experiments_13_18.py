import unittest

import numpy as np
import torch

from experiments.evaluation.factorial_ptq import books, encode
from experiments.training.pack_mlp import pack, unpack, SHAPES


class ExperimentContracts(unittest.TestCase):
    def test_factorial_limit_identities(self):
        weight=torch.tensor([[.1,.2,-.3,.5,.2,-.4,.7,.1]],dtype=torch.float32)
        h=torch.tensor([.5,1,2,3,.8,1.5,.25,4],dtype=torch.float32)
        book=torch.from_numpy(books()["mul120"])
        for alpha,left,right in ((0,"A","B"),(0,"C","D"),(1,"B","D")):
            scale=(h.sqrt()/torch.exp(h.sqrt().log().mean())).pow(alpha)
            a,codes_a,_,_=encode(weight,h,book,scale,left)
            b,codes_b,_,_=encode(weight,h,book,scale,right)
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(codes_a,codes_b)

    def test_packed_mlp_independent_reload(self):
        scale=[.05,.04,.03]
        for method in ("scalar2","mul120_2bpw"):
            state={}
            book=books()["mul120"]
            for i,(rows,cols) in enumerate(SHAPES):
                if method=="scalar2":
                    values=np.asarray([-3,-1,1,3],dtype=np.float32)/np.sqrt(np.float32(5))
                    weight=np.resize(values,rows*cols).reshape(rows,cols)*scale[i]
                else:
                    weight=np.resize(book,rows*cols).reshape(rows,cols)*scale[i]
                state[f"layers.{i}.weight"]=torch.from_numpy(weight.copy())
                state[f"layers.{i}.bias"]=torch.zeros(rows)
            result=unpack(pack(state,scale,method))
            for key in state:
                torch.testing.assert_close(result[key],state[key],rtol=1e-5,atol=1e-6)


if __name__=="__main__":unittest.main()
