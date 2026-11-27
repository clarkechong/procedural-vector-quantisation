import unittest

import torch

from experiments.evaluation.seedlm_model import perplexity


class ShiftedEvaluator(unittest.TestCase):
    def test_shift_count_and_context_reset(self):
        class Model:
            def __call__(self, tokens, use_cache=False):
                batch, length=tokens.shape
                logits=torch.zeros(batch,length,4)
                logits.scatter_(2,((tokens+1)%4)[...,None],2.)
                return type("Output",(),{"logits":logits})()
        tokens=torch.tensor([0,1,2,3,0,1,2,3,0])
        nll,count=perplexity(Model(),tokens,4,2)
        self.assertEqual(count,6)
        windows=[tokens[:4],tokens[4:8]]
        manual=[]
        for window in windows:
            logits=Model()(window[None]).logits[:,:-1]
            manual.append(torch.nn.functional.cross_entropy(logits.reshape(-1,4),window[1:],reduction="sum"))
        self.assertAlmostEqual(nll,float(sum(manual)/6),places=7)
