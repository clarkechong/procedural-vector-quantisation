"""Matched two-bit scalar projection baseline with per-matrix scale."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.evaluation.seedlm_model import add_evaluation_arguments, run


class ScalarTwoBit:
    def __init__(self):
        self.matrix_count = 0

    def encode_matrix(self, matrix):
        values = np.asarray(matrix, dtype=np.float32)
        scale = np.sqrt(np.mean(np.square(values, dtype=np.float64))/5)
        for _ in range(3):
            indices = np.clip(np.rint((values/scale+3)/2),0,3).astype(np.uint8)
            levels = (2*indices.astype(np.int16)-3).astype(np.float32)
            denominator = float(np.sum(np.square(levels, dtype=np.float64)))
            scale = max(float(np.sum(values*levels, dtype=np.float64)/denominator),1e-12)
        indices = np.clip(np.rint((values/scale+3)/2),0,3).astype(np.uint8)
        levels = (2*indices.astype(np.int16)-3).astype(np.float32)
        restored = levels*np.float32(scale)
        bits = np.empty((*indices.shape,2),dtype=np.uint8)
        bits[...,0]=indices&1
        bits[...,1]=indices>>1
        packed=np.packbits(bits.reshape(indices.shape[0],-1),axis=1,bitorder="little")
        self.matrix_count += 1
        return restored, np.float32(scale).tobytes()+packed.tobytes()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,default=Path("data/smollm2-135m.safetensors"))
    parser.add_argument("--data",type=Path,default=Path("data/smollm2"))
    parser.add_argument("--out",type=Path,default=Path("experiments/results/activation_ptq/scalar_model_smoke"))
    parser.add_argument("--tokens",type=int,default=128)
    parser.add_argument("--context",type=int,default=128)
    parser.add_argument("--batch",type=int,default=1)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--calibration-tokens",type=int,default=0)
    parser.add_argument("--methods",nargs="+",choices=("original","scalar_2bit"),default=("original","scalar_2bit"))
    add_evaluation_arguments(parser)
    args=parser.parse_args()
    args.seeds=0;args.shortlist=0;args.neighborhood=0
    run(args,{"scalar_2bit":ScalarTwoBit},
        {"codec":"four uniform symmetric levels -3,-1,1,3",
         "scale":"one fitted float32 per matrix, three alternating passes",
         "packing":"two bits per weight, byte-aligned rows"})
    (args.out/"codec.json").write_text(json.dumps(dict(levels=[-3,-1,1,3],
      scale="shared per matrix, three alternating passes",packing="low-bit first, byte-aligned rows"),indent=2)+"\n")


if __name__=="__main__":
    main()
