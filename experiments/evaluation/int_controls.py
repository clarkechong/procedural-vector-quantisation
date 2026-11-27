"""Matched model-level symmetric INT4 and INT8 scalar controls."""

import argparse
from pathlib import Path

import numpy as np

from experiments.evaluation.seedlm_model import add_evaluation_arguments, run


class UniformScalar:
    def __init__(self, bits, iterations=3):
        self.bits=bits
        self.iterations=iterations
        self.levels=np.arange(-(1<<bits)+1,1<<bits,2,dtype=np.float32)

    def encode_matrix(self,matrix):
        values=np.asarray(matrix,dtype=np.float32)
        scale=np.sqrt(float(np.mean(values.astype(np.float64)**2))/float(np.mean(self.levels**2)))
        for _ in range(self.iterations):
            indices=np.clip(np.rint((values/scale-self.levels[0])/2),0,len(self.levels)-1).astype(np.uint8)
            chosen=self.levels[indices]
            denominator=float(np.sum(chosen.astype(np.float64)**2))
            scale=max(float(np.sum(values*chosen,dtype=np.float64)/denominator),1e-12)
        indices=np.clip(np.rint((values/scale-self.levels[0])/2),0,len(self.levels)-1).astype(np.uint8)
        restored=self.levels[indices]*np.float32(scale)
        if self.bits==4:
            pair=np.empty((indices.shape[0],indices.shape[1]//2),dtype=np.uint8)
            pair[:]=indices[:,0::2] | (indices[:,1::2]<<4)
            packed=pair.tobytes()
        else:
            packed=indices.tobytes()
        return restored,np.float32(scale).tobytes()+packed


class GroupedScalar:
    def __init__(self,bits,group):
        self.bits=bits
        self.group=group
        self.q=(1<<(bits-1))-1

    def encode_matrix(self,matrix):
        values=np.asarray(matrix,dtype=np.float32)
        rows,columns=values.shape
        group=columns if self.group=="channel" else self.group
        if columns%group:
            raise ValueError("group must divide matrix width")
        blocks=values.reshape(rows,-1,group)
        maximum=np.max(np.abs(blocks),axis=-1,keepdims=True)
        base=np.maximum(maximum/self.q,1e-12)
        best_error=np.full(base.shape,np.inf,dtype=np.float64)
        best_scale=base.copy()
        for factor in (0.25,0.35,0.5,0.7,1.0):
            scale=base*factor
            codes=np.clip(np.rint(blocks/scale),-self.q,self.q)
            error=np.sum((blocks-codes*scale).astype(np.float64)**2,axis=-1,keepdims=True)
            choose=error<best_error
            best_error=np.where(choose,error,best_error)
            best_scale=np.where(choose,scale,best_scale)
        signed=np.clip(np.rint(blocks/best_scale),-self.q,self.q).astype(np.int16)
        restored=(signed*best_scale).reshape(values.shape).astype(np.float32)
        encoded=(signed+self.q).astype(np.uint8).reshape(values.shape)
        if self.bits==4:
            packed=(encoded[:,0::2] | (encoded[:,1::2]<<4)).tobytes()
        else:
            packed=encoded.tobytes()
        return restored,best_scale.astype(np.float32).tobytes()+packed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,default=Path("data/smollm2-135m.safetensors"))
    parser.add_argument("--data",type=Path,default=Path("data/smollm2"))
    parser.add_argument("--out",type=Path,default=Path("results/e04-07-reference-optimization/activation-ptq/int-controls"))
    parser.add_argument("--tokens",type=int,default=2048)
    parser.add_argument("--context",type=int,default=128)
    parser.add_argument("--batch",type=int,default=1)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--calibration-tokens",type=int,default=0)
    parser.add_argument("--methods",nargs="+",choices=["original","int4","int8",
                                                "int4_group64","int8_channel","int8_fitted"],
                        default=["original","int4","int8"])
    add_evaluation_arguments(parser)
    args=parser.parse_args()
    args.seeds=0;args.shortlist=0;args.neighborhood=0
    run(args,{"int4":lambda:UniformScalar(4),"int8":lambda:UniformScalar(8),
              "int8_fitted":lambda:UniformScalar(8,50),
              "int4_group64":lambda:GroupedScalar(4,64),
              "int8_channel":lambda:GroupedScalar(8,"channel")},
        {"codec":"symmetric uniform signed levels; one fitted FP32 scale per matrix",
         "packing":"INT4 two nibbles per byte, INT8 one byte per weight"})


if __name__=="__main__":main()
