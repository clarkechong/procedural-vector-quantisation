"""Memory-map local Hugging Face SafeTensors without loading a model."""

import json
import math
import struct
from pathlib import Path

import numpy as np


PROJECTIONS = {
    **{name: "mlp" for name in ("gate_proj", "up_proj", "down_proj")},
    **{name: "self_attn" for name in ("q_proj", "k_proj", "v_proj", "o_proj")},
}


class Weights:
    def __init__(self, checkpoint):
        checkpoint = Path(checkpoint)
        if checkpoint.is_dir():
            index = checkpoint / "model.safetensors.index.json"
            if index.is_file():
                weight_map = json.loads(index.read_text())["weight_map"]
                files = [checkpoint / name for name in sorted(set(weight_map.values()))]
            else:
                files = sorted(checkpoint.glob("*.safetensors"))
        else:
            files = [checkpoint]

        self.tensors = {}
        self.files = {}
        for path in files:
            with path.open("rb") as handle:
                length, = struct.unpack("<Q", handle.read(8))
                if not 0 < length <= min(100_000_000, path.stat().st_size - 8):
                    raise ValueError(f"Invalid SafeTensors header size in {path}")
                header = json.loads(handle.read(length))
            for name, metadata in header.items():
                if name != "__metadata__":
                    if name in self.tensors:
                        raise ValueError(f"Duplicate tensor {name}")
                    self.tensors[name] = (path, 8 + length, metadata)
            self.files[path.name] = path.stat().st_size
        if not self.tensors:
            raise ValueError(f"No SafeTensors weights found at {checkpoint}")

    @property
    def layers(self):
        return sorted({
            int(name.split(".")[2]) for name in self.tensors
            if name.startswith("model.layers.") and name.endswith(".self_attn.q_proj.weight")
        })

    def get(self, layer, projection):
        name = f"model.layers.{layer}.{PROJECTIONS[projection]}.{projection}.weight"
        path, data_start, metadata = self.tensors[name]
        dtype = {"F32": "<f4", "F16": "<f2", "BF16": "<u2"}[metadata["dtype"]]
        shape = tuple(metadata["shape"])
        start, end = metadata["data_offsets"]
        if (len(shape) != 2 or start < 0 or end - start != math.prod(shape) * np.dtype(dtype).itemsize
                or data_start + end > self.files[path.name]):
            raise ValueError(f"Invalid tensor layout for {name}")
        raw = np.memmap(path, mode="r", dtype=dtype, offset=data_start + start, shape=shape)
        if metadata["dtype"] == "BF16":
            return (raw.astype(np.uint32) << 16).view(np.float32)
        return np.asarray(raw, dtype=np.float32)
