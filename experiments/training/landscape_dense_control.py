"""Independent dense-init divergence control for the paired MLP landscape run."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.training.landscape_mlp import (
    align_to, clone_state, load_layers, model_for, state_ce, state_distance, train, write_csv,
)
from experiments.training.mnist_qat import MLP, data_split


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--result",type=Path,default=Path("experiments/results/experiments-13-18/landscape_mlp"))
    args=parser.parse_args()
    torch.set_num_threads(4)
    source = args.result
    config = json.loads((source/"config.json").read_text())
    epochs = config["epochs"]
    image, label, _, _, train_np, valid_np, _ = data_split()
    x = torch.as_tensor(image.reshape(60000,-1), dtype=torch.float32, device="cuda")/255
    y = torch.as_tensor(label.astype(np.int64), device="cuda")
    train_ids = torch.as_tensor(train_np, device="cuda")
    valid_ids = torch.as_tensor(valid_np, device="cuda")
    rows, connectivity = [], []
    for seed in config["seeds"]:
        torch.manual_seed(seed+10000)
        independent = MLP().to("cuda")
        init = clone_state(independent.state_dict())
        scales = [float(layer.weight.detach().square().mean().sqrt()) for layer in independent.layers]
        run_dir = Path(config["checkpoint_dir"])/f"seed{seed}"
        model = model_for("dense", scales, "cuda")
        load_layers(model, init)
        endpoint = train(model, x, y, train_ids, valid_ids, seed=seed+10000,
                         first=0, last=epochs, out=run_dir,
                         label="dense_independent", rows=rows)
        paired = torch.load(run_dir/f"dense_epoch{epochs}.pt", weights_only=False)["decoded"]
        aligned = align_to(paired, endpoint)
        evaluator = model_for("dense", scales, "cuda")
        before = state_ce(endpoint, x, y, valid_ids, evaluator)
        after = state_ce(aligned, x, y, valid_ids, evaluator)
        assert abs(before-after) < 1e-5
        for layout, state in (("raw", endpoint), ("aligned", aligned)):
            for step in range(21):
                t = step/20
                middle = {key:(1-t)*paired[key]+t*state[key] for key in paired}
                connectivity.append(dict(seed=seed, method="dense_independent",layout=layout,t=t,
                                         validation_ce=state_ce(middle,x,y,valid_ids,evaluator),
                                         weight_nmse=state_distance(paired,state)))
    write_csv(source/"dense_control_curves.csv", rows)
    write_csv(source/"dense_control_connectivity.csv", connectivity)
    (source/"dense_control_config.json").write_text(json.dumps(dict(
        seeds=config["seeds"], independent_seed_offset=10000, epochs=epochs,
        batch_size=256, optimizer="Adam lr 0.001", split="same exposed validation",
        alignment="incoming-row cosine Hungarian, legal ReLU permutations"),indent=2)+"\n")


if __name__ == "__main__":
    main()
