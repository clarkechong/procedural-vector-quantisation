"""Paired MLP scratch/FT, release, connectivity and cross-codec pilot.

Run with host MI300X access.  Official MNIST test is intentionally unused:
the existing 5k training-validation split is an exposed exploratory split.
"""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F

from experiments.training.mnist_qat import MLP, data_split, evaluate
from experiments.training.low_rate_qat import QuantMLP, make_book, storage


METHODS = ("dense", "scalar2", "mul120_2bpw", "dense_projected")


def clone_state(state):
    return {key: value.detach().clone() for key, value in state.items()}


def model_for(method, scales, device):
    if method == "mul120_2bpw":
        return QuantMLP("procedural", scales, make_book("mul120", 4), 4).to(device)
    if method == "scalar2":
        return QuantMLP("scalar2", scales).to(device)
    return QuantMLP("dense", scales).to(device)


def decoded_state(model):
    state = clone_state(model.state_dict())
    if model.method != "dense":
        with torch.no_grad():
            for i, layer in enumerate(model.layers):
                state[f"layers.{i}.weight"] = model.quantize(layer.weight, model.scales[i]).detach().clone()
    return {key: value for key, value in state.items() if key.startswith("layers.")}


def load_layers(model, state):
    model.load_state_dict(state, strict=False)
    return model


def train(model, x, y, train_ids, valid_ids, *, seed, first, last, out, label, rows):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    started = time.perf_counter()
    for epoch in range(first, last + 1):
        ce, accuracy = evaluate(model, x[valid_ids], y[valid_ids])
        rows.append(dict(seed=seed, branch=label, epoch=epoch,
                         validation_ce=ce, validation_accuracy=accuracy,
                         elapsed_s=time.perf_counter()-started))
        state = decoded_state(model)
        torch.save(dict(epoch=epoch, decoded=state,
                        shadow=clone_state(model.state_dict()),
                        optimizer=optimizer.state_dict()), out/f"{label}_epoch{epoch}.pt")
        print(seed, label, epoch, ce, flush=True)
        if epoch == last:
            break
        model.train()
        order = torch.randperm(len(train_ids),
                               generator=torch.Generator().manual_seed(seed*1000+epoch)).to(x.device)
        for chosen in order.split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[train_ids[chosen]]), y[train_ids[chosen]])
            loss.backward()
            optimizer.step()
    return decoded_state(model)


def align_to(reference, target):
    """Legal ReLU hidden-unit permutations, matched on incoming weight cosine."""
    aligned = clone_state(target)
    for layer in (0, 1):
        key = f"layers.{layer}.weight"
        a = F.normalize(reference[key].float(), dim=1).cpu().numpy()
        b = F.normalize(aligned[key].float(), dim=1).cpu().numpy()
        rows, cols = linear_sum_assignment(-(a @ b.T))
        assert np.array_equal(rows, np.arange(len(rows)))
        permutation = torch.as_tensor(cols, device=aligned[key].device)
        aligned[key] = aligned[key][permutation]
        bias = f"layers.{layer}.bias"
        aligned[bias] = aligned[bias][permutation]
        next_key = f"layers.{layer+1}.weight"
        aligned[next_key] = aligned[next_key][:, permutation]
    return aligned


@torch.inference_mode()
def state_ce(state, x, y, valid_ids, template):
    load_layers(template, state)
    return evaluate(template, x[valid_ids], y[valid_ids])[0]


def state_distance(a, b):
    keys = [key for key in a if key.endswith(".weight")]
    numerator = sum(float((a[k]-b[k]).square().sum()) for k in keys)
    denominator = sum(float(a[k].square().sum()) for k in keys)
    return numerator/denominator


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(set().union(*(row.keys() for row in rows))))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 17, 23])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--out", type=Path,
                        default=Path("experiments/results/experiments-13-18/landscape_mlp"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("MI300X ROCm device required")
    args.out.mkdir(parents=True, exist_ok=True)
    image, label, _, _, train_np, valid_np, manifest = data_split()
    x = torch.as_tensor(image.reshape(60000, -1), dtype=torch.float32, device="cuda")/255
    y = torch.as_tensor(label.astype(np.int64), device="cuda")
    train_ids = torch.as_tensor(train_np, device="cuda")
    valid_ids = torch.as_tensor(valid_np, device="cuda")
    (args.out/"split_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    curves, connectivity, cross, distances = [], [], [], []
    switch = args.epochs*3//4
    release_epochs = max(2, args.epochs//4)
    for seed in args.seeds:
        run_dir = Path("runs/experiments-13-18")/args.out.name/f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(seed)
        initial_model = MLP().to("cuda")
        initial = clone_state(initial_model.state_dict())
        scales = [float(layer.weight.detach().square().mean().sqrt()) for layer in initial_model.layers]
        torch.save(initial, run_dir/"initial.pt")
        projected = model_for("mul120_2bpw", scales, "cuda")
        load_layers(projected, initial)
        projected_initial = decoded_state(projected)
        final = {}
        for method in METHODS:
            model = model_for(method, scales, "cuda")
            load_layers(model, projected_initial if method == "dense_projected" else initial)
            final[method] = train(model, x, y, train_ids, valid_ids, seed=seed,
                                  first=0, last=args.epochs, out=run_dir,
                                  label=method, rows=curves)
            write_csv(args.out/"curves.csv", curves)
            del model
        # Paired dense-75% -> QAT and dense reset continuations.
        saved = torch.load(run_dir/f"dense_epoch{switch}.pt", weights_only=False)
        for method in ("dense", "scalar2", "mul120_2bpw"):
            model = model_for(method, scales, "cuda")
            load_layers(model, saved["decoded"])
            label_ft = f"ft_{method}"
            final[label_ft] = train(model, x, y, train_ids, valid_ids, seed=seed,
                                    first=switch, last=args.epochs, out=run_dir,
                                    label=label_ft, rows=curves)
            write_csv(args.out/"curves.csv", curves)
            del model
        # Remove the actual deployed constraint and train with a common reset.
        for method in ("dense", "scalar2", "mul120_2bpw", "ft_mul120_2bpw"):
            model = model_for("dense", scales, "cuda")
            load_layers(model, final[method])
            label_release = f"release_{method}"
            final[label_release] = train(model, x, y, train_ids, valid_ids, seed=seed,
                                         first=args.epochs, last=args.epochs+release_epochs,
                                         out=run_dir, label=label_release, rows=curves)
            write_csv(args.out/"curves.csv", curves)
            del model
        evaluator = model_for("dense", scales, "cuda")
        ref = final["dense"]
        for method in ("scalar2", "mul120_2bpw", "dense_projected",
                       "ft_scalar2", "ft_mul120_2bpw", "release_scalar2",
                       "release_mul120_2bpw", "release_ft_mul120_2bpw"):
            target = final[method]
            aligned = align_to(ref, target)
            before = state_ce(target, x, y, valid_ids, evaluator)
            after = state_ce(aligned, x, y, valid_ids, evaluator)
            if abs(before-after) > 1e-5:
                raise AssertionError(f"alignment changed function: {method} {before} {after}")
            distances.append(dict(seed=seed, method=method,
                                  raw_nmse=state_distance(ref, target),
                                  aligned_nmse=state_distance(ref, aligned),
                                  validation_ce=before))
            for layout, endpoint in (("raw", target), ("aligned", aligned)):
                for step in range(21):
                    t = step/20
                    mixed = {key: (1-t)*ref[key]+t*endpoint[key] for key in ref}
                    ce = state_ce(mixed, x, y, valid_ids, evaluator)
                    connectivity.append(dict(seed=seed, method=method, layout=layout,
                                             t=t, validation_ce=ce))
        write_csv(args.out/"distances.csv", distances)
        write_csv(args.out/"connectivity.csv", connectivity)
        # Frozen independent-codec transfer.  Same scales from paired W0.
        for source_name in ("dense", "scalar2", "mul120_2bpw",
                            "release_mul120_2bpw", "release_scalar2"):
            source = final[source_name]
            source_ce = state_ce(source, x, y, valid_ids, evaluator)
            for codec_name in ("scalar2", "mul120_2bpw", "scalar4"):
                codec = (model_for(codec_name, scales, "cuda") if codec_name != "scalar4"
                         else QuantMLP("scalar4", scales).to("cuda"))
                load_layers(codec, source)
                encoded = decoded_state(codec)
                ce = state_ce(encoded, x, y, valid_ids, evaluator)
                cross.append(dict(seed=seed, source=source_name, codec=codec_name,
                                  source_ce=source_ce, compressed_ce=ce, delta_ce=ce-source_ce,
                                  weight_nmse=state_distance(source, encoded),
                                  stored_bits=storage(codec_name, 4)))
        write_csv(args.out/"cross_codec.csv", cross)
        print("completed seed", seed, flush=True)
    (args.out/"config.json").write_text(json.dumps(dict(
        seeds=args.seeds, epochs=args.epochs, switch_epoch=switch,
        release_epochs=release_epochs, architecture=[784,256,128,10],
        optimizer="Adam, lr=0.001, reset at transition/release",
        batch_size=256, scales="frozen layer RMS at paired dense W0",
        scratch_methods=METHODS, validation="exposed 5k training holdout",
        checkpoint_dir=f"runs/experiments-13-18/{args.out.name}",
        torch=torch.__version__, hip=torch.version.hip), indent=2)+"\n")


if __name__ == "__main__":
    main()
