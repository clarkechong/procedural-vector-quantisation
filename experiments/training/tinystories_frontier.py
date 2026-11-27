"""Paired TinyStories rate frontier with matched scratch and dense-to-QAT controls."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from experiments.training.tinystories_transformer import TRAIN, VALID, QLinear, Transformer, evaluate, storage


SCRATCH = ("dense", "scalar4", "scalar2", "binary", "procedural2", "procedural1", "procedural0.5")


def encode_stories(column, indices, context=256):
    content = b"\0".join(column[int(i)].as_py().encode("utf-8", errors="replace") for i in indices)
    count = (len(content) - 1) // (context + 1)
    array = np.frombuffer(content[:count * (context + 1)], dtype=np.uint8).copy().reshape(count, context + 1)
    return torch.from_numpy(array.astype(np.int64))


def manifest(path, count, seed, confirmation=0):
    column = pq.read_table(path, columns=["text"]).column(0)
    order = np.random.default_rng(seed).permutation(len(column))
    selected = order[:count]
    heldout = order[count:count + confirmation]
    if len(heldout) != confirmation:
        raise ValueError("not enough distinct confirmation stories")
    return (encode_stories(column, selected), encode_stories(column, heldout) if confirmation else None,
            selected, heldout)


def train_arm(method, initial, data, indices, validation, confirmation, args, start=0, stop=None):
    stop = args.steps if stop is None else stop
    model = Transformer(method).to(args.device)
    model.load_state_dict(initial, strict=False)
    for layer in model.modules():
        if isinstance(layer, QLinear):
            layer.scale = float(layer.weight.detach().square().mean().sqrt())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    rows = []
    begin = time.perf_counter()
    checkpoint = None
    for step in range(start, stop + 1):
        if step % args.eval_every == 0 or step == stop:
            ce = evaluate(model, validation)
            rows.append(dict(method=method, seed=args.seed, step=step, validation_ce=ce,
                             validation_ppl=float(np.exp(ce)), elapsed_s=time.perf_counter() - begin))
            print(args.seed, method, step, ce, flush=True)
        if step == stop:
            break
        if method == "dense" and step == args.finetune_start:
            checkpoint = {key: value.detach().clone() for key, value in model.state_dict().items()}
        model.train()
        sample = data[indices[step]]
        optimizer.zero_grad(set_to_none=True)
        logits = model(sample[:, :-1])
        F.cross_entropy(logits.reshape(-1, 256), sample[:, 1:].reshape(-1)).backward()
        optimizer.step()
    confirmation_ce = evaluate(model, confirmation)
    size = storage(model, method)
    return rows, checkpoint, dict(method=method, seed=args.seed, selection_ce=rows[-1]["validation_ce"],
                                  confirmation_ce=confirmation_ce, confirmation_ppl=float(np.exp(confirmation_ce)),
                                  **size)


def run(args):
    torch.set_num_threads(4)
    if args.require_gpu and not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    train, _, train_ids, _ = manifest(TRAIN, args.train_stories, 20261130)
    selection, confirmation, selection_ids, confirmation_ids = manifest(VALID, args.valid_stories, 20261201,
                                                                        args.confirm_stories)
    train = train.to(args.device)
    selection = selection.to(args.device)
    confirmation = confirmation.to(args.device)
    all_rows, summaries = [], []
    for seed in args.seeds:
        args.seed = seed
        torch.manual_seed(seed)
        base = Transformer("dense")
        initial = {key: value.detach().clone() for key, value in base.state_dict().items()}
        rng = torch.Generator(device="cpu").manual_seed(seed * 1000 + 19)
        indices = torch.randint(len(train), (args.steps, args.batch_size), generator=rng).to(args.device)
        dense_checkpoint = None
        for method in args.methods:
            rows, checkpoint, summary = train_arm(method, initial, train, indices, selection, confirmation, args)
            if method == "dense":
                dense_checkpoint = checkpoint
            all_rows.extend(rows)
            summaries.append(summary)
            with (out / "curves.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=all_rows[0].keys())
                writer.writeheader(); writer.writerows(all_rows)
            with (out / "summary.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
                writer.writeheader(); writer.writerows(summaries)
        if dense_checkpoint is None:
            raise ValueError("dense arm required for fine-tuning controls")
        for method in args.finetune_methods:
            rows, _, summary = train_arm(method, dense_checkpoint, train, indices, selection, confirmation,
                                         args, start=args.finetune_start, stop=args.steps)
            name = "dense_to_" + method
            for row in rows:
                row["method"] = name
            summary["method"] = name
            all_rows.extend(rows)
            summaries.append(summary)
            with (out / "curves.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=all_rows[0].keys())
                writer.writeheader(); writer.writerows(all_rows)
            with (out / "summary.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
                writer.writeheader(); writer.writerows(summaries)
    out.joinpath("config.json").write_text(json.dumps(dict(seeds=args.seeds, methods=args.methods,
        finetune_methods=args.finetune_methods, steps=args.steps, finetune_start=args.finetune_start,
        batch_size=args.batch_size, lr=args.lr, train_stories=args.train_stories,
        validation_stories=args.valid_stories, confirmation_stories=args.confirm_stories,
        train_ids_sha256=hashlib.sha256(train_ids.tobytes()).hexdigest(),
        validation_ids_sha256=hashlib.sha256(selection_ids.tobytes()).hexdigest(),
        confirmation_ids_sha256=hashlib.sha256(confirmation_ids.tobytes()).hexdigest(),
        disjoint_validation_confirmation=not bool(np.intersect1d(selection_ids, confirmation_ids).size),
        train_sequences=len(train), validation_sequences=len(selection), confirmation_sequences=len(confirmation),
        architecture="4 layers, 128 hidden, 4 heads, FFN 512, byte tokenizer, context 256",
        device=str(args.device), note="Same initial tensors and sampled sequence IDs within each seed. Dense-to-QAT uses a fresh AdamW optimizer at finetune_start. Public TinyStories validation split is exposed, not blind."),
        indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 17, 23])
    parser.add_argument("--methods", nargs="+", default=list(SCRATCH))
    parser.add_argument("--finetune-methods", nargs="+", default=["scalar2", "procedural2", "binary", "procedural1"])
    parser.add_argument("--steps", type=int, default=6400)
    parser.add_argument("--finetune-start", type=int, default=4800)
    parser.add_argument("--eval-every", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-stories", type=int, default=20000)
    parser.add_argument("--valid-stories", type=int, default=1000)
    parser.add_argument("--confirm-stories", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=.0003)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("results/e19-21-qtip-training/tinystories-frontier"))
    run(parser.parse_args())
