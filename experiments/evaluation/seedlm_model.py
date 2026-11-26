"""Evaluate reconstructed Llama-family weights on WikiText-2."""
import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
from safetensors.torch import load_file
from tokenizers import Tokenizer
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from experiments.encoding.model_baselines import IndexedShortlist, Shortlist, quantize_matrix
from experiments.manifests.weights import PROJECTIONS, Weights


def token_ids(data, split, tokenizer, bos=None):
    source = data / f"{split}-00000-of-00001.parquet"
    lines = pq.read_table(source, columns=["text"])["text"].to_pylist()
    tokens = tokenizer.encode("\n\n".join(lines), add_special_tokens=False).ids
    return torch.tensor(tokens if bos is None else [bos, *tokens], dtype=torch.long)


def submodules(model, layers):
    for layer in layers:
        for name, group in PROJECTIONS.items():
            yield layer, name, getattr(getattr(model.model.layers[layer], group), name)


def calibration(model, tokens, length, device):
    captured = {}
    handles = []
    layer_count = model.config.num_hidden_layers
    layers = sorted({0, layer_count // 2, layer_count - 1})
    for layer, name, module in submodules(model, layers):
        def record(_, args, key=(layer, name)):
            values = args[0].detach().to(device="cpu", dtype=torch.float32)
            captured.setdefault(key, []).append(values.numpy().reshape(-1, values.shape[-1]))
        handles.append(module.register_forward_pre_hook(record))
    try:
        for chunk in tokens.split(length):
            model.model(input_ids=chunk[None].to(device), use_cache=False)
    finally:
        for hook in handles:
            hook.remove()
    return {key: np.concatenate(inputs) for key, inputs in captured.items()}


def perplexity(model, tokens, context, batch_size, device="cpu"):
    windows = [window for window in tokens.split(context) if len(window) >= 2]
    if not windows:
        raise ValueError("Need at least two evaluation tokens")
    loss_sum = 0.0
    count = 0
    full = len(windows) - int(len(windows[-1]) != context)
    batches = [torch.stack(windows[i:min(i + batch_size, full)]) for i in range(0, full, batch_size)]
    if full < len(windows):
        batches.append(windows[-1][None])
    for batch in batches:
        batch = batch.to(device)
        logits = model(batch, use_cache=False).logits[:, :-1].float()
        targets = batch[:, 1:]
        loss_sum += torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                      targets.reshape(-1), reduction="sum").item()
        count += targets.numel()
    return loss_sum / count, count


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_model(checkpoint, assets, dtype):
    config = LlamaConfig.from_json_file(str(assets / "config.json"))
    if checkpoint.is_dir():
        model, loading = LlamaForCausalLM.from_pretrained(
            str(checkpoint), config=config, dtype=dtype, local_files_only=True,
            use_safetensors=True, output_loading_info=True,
        )
        missing, unexpected = loading["missing_keys"], loading["unexpected_keys"]
    else:
        model = LlamaForCausalLM(config).to(dtype=dtype)
        missing, unexpected = model.load_state_dict(load_file(str(checkpoint)), strict=False)
    allowed_missing = {"lm_head.weight"} if model.config.tie_word_embeddings else set()
    if set(missing) - allowed_missing or unexpected:
        raise ValueError(f"Unexpected checkpoint layout: {missing}, {unexpected}")
    model.tie_weights()
    return model.eval()


@contextmanager
def inference_weights(model, device):
    if device.type == "cpu":
        yield
        return
    from accelerate import cpu_offload
    from accelerate.hooks import remove_hook_from_submodules

    cpu_offload(model, execution_device=device)
    try:
        yield
    finally:
        remove_hook_from_submodules(model)
        model.tie_weights()


def add_evaluation_arguments(parser):
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="CUDA streams weights from host RAM for inference")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--windows", type=int, default=0,
                        help="Score exactly this many complete test windows; zero includes the short tail")
    parser.add_argument("--bos", action="store_true", help="Prepend one checkpoint BOS token to each split")


@torch.inference_mode()
def run(args, extra_encoders=None, extra_metadata=None):
    extra_encoders = extra_encoders or {}
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable to this PyTorch environment")
    assets = args.checkpoint if args.checkpoint.is_dir() else args.data
    checkpoint = Weights(args.checkpoint)
    model = load_model(args.checkpoint, assets, getattr(torch, args.dtype))
    config = model.config
    tokenizer = Tokenizer.from_file(str(assets / "tokenizer.json"))
    bos = config.bos_token_id if args.bos else None
    if args.bos and not isinstance(bos, int):
        raise ValueError("The checkpoint needs a single BOS token ID for --bos")
    tokens = token_ids(args.data, "test", tokenizer, bos)
    if args.tokens:
        tokens = tokens[:args.tokens]
    available_tokens = len(tokens)
    if args.windows:
        required = args.windows * args.context
        if available_tokens < required:
            raise ValueError(f"Need {required:,} tokens for {args.windows} windows, got {available_tokens:,}")
        tokens = tokens[:required]
    activations = {}
    splits = ["test"]
    if args.calibration_tokens:
        validation = token_ids(args.data, "validation", tokenizer, bos)[:args.calibration_tokens]
        with inference_weights(model, device):
            activations = calibration(model, validation, args.context, device)
        splits.append("validation")
    shortlist = Shortlist(args.seeds, args.shortlist)
    indexed = None
    shards = sorted({entry[0] for entry in checkpoint.tensors.values()})
    index = assets / "model.safetensors.index.json"
    if args.checkpoint.is_dir() and index.is_file():
        shards.append(index)
    sources = {path.name: digest(path) for path in [
        *shards, assets / "config.json", assets / "tokenizer.json",
        *(args.data / f"{split}-00000-of-00001.parquet" for split in splits),
    ]}
    (args.out / "model_config.json").write_text(json.dumps(dict(
        context=args.context, calibration_tokens=args.calibration_tokens, inputs=len(tokens),
        available_inputs=available_tokens, windows=args.windows, bos=bos, device=args.device, dtype=args.dtype,
        model_parameters=sum(value.numel() for value in model.parameters()),
        candidates=args.seeds if set(args.methods) & {"paper", "range", "refine"} else None,
        shortlist=args.shortlist,
        indexed_neighborhood=args.neighborhood if "indexed_refine" in args.methods else None, sources=sources,
        torch=torch.__version__, numpy=np.__version__, **(extra_metadata or {})), indent=2) + "\n")
    with (args.out / "perplexity.csv").open("w", newline="") as results, (
            args.out / "activation_error.csv").open("w", newline="") as details:
        writer = csv.writer(results)
        writer.writerow(("method", "projection_weights", "packed_bytes", "projection_bpw", "weight_nmse",
                         "nll", "perplexity", "scored_tokens", "encode_seconds", "eval_seconds"))
        metrics = csv.writer(details)
        metrics.writerow(("method", "layer", "projection", "weight_nmse", "output_nmse",
                          "weight_sse", "weight_energy", "output_sse", "output_energy"))
        projection_size = sum(module.weight.numel() for _, _, module in
                              submodules(model, range(config.num_hidden_layers)))
        for method in args.methods:
            print(f"{method}: preparing model weights", flush=True)
            started = time.perf_counter()
            custom = extra_encoders[method]() if method in extra_encoders else None
            payload_log = None
            payload_writer = None
            if custom is not None:
                payload_log = (args.out / f"{method}_payloads.csv").open("w", newline="")
                payload_writer = csv.writer(payload_log)
                payload_writer.writerow(("layer", "projection", "rows", "columns", "payload_bytes", "sha256"))
            if method == "indexed_refine" and indexed is None:
                indexed = IndexedShortlist(args.neighborhood, args.shortlist)
            encoder = indexed if method == "indexed_refine" else shortlist
            packed_bytes, size, total_error, total_energy = 0, 0, 0., 0.
            if method == "original":
                size = projection_size
                for layer, name, _ in submodules(model, range(config.num_hidden_layers)):
                    key = f"model.layers.{layer}.{PROJECTIONS[name]}.{name}.weight"
                    start, end = checkpoint.tensors[key][2]["data_offsets"]
                    packed_bytes += end - start
            else:
                for layer, name, module in submodules(model, range(config.num_hidden_layers)):
                    original = checkpoint.get(layer, name)
                    if custom is None:
                        restored, payload = quantize_matrix(original, method, encoder)
                    else:
                        restored, payload = custom.encode_matrix(original)
                    # A sharded model may reference memory-mapped source weights.
                    # Replace the parameter to keep the source pristine for each method.
                    module.weight = torch.nn.Parameter(torch.from_numpy(restored).to(module.weight.dtype),
                                                       requires_grad=False)
                    size += original.size
                    packed_bytes += len(payload)
                    if payload_writer is not None:
                        payload_writer.writerow((layer, name, *original.shape, len(payload),
                                                 hashlib.sha256(payload).hexdigest()))
                        payload_log.flush()
                    error = restored - original
                    weight_error = float(np.square(error, dtype=np.float64).sum())
                    weight_energy = float(np.square(original, dtype=np.float64).sum())
                    total_error += weight_error
                    total_energy += weight_energy
                    if (layer, name) in activations:
                        inputs = activations[layer, name]
                        output_error = float(np.square(inputs @ error.T, dtype=np.float64).sum())
                        output_energy = float(np.square(inputs @ original.T, dtype=np.float64).sum())
                        metrics.writerow((method, layer, name, weight_error / weight_energy,
                                          output_error / output_energy, weight_error, weight_energy,
                                          output_error, output_energy))
                    if hasattr(custom, "kernel_seconds"):
                        elapsed = time.perf_counter() - started
                        print(f"{method}: encoded {size:,}/{projection_size:,} weights in {elapsed:.1f}s "
                              f"({size / max(elapsed, 1e-9):,.1f} weights/s)", flush=True)
                    if name == "o_proj" and (layer + 1) % 10 == 0:
                        print(f"{method}: encoded {layer + 1}/{config.num_hidden_layers} layers",
                              flush=True)
                details.flush()
            if payload_log is not None:
                payload_log.close()
            if method == "synthesized_3p5" and (size != 106168320 or packed_bytes != 46448640):
                raise AssertionError(f"Unexpected 3.5-bpw payload size: {size} weights, {packed_bytes} bytes")
            if custom is not None:
                if hasattr(custom, "kernel_seconds"):
                    timing = dict(setup_seconds=custom.setup_seconds,
                                  jit_seconds=custom.jit_seconds,
                                  transfer_seconds=custom.transfer_seconds,
                                  kernel_seconds=custom.kernel_seconds,
                                  blocks_encoded=custom.blocks_encoded,
                                  seed_bits=custom.seed_bits,
                                  seed_tile=custom.seed_tile,
                                  target_tile=custom.target_tile,
                                  score_backend=custom.score_backend,
                                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
                    (args.out / f"{method}_encoder_timing.json").write_text(
                        json.dumps(timing, indent=2) + "\n")
                del custom
            encode_seconds = time.perf_counter() - started
            print(f"{method}: evaluation on {len(tokens):,} input tokens after {encode_seconds:.1f}s encoding",
                  flush=True)
            started = time.perf_counter()
            with inference_weights(model, device):
                nll, count = perplexity(model, tokens, args.context, args.batch, device)
            writer.writerow((method, size, packed_bytes, 8 * packed_bytes / size,
                             total_error / total_energy if total_energy else 0, nll, np.exp(nll), count,
                             encode_seconds, time.perf_counter() - started))
            results.flush()
            details.flush()
            print(f"{method}: PPL {np.exp(nll):.5f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=255)
    parser.add_argument("--shortlist", type=int, default=8)
    parser.add_argument("--neighborhood", type=int, default=192)
    parser.add_argument("--tokens", type=int, default=0, help="Zero uses the entire test split")
    parser.add_argument("--calibration-tokens", type=int, default=512)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch", type=int, default=2)
    add_evaluation_arguments(parser)
    parser.add_argument("--methods", nargs="+", choices=("original", "int4_group64", "paper", "range", "refine",
                                                        "indexed_refine"),
                        default=("original", "int4_group64", "paper", "refine"))
    args = parser.parse_args()
    if not (2 <= args.context and 0 < args.shortlist <= args.seeds < 65536
            and args.shortlist <= args.neighborhood <= 65535
            and args.tokens != 1 and args.tokens >= 0 and args.calibration_tokens >= 0
            and args.windows >= 0 and args.threads > 0 and args.batch > 0):
        parser.error("Invalid token or seed counts")
    if "original" in args.methods[1:]:
        parser.error("Pass original first so the model has the original weights")
    run(args)


if __name__ == "__main__":
    main()
