"""Finite-difference and exact-expectation surrogates for an eight-bit codebook."""

import numpy as np
import torch


def vertex_jacobian(codebook, seeds):
    """J[i,j]=G_i(bit j=1)-G_i(bit j=0), at other bits of seed."""
    book = np.asarray(codebook)
    seeds = np.asarray(seeds, dtype=np.int64).reshape(-1)
    if book.shape[0] != 256:
        raise ValueError("expected an eight-bit codebook")
    columns = []
    for bit in range(8):
        off = seeds & ~(1 << bit)
        columns.append(book[off | (1 << bit)]-book[off])
    return np.stack(columns, axis=-1)


def exact_expectation(codebook, logits):
    """Differentiable mean under independent Bernoulli seed bits."""
    if logits.shape[-1] != 8:
        raise ValueError("expected eight logits")
    states = torch.arange(256, device=logits.device)
    bits = ((states[:, None] >> torch.arange(8, device=logits.device)) & 1).to(logits.dtype)
    log_probability = (bits * torch.nn.functional.logsigmoid(logits[..., None, :]) +
                       (1-bits) * torch.nn.functional.logsigmoid(-logits[..., None, :])).sum(dim=-1)
    return torch.exp(log_probability) @ torch.as_tensor(codebook, dtype=logits.dtype, device=logits.device)
