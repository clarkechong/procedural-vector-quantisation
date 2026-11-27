"""Independent SmoothQuant channel scaling, matching upstream clamps."""

import torch


def scales(weights, activation_max, alpha):
    """weights is a sequence of output-by-input matrices sharing one input."""
    weight_max = torch.stack([w.abs().amax(dim=0) for w in weights]).amax(dim=0).clamp(min=1e-5)
    return (activation_max.pow(alpha) / weight_max.pow(1 - alpha)).clamp(min=1e-5)


def transform(weights, activation_max, alpha):
    scale = scales(weights, activation_max, alpha)
    return [w * scale[None] for w in weights], scale
