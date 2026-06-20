"""INT8/INT4 quantization helper for the LGM/diffusion backbones (≤4GB VRAM).

Replaces ``nn.Linear`` modules in-place with bitsandbytes 8-bit / 4-bit linears.
This is the explicit VRAM-budget mechanism referenced by
:class:`atanor_core.generation.lgm.LGMGenerator` (PRD §2.2). It is a no-op (and
silently leaves fp16) when bitsandbytes is unavailable, so quantization is an
optimization, never a hard requirement.

Lazy/optional: only imported on the GPU generation path; never touched by the
CPU/numpy PoC tests.
"""

from __future__ import annotations

from typing import Any


def quantize_linears(module: Any, bits: int = 8) -> Any:
    """Swap nn.Linear -> bitsandbytes {8,4}-bit linears in-place.

    Args:
        module: a torch nn.Module (diffusion UNet or LGM backbone).
        bits: 8 (Linear8bitLt) or 4 (Linear4bit / NF4).

    Returns the same module (mutated). Raises only on a torch/bnb-internal error;
    a missing bitsandbytes import is handled by the caller (falls back to fp16).
    """
    import torch.nn as nn
    import bitsandbytes as bnb

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if bits == 4:
                new = bnb.nn.Linear4bit(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=None,
                    quant_type="nf4",
                )
            else:
                new = bnb.nn.Linear8bitLt(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    has_fp16_weights=False,
                )
            new.weight.data = child.weight.data
            if child.bias is not None:
                new.bias.data = child.bias.data
            setattr(module, name, new)
        else:
            quantize_linears(child, bits=bits)  # recurse
    return module
