"""Pure SGF domain layer (numpy only, no torch/gsplat)."""

from .sgf import (
    C0,
    Cartridge,
    DeformationCoeffs,
    GaussianField,
    rgb_to_sh_dc,
    sh_dc_to_rgb,
)

__all__ = [
    "C0",
    "Cartridge",
    "DeformationCoeffs",
    "GaussianField",
    "rgb_to_sh_dc",
    "sh_dc_to_rgb",
]
