"""Compression adapters (GaussianField <-> Cartridge).

* :class:`MockCodec` — identity compress/decompress (PRD §7.2). It does NOT
  encode anything; it wraps the field in a Cartridge unchanged. The
  ``estimate_compressed_bytes`` method gives an **honest estimate** of what a
  real codec would achieve, clearly labeled as an estimate (not a real size).
* Real path (PRD §3.3 / §8): Self-Organizing Gaussians (SOG) + LightGaussian
  for <1MB cartridges. Documented here; not implemented in the PoC.
"""

from __future__ import annotations

from ..domain.sgf import Cartridge, GaussianField


class MockCodec:
    """Identity codec. Honest size estimate, but no real encoding.

    Real adapter would replace this with Self-Organizing Gaussians (2D grid
    sort + image codec) plus LightGaussian pruning/distillation. See PRD §8.
    """

    def __init__(self, ratio: float = 20.0) -> None:
        # Typical SOG+LightGaussian compression ratio used only for *estimates*.
        self.ratio = float(ratio)

    def compress(self, name: str, field: GaussianField) -> Cartridge:
        # Identity: no actual compression performed (PoC mock).
        return Cartridge(
            name=name,
            field=field,
            meta={
                "codec": "MockCodec(identity)",
                "raw_bytes": field.nbytes(),
                "estimated_compressed_bytes": self.estimate_compressed_bytes(field),
                "estimate_only": True,
            },
        )

    def decompress(self, cartridge: Cartridge) -> GaussianField:
        return cartridge.field

    def estimate_compressed_bytes(self, field: GaussianField) -> int:
        """Honest *estimate* (not a real encode) of a real codec's output."""
        return int(field.nbytes() / max(self.ratio, 1.0))
