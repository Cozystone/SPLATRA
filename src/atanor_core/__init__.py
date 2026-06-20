"""atanor-hologram-core: 3D Gaussian hologram + generation engine.

Zero-config CPU PoC entrypoint: :func:`build_default_engine` wires the mock /
pure-numpy adapters into the unified state machine. No torch, no gsplat, no
model downloads required.
"""

from __future__ import annotations

__version__ = "0.1.0"


def build_default_engine(coalesce_turbulence: float = 0.35):
    """Construct a CPU/numpy-only :class:`HologramEngine` (PoC defaults).

    Adapters (all PoC / mock — see PRD §7.2):
        mapper      -> GraphMapper (PCA fallback, no UMAP needed)
        generator   -> MockGenerator (average-color blob; explicitly a mock)
        rasterizer  -> CPURasterizer (painter splat; coverage, not quality)
        compressor  -> MockCodec (identity + honest size estimate)
        verifier    -> PSNRGate (real PSNR over held-out views)
    """
    from .compression.codec import MockCodec
    from .generation.generator import MockGenerator
    from .mapping.graph_mapper import GraphMapper
    from .state.machine import HologramEngine
    from .state.rasterizer import CPURasterizer
    from .verification.psnr_gate import PSNRGate

    rasterizer = CPURasterizer()
    return HologramEngine(
        mapper=GraphMapper(use_umap=False),
        generator=MockGenerator(),
        rasterizer=rasterizer,
        compressor=MockCodec(),
        verifier=PSNRGate(rasterizer),
        coalesce_turbulence=coalesce_turbulence,
    )


__all__ = ["__version__", "build_default_engine"]
