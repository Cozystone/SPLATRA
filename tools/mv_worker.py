"""Run the multi-view reconstruction in its own process.

Zero123++ cannot be loaded into the API process. The server keeps Stable Diffusion
resident under ``enable_model_cpu_offload``, and loading a second pipeline
alongside an offloaded one is a known diffusers failure (huggingface/diffusers
#5281): accelerate's hooks leave modules on the meta device, so the load dies with
"Cannot copy out of meta tensor" — and worse, it corrupts the shared pipeline, so
even the fast generator starts failing until the server restarts.

Isolation fixes both halves. This worker owns its own CUDA context, does the
reconstruction, writes a point cloud to disk and exits; a crash here cannot touch
the server's models.

    python tools/mv_worker.py "a pikachu" out/mv_result.npz
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def main(argv):
    if len(argv) < 3:
        print("usage: mv_worker.py <prompt> <out.npz>", file=sys.stderr)
        return 2
    prompt, out_path = argv[1], argv[2]
    # never inherit the server's low-VRAM offload: it is what breaks the load
    os.environ["SPLATRA_LOWVRAM"] = "0"

    import numpy as np

    from atanor_core.generation.multiview import MultiViewGenerator

    field = MultiViewGenerator().generate(prompt)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(
        out_path,
        means=field.means, scales=field.scales, quats=field.quats,
        opacities=field.opacities, sh=field.sh,
        sh_degree=np.int32(field.sh_degree),
    )
    print("OK %d" % field.means.shape[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
