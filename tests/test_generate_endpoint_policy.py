from __future__ import annotations

import time

from apps import plugin_api
from atanor_core.generation.text_to_3d import _default_model


class _NoStartThread:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def start(self) -> None:
        return None


def test_gpu_quality_prompt_shape_word_uses_real_generator_not_primitive(monkeypatch) -> None:
    monkeypatch.setattr(plugin_api, "_USE_TRIPOSR", False)
    monkeypatch.setattr(plugin_api, "_USE_MV", False)
    monkeypatch.setattr(plugin_api, "_USE_SD", True)
    monkeypatch.setattr(plugin_api.threading, "Thread", _NoStartThread)

    payload = plugin_api.generate_3d_object(
        plugin_api.GenerateRequest(prompt="a blue orb", quality="gpu"),
    )

    assert payload["status"] == "generating"
    assert payload["cache"] == "real_generator_pending"
    assert payload["shape"] == "real_generator:gpu"


def test_explicit_shape_still_requests_procedural_primitive(monkeypatch) -> None:
    monkeypatch.setattr(plugin_api, "_USE_TRIPOSR", False)
    monkeypatch.setattr(plugin_api, "_USE_MV", False)
    monkeypatch.setattr(plugin_api, "_USE_SD", True)
    calls: list[dict] = []

    def fake_generate(name, mv_images, cam_rays=None):
        calls.append({"name": name, "cam_rays": cam_rays})
        return "hit"

    monkeypatch.setattr(plugin_api._engine, "generate_3d_object", fake_generate)

    payload = plugin_api.generate_3d_object(
        plugin_api.GenerateRequest(prompt="a blue orb", quality="gpu", shape="sphere"),
    )

    assert payload["status"] == "displayed"
    assert payload["shape"] == "sphere"
    assert calls and calls[0]["cam_rays"] == {"shape": "sphere"}


def test_cuda_default_model_prefers_interactive_sd_turbo(monkeypatch) -> None:
    monkeypatch.delenv("SPLATRA_SD_MODEL", raising=False)
    monkeypatch.delenv("SPLATRA_SD_USE_SDXL", raising=False)

    assert _default_model("cuda") == "stabilityai/sd-turbo"


def test_real_generator_job_times_out_without_hot_swap(monkeypatch) -> None:
    job_id = "timeout-job"
    monkeypatch.setattr(plugin_api, "_REAL_JOB_MAX_SECONDS", 0.001)
    plugin_api._jobs[job_id] = {
        "name": "slow-object",
        "done": False,
        "cache": "real_generator_pending",
        "shape": "real_generator:gpu",
        "phase": "generating",
        "created_at": time.time() - 10.0,
        "timeout_seconds": 0.001,
        "error": None,
    }

    payload = plugin_api.get_job(job_id)

    assert payload["done"] is True
    assert payload["cache"] == "real_generator_timeout"
    assert payload["phase"] == "timeout"
    assert "exceeded" in payload["error"]
    assert plugin_api._jobs[job_id]["verified"] is False
    assert plugin_api._jobs[job_id]["hot_swap"] is False
