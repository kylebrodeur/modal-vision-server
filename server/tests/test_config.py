"""Smoke tests for the Modal vision service application constants.

These tests parse and exercise the module-level config surface of ``app.py``
without importing ``modal`` or ``torch`` (heavy ML deps belong to the image,
not the laptop, per pyproject.toml). Instead of importing app.py, we execute
its prelude (the config-constants block) in an isolated namespace so the
env-overridable defaults stay testable without any heavy dep.
"""

from __future__ import annotations

import os
import sys
import types
import unittest.mock as mock
from pathlib import Path

import pytest

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def _load_prelude(env_overrides: dict[str, str] | None = None):
    """Execute only the config-constant prelude of app.py in an isolated ns."""
    source = APP_PATH.read_text()
    # Truncate the source at the first `@app.cls` decorator so only the plain
    # config constants module preamble executes.
    cut = source.index("@app.cls")
    prelude = source[:cut]
    # Stub the heavy deps so the prelude's imports execute locally.
    modal_stub = types.ModuleType("modal")
    from types import SimpleNamespace

    modal_stub.Volume = SimpleNamespace(from_name=lambda *a, **kw: None)
    class _ChainStub:
        def __getattr__(self, name):
            return _ChainStub()
        def __call__(self, *_args, **_kwargs):
            return _ChainStub()
    class _ChainStub:
        def __getattr__(self, name):
            return _ChainStub()
        def __call__(self, *_args, **_kwargs):
            return _ChainStub()

    modal_stub.Image = _ChainStub()
    modal_stub.App = _ChainStub()
    stubs = {
        "modal": modal_stub,
        "torch": types.ModuleType("torch"),
    }
    sys.modules.setdefault("modal", stubs["modal"])
    sys.modules.setdefault("torch", stubs["torch"])
    ns: dict[str, object] = {}
    with mock.patch.dict(os.environ, env_overrides or {}, clear=False):
        exec(compile(prelude, str(APP_PATH), "exec"), ns)
    return ns


def test_default_app_name():
    ns = _load_prelude()
    assert ns["APP_NAME"] == "modal-vision-server"


def test_default_gpu_is_t4_lowercased():
    ns = _load_prelude()
    assert ns["GPU_TYPE"] == "t4"


def test_cpu_when_gpu_env_empty():
    ns = _load_prelude({"MODAL_VISION_GPU": ""})
    assert ns["GPU_TYPE"] is None


def test_cpu_when_gpu_env_none():
    ns = _load_prelude({"MODAL_VISION_GPU": "none"})
    assert ns["GPU_TYPE"] is None


def test_cpu_when_gpu_env_cpu():
    ns = _load_prelude({"MODAL_VISION_GPU": "cpu"})
    assert ns["GPU_TYPE"] is None


def test_default_model_is_bioclip2_namespace():
    ns = _load_prelude()
    assert ns["MODAL_VISION_MODEL"] == "hf-hub:imageomics/bioclip-2"


def test_default_reference_thresholds():
    ns = _load_prelude()
    assert ns["REFERENCE_SCALE"] == pytest.approx(60.0)
    assert ns["REFERENCE_MAX_LOGITS"] == pytest.approx(6.0)
    assert ns["REFERENCE_NULL"] == pytest.approx(0.80)


def test_adaptive_thresholds_default():
    ns = _load_prelude()
    assert ns["ADAPTIVE_CONFIDENCE_THRESHOLD"] == pytest.approx(0.55)
    assert ns["ADAPTIVE_MARGIN_THRESHOLD"] == pytest.approx(0.30)


def test_overrides_flow_through_env():
    ns = _load_prelude(
        {
            "MODAL_VISION_ADAPTIVE_CONFIDENCE_THRESHOLD": "0.8",
            "MODAL_VISION_REFERENCE_SCALE": "100.5",
        }
    )
    assert ns["ADAPTIVE_CONFIDENCE_THRESHOLD"] == pytest.approx(0.8)
    assert ns["REFERENCE_SCALE"] == pytest.approx(100.5)


def test_default_rate_limit_window():
    ns = _load_prelude()
    assert ns["RATE_LIMIT_MAX"] == 30
    assert ns["RATE_LIMIT_WINDOW"] == 60


def test_sam_repo_and_cache_paths():
    ns = _load_prelude()
    assert ns["SAM_HF_REPO"] == "facebook/sam2.1-hiera-tiny"
    assert ns["CACHE_DIR"] == "/root/cache"
