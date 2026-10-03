"""Tests for the lifecycle hook wiring on the API routes.

The vision service fires the closed tag set from server/app.py at three
points: (request.pre, request.post) around every route and identify.post
after a successful /v1/identify. These tests drive the real FastAPI app
built by VisionService.web (torch/metric touches stubbed) and assert the
fire contract: ordered tags on identify, route-tag coverage on health, and
contained errors -- raising hooks never break the response.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

_SERVER_DIR = str(Path(__file__).resolve().parent.parent)
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

# app.py imports torch at module level (heavy dep, not on the laptop per the
# pyproject comment in test_config.py). Stub it minimally: only the names
# app.py touches at import/definition time (randn/zeros in modal.enter).
try:
    import torch  # noqa: F401  (real torch if present, e.g. CI image)
except ModuleNotFoundError:
    _torch_stub = ModuleType("torch")
    _torch_stub.randn = lambda *a, **kw: None
    _torch_stub.zeros = lambda *a, **kw: None
    _torch_stub.no_grad = lambda: None
    sys.modules["torch"] = _torch_stub

import app  # noqa: E402


def _build_api():
    """Build the real FastAPI app from VisionService.web with a fake self."""
    real_cls = app.VisionService._get_user_cls()
    web_fn = real_cls.web._get_raw_f()

    class _FakeSelf:
        device = "cpu"
        model = None

        def _infer(self, images, segment, candidates, references, adaptive):
            return {"ok": True, "model": "test-model", "predictions": []}

    return web_fn(_FakeSelf())


@pytest.fixture()
def client():
    """TestClient over the real app with a token configured for auth.

    web() reads API_TOKEN at app-build time, so the patch must wrap the
    build, not just the requests.
    """
    from fastapi.testclient import TestClient

    with patch.dict("os.environ", {"API_TOKEN": "test-token"}):
        api = _build_api()
        with TestClient(api) as c:
            yield c


@pytest.fixture(autouse=True)
def _clear_hooks():
    """Isolate hook registrations between tests."""
    app.hooks.clear()
    yield
    app.hooks.clear()


_AUTH = {"authorization": "Bearer test-token"}


class RecordingHooks:
    """Captures (tag, args) in fire order."""

    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []
        app.hooks.register("request.pre", lambda *a: self.calls.append(("request.pre", a)))
        app.hooks.register("request.post", lambda *a: self.calls.append(("request.post", a)))
        app.hooks.register("identify.post", lambda *a: self.calls.append(("identify.post", a)))

    @property
    def tags(self):
        return [tag for tag, _ in self.calls]


def test_identify_fires_tags_in_order(client):
    rh = RecordingHooks()
    resp = client.post("/v1/identify", json={"images": ["x"]}, headers=_AUTH)
    assert resp.status_code == 200
    assert rh.tags == ["request.pre", "identify.post", "request.post"]


def test_identify_post_args_carry_model_and_took_ms(client):
    seen = {}
    app.hooks.register(
        "identify.post",
        lambda model, took_ms: seen.update(model=model, took_ms=took_ms),
    )
    client.post("/v1/identify", json={"images": ["x"]}, headers=_AUTH)
    # model comes from the inference result's "model" field; took_ms is the
    # integer latency the route measured around self._infer.
    assert seen["model"] == "test-model"
    assert isinstance(seen["took_ms"], int)


def test_health_fires_pre_then_post(client):
    rh = RecordingHooks()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert rh.calls == [
        ("request.pre", ("GET", "/health")),
        ("request.post", ("GET", "/health", 200)),
    ]


def test_warm_fires_pre_then_post(client):
    rh = RecordingHooks()
    resp = client.post("/warm", headers=_AUTH)
    # FakeSelf.model is None -> warm's encode_image raises -> 503 path.
    assert resp.status_code == 503
    assert rh.calls == [
        ("request.pre", ("POST", "/warm")),
        ("request.post", ("POST", "/warm", 503)),
    ]


def test_raise_in_request_pre_does_not_break_identify(client):
    def _boom(*_args):
        raise RuntimeError("hook blew up")

    app.hooks.register("request.pre", _boom)
    resp = client.post("/v1/identify", json={"images": ["x"]}, headers=_AUTH)
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert app.hooks.last_errors("request.pre"), "the raise must be recorded"


def test_raise_in_identify_post_does_not_break_identify(client):
    def _boom(*_args):
        raise RuntimeError("hook blew up")

    app.hooks.register("identify.post", _boom)
    resp = client.post("/v1/identify", json={"images": ["x"]}, headers=_AUTH)
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert app.hooks.last_errors("identify.post")


def test_raise_in_request_post_does_not_break_health(client):
    def _boom(*_args):
        raise RuntimeError("hook blew up")

    app.hooks.register("request.post", _boom)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert app.hooks.last_errors("request.post")


def test_closed_tag_set_is_documented_tuple():
    assert app.hooks.tags == ("request.pre", "request.post", "identify.post")
    assert app.hooks.name == "modal-vision-server"


def test_unknown_tag_refuses_registration():
    def _never(_tag):  # pragma: no cover - never invoked
        raise AssertionError

    with pytest.raises(ValueError, match="unknown hook tag"):
        app.hooks.register("boot.pre", _never)
