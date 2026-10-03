"""Tests for the pluggable fast gates.

Pure stdlib: no Modal, torch, PIL, or network. The bundled flat-background
gate is exercised with a duck-typed solid-color stand-in for a PIL image so
these tests also run in the laptop dev env (which has no Pillow).
"""

from __future__ import annotations

import urllib.error
from pathlib import Path

import pytest

from fastgate import (
    DeterministicGate,
    ExternalGate,
    GateResult,
    SelfGate,
    make_fast_gate,
)

SERVER_DIR = Path(__file__).resolve().parent.parent
FLAT_GATE_PATH = SERVER_DIR / "plugins" / "fastgates" / "flat_background_gate.py"


class _SolidImage:
    """Duck-typed stand-in for a solid-color PIL image (convert/resize/getdata)."""

    def __init__(self, rgb, size=(100, 100)):
        self._rgb = rgb
        self._size = size

    def convert(self, _mode):
        return self

    def resize(self, size):
        return type(self)(self._rgb, size)

    def getdata(self):
        return [self._rgb] * (self._size[0] * self._size[1])


# --- SelfGate ---------------------------------------------------------------


def test_self_gate_decisive_skips():
    gate = SelfGate(0.55, 0.30)
    result = gate.decide([{"score": 0.99}, {"score": 0.10}], {})
    assert result.decision == "skip"
    assert result.reason == "decisive fast pass"


def test_self_gate_thin_margin_segments():
    gate = SelfGate(0.55, 0.30)
    result = gate.decide([{"score": 0.99}, {"score": 0.95}], {})
    assert result.decision == "segment"
    assert result.reason == "ambiguous fast pass"


def test_self_gate_single_prediction_uses_top1_as_margin():
    gate = SelfGate(0.55, 0.30)
    # margin = top1 - 0 = 0.6 >= 0.30 and 0.6 >= 0.55 -> skip
    assert gate.decide([{"score": 0.6}], {}).decision == "skip"
    # ...but a single weak prediction must still segment.
    assert gate.decide([{"score": 0.5}], {}).decision == "segment"


def test_self_gate_empty_predictions_segment():
    assert SelfGate(0.55, 0.30).decide([], {}).decision == "segment"


# --- DeterministicGate ------------------------------------------------------


def test_deterministic_gate_bundled_flat_gate_skips_solid_image():
    gate = DeterministicGate(str(FLAT_GATE_PATH))
    result = gate.decide(_SolidImage((200, 220, 205)), {})
    assert result.decision == "skip"
    assert "flat" in result.reason


def test_deterministic_gate_bundled_flat_gate_undecided_on_structured_image():
    gate = DeterministicGate(str(FLAT_GATE_PATH))

    class _GradientImage(_SolidImage):
        def getdata(self):
            w, h = self._size  # checkerboard: half the pixels far from the mean
            return [
                (10, 10, 10) if (i % 2 == 0) else (245, 245, 245)
                for i in range(w * h)
            ]

    result = gate.decide(_GradientImage((0, 0, 0)), {})
    assert result.decision == "undecided"


def test_deterministic_gate_syntax_error_fails_open(tmp_path):
    script = tmp_path / "broken_gate.py"
    script.write_text("def decide(:\n    pass\n")
    gate = DeterministicGate(str(script))
    result = gate.decide(_SolidImage((1, 2, 3)), {})
    assert result.decision == "undecided"
    assert result.reason.startswith("gate error:")
    # The load error is cached: a second construction must fail open with the
    # same reason, never silently succeed or raise.
    again = DeterministicGate(str(script))
    assert again.decide(_SolidImage((1, 2, 3)), {}).reason.startswith("gate error:")


def test_deterministic_gate_missing_decide_raises_at_construct(tmp_path):
    script = tmp_path / "no_decide.py"
    script.write_text("X = 1\n")
    with pytest.raises(ValueError, match="decide"):
        DeterministicGate(str(script))


def test_deterministic_gate_runtime_error_fails_open(tmp_path):
    script = tmp_path / "boom_gate.py"
    script.write_text("def decide(image, meta):\n    raise RuntimeError('boom')\n")
    gate = DeterministicGate(str(script))
    result = gate.decide(_SolidImage((1, 2, 3)), {})
    assert result.decision == "undecided"
    assert result.reason.startswith("gate error:")


def test_deterministic_gate_none_and_unknown_decisions_are_undecided(tmp_path):
    none_script = tmp_path / "none_gate.py"
    none_script.write_text("def decide(image, meta):\n    return None\n")
    assert DeterministicGate(str(none_script)).decide(object(), {}).decision == "undecided"

    weird_script = tmp_path / "weird_gate.py"
    weird_script.write_text("def decide(image, meta):\n    return {'decision': 'explode'}\n")
    assert DeterministicGate(str(weird_script)).decide(object(), {}).decision == "undecided"


# --- ExternalGate -----------------------------------------------------------


def test_external_gate_url_error_fails_open(monkeypatch):
    def _raise(*_args, **_kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", _raise)
    result = ExternalGate("http://127.0.0.1:9/decide").decide(
        {"predictions": []}, {"segment_requested": True}
    )
    assert result.decision == "undecided"
    assert result.reason == "gate endpoint unavailable"


def test_external_gate_bad_json_fails_open(monkeypatch):
    class _FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"not json at all"

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: _FakeResponse())
    result = ExternalGate("http://gate.invalid/decide").decide(None, {})
    assert result.decision == "undecided"
    assert result.reason == "gate endpoint unavailable"


def test_external_gate_valid_response_maps_decision(monkeypatch):
    class _FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"decision": "skip", "reason": "endpoint says isolated"}'

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: _FakeResponse())
    result = ExternalGate("http://gate.invalid/decide").decide(None, {})
    assert result.decision == "skip"
    assert result.reason == "endpoint says isolated"


# --- make_fast_gate ---------------------------------------------------------


def test_make_fast_gate_none_returns_none():
    assert make_fast_gate("none", 0.55, 0.30) is None


def test_make_fast_gate_self():
    gate = make_fast_gate("self", 0.6, 0.2)
    assert isinstance(gate, SelfGate)
    assert gate.confidence == pytest.approx(0.6)
    assert gate.margin == pytest.approx(0.2)


def test_make_fast_gate_deterministic_requires_script():
    with pytest.raises(ValueError, match="fast_gate_script"):
        make_fast_gate("deterministic", 0.55, 0.30)


def test_make_fast_gate_external_requires_url():
    with pytest.raises(ValueError, match="fast_gate_url"):
        make_fast_gate("external", 0.55, 0.30)


def test_make_fast_gate_cheap_clip_requires_model():
    with pytest.raises(ValueError, match=r"cheap_clip.*requires a model"):
        make_fast_gate("cheap_clip", 0.55, 0.30, device="cpu")


def test_make_fast_gate_cheap_clip_requires_device():
    with pytest.raises(ValueError, match=r"cheap_clip.*requires a device"):
        make_fast_gate("cheap_clip", 0.55, 0.30, cheap_clip_hf_id="openai/ViT-B-32")



def test_make_fast_gate_cheap_clip_constructs(monkeypatch):
    class _FakeModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

    class _FakeOpenClip:
        @staticmethod
        def create_model_and_transforms(_hf_id):
            return _FakeModel(), None, object()

        @staticmethod
        def get_tokenizer(_hf_id):
            return object()

    class _FakeTorch:
        pass

    class _FakeImage:
        pass

    fake_pil = type("_FakePIL", (), {"Image": _FakeImage})
    monkeypatch.setitem(__import__("sys").modules, "open_clip", _FakeOpenClip)
    monkeypatch.setitem(__import__("sys").modules, "torch", _FakeTorch)
    monkeypatch.setitem(__import__("sys").modules, "PIL", fake_pil)
    monkeypatch.setitem(__import__("sys").modules, "PIL.Image", fake_pil)
    gate = make_fast_gate(
        "cheap_clip", 0.6, 0.2, cheap_clip_hf_id="openai/ViT-B-32", device="cpu"
    )
    assert gate.hf_id == "openai/ViT-B-32"
    assert gate.confidence == pytest.approx(0.6)
    assert gate.margin == pytest.approx(0.2)

def test_make_fast_gate_unknown_kind():
    with pytest.raises(ValueError, match="unknown fast gate kind"):
        make_fast_gate("telepathy", 0.55, 0.30)


def test_gate_result_defaults():
    result = GateResult("skip")
    assert result.reason == ""
    assert result.meta == {}
    # Default meta must not be shared across instances.
    other = GateResult("segment")
    other.meta["k"] = 1
    assert result.meta == {}
