"""Tests for the model-card configuration layer.

Pure stdlib: no Modal, torch, or network. load_card() reads os.environ, so
every test drives env through monkeypatch.
"""

from __future__ import annotations

import json

import pytest

import model_card
from model_card import ModelCard, load_card

# Every env var load_card consults; cleared per-test so machine state
# (e.g. a developer's MODAL_VISION_REFERENCE_SCALE) cannot leak in.
CARD_ENV_VARS = [
    "MODAL_VISION_MODEL_CARD",
    "MODAL_VISION_MAIN_MODEL",
    "MODAL_VISION_SEGMENTER",
    "MODAL_VISION_FAST_GATE",
    "MODAL_VISION_FAST_GATE_SCRIPT",
    "MODAL_VISION_FAST_GATE_URL",
    "MODAL_VISION_REFERENCE_SCALE",
    "MODAL_VISION_REFERENCE_MAX_LOGITS",
    "MODAL_VISION_REFERENCE_NULL",
    "MODAL_VISION_ADAPTIVE_CONFIDENCE_THRESHOLD",
    "MODAL_VISION_ADAPTIVE_MARGIN_THRESHOLD",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in CARD_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_defaults_reproduce_service_constants():
    card = ModelCard()
    assert card.main_model_hf_id is None
    assert card.backend == "open_clip"
    assert card.prompt_template == "a photo of {label}, a type of plant"
    assert card.similarity_scale == pytest.approx(100.0)
    assert card.reference_scale == pytest.approx(60.0)
    assert card.reference_max_logit_add == pytest.approx(6.0)
    assert card.reference_null_floor == pytest.approx(0.80)
    assert card.segmenter_kind == "sam2"
    assert card.segmenter_hf_repo == "facebook/sam2.1-hiera-tiny"
    assert card.segmenter_ckpt == "sam2.1_hiera_tiny.pt"
    assert card.segmenter_config == "configs/sam2.1/sam2.1_hiera_t.yaml"
    assert card.fast_gate_kind == "self"
    assert card.fast_gate_confidence == pytest.approx(0.55)
    assert card.fast_gate_margin == pytest.approx(0.30)
    assert card.fast_gate_script is None
    assert card.fast_gate_url is None
    assert card.label_defaults is None


def test_load_card_no_env_is_all_defaults():
    assert load_card() == ModelCard()


def test_json_card_overrides_only_the_fields_it_carries(tmp_path, monkeypatch):
    card_file = tmp_path / "card.json"
    card_file.write_text(
        json.dumps(
            {
                "main_model_hf_id": "hf-hub:acme/vision-x",
                "backend": "transformers",
                "fast_gate_kind": "external",
                "fast_gate_url": "http://gate.internal/decide",
                "reference_scale": 42.5,
                "unknown_future_key": "ignored",
            }
        )
    )
    monkeypatch.setenv("MODAL_VISION_MODEL_CARD", str(card_file))
    card = load_card()
    assert card.main_model_hf_id == "hf-hub:acme/vision-x"
    assert card.backend == "transformers"
    assert card.fast_gate_kind == "external"
    assert card.fast_gate_url == "http://gate.internal/decide"
    assert card.reference_scale == pytest.approx(42.5)
    # Untouched fields stay at the defaults.
    assert card.segmenter_kind == "sam2"
    assert card.fast_gate_margin == pytest.approx(0.30)
    assert not hasattr(card, "unknown_future_key")


def test_bad_card_json_raises_value_error(tmp_path, monkeypatch):
    card_file = tmp_path / "card.json"
    card_file.write_text("{not json")
    monkeypatch.setenv("MODAL_VISION_MODEL_CARD", str(card_file))
    with pytest.raises(ValueError, match="MODAL_VISION_MODEL_CARD"):
        load_card()


def test_missing_card_file_raises_value_error(monkeypatch):
    monkeypatch.setenv("MODAL_VISION_MODEL_CARD", "/nonexistent/card.json")
    with pytest.raises(ValueError, match="MODAL_VISION_MODEL_CARD"):
        load_card()


def test_env_overrides_beat_card_and_defaults(tmp_path, monkeypatch):
    card_file = tmp_path / "card.json"
    card_file.write_text(
        json.dumps(
            {
                "main_model_hf_id": "hf-hub:card/model",
                "segmenter_kind": "sam2",
                "fast_gate_kind": "deterministic",
            }
        )
    )
    monkeypatch.setenv("MODAL_VISION_MODEL_CARD", str(card_file))
    monkeypatch.setenv("MODAL_VISION_MAIN_MODEL", "hf-hub:env/model")
    monkeypatch.setenv("MODAL_VISION_SEGMENTER", "none")
    monkeypatch.setenv("MODAL_VISION_FAST_GATE", "external")
    card = load_card()
    assert card.main_model_hf_id == "hf-hub:env/model"
    assert card.segmenter_kind == "none"
    assert card.fast_gate_kind == "external"


def test_env_segmenter_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("MODAL_VISION_SEGMENTER", "NoNe")
    assert load_card().segmenter_kind == "none"


@pytest.mark.parametrize("value", ["junk", "sam3", "", "SAM"])
def test_invalid_segmenter_env_raises(monkeypatch, value):
    monkeypatch.setenv("MODAL_VISION_SEGMENTER", value)
    with pytest.raises(ValueError, match="MODAL_VISION_SEGMENTER"):
        load_card()


@pytest.mark.parametrize("value", ["bogus", "SELFISH"])
def test_invalid_fast_gate_env_raises(monkeypatch, value):
    monkeypatch.setenv("MODAL_VISION_FAST_GATE", value)
    with pytest.raises(ValueError, match="MODAL_VISION_FAST_GATE"):
        load_card()


def test_gate_kind_none_is_legal(monkeypatch):
    monkeypatch.setenv("MODAL_VISION_FAST_GATE", "none")
    assert load_card().fast_gate_kind == "none"


def test_deprecated_env_fallbacks_apply_without_card(monkeypatch):
    monkeypatch.setenv("MODAL_VISION_REFERENCE_SCALE", "65")
    monkeypatch.setenv("MODAL_VISION_REFERENCE_MAX_LOGITS", "7.5")
    monkeypatch.setenv("MODAL_VISION_REFERENCE_NULL", "0.75")
    monkeypatch.setenv("MODAL_VISION_ADAPTIVE_CONFIDENCE_THRESHOLD", "0.66")
    monkeypatch.setenv("MODAL_VISION_ADAPTIVE_MARGIN_THRESHOLD", "0.22")
    card = load_card()
    assert card.reference_scale == pytest.approx(65.0)
    assert card.reference_max_logit_add == pytest.approx(7.5)
    assert card.reference_null_floor == pytest.approx(0.75)
    assert card.fast_gate_confidence == pytest.approx(0.66)
    assert card.fast_gate_margin == pytest.approx(0.22)


def test_deprecated_env_does_not_override_explicit_card_value(tmp_path, monkeypatch):
    """Deprecated fallbacks preserve OLD deployments only: when a card (or the
    modern env) sets the field, the legacy var must not silently win."""
    card_file = tmp_path / "card.json"
    card_file.write_text(json.dumps({"reference_scale": 45.0, "fast_gate_confidence": 0.9}))
    monkeypatch.setenv("MODAL_VISION_MODEL_CARD", str(card_file))
    monkeypatch.setenv("MODAL_VISION_REFERENCE_SCALE", "65")
    monkeypatch.setenv("MODAL_VISION_ADAPTIVE_CONFIDENCE_THRESHOLD", "0.11")
    card = load_card()
    assert card.reference_scale == pytest.approx(45.0)
    assert card.fast_gate_confidence == pytest.approx(0.9)


def test_to_dict_has_no_secrets_and_serializes():
    card = ModelCard()
    rendered = json.loads(json.dumps(card.to_dict()))
    assert rendered["segmenter_kind"] == "sam2"
    assert rendered["fast_gate_kind"] == "self"
    assert all("token" not in key.lower() for key in rendered)


def test_card_is_frozen():
    import dataclasses

    card = ModelCard()
    with pytest.raises(dataclasses.FrozenInstanceError):
        card.backend = "transformers"


def test_prelude_card_defaults(monkeypatch):
    """The card built by app.py's prelude equals the plain defaults."""
    reloaded = load_card({})  # empty env: no card file, no overrides
    assert reloaded == ModelCard()
    assert model_card.CARD_ENV == "MODAL_VISION_MODEL_CARD"
