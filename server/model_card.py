"""Model-card configuration for the vision service.

A model card is a small JSON registry describing which encoder backend,
segmenter, and fast gate this deployment uses, plus the calibration geometry
that travels with the model (similarity scale, reference floors). Precedence:

    env override  >  card JSON (MODAL_VISION_MODEL_CARD)  >  dataclass defaults

Deprecated legacy service env vars (MODAL_VISION_REFERENCE_SCALE etc.) apply
only when neither the card nor the modern env sets the field, so existing
deployments keep working unchanged.

With no card and no env, every field below reproduces the service's original
hardcoded constants byte-for-byte.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

CARD_ENV = "MODAL_VISION_MODEL_CARD"

_SEGMENTER_KINDS = ("none", "sam2")
_FAST_GATE_KINDS = ("self", "cheap_clip", "deterministic", "external", "none")


@dataclass(frozen=True)
class ModelCard:
    """Everything a deployment swizzle needs, in one immutable record."""

    # Main classifier. None means the caller falls back to MODAL_VISION_MODEL.
    main_model_hf_id: str | None = None
    backend: str = "open_clip"
    prompt_template: str = "a photo of {label}, a type of plant"
    similarity_scale: float = 100.0

    # Few-shot reference-matching calibration (model-coupled geometry).
    reference_scale: float = 60.0
    reference_max_logit_add: float = 6.0
    reference_null_floor: float = 0.80

    # Segmenter: "none" disables SAM entirely.
    segmenter_kind: str = "sam2"
    segmenter_hf_repo: str = "facebook/sam2.1-hiera-tiny"
    segmenter_ckpt: str = "sam2.1_hiera_tiny.pt"
    segmenter_config: str = "configs/sam2.1/sam2.1_hiera_t.yaml"

    # Fast gate for the adaptive path. "self" = classify full image first.
    fast_gate_kind: str = "self"
    fast_gate_confidence: float = 0.55
    fast_gate_margin: float = 0.30
    fast_gate_script: str | None = None
    fast_gate_url: str | None = None
    cheap_clip_hf_id: str | None = None

    label_defaults: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serializable view for /health introspection. Contains no secrets."""
        return asdict(self)


def load_card(env: Mapping[str, str] | None = None) -> ModelCard:
    """Build the effective ModelCard from card JSON + env + defaults."""
    env = os.environ if env is None else env
    field_names = {f.name for f in fields(ModelCard)}
    values: dict[str, Any] = {}
    explicit: set[str] = set()

    # (a) Card JSON. Only keys matching field names are applied.
    card_path = env.get(CARD_ENV)
    if card_path:
        try:
            with open(card_path) as handle:
                raw = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"{CARD_ENV}: could not load model card JSON from {card_path!r}: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise ValueError(f"{CARD_ENV}: card file {card_path!r} must contain a JSON object")
        for key, value in raw.items():
            if key in field_names:
                values[key] = value
                explicit.add(key)

    # (b) Single-line env overrides; they win over the card.
    def _take(var: str, name: str, cast=lambda v: v):
        value = env.get(var)
        if value is not None:
            values[name] = cast(value)
            explicit.add(name)

    _take("MODAL_VISION_MAIN_MODEL", "main_model_hf_id")
    _take("MODAL_VISION_FAST_GATE_SCRIPT", "fast_gate_script")
    _take("MODAL_VISION_FAST_GATE_URL", "fast_gate_url")
    _take("MODAL_VISION_CHEAP_CLIP_MODEL", "cheap_clip_hf_id")

    segmenter = env.get("MODAL_VISION_SEGMENTER")
    if segmenter is not None:
        kind = segmenter.strip().lower()
        if kind not in _SEGMENTER_KINDS:
            raise ValueError(
                f"MODAL_VISION_SEGMENTER must be one of {_SEGMENTER_KINDS}; got {segmenter!r}"
            )
        values["segmenter_kind"] = kind
        explicit.add("segmenter_kind")

    gate = env.get("MODAL_VISION_FAST_GATE")
    if gate is not None:
        kind = gate.strip().lower()
        if kind not in _FAST_GATE_KINDS:
            raise ValueError(
                f"MODAL_VISION_FAST_GATE must be one of {_FAST_GATE_KINDS}; got {gate!r}"
            )
        values["fast_gate_kind"] = kind
        explicit.add("fast_gate_kind")

    # (c) Deprecated legacy fallbacks: only when the card/env did not set the
    # corresponding field, preserving existing deployments.
    legacy = (
        ("MODAL_VISION_REFERENCE_SCALE", "reference_scale"),
        ("MODAL_VISION_REFERENCE_MAX_LOGITS", "reference_max_logit_add"),
        ("MODAL_VISION_REFERENCE_NULL", "reference_null_floor"),
        ("MODAL_VISION_ADAPTIVE_CONFIDENCE_THRESHOLD", "fast_gate_confidence"),
        ("MODAL_VISION_ADAPTIVE_MARGIN_THRESHOLD", "fast_gate_margin"),
    )
    for var, name in legacy:
        value = env.get(var)
        if name not in explicit and value is not None:
            values[name] = float(value)

    # Kinds are validated wherever they came from, so a bad card fails fast.
    if "segmenter_kind" in values and values["segmenter_kind"] not in _SEGMENTER_KINDS:
        raise ValueError(
            f"model card segmenter_kind must be one of {_SEGMENTER_KINDS}; "
            f"got {values['segmenter_kind']!r}"
        )
    if "fast_gate_kind" in values and values["fast_gate_kind"] not in _FAST_GATE_KINDS:
        raise ValueError(
            f"model card fast_gate_kind must be one of {_FAST_GATE_KINDS}; "
            f"got {values['fast_gate_kind']!r}"
        )

    return ModelCard(**values)
