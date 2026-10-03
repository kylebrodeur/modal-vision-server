"""Pluggable fast gates for the adaptive segmentation decision.

All gates return the same GateResult so the request/response API never
changes. Decisions:

- "skip":     the specimen is isolated enough; skip the SAM segmentation pass
- "segment":  force the segmented path
- "undecided": the gate cannot say; the caller falls back to the card's
  SelfGate threshold semantics

This module is stdlib-only on purpose: gates must construct and run cheaply,
including off-GPU. torch/numpy/httpx/requests are forbidden here.
"""

from __future__ import annotations

import importlib.util
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

_VALID_DECISIONS = ("skip", "segment", "undecided")


@dataclass
class GateResult:
    decision: str  # "skip" | "segment" | "undecided"
    reason: str = ""
    meta: dict = field(default_factory=dict)


class FastGate:
    """Base class: decide(subject, context) -> GateResult."""

    def decide(self, subject: Any, context: dict) -> GateResult:
        raise NotImplementedError


class SelfGate(FastGate):
    """The original behavior: the main model classifies the full image first,
    and the top-1/top-2 confidence geometry decides whether SAM is needed."""

    def __init__(self, confidence: float, margin: float):
        self.confidence = float(confidence)
        self.margin = float(margin)

    def decide(self, subject: Any, context: dict) -> GateResult:
        predictions = list(subject or [])
        if not predictions:
            return GateResult("segment", reason="no predictions to judge")
        top = float(predictions[0].get("score", 0.0))
        second = float(predictions[1].get("score", 0.0)) if len(predictions) > 1 else 0.0
        if top >= self.confidence and (top - second) >= self.margin:
            return GateResult("skip", reason="decisive fast pass")
        return GateResult("segment", reason="ambiguous fast pass")

class CheapClipGate(FastGate):
    """Classify the full image with a smaller CLIP model before deciding SAM."""

    def __init__(self, hf_id: str, device: str, confidence: float, margin: float):
        import open_clip
        import torch
        from PIL.Image import Image

        self.hf_id = hf_id
        self.device = device
        self.confidence = float(confidence)
        self.margin = float(margin)
        self._torch = torch
        self._image_type = Image
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(hf_id)
        self.tokenizer = open_clip.get_tokenizer(hf_id)
        self.model.to(device).eval()

    @staticmethod
    def _decision(scores: list[float], confidence: float, margin: float) -> GateResult:
        if not scores:
            return GateResult("undecided", reason="no cheap_clip scores")
        top = scores[0]
        second = scores[1] if len(scores) > 1 else 0.0
        if top >= confidence and (top - second) >= margin:
            return GateResult("skip", reason="cheap_clip gate decisive")
        return GateResult("segment", reason="cheap_clip gate ambiguous")

    def decide(self, subject: Any, context: dict) -> GateResult:
        if not isinstance(subject, self._image_type):
            return GateResult("undecided", reason="cheap_clip gate requires a PIL image")
        labels = list((context or {}).get("taxa") or [])
        if not labels:
            return GateResult("undecided", reason="cheap_clip gate requires taxa")
        try:
            image = self.preprocess(subject).unsqueeze(0).to(self.device)
            texts = self.tokenizer(labels).to(self.device)
            with self._torch.no_grad():
                image_features = self.model.encode_image(image)
                text_features = self.model.encode_text(texts)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                scores = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]
            ranked = sorted((float(score) for score in scores.tolist()), reverse=True)
            return self._decision(ranked, self.confidence, self.margin)
        except Exception:
            return GateResult("undecided", reason="cheap_clip gate failed")


_GATE_MODULE_CACHE: dict[str, Any] = {}


class DeterministicGate(FastGate):
    """Pure-Python plugin gate: a script exposing decide(image, meta).

    Loading never spawns heavy deps; any failure (missing file, syntax error,
    import error, runtime error, unknown decision) resolves to "undecided" so
    the service fails open. A script that loads but lacks decide() is a
    configuration error and raises ValueError at construction.
    """

    def __init__(self, script_path: str):
        self.script_path = script_path
        entry = _GATE_MODULE_CACHE.get(script_path)
        if entry is None:
            module, error = None, None
            try:
                spec = importlib.util.spec_from_file_location(
                    f"modal_vision_fastgate_{abs(hash(script_path))}", script_path
                )
                if spec is None or spec.loader is None:
                    raise ImportError(f"cannot load gate script {script_path!r}")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except Exception as exc:
                error, module = exc, None
            entry = {"module": module, "error": error}
            _GATE_MODULE_CACHE[script_path] = entry
        self._module = entry["module"]
        self._load_error = entry["error"]
        if self._module is not None and not callable(getattr(self._module, "decide", None)):
            raise ValueError(
                f"fast-gate script {script_path!r} must define a decide(image, meta) function"
            )

    def decide(self, subject: Any, context: dict) -> GateResult:
        if self._module is None:
            return GateResult("undecided", reason=f"gate error: {self._load_error}")
        try:
            result = self._module.decide(subject, dict(context or {}))
        except Exception as exc:
            return GateResult("undecided", reason=f"gate error: {exc}")
        if result is None:
            return GateResult("undecided", reason="gate returned no decision")
        if not isinstance(result, dict):
            return GateResult("undecided", reason="gate returned a non-dict result")
        decision = str(result.get("decision", "")).strip().lower()
        if decision not in _VALID_DECISIONS:
            return GateResult("undecided", reason=f"gate returned unknown decision {decision!r}")
        meta = {k: v for k, v in result.items() if k not in ("decision", "reason")}
        return GateResult(decision, reason=str(result.get("reason", "")), meta=meta)


class ExternalGate(FastGate):
    """POST the decision to any external endpoint; fail open on trouble."""

    def __init__(self, url: str, timeout: float = 5.0):
        self.url = url
        self.timeout = float(timeout)

    def decide(self, subject: Any, context: dict) -> GateResult:
        body: dict[str, Any] = {"context": context or {}}
        if subject is not None:
            body["subject"] = subject
        headers = {"Content-Type": "application/json"}
        token = os.environ.get("MODAL_VISION_FAST_GATE_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if response.status != 200:
                    return GateResult("undecided", reason="gate endpoint unavailable")
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError):
            return GateResult("undecided", reason="gate endpoint unavailable")
        decision = ""
        if isinstance(data, dict):
            decision = str(data.get("decision", "")).strip().lower()
        if decision not in _VALID_DECISIONS:
            return GateResult("undecided", reason="gate endpoint unavailable")
        return GateResult(decision, reason=str(data.get("reason", "")))


def make_fast_gate(
    kind: str,
    confidence: float,
    margin: float,
    script_path: str | None = None,
    url: str | None = None,
    cheap_clip_hf_id: str | None = None,
    device: str | None = None,
) -> FastGate | None:
    kind = (kind or "self").strip().lower()
    if kind == "none":
        return None
    if kind == "self":
        return SelfGate(confidence, margin)
    if kind == "deterministic":
        if not script_path:
            raise ValueError(
                "fast_gate_kind 'deterministic' requires a script: set the model card's "
                "fast_gate_script field or the MODAL_VISION_FAST_GATE_SCRIPT env var"
            )
        return DeterministicGate(script_path)
    if kind == "external":
        if not url:
            raise ValueError(
                "fast_gate_kind 'external' requires a URL: set the model card's "
                "fast_gate_url field or the MODAL_VISION_FAST_GATE_URL env var"
            )
        return ExternalGate(url)
    if kind == "cheap_clip":
        if not cheap_clip_hf_id:
            raise ValueError(
                "fast_gate_kind 'cheap_clip' requires a model: set the model card's "
                "cheap_clip_hf_id field or the MODAL_VISION_CHEAP_CLIP_MODEL env var"
            )
        if not device:
            raise ValueError("fast_gate_kind 'cheap_clip' requires a device")
        return CheapClipGate(cheap_clip_hf_id, device, confidence, margin)
    raise ValueError(f"unknown fast gate kind: {kind!r}")
