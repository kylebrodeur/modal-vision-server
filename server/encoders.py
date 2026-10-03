"""Pluggable encoder backends selected by the model card.

The heavy ML deps (torch, open_clip, transformers) are imported INSIDE the
constructors, never at module top: importing this module must cost nothing,
so the service's config surface and tests stay laptop-friendly.

Every backend exposes the same surface: ``model`` / ``preprocess`` /
``tokenizer`` attributes (rebound onto the service for compatibility) plus
``encode_image`` / ``encode_texts`` / ``embed_image_raw`` methods returning
numpy rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from model_card import ModelCard

if TYPE_CHECKING:
    import numpy as np


class Encoder(Protocol):
    """What the service needs from any classifier backend."""

    model: Any
    preprocess: Any
    tokenizer: Any
    device: str
    prompt_template: str

    def encode_image(self, pil) -> np.ndarray:
        """L2-normalized image embedding for one PIL image."""
        ...

    def encode_texts(self, labels: list[str]) -> np.ndarray:
        """L2-normalized embeddings for labels run through the prompt template."""
        ...

    def embed_image_raw(self, pil) -> np.ndarray:
        """Unnormalized image embedding (used for reference centroids)."""
        ...


class OpenClipEncoder:
    """open_clip backend (the original stack: BioCLIP-family checkpoints)."""

    def __init__(self, hf_id: str, device: str, prompt_template: str):
        import open_clip

        self.device = device
        self.prompt_template = prompt_template
        model, _, preprocess = open_clip.create_model_and_transforms(hf_id)
        self.model = model.to(device).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(hf_id)

    def _encode(self, pil):
        import torch

        tensor = self.preprocess(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model.encode_image(tensor)

    def encode_image(self, pil) -> np.ndarray:
        import torch

        with torch.no_grad():
            feats = self._encode(pil)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].cpu().numpy()

    def encode_texts(self, labels: list[str]) -> np.ndarray:
        import torch
        from torch.nn import functional as torch_functional

        prompts = [self.prompt_template.format(label=label) for label in labels]
        tokens = self.tokenizer(prompts).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_text(tokens)
            feats = torch_functional.normalize(feats, dim=-1)
        return feats.cpu().numpy()

    def embed_image_raw(self, pil) -> np.ndarray:
        import torch

        with torch.no_grad():
            feats = self._encode(pil)
        return feats[0].cpu().numpy()


class TransformersEncoder:
    """transformers AutoModel backend.

    Text capability is detected from the model config: CLIP-family configs get
    full text+image encoding; anything else is image-only, which means
    zero-shot classification by prompt is unavailable and labels must be
    supported by reference images instead.
    """

    def __init__(self, hf_id: str, device: str, prompt_template: str):
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        self.device = device
        self.prompt_template = prompt_template
        self.model = AutoModel.from_pretrained(hf_id).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(hf_id)
        architectures = getattr(self.model.config, "architectures", None) or []
        self._text_capable = any("CLIP" in arch for arch in architectures)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id) if self._text_capable else None
        # Attribute-compatible with the open_clip preprocess contract:
        # PIL image -> [C, H, W] tensor.
        self.preprocess = self._preprocess

    def _preprocess(self, pil):
        inputs = self.processor(images=pil, return_tensors="pt")
        return inputs["pixel_values"][0]

    def _encode(self, pil):
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        if hasattr(self.model, "get_image_features"):
            return self.model.get_image_features(**inputs)
        output = self.model(**inputs)
        return getattr(output, "image_embeds", output.pooler_output)

    def encode_image(self, pil) -> np.ndarray:
        import torch

        with torch.no_grad():
            feats = self._encode(pil)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].cpu().numpy()

    def encode_texts(self, labels: list[str]) -> np.ndarray:
        if not self._text_capable or self.tokenizer is None:
            raise NotImplementedError(
                "image-only encoder: classification requires reference images "
                "(references=...) per label"
            )
        import torch

        prompts = [self.prompt_template.format(label=label) for label in labels]
        tokens = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            if hasattr(self.model, "get_text_features"):
                feats = self.model.get_text_features(**tokens)
            else:
                feats = self.model(**tokens).text_embeds
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()

    def embed_image_raw(self, pil) -> np.ndarray:
        import torch

        with torch.no_grad():
            feats = self._encode(pil)
        return feats[0].cpu().numpy()


def encoder_factory(card: ModelCard, fallback_hf_id: str, device: str) -> Encoder:
    """Dispatch on card.backend; hf_id falls back to the service default."""
    hf_id = card.main_model_hf_id or fallback_hf_id
    backend = (card.backend or "open_clip").strip().lower()
    if backend == "open_clip":
        return OpenClipEncoder(hf_id, device, card.prompt_template)
    if backend == "transformers":
        return TransformersEncoder(hf_id, device, card.prompt_template)
    raise ValueError(f"unknown encoder backend {card.backend!r} (expected 'open_clip' or 'transformers')")
