"""
Modal Vision Server: generic vision inference on Modal.

Pick your main model (open_clip or transformers backend), your segmenter
(SAM 2 or none), and your fast gate (self / cheap_clip / deterministic /
external / none) via a model card (registry/*.json + MODAL_VISION_* envs).
Few-shot reference matching, scale-to-zero, auth via Modal Secret. The
shipped example card is a BioCLIP plant stack (plants + SAM 2 + few-shot).

Deploy:  modal deploy server/app.py
Smoke:   modal run   server/app.py     (drop .jpg/.png in examples/fixtures/)
Read:    registry/README.md for the card reference; README.md for the three
         choice axes and the API contract.
"""

import base64
import contextlib
import hashlib
import io
import os
import time
from collections import defaultdict

import modal
import torch

import model_card
import vm_metrics
from libs.hooks import Hooks

APP_NAME = "modal-vision-server"
SERVICE_VERSION = "1.2.0"
_gpu_env = os.environ.get("MODAL_VISION_GPU", "T4").strip().lower()
GPU_TYPE = None if _gpu_env in ("", "none", "cpu") else _gpu_env  # CPU mode when unset-able
VOLUME_NAME = "modal-vision-weights"
AUTH_SECRET_NAME = "modal-vision-secret"
CACHE_DIR = "/root/cache"

# Vision Model (ViT-L/14, TreeOfLife-200M): +18.1% species classification over
# Vision Model 1, and emergent intra-species variation separation -- the property
# that discriminates within a confusable morphological cluster.
MODAL_VISION_MODEL = os.environ.get("MODAL_VISION_MODEL", "hf-hub:imageomics/bioclip-2")
# Few-shot reference-matching calibration. Text logits use the vision model's 100x
# cosine scale; observed intra-cluster logit gaps between confusables are
# O(1-3). The reference term is a raw-cosine margin scaled by REFERENCE_SCALE
# and hard-capped at REFERENCE_MAX_LOGITS, so it can reorder confusables but
# never steamroll decisive text evidence. Calibrated live on the sanderiana
# cluster (reference cosine 0.89 vs text-vs-reference logit gap ~1.5).
REFERENCE_SCALE = float(os.environ.get("MODAL_VISION_REFERENCE_SCALE", "60"))
REFERENCE_MAX_LOGITS = float(os.environ.get("MODAL_VISION_REFERENCE_MAX_LOGITS", "6"))
# Cosine floor for the single-taxon reference case (no cross-taxon baseline
# available): measured same-plant photo pairs sit well above this.
REFERENCE_NULL = float(os.environ.get("MODAL_VISION_REFERENCE_NULL", "0.80"))

# Adaptive segmentation: a fast full-image classification decides whether the
# specimen is isolated enough to skip SAM. If the top prediction is decisive,
# the expensive segmentation pass is avoided. If the top-two are close, SAM is
# used as a tie-breaker. Thresholds are intentionally conservative because a
# wrong fast answer is worse than a slow correct one.
ADAPTIVE_SEGMENT = os.environ.get("MODAL_VISION_ADAPTIVE_SEGMENT", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ADAPTIVE_CONFIDENCE_THRESHOLD = float(os.environ.get("MODAL_VISION_ADAPTIVE_CONFIDENCE_THRESHOLD", "0.55"))
ADAPTIVE_MARGIN_THRESHOLD = float(os.environ.get("MODAL_VISION_ADAPTIVE_MARGIN_THRESHOLD", "0.30"))

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "MODAL_VISION_ALLOWED_ORIGINS",
        "http://localhost:8080,http://127.0.0.1:8080",
    ).split(",")
    if o.strip()
]
RATE_LIMIT_MAX = int(os.environ.get("MODAL_VISION_RATE_LIMIT_MAX", "30"))
RATE_LIMIT_WINDOW = int(os.environ.get("MODAL_VISION_RATE_LIMIT_WINDOW", "60"))

SAM_HF_REPO = "facebook/sam2.1-hiera-tiny"
SAM_CKPT_FILE = "sam2.1_hiera_tiny.pt"
SAM_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

# Model card: JSON registry + env overrides layered over the constants above.
# With no card and no env, every card field equals the hardcoded defaults, so
# legacy deployments behave byte-identically.
_CARD = model_card.load_card()

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "git")
    .pip_install(
        "torch>=2.5.1",
        "torchvision>=0.20.0",
        "open_clip_torch>=2.26.0",
        "fastapi[standard]>=0.115.0",
        "pillow>=10.4.0",
        "huggingface_hub>=0.25.0",
        "hf_transfer>=0.1.8",
        "numpy>=1.26.0",
        "opencv-python-headless>=4.10.0",
        "git+https://github.com/facebookresearch/sam2.git",
    )
    .env({"HF_HOME": CACHE_DIR, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App(APP_NAME, image=image)
weights_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

DEFAULT_TAXA = [
    "Alocasia zebrina",
    "Alocasia micholitziana",
    "Alocasia longiloba",
    "Alocasia sanderiana",
    "Alocasia baginda",
    "Alocasia reginula",
    "Monstera deliciosa",
    "Monstera adansonii",
    "Monstera dubia",
    "Philodendron hederaceum",
    "Philodendron billietiae",
    "Philodendron verrucosum",
    "Epipremnum aureum",
    "Scindapsus pictus",
    "Ficus elastica",
    "Ficus lyrata",
]

# Lifecycle hooks (family extension seam; vendored in server/libs/hooks.py).
# Closed tag set for this package: request.pre / request.post fire on every
# API route (method, path / method, path, status); identify.post fires after
# a successful /v1/identify (model, took_ms). New tags ship in a release.
hooks = Hooks(
    ("request.pre", "request.post", "identify.post"),
    name="modal-vision-server",
)


@app.cls(
    gpu=GPU_TYPE,
    # Keep scale-to-zero by default. Set MODAL_VISION_MIN_CONTAINERS=1 only when
    min_containers=int(os.environ.get("MODAL_VISION_MIN_CONTAINERS", "0")),
    timeout=int(os.environ.get("MODAL_VISION_TIMEOUT", "120")),
    secrets=[modal.Secret.from_name(AUTH_SECRET_NAME)],
)
class VisionService:
    @modal.enter()
    def initialize(self):
        import torch

        import encoders
        import fastgate

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder = encoders.encoder_factory(_CARD, fallback_hf_id=MODAL_VISION_MODEL, device=self.device)
        self.encoder = encoder
        self.model = encoder.model
        self.preprocess = encoder.preprocess
        self.tokenizer = encoder.tokenizer

        # Calibration geometry travels with the model card; the card's field
        # defaults are the legacy module constants, so this is a no-op without
        # a card and honors explicit values (including zeros) with one.
        self._similarity_scale = _CARD.similarity_scale
        self._reference_scale = _CARD.reference_scale
        self._reference_max_logit_add = _CARD.reference_max_logit_add
        self._reference_null_floor = _CARD.reference_null_floor

        if _CARD.segmenter_kind == "none":
            self.sam_model = None
            self.mask_generator = None
        else:
            from huggingface_hub import hf_hub_download
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2

            ckpt = hf_hub_download(_CARD.segmenter_hf_repo, _CARD.segmenter_ckpt, cache_dir=CACHE_DIR)
            sam2 = build_sam2(_CARD.segmenter_config, ckpt, device=self.device, apply_postprocessing=False)
            self.sam_model = sam2
            self.mask_generator = SAM2AutomaticMaskGenerator(
                sam2, points_per_side=16, pred_iou_thresh=0.8, stability_score_thresh=0.9
            )

        self._fast_gate = fastgate.make_fast_gate(
            kind=_CARD.fast_gate_kind,
            confidence=_CARD.fast_gate_confidence,
            margin=_CARD.fast_gate_margin,
            script_path=_CARD.fast_gate_script,
            url=_CARD.fast_gate_url,
            cheap_clip_hf_id=_CARD.cheap_clip_hf_id,
            device=self.device,
        )
        # Threshold fallback for gates that answer "undecided"; card defaults
        # are exactly the historical adaptive thresholds.
        self._fallback_gate = fastgate.SelfGate(
            _CARD.fast_gate_confidence,
            _CARD.fast_gate_margin,
        )

        # Warm the default taxa space once at boot so requests
        # without candidates/references (the cache path) always have it.
        if _CARD.label_defaults:
            self.cached_taxa, self.cached_text_features = self._taxa_embeddings(list(_CARD.label_defaults))
        else:
            self.cached_taxa, self.cached_text_features = None, None
        # Reference centroids and text-embedding sets memoized for the
        # container's lifetime: they rarely change and re-encoding them per
        # request would tax the latency budget.
        self._reference_cache: dict = {}
        self._text_cache: dict = {}

        # Opt-in metrics drop-in: MODAL_VISION_METRICS=1 enables emission via
        # the vendored vm_metrics module. Absent = zero overhead, no code
        # path changes on the identify route.
        self._metrics_on: bool = os.environ.get("MODAL_VISION_METRICS", "").strip() == "1"

        dummy = torch.randn(1, 3, 224, 224, device=self.device)
        with torch.no_grad():
            self.model.encode_image(dummy)
        weights_volume.commit()

    def _taxa_embeddings(self, taxa_list: list[str]):
        """Return (taxa, L2-normalized text features) for one request.

        Keyed by the exact taxon tuple: a request that supplies its own
        candidate or reference taxa can never corrupt another request's
        classification space (the previous design mutated shared state).
        """
        import numpy as np
        import torch

        key = tuple(taxa_list)
        cached = self._text_cache.get(key)
        if cached is not None:
            return cached[0], cached[1]
        # The prompt template lives in the encoder (owned by the model card).
        text_embeds = self.encoder.encode_texts(list(taxa_list))
        feats = torch.from_numpy(np.asarray(text_embeds, dtype=np.float32)).to(self.device)
        entry = (list(taxa_list), feats)
        self._text_cache[key] = entry
        return entry

    def _segment(self, img):
        """Return (specimen_on_black PIL.Image, produced_mask bool).

        Live failure this guards against: a carpet/floor fills the frame, so
        an area-dominated mask score picks the *background* as the specimen
        and the vision model then classifies floor texture. Selection therefore has a
        hard sanity gate -- the mask must be a plausible single object
        (not near-full-frame, not a sliver) -- and, among the survivors,
        prefers the mask whose crop best matches its own bounding box.
        """
        import numpy as np
        from PIL import Image as _Image

        if self.mask_generator is None:
            raise ValueError("segmentation disabled (segmenter=none)")

        arr = np.array(img)
        masks = self.mask_generator.generate(arr)
        if not masks:
            return img, False
        h, w = arr.shape[:2]
        frame = float(h * w)
        cx, cy = w / 2.0, h / 2.0

        def candidate_masks():
            for mask in masks:
                area = float(mask["area"])
                frac = area / frame
                if frac < 0.02 or frac > 0.9:
                    continue  # sliver or near-full-frame (background)
                x, y, bw, bh = mask["bbox"]
                if bw < 32 or bh < 32:
                    continue
                yield mask, area, frac, x, y, bw, bh

        def score(entry):
            _mask, area, frac, x, y, bw, bh = entry
            # Tightness: how much of the bbox the mask fills. A background
            # region scores high on area but low on tightness relative to a
            # coherent leaf; bbox fill directly measures that.
            tightness = area / float(max(bw * bh, 1))
            mcx, mcy = x + bw / 2.0, y + bh / 2.0
            dist = ((mcx - cx) ** 2 + (mcy - cy) ** 2) ** 0.5
            centrality = 1.0 / (1.0 + dist / max(w, h))
            # Prefer leaf-scale objects: not tiny, not the whole frame.
            scale = 1.0 - abs(frac - 0.25)
            return tightness * 2.0 + centrality + scale

        entries = list(candidate_masks())
        if not entries:
            return img, False
        best = max(entries, key=score)[0]["segmentation"]
        out = arr.copy()
        out[~best] = 0
        return _Image.fromarray(out), True

    def _load_image(self, data_url: str):
        from PIL import Image

        raw = data_url.split(",")[-1]
        return Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")

    def _encode_image(self, img):
        import torch

        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0]

    def _classify(self, img, taxa, text_features, references: dict | None = None):
        """Run vision classification on a prepared PIL image.

        Returns the standard inference result dict plus the raw top-k scores
        so an adaptive caller can decide whether segmentation was necessary.
        """
        import torch

        feats = self._encode_image(img).unsqueeze(0)
        with torch.no_grad():
            text_logits = (self._similarity_scale * feats @ text_features.T).to(torch.float32)

        reference_scores: dict[str, float] = {}
        if references:
            centroids = []
            query = feats[0]
            for taxon in taxa:
                centroid = self._reference_centroid(references.get(taxon) or [])
                if centroid is None:
                    centroids.append(None)
                    continue
                score = float(torch.dot(query, centroid).item())
                reference_scores[taxon] = round(score, 4)
                centroids.append(score)
            if any(c is not None for c in centroids):
                present = [c for c in centroids if c is not None]
                baseline = sum(present) / len(present)
                for index, value in enumerate(centroids):
                    if value is None:
                        continue
                    margin = max(value - self._reference_null_floor, 0.0) if len(present) == 1 else value - baseline
                    add = max(
                        -self._reference_max_logit_add,
                        min(self._reference_max_logit_add, self._reference_scale * margin),
                    )
                    text_logits[0, index] += add

        sim = text_logits.softmax(dim=-1)[0]
        k = min(5, len(taxa))
        scores, idx = sim.topk(k)
        predictions = [
            {
                "name": taxa[i],
                "score": round(s, 4),
                "common_names": [],
                "reference_score": reference_scores.get(taxa[i]),
            }
            for s, i in zip(scores.tolist(), idx.tolist(), strict=False)
        ]
        return predictions, reference_scores

    def _metric(self, name: str, value: float = 1.0, tags: dict[str, str] | None = None) -> None:
        """Emit one metrics point when MODAL_VISION_METRICS=1; no-op otherwise.

        Belt+suspenders around vm_metrics (which already never raises): a
        metrics hiccup must never touch /v1/identify latency or correctness.
        """
        if not getattr(self, "_metrics_on", False):
            return
        merged = {"device": os.environ.get("MODAL_VISION_DEVICE_TAG", APP_NAME)}
        if tags:
            merged.update(tags)
        with contextlib.suppress(Exception):
            vm_metrics.write_metric(name, value, tags=merged)

    def _infer(
        self,
        images: list[str],
        segment: bool,
        candidates: list[str] | None,
        references: dict | None = None,
        adaptive: bool = False,
    ):
        """Classify the specimen; optional few-shot references per taxon.

        When `adaptive` is true, a fast full-image classification is performed
        first. If the top prediction is decisive (confidence and margin above
        the configured thresholds), that result is returned without running
        the expensive SAM segmentation. Otherwise the request falls back to the
        segmented path. This is intentionally conservative: a wrong fast
        answer is worse than a slow correct one.

        Reference matching (rung 4): each taxon's reference photos are encoded
        once (memoized on the volume by content hash) and averaged into a
        single L2-normalized centroid. The centroid's cosine similarity adds a
        bounded number of logits to that taxon's text logit, so text evidence
        still dominates unless a reference genuinely matches better -- the
        knowledge-inversion fix for clusters where morphology beats text.
        """

        _t0 = time.perf_counter()
        self._metric(
            "vision_request",
            tags={"segment_requested": str(segment).lower(), "gate": _CARD.fast_gate_kind},
        )

        # Taxon space: an explicit candidate list wins; otherwise the curated
        # defaults are unioned with any taxa that have reference photos, so a
        # reference pass never shrinks the classification space.
        if candidates:
            taxa = list(candidates)
        elif references:
            taxa = list(DEFAULT_TAXA)
            for name in references:
                if name not in taxa:
                    taxa.append(name)
        else:
            taxa = None
        if taxa:
            taxa, text_features = self._taxa_embeddings(taxa)
        else:
            taxa, text_features = self.cached_taxa, self.cached_text_features

        img = self._load_image(images[0])

        # Adaptive fast path: a pluggable gate decides whether the expensive
        # SAM segmentation pass is needed. The default "self" gate reproduces
        # the historical behavior exactly: classify the full image, skip SAM
        # when the top prediction is decisive under the card's thresholds.
        # "full" = no gate ran (adaptive off or gate disabled): full pipeline.
        gate_decision = "full"
        if adaptive and self._fast_gate is not None:
            ctx = {"segment_requested": segment, "taxa": taxa}

            def fast_result(predictions, reference_scores):
                return {
                    "version": SERVICE_VERSION,
                    "model": MODAL_VISION_MODEL,
                    "predictions": predictions,
                    "reference_scores": reference_scores,
                    "specimen_crop": None,
                    "segmented": False,
                    "adaptive_skip": True,
                }

            if _CARD.fast_gate_kind == "deterministic":
                gate_result = self._fast_gate.decide(img, ctx)
                gate_decision = gate_result.decision
                if gate_result.decision == "skip":
                    predictions, reference_scores = self._classify(img, taxa, text_features, references)
                    self._metric(
                        "vision_gate_decision",
                        tags={"decision": gate_decision, "kind": _CARD.fast_gate_kind},
                    )
                    self._metric("vision_identify_seconds", time.perf_counter() - _t0, tags={"segmented": "false"})
                    return fast_result(predictions, reference_scores)
                if gate_result.decision == "segment":
                    segment = True
            else:
                if _CARD.fast_gate_kind == "cheap_clip":
                    subject = img
                else:
                    fast_predictions, fast_reference_scores = self._classify(img, taxa, text_features, references)
                    subject = (
                        fast_predictions
                        if _CARD.fast_gate_kind == "self"
                        else {
                            "predictions": fast_predictions,
                            "reference_scores": fast_reference_scores,
                        }
                    )
                gate_result = self._fast_gate.decide(subject, ctx)
                decision = gate_result.decision
                if decision == "undecided" and _CARD.fast_gate_kind != "self":
                    # Fail soft to the card's threshold semantics.
                    fallback_predictions, fallback_reference_scores = self._classify(
                        img, taxa, text_features, references
                    )
                    decision = self._fallback_gate.decide(fallback_predictions, ctx).decision
                    if _CARD.fast_gate_kind == "cheap_clip":
                        fast_predictions = fallback_predictions
                        fast_reference_scores = fallback_reference_scores
                gate_decision = decision
                if decision == "skip":
                    if _CARD.fast_gate_kind == "cheap_clip":
                        fast_predictions, fast_reference_scores = self._classify(img, taxa, text_features, references)
                    self._metric(
                        "vision_gate_decision",
                        tags={"decision": gate_decision, "kind": _CARD.fast_gate_kind},
                    )
                    self._metric("vision_identify_seconds", time.perf_counter() - _t0, tags={"segmented": "false"})
                    return fast_result(fast_predictions, fast_reference_scores)
                if decision == "segment" and _CARD.fast_gate_kind != "self":
                    # Non-self gates may force segmentation; the self gate
                    # keeps the historical "respect the request flag" flow.
                    segment = True
        self._metric(
            "vision_gate_decision",
            tags={"decision": gate_decision, "kind": _CARD.fast_gate_kind},
        )
        segmented = False
        specimen_crop = None
        if segment:
            try:
                seg_img, segmented = self._segment(img)
                if segmented:
                    img = seg_img
                    buf = io.BytesIO()
                    seg_img.save(buf, format="PNG")
                    specimen_crop = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            except Exception:
                segmented = False

        predictions, reference_scores = self._classify(img, taxa, text_features, references)
        self._metric(
            "vision_identify_seconds",
            time.perf_counter() - _t0,
            tags={"segmented": str(segmented).lower()},
        )
        return {
            "version": SERVICE_VERSION,
            "model": MODAL_VISION_MODEL,
            "predictions": predictions,
            "reference_scores": reference_scores,
            "specimen_crop": specimen_crop,
            "segmented": segmented,
            "adaptive_skip": False,
        }

    def _reference_embedding(self, urls: list[str], taxa: list[str]):
        """Mean L2-normalized embedding for a taxon's reference photos."""
        if not urls:
            return None
        key = hashlib.sha256("|".join(sorted(urls)).encode()).hexdigest()[:32]
        cached = self._reference_cache.get(key)
        if cached is not None:
            return cached
        vectors = []
        for url in urls[:6]:
            try:
                vectors.append(self._encode_image(self._load_image(url)))
            except Exception:
                continue
        if not vectors:
            return None
        centroid = torch.stack(vectors).mean(dim=0)
        centroid = centroid / centroid.norm()
        self._reference_cache[key] = centroid
        return centroid

    def _reference_centroid(self, urls: list[str]) -> "object | None":
        """Alias of _reference_embedding (name clarity: it returns a normalized centroid)."""
        return self._reference_embedding(urls, taxa=[])

    @modal.method()
    def infer(
        self,
        images: list[str],
        segment: bool = True,
        candidates: list[str] | None = None,
        references: dict | None = None,
        adaptive: bool = False,
    ):
        return self._infer(images, segment, candidates, references, adaptive)

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, Header, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel

        class VisionRequest(BaseModel):
            images: list[str]
            candidates: list[str] | None = None
            references: dict | None = None
            segment: bool = True
            adaptive: bool = False

        api = FastAPI(title="Modal Vision Server", version="1.0.0")
        api.add_middleware(
            CORSMiddleware,
            allow_origins=ALLOWED_ORIGINS,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["content-type", "authorization"],
            max_age=86400,
        )
        api_token = os.environ.get("API_TOKEN", "")
        rate: dict[str, list[float]] = defaultdict(list)

        def authorize(authorization: str, origin: str) -> JSONResponse | None:
            if not api_token:
                return JSONResponse(status_code=503, content={"error": "Server auth not configured"})
            if authorization != f"Bearer {api_token}":
                return JSONResponse(status_code=401, content={"error": "Invalid or missing bearer token"})
            if origin and origin not in ALLOWED_ORIGINS:
                return JSONResponse(status_code=403, content={"error": "Origin not allowed"})
            return None

        def admit(ip: str) -> JSONResponse | None:
            now = time.time()
            hist = [t for t in rate[ip] if t > now - RATE_LIMIT_WINDOW]
            if len(hist) >= RATE_LIMIT_MAX:
                return JSONResponse(
                    status_code=429,
                    content={"error": "Rate limit"},
                    headers={"Retry-After": str(RATE_LIMIT_WINDOW)},
                )
            hist.append(now)
            rate[ip] = hist
            return None

        @api.get("/health")
        async def health():
            hooks.fire("request.pre", "GET", "/health")
            try:
                return {
                    "status": "ok",
                    "service": APP_NAME,
                    "version": SERVICE_VERSION,
                    "model": MODAL_VISION_MODEL,
                    "main_model": (_CARD.main_model_hf_id or MODAL_VISION_MODEL),
                    "segmenter": _CARD.segmenter_kind,
                    "fast_gate": _CARD.fast_gate_kind,
                }
            finally:
                hooks.fire("request.post", "GET", "/health", 200)

        @api.post("/warm")
        async def warm(
            req: Request,
            authorization: str = Header(default=""),
            origin: str = Header(default=""),
        ):
            hooks.fire("request.pre", "POST", "/warm")
            denied = authorize(authorization, origin)
            if denied is not None:
                hooks.fire("request.post", "POST", "/warm", denied.status_code)
                return denied
            ip = req.headers.get("x-forwarded-for", "").split(",")[0].strip() or (
                req.client.host if req.client else "unknown"
            )
            limited = admit(ip)
            if limited is not None:
                hooks.fire("request.post", "POST", "/warm", limited.status_code)
                return limited
            try:
                import torch

                dummy = torch.zeros((1, 3, 224, 224), device=self.device)
                with torch.no_grad():
                    self.model.encode_image(dummy)
            except Exception as exc:
                hooks.fire("request.post", "POST", "/warm", 503)
                return JSONResponse(status_code=503, content={"error": f"Warm failed: {exc}"})
            hooks.fire("request.post", "POST", "/warm", 200)
            return {"warm": True, "service": APP_NAME, "model": MODAL_VISION_MODEL}

        @api.post("/v1/identify")
        async def identify(
            req: Request,
            payload: VisionRequest,
            authorization: str = Header(default=""),
            origin: str = Header(default=""),
        ):
            hooks.fire("request.pre", "POST", "/v1/identify")
            denied = authorize(authorization, origin)
            if denied is not None:
                hooks.fire("request.post", "POST", "/v1/identify", denied.status_code)
                return denied
            ip = req.headers.get("x-forwarded-for", "").split(",")[0].strip() or (
                req.client.host if req.client else "unknown"
            )
            limited = admit(ip)
            if limited is not None:
                hooks.fire("request.post", "POST", "/v1/identify", limited.status_code)
                return limited
            if not payload.images:
                hooks.fire("request.post", "POST", "/v1/identify", 400)
                return JSONResponse(status_code=400, content={"error": "No images provided"})
            _t0 = time.perf_counter()
            try:
                result = self._infer(
                    payload.images[:5],
                    payload.segment,
                    payload.candidates,
                    payload.references,
                    payload.adaptive,
                )
                took_ms = round((time.perf_counter() - _t0) * 1000)
                hooks.fire("identify.post", result.get("model", MODAL_VISION_MODEL), took_ms)
            except Exception as exc:
                hooks.fire("request.post", "POST", "/v1/identify", 500)
                return JSONResponse(status_code=500, content={"error": f"Vision failure: {exc}"})
            hooks.fire("request.post", "POST", "/v1/identify", 200)
            return result

        return api


@app.local_entrypoint()
def main():
    import pathlib

    assets = pathlib.Path(__file__).resolve().parents[1] / "examples" / "fixtures"
    if not assets.is_dir():
        print(
            f"No local fixtures found at {assets} -- drop any .jpg/.png specimen\n"
            "photos there (see examples/fixtures/README.md; the smoke entrypoint\n"
            "expects pseudo-sanderiana-1.jpg, -2.jpg, and -3.jpg), then rerun."
        )
        return

    def data_url(name: str) -> str:
        return "data:image/jpeg;base64," + base64.b64encode((assets / name).read_bytes()).decode()

    svc = VisionService()
    photo = data_url("pseudo-sanderiana-1.jpg")
    print("=== text-only (no references) ===")
    out = svc.infer.remote(images=[photo], segment=True)
    print("version:", out["version"], "| model:", out["model"])
    print("segmented:", out["segmented"], "crop_len:", len(out.get("specimen_crop") or ""))
    for prediction in out["predictions"]:
        print(f"  {prediction['score']:.4f}  {prediction['name']}")

    print("=== with reference photos (few-shot) ===")
    references = {
        "Alocasia sanderiana": [
            data_url("pseudo-sanderiana-2.jpg"),
            data_url("pseudo-sanderiana-3.jpg"),
        ]
    }
    out = svc.infer.remote(images=[photo], segment=True, references=references)
    for prediction in out["predictions"]:
        print(f"  {prediction['score']:.4f}  {prediction['name']}  (ref={prediction.get('reference_score')})")
