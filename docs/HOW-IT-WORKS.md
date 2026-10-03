# Vision Server: How It Works

The deep-dive doc for the generic stack (README covers the three choice axes;
this covers the mechanics under them). Everything here is verified against the
running code; the file names below live in `server/`.

## The request path

```
POST /v1/identify {images: [data_url], candidates?, references?, segment, adaptive}
  -> _load_image            (base64 data URL -> PIL RGB; http(s) is not fetched)
  -> [adaptive? gate pre-pass]  (fastgate.py; may skip everything below)
  -> [segment? _segment]    (SAM 2 mask generation, best scored plausible mask)
  -> _encode_image          (encoder adapter; L2-normalized)
  -> _classify              (text/label space vs image embedding)
       + reference centroids (few-shot rung; bounded logit add)
  -> {predictions, segmented, adaptive_skip, ...}
```

## The three choices (where they execute)

**Main model** (`model_card.py` -> `encoders.py`). `initialize()` calls
`encoder_factory(card, fallback_hf_id, device)`. The adapter owns model load,
preprocess, tokenizer, and the prompt template. `OpenClipEncoder` is the
original path (`create_model_and_transforms` + `get_tokenizer`); text
embeddings run through `prompt_template.format(label=label)` then
`F.normalize`. `TransformersEncoder` uses `AutoProcessor`; when
`config.architectures` says image-only (DINOv2), `encode_texts` raises
NotImplementedError. Because the identify path encodes the label space before
reference matching can kick in, the shipped
`registry/dinov2-objects.json` demo card is currently **not executable**
through `/v1/identify`: image-only classification with reference-only labels
is a planned extension, not live behavior.

**Segmenter** (also `initialize()`). Two modes: `sam2` downloads the card's
checkpoint to the weights Volume and builds `build_sam2`; `_segment` runs
mask generation and, among candidates that pass the hard isolation gates
(no sliver, no near-full-frame), keeps the best composite-scored mask
(bbox tightness, centrality, object-scale preference) — not simply the
largest one. `none` skips SAM entirely
(no download in the boot path), sets `sam_model = None`; a request with
`segment: true` then returns `segmented: false` rather than 500 (the guard
raises a caught ValueError). GPU is optional in config (`MODAL_VISION_GPU="";
none|cpu` are equivalent); classify-only cards are CPU-eligible.

**Fast gate** (`fastgate.py`; the adaptive pre-pass). Built once in
`initialize()` from the card; the request's `adaptive: true` flag decides
whether the gate runs at all. All gates return `GateResult(decision, reason)`:

- `self`: classify the full image with the main model first. Decisive
  (top1 >= `fast_gate_confidence` 0.55 AND top1-top2 margin >= `fast_gate_margin`
  0.30; top1 counts as the margin when there is only one label) skips
  segmentation: `adaptive_skip: true, segmented: false`, same classification
  body as the full path. Otherwise falls through. This reproduces the
  historical behavior byte-for-byte (the card's defaults are those constants).
- `deterministic`: a pure-Python plugin at `fast_gate_script` with
  `decide(image, meta) -> dict | None`; runs before any encode, microseconds.
  The bundled `plugins/fastgates/flat_background_gate.py` demonstrates the
  shape (flat-lay/scanner detection via corner-color + center-gradient).
  Any script error loads once, logs, and returns `undecided` (fail-open);
  the request then follows the self-gate threshold semantics.
- `cheap_clip`: a small open_clip encoder (`cheap_clip_hf_id`, default
  unset to avoid silent downloads; `MODAL_VISION_CHEAP_CLIP_MODEL` env) runs
  the label space on the image; decisive under its thresholds skips, ambiguous
 -but-close forces segmentation, everything else falls through to the main
  model's own thresholds. Lazy imports; construction without a device or model
  raises a loud ValueError at boot rather than a runtime surprise.
- `external`: POSTs the context to `fast_gate_url` with a bearer from
  `MODAL_VISION_FAST_GATE_TOKEN` when set; 5s timeout; bad-JSON/timeout maps
  to `undecided` (fail-open, never a failed request).
- `none`: no gate at all; adaptive requests take the full path.

Non-self gates may FORCE the segmented path on a `segment` decision; the
self gate never does (it only skips or falls through), keeping the historical
flow intact.

## Reference matching (the few-shot rung)

Per-label reference images are encoded once and cached in a process-local
dict (`_reference_cache`) for the container's lifetime; they are NOT
persisted to the weights Volume. The centroid's cosine adds a bounded
number of logits (`reference_scale * similarity`, capped at
`reference_max_logit_add`) to that label's text logit, so text evidence
dominates unless a reference genuinely matches better. `reference_null_floor` covers the single-reference case where
no cross-label baseline exists. These constants are per-model (they encode the
model's logit geometry); they travel in the card.

## Volume and caching

- `modal-vision-weights` Volume: model weights (main + SAM checkpoint).
  Reference centroids live only in a process-local dict (_reference_cache) on
  the running container; they are not written to the Volume. First boot
  downloads; later boots read from disk.
- The dummy encode at startup pre-compiles CUDA kernels so the first real
  request hits compiled paths (`/warm` does the same thing over HTTP).

## Ops (what `mtk doctor` sees)

`/health` (no auth) returns status + `main_model` + `segmenter` +
`fast_gate` (the card view, so a doctor probe answers "what is this
deployment configured as?" without auth). Protected routes return 503 when
the auth secret is missing (fail-closed). Rate limiting is per-IP
(`RATE_LIMIT_MAX`/`WINDOW`).

## Eval

`eval/harness.py` scores this service against the Pl@ntNet API on
Wikimedia-sourced labeled images (per-taxon top-1 accuracy). Its committed
`results.json` holds the BioCLIP-plant run (+18.1% headline). For a
different card, run the same harness against the new deployment; the
comparison then measures your model choice, which is the point.

## Metrics (opt-in)

`server/vm_metrics.py` is the vendored copy of the toolkit's canonical
metrics writer (stdlib-only, never raises; the only delta is the env var
name, so each Modal deploy tree stays self-contained). Set
`MODAL_VISION_METRICS=1` to enable emission; `MODAL_VISION_VM_URL` picks the
endpoint (default `http://localhost:8428`); `MODAL_VISION_DEVICE_TAG`
labels each point's `device` tag. Emission points, all on the identify path:

- Request entry: `vision_request` (tags `segment_requested`, `gate`).
- Gate outcome, before returning or falling through: `vision_gate_decision`
  (tags `decision` = `skip` | `segment` | `undecided` | `full`, `kind` =
  the card's fast-gate kind). Plot skip rate by `kind` to see what the gate
  is saving.
- Request exit: `vision_identify_seconds` (tag `segmented`).

With the env off every emission site is a single boolean check.

## Known open items

- cheap_clip's model download happens at gate construction (boot), not lazy
  per-request; a card with `cheap_clip` pays one download per cold boot.