# Model Cards

A model card is a JSON file that fully describes one vision stack: the main
classification model and its backend, the segmenter, the fast gate, and the
default labels. One card replaces a pile of individual environment variables,
and it is reviewable, diffable, and shareable.

## Card contents

Every card carries these fields:

- `main_model_hf_id`: Hugging Face model id (open_clip `hf-hub:` ids and plain `org/name` ids both work, depending on the backend).
- `backend`: `open_clip` or `transformers`.
- `prompt_template`: text template with `{label}` substituted per class. Image-only encoders ignore it.
- `similarity_scale`: logit scale applied to text similarity.
- `reference_scale`, `reference_max_logit_add`, `reference_null_floor`: few-shot reference-matching calibration.
- `segmenter_kind`, `segmenter_hf_repo`, `segmenter_ckpt`, `segmenter_config`: segmentation model and weights.
- `fast_gate_kind`, `fast_gate_confidence`, `fast_gate_margin`, `fast_gate_script`, `fast_gate_url`: adaptive gate.
- `label_defaults`: default labels when a request omits them (`null` means labels always come from the request).

## Resolution order

Configuration resolves in this order, first hit wins per choice:

1. `MODAL_VISION_MODEL_CARD`: name of a card in this directory (for example `bioclip-plants`) or a path to a card JSON file. The card sets all three choices at once.
2. Single-line envs: `MODAL_VISION_MAIN_MODEL`, `MODAL_VISION_SEGMENTER`, `MODAL_VISION_FAST_GATE` override the corresponding choice individually.
3. Legacy envs (`MODAL_VISION_MODEL`, `MODAL_VISION_ADAPTIVE_SEGMENT`, and friends) are still respected when nothing above them is set.
4. Built-in defaults, which are exactly the shipped `bioclip-plants` values.

Use a card when you are changing two or more choices together, or when the
combination should be checked into version control. Use the single-line envs
for one-off experiments (for example, trying `MODAL_VISION_SEGMENTER=none` on
the default stack).

## Sample cards

- `bioclip-plants.json`: the deployment default. BioCLIP 2 via open_clip over 16 houseplant taxa, SAM 2 segmentation, and the self fast gate. Loading it is byte-identical to the shipped defaults.
- `dinov2-objects.json`: an image-only demonstration card. DINOv2 via transformers, no segmenter, no fast gate, CPU-eligible. There is no text encoder, so classification requires few-shot reference images passed as `references` per label at request time, and labels come entirely from the request (`label_defaults` is `null`).

## Cards never carry secrets

Cards are configuration, not credentials. Never put tokens, API keys, or
service-account material in a card. Auth stays in Modal secrets (the auth
secret and, when the model card points at a gated repo, a Hugging Face token),
and the fast-gate `external` URL field names an endpoint only; any credentials
it needs live in the Modal secret too.
