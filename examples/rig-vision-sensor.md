# Vision as a Sensor: wiring modal-vision-server into a camera rig

A camera photo becomes a metric. This guide wires any camera-driven rig (a
plant cam, a greenhouse box, a workshop frame) to read a value out of the
photo through the **generic vision server**: capture, crop, POST, then write
the parsed value into VictoriaMetrics as a normal sensor series. The sensor's
"skill" (which model, which labels, few-shot references) is a model card, not
a new app.

The flow, using the rig's existing primitives:

```
capture_photo.sh            (rig: takes the photo, queues a pending crop)
        |
lib/vision_sensor.sh        (NEW: crop ROI -> POST /v1/identify -> parse)
        |
Modal vision server         (https://<ws>--modal-vision...modal.run/v1/identify)
        |
write_metric()              (vm_write.sh: taxon label + confidence as a series)
```

## What the generic server provides that a bespoke OCR app doesn't

| Dimension | Bespoke OCR app (per-deployment) | Vision server (generic) |
| :--- | :--- | :--- |
| Model choice | baked (`minicpm-v`, OCR prompt) | card (`fast_gate`, `backend`, `prompt_template`) |
| Labels | none (free-form OCR) | card `label_defaults` or per-request `candidates` |
| Confidence | model mood | `score` per prediction + `adaptive_skip` |
| Few-shot | no | `references` per label (calibrated centroids) |
| Cost posture | 2min cold boot, warm 5s | same scale-to-zero; `deterministic` gate pre-pass can decide flat-lay photos in microseconds before any GPU work |
| Changing the sensor | new app + redeploy | edit a card + redeploy (or swap cards by env) |

## One-time deploy (if the server isn't already up)

```bash
cd modal-vision-server
uv run --project server modal deploy server/app.py
# -> prints the endpoint URL: https://<workspace>--modal-vision-server-visionservice-web.modal.run
```

Auth: the endpoint enforces its own bearer (`API_TOKEN` from the
`modal-vision-secret` Secret). Store both the URL and the token in the rig's
`rig.env`:

```bash
VISION_SENSOR_URL=https://<workspace>--modal-vision-server-visionservice-web.modal.run
VISION_SENSOR_TOKEN=<hex token>
```

## The sensor script (`lib/vision_sensor.sh`)

New file; mirrors the split-primitive shape (crop local, query out-of-band) (crop local,
query out-of-band so the ~2min cold start never blocks the capture container)
and its return-code contract (0 success, 2 config missing, 3 bad input, 4
network/transient, 5 no-prediction definitive, 1 other):

```bash
#!/usr/bin/env bash
# Reference implementation of vision-as-a-sensor for a camera rig: sends an
# already-cropped plant photo to the deployed modal vision server and parses
# the result. Companion to rig-vision-sensor.md (same directory), which
# explains the wiring into capture_photo.sh / read_pending_* / vm_write.sh.
#
# CONTRACT (mirrors the rig's existing sensor-reader shape so its retry/discard
# logic works unchanged):
#   vision_sensor_query <cropped.jpg>
#   Returns 0 on success with VISION_TAXON / VISION_CONFIDENCE /
#   VISION_SEGMENTED / VISION_ADAPTIVE_SKIP set; the other return codes match
#   existing sensor readers' meanings (2 = config missing/unauthorized, 3 = bad input
#   image, 4 = transient network/server, 5 = 200 but unusable payload,
#   1 = other local failure).
#
# Env:
#   VISION_SENSOR_URL    endpoint base (e.g. https://ws--...modal.run). Required.
#   VISION_SENSOR_TOKEN  bearer token from the modal-vision-secret. Required.
#   VISION_CANDIDATES    optional comma-separated label list; when set, the
#                        request carries `candidates` (multi-plant rigs). When
#                        unset the request uses the card's label_defaults.
#   VISION_REFERENCES_JSON  optional path to a file mapping label -> [urls];
#                        few-shot mode (see rig-vision-sensor.md's growth-stage
#                        note). Optional.
#
# NOTE: this reference implementation keeps curl simple and readable. The
# production rig version should follow the other sensor readers' exact guards
# (|| var="" per command substitution; set -euo pipefail-safe call sites).
set -euo pipefail

vision_sensor_query() {
  VISION_TAXON=""; VISION_CONFIDENCE=""; VISION_SEGMENTED=""; VISION_ADAPTIVE_SKIP=""

  local image_jpg="${1:?usage: vision_sensor_query <cropped.jpg>}"
  [[ -f "${image_jpg}" ]] || return 3

  # Config: both URL and token must be set (transient: fixable later).
  [[ -n "${VISION_SENSOR_URL:-}" && -n "${VISION_SENSOR_TOKEN:-}" ]] || return 2

  # Encode + build the request body (one file so curl reads it atomically).
  local b64 payload http_code body_file body
  b64="$(base64 -i "${image_jpg}" 2>/dev/null)" || return 3
  [[ -n "${b64}" ]] || return 3

  local candidates_json="null" references_json="null"
  if [[ -n "${VISION_CANDIDATES:-}" ]]; then
    # shellcheck disable=SC2086
    candidates_json="$(echo "${VISION_CANDIDATES}" | tr ',' '\n' | \
      sed 's/.*/"&"/' | paste -sd, - | sed 's/^/[/' | sed 's/$/]/')"
  fi
  if [[ -n "${VISION_REFERENCES_JSON:-}" && -f "${VISION_REFERENCES_JSON}" ]]; then
    references_json="$(cat "${VISION_REFERENCES_JSON}")"
  fi

  payload="$(mktemp /tmp/vision_req.XXXXXX.json)"
  printf '{"images":["data:image/jpeg;base64,%s"],"segment":false,"adaptive":true,"candidates":%s,"references":%s}' \
    "${b64}" "${candidates_json}" "${references_json}" > "${payload}"

  body_file="$(mktemp /tmp/vision_resp.XXXXXX.json)"
  http_code="$(curl -s -o "${body_file}" -w '%{http_code}' -m 300 \
    -X POST "${VISION_SENSOR_URL%/}/v1/identify" \
    -H "Authorization: Bearer ${VISION_SENSOR_TOKEN}" \
    -H "Content-Type: application/json" \
    -d @"${payload}")" || http_code=""
  rm -f "${payload}"
  body="$( [ -f "${body_file}" ] && { cat "${body_file}"; rm -f "${body_file}"; } )"

  case "${http_code}" in
    ""|000) return 4 ;;    # curl failed outright (network/timeout)
    200) ;;
    401|403) return 2 ;;   # auth: config issue, transient in the rig's model
    429) sleep 5; return 4 ;;
    5*) return 4 ;;
    *) return 4 ;;
  esac

  # Parse the standard /v1/identify response: predictions[] with name+score,
  # plus segmented/adaptive_skip booleans. Empty predictions = unusable crop
  # (definitive: identical bytes reproduce this; caller discards).
  VISION_TAXON="$(printf '%s' "${body}" | python3 -c "
import json,sys
try:
    preds = json.load(sys.stdin).get('predictions') or []
    print(preds[0].get('name','') if preds else '')
except Exception:
    pass
" 2>/dev/null || true)" || VISION_TAXON=""
  VISION_CONFIDENCE="$(printf '%s' "${body}" | python3 -c "
import json,sys
try:
    preds = json.load(sys.stdin).get('predictions') or []
    print(f\"{preds[0].get('score',0):.4f}\" if preds else '')
except Exception:
    pass
" 2>/dev/null || true)" || VISION_CONFIDENCE=""
  VISION_SEGMENTED="$(printf '%s' "${body}" | python3 -c "
import json,sys
try:
    print('true' if json.load(sys.stdin).get('segmented') else 'false')
except Exception:
    pass
" 2>/dev/null || true)" || VISION_SEGMENTED="false"
  VISION_ADAPTIVE_SKIP="$(printf '%s' "${body}" | python3 -c "
import json,sys
try:
    print('true' if json.load(sys.stdin).get('adaptive_skip') else 'false')
except Exception:
    pass
" 2>/dev/null || true)" || VISION_ADAPTIVE_SKIP="false"

  [[ -n "${VISION_TAXON}" ]] || return 5
  return 0
}

# When sourced, exporting the contract makes the values visible to callers
# (the read_pending_* script uses them for write_metric()).
export VISION_TAXON VISION_CONFIDENCE VISION_SEGMENTED VISION_ADAPTIVE_SKIP
```

This is the maintained reference implementation, kept in sync at
[`examples/vision_sensor_query.sh`](vision_sensor_query.sh) in this repo:
ONE curl per request, body captured to a file, the HTTP check coded off
`curl -w '%{http_code}' -o body_file -d @payload`. Embed a copy of that
file in the rig (as `lib/vision_sensor.sh`); don't retype the shell.

## The writer loop (`read_pending_vision.sh`)

```bash
#!/usr/bin/env bash
# systemd-rig-vision-read.service: reads the pending plant crop with a retry
# contract identical to the rig's other sensor readers (definitive vs transient; depth-1 queue).
source "${SCRIPT_DIR}/lib/vm_write.sh"
source "${SCRIPT_DIR}/lib/vision_sensor.sh"

PENDING_JPG="$(dirname "${PHOTO_DIR}")/vision_pending.jpg"
# metric names, one series per taxon so Grafana can filter:
status=0
vision_sensor_query "${PENDING_JPG}" || status=$?
case "${status}" in
  0) write_metric "vision_taxon_conf" "${VISION_CONFIDENCE}" ; \
     write_metric "vision_taxon_score_${VISION_TAXON}" "1" ; \
     rm -f "${PENDING_JPG}" ;;
  3|5) rm -f "${PENDING_JPG}" ;;   # definitive: discard, wait for the next cycle
  *) : ;;                          # transient: leave for the next run
esac
```

Tag semantics: one point per capture, `device=${RIG_DEVICE_TAG}` groups it
under the rig; `vision_taxon_score_<taxon>=1` builds the presence series that
Grafana can time-filter.

## The model card side (what to tune for a rig, not a house plant)

The rig's camera sees one plant at a fixed distance with strong
day/night cycles: two card knobs matter most.

1. **A deterministic fast gate beats few-shot for this data.** The rig's
   crops are geometrically stable; a gate script keyed on the crop's flat
   background (the bundled
   [`flat_background_gate.py`](https://github.com/kylebrodeur/modal-vision-server/blob/main/server/plugins/fastgates/flat_background_gate.py))
   decides most daytime frames in microseconds and only falls back to the
   classifier at night (when the LED band color changes the corner average).
   Point `fast_gate_kind: "deterministic"` +
   `fast_gate_script: "/root/plugins/fastgates/flat_background_gate.py"` in a
   rig card.

2. **Labels come from a per-cycle card, not the request.** Keep the
   card's `label_defaults` to the ONE plant being monitored (rig cards are
   rig-scoped, like `RIG_DEVICE_TAG`). A request without `candidates` then
   classifies against that single label and the confidence series becomes a
   real state signal. Multi-plant rigs: one card per rig, one env var
   (`MODAL_VISION_MODEL_CARD` baked per deploy) or `candidates` at request
   time.

3. **Optional: few-shot per growth stage.** Five reference images per
   stage (seedling/vegetative/flowering) lets the same card act as a 3-class
   stage classifier through `references` at request time; the centroid
   calibration constants in the card carry over (they're model properties).
   The rig's own time-series then gets a `vision_stage` series.

## Cost shape (why this is a sensor, not a job)

- Scale-to-zero: between capture cycles the endpoint costs nothing.
- The deterministic gate's skip path costs **microseconds with no GPU**: the
  expensive encode happens only at night or when the gate falls through.
- Model-card changes still need a redeploy: the card is parsed once at image
  import (`load_card()` at module load), so edits don't reach a running
  container. What card edits buy you is that a swap stays a config-file
  change, not a code change. `mtk doctor` sees it as one more package lane;
  the fleet dashboard's vision card shows the model + fast_gate posture.

## Files this example assumes you have

| Rig side (existing) | Vision side (this repo) |
| :--- | :--- |
| `capture_photo.sh` (pending-file queue) | `registry/bioclip-plants.json` (or a rig card) |
| `lib/vm_write.sh` (`write_metric`) | `server/app.py` deployed |
| `systemd/rig-<sensor>-read.service` (copy the unit shape) | `/v1/identify` + bearer secret |
| `rig.env` (URL/secret envs) | `examples/vision_sensor_query.sh` reference impl |

The example script is written to be COPIED into the rig scaffold
(`scaffold/lib/vision_sensor.sh`), not sourced cross-repo: the rig
keeps its own vendored primitives, matching the other sensor readers' pattern.