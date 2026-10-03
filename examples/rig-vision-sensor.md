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
# vision_sensor_query(): sends an already-cropped plant photo to the
# modal vision server and sets VISION_TAXON / VISION_CONFIDENCE / VISION_SEGMENTED.
# Return codes mirror the rig's existing reader contract (same retry/discard logic
# in the caller): 0 ok, 2 no config, 3 bad input, 4 transient network,
# 5 200-with-unusable-payload, 1 other local failure.
vision_sensor_query() {
  local image_jpg="${1:?usage: vision_sensor_query <cropped.jpg>}"
  [[ -n "${VISION_SENSOR_URL:-}" && -n "${VISION_SENSOR_TOKEN:-}" ]] || return 2

  local payload b64 http_code body
  b64="$(base64 -i "${image_jpg}" 2>/dev/null)" || { return 3; }
  [[ -n "${b64}" ]] || return 3
  payload="$(mktemp)"
  printf '{"images":["data:image/jpeg;base64,%s"],"segment":false,"adaptive":true}' \
    "${b64}" > "${payload}"

  http_code="$(curl -s -o /dev/null -w '%{http_code}' -m 180 \
    -X POST "${VISION_SENSOR_URL}/v1/identify" \
    -H "Authorization: Bearer ${VISION_SENSOR_TOKEN}" \
    -H "Content-Type: application/json" -d @"${payload}")" || http_code=""
  rm -f "${payload}"
  case "${http_code}" in
    ""|000) return 4 ;;
    200) ;;
    401|403) return 2 ;;   # auth misconfig: fixable later, transient
    5*) return 4 ;;
    *) return 4 ;;
  esac

  body="$(curl -s -m 30 -X POST "${VISION_SENSOR_URL}/v1/identify" \
    -H "Authorization: Bearer ${VISION_SENSOR_TOKEN}" \
    -H "Content-Type: application/json" -d @"${payload}")" || body=""
  VISION_TAXON="$(echo "${body}" | python3 -c "
import json,sys
try:
    p = json.load(sys.stdin).get('predictions') or []
    print(p[0].get('name','') if p else '')
except Exception:
    pass
")"
  VISION_CONFIDENCE="$(echo "${body}" | python3 -c "
import json,sys
try:
    p = json.load(sys.stdin).get('predictions') or []
    print(p[0].get('score','') if p else '')
except Exception:
    pass
")"
  [[ -n "${VISION_TAXON}" ]] || return 5
  return 0
}
```

(The double-curl above is for exposition; the deployed version should keep ONE
curl, capture the body, and code the http check off `curl -w '%{http_code}'
-o body_file -d @payload` in one curl. The point of the example is the
CONTRACT, not the shell.

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
- Model-card changes are free (config), so per-plant model swaps are config
  edits, not new deployments. `mtk doctor` sees it as one more package lane;
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