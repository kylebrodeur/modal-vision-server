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