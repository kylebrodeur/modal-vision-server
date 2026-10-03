"""Minimal vision classification example against the deployed Modal Vision Server.

Posts one base64-encoded local image to the live `POST /v1/identify` route
(the server accepts data URLs only; it does not fetch http(s) URLs itself).

Usage:
    export MODAL_VISION_URL="https://<workspace>--modal-vision-server-visionservice-web.modal.run"
    export MODAL_VISION_API_TOKEN="<your auth token>"
    export IMAGE_PATH="<path to a local .jpg/.png>"
    # Optional comma-separated label list; omit to use the card's label_defaults.
    export CANDIDATES="Monstera deliciosa,Monstera adansonii"
    uv run examples/classify_example.py
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import pathlib
import sys
import urllib.request

BASE = os.environ.get("MODAL_VISION_URL", "").rstrip("/")
TOKEN = os.environ.get("MODAL_VISION_API_TOKEN", "")
IMAGE_PATH = os.environ.get("IMAGE_PATH", "")
CANDIDATES = os.environ.get("CANDIDATES", "")

if not BASE or not TOKEN or not IMAGE_PATH:
    print(
        "Set MODAL_VISION_URL, MODAL_VISION_API_TOKEN, and IMAGE_PATH first.",
        file=sys.stderr,
    )
    sys.exit(1)

path = pathlib.Path(IMAGE_PATH)
if not path.is_file():
    print(f"No such image: {path}", file=sys.stderr)
    sys.exit(1)

# The server's _load_image() only base64-decodes `data:<mime>;base64,<payload>`
# strings, so the client encodes the file itself.
mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
data_url = (
    "data:" + mime + ";base64," + base64.b64encode(path.read_bytes()).decode("ascii")
)

payload: dict = {"images": [data_url]}
if CANDIDATES:
    payload["candidates"] = [c.strip() for c in CANDIDATES.split(",") if c.strip()]

req = urllib.request.Request(
    f"{BASE}/v1/identify",
    data=json.dumps(payload).encode("utf-8"),
    method="POST",
    headers={
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
    },
)
with urllib.request.urlopen(req) as response:
    body = json.loads(response.read().decode("utf-8"))

for prediction in body.get("predictions", []):
    print(f"  {prediction['score']:.3f}  {prediction['name']}")
