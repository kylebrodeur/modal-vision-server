"""Minimal vision classification example against the deployed Modal Vision Server.

Usage:
    export MODAL_VISION_MODAL_URL="https://<workspace>--modal-vision-server.modal.run"
    export MODAL_VISION_API_TOKEN="<your auth token>"
    export PHOTO_URL="<a publicly reachable image URL>"
    uv run examples/classify_example.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

BASE = os.environ.get("MODAL_VISION_MODAL_URL", "").rstrip("/")
TOKEN = os.environ.get("MODAL_VISION_API_TOKEN", "")
PHOTO_URL = os.environ.get("PHOTO_URL", "")

if not BASE or not TOKEN or not PHOTO_URL:
    print(
        "Set MODAL_VISION_MODAL_URL, MODAL_VISION_API_TOKEN, and PHOTO_URL first.",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    payload = {"photo_url": PHOTO_URL, "topk": 5, "segment": False}
    req = urllib.request.Request(
        f"{BASE}/identify",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req) as response:
        body = json.loads(response.read().decode("utf-8"))

    for label, score in body.get("taxa", []):
        print(f"  {score:.3f}  {label}")


if __name__ == "__main__":
    main()