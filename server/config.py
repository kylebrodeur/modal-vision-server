"""Configuration constants and env-overridable knobs for the Modal vision service.

Everything is env-overridable; config values are read once at import time so a
deployed container stays consistent for its lifetime.

Override any value at deploy time with an environment variable of the same
name (e.g. ``MODAL_VISION_GPU=A10G modal deploy server/app.py``).
"""

from __future__ import annotations

import os

# ── App identity ─────────────────────────────────────────────────────────────
# Keep these stable: renaming a Volume or App orphans the data behind it.

APP_NAME = os.environ.get("MODAL_VISION_APP_NAME", "modal-vision-server")

# Persistent Volume for model weights so cold starts don't re-download.
WEIGHTS_VOLUME_NAME = os.environ.get("MODAL_VISION_WEIGHTS_VOLUME", "modal-vision-weights")
WEIGHTS_DIR = "/root/cache"

# ── Secrets ─────────────────────────────────────────────────────────────────
AUTH_SECRET_NAME = os.environ.get("MODAL_VISION_AUTH_SECRET", "vision-auth-secret")
HF_SECRET_NAME = os.environ.get("MODAL_VISION_HF_SECRET", "huggingface-secret")

# ── Compute ──────────────────────────────────────────────────────────────────
_gpu_env = os.environ.get("MODAL_VISION_GPU", "T4")
GPU = _gpu_env.strip().lower() or None

# ── Model defaults ──────────────────────────────────────────────────────────
VISION_MODEL = os.environ.get("MODAL_VISION_MODEL", "hf-hub:imageomics/bioclip-2")

# ── Rate limiting ───────────────────────────────────────────────────────────
RATE_LIMIT_MAX = int(os.environ.get("MODAL_VISION_RATE_LIMIT_MAX", "30"))
RATE_LIMIT_WINDOW = int(os.environ.get("MODAL_VISION_RATE_LIMIT_WINDOW", "60"))
