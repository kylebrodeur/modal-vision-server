"""Deterministic fast-gate plugins.

Each module defines ``decide(image, meta) -> dict | None`` and must stay
pure-Python (no GPU deps, no network): these run in microseconds before the
expensive segmentation pass.
"""
