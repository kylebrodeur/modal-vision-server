"""Smoke tests for the Modal vision service application constants."""

import os
from unittest.mock import patch

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import app


class TestVisionConfig:
    def test_default_app_name(self):
        assert app.APP_NAME == "modal-vision-server"

    def test_default_gpu(self):
        assert app.GPU_TYPE == "t4"

    def test_cpu_when_empty(self):
        with patch.dict(os.environ, {"MODAL_VISION_GPU": ""}):
            import importlib

            importlib.reload(app)
            assert app.GPU_TYPE is None

    def test_default_model(self):
        assert app.MODAL_VISION_MODEL == "hf-hub:imageomics/bioclip-2"

    def test_auth_secret_name(self):
        assert app.AUTH_SECRET_NAME == "modal-vision-secret"

    def test_rate_limit_defaults(self):
        assert app.RATE_LIMIT_MAX == 30
        assert app.RATE_LIMIT_WINDOW == 60

    def test_weights_volume_name(self):
        assert app.VOLUME_NAME == "modal-vision-weights"