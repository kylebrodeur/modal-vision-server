"""Put the server package directory on sys.path for flat imports.

Inside the Modal container server/ is importable directly (``import app``
style); mirror that here so tests can import model_card / encoders / fastgate
and so test_config.py can exec the app.py prelude unchanged.
"""

import sys
from pathlib import Path

_SERVER_DIR = str(Path(__file__).resolve().parent)
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)
