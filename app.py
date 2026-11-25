"""Where hosts that look for a FastAPI `app` (such as Vercel) find it.

Everything is in src/delivery_eta; this only points there. It works whether or not the host
installed the package, and from whatever folder the host starts it in.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("ETA_MODEL_DIR", str(ROOT / "model"))

from delivery_eta.api.app import create_app  # noqa: E402

app = create_app()
