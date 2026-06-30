"""Entry point for the Flask game server.

Usage (from project root):
    python src/backend/scripts/run.py              # Start on http://127.0.0.1:5000
    python src/backend/scripts/run.py --debug      # Start with debug mode (auto-reload)

Before running in production mode, build the React frontend first:
    cd src/frontend && npm run build
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `src.*` imports resolve.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.backend.controller.api import create_app

app = create_app()

if __name__ == "__main__":
    debug = "--debug" in sys.argv
    app.run(host="127.0.0.1", port=5000, debug=debug)
