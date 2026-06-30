"""Entry point for the Flask game server.

Usage:
    python run.py              # Start on http://127.0.0.1:5000
    python run.py --debug      # Start with debug mode (auto-reload)

Before running in production mode, build the React frontend first:
    cd src/frontend && npm run build
"""

import sys

from src.backend.controller.api import create_app

app = create_app()

if __name__ == "__main__":
    debug = "--debug" in sys.argv
    app.run(host="127.0.0.1", port=5000, debug=debug)
