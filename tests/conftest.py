"""Shared fixtures for all backend tests.

Fixtures defined here are auto-discovered by pytest and available
to all test files in the tests/ directory tree without explicit imports.
"""

import importlib
import json
import os

import pytest

from src.backend.app.app_manager import AppManager
from src.backend.game.session_store import SessionStore
from src.backend.log.structured_logger import LoggerBuilder


# ---------------------------------------------------------------------------
# LoggerBuilder reset (autouse — every test gets a clean singleton)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_logger_builder():
    """Reset LoggerBuilder singleton before and after every test.

    Ensures each test starts with a clean LoggerBuilder state,
    preventing cross-test contamination of the singleton.
    """
    LoggerBuilder._reset()
    yield
    LoggerBuilder._reset()


# ---------------------------------------------------------------------------
# AppManager
# ---------------------------------------------------------------------------


@pytest.fixture
def app_manager(tmp_path):
    """Create an AppManager with a temporary SessionStore.

    Uses tmp_path so no real data files are affected.
    """
    saves_dir = tmp_path / "saves"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"save_dir": str(saves_dir)}))
    store = SessionStore(config_path=str(config_path))
    return AppManager(store)


# ---------------------------------------------------------------------------
# Flask test client
# ---------------------------------------------------------------------------


@pytest.fixture
def api_client(tmp_path):
    """Create a Flask test client with a temporary SessionStore.

    Creates a fresh Flask app instance per test, with saves directory
    isolated to tmp_path. No real files are affected.
    """
    from src.backend.controller.api import create_app
    saves_dir = tmp_path / "saves"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"save_dir": str(saves_dir)}))
    app = create_app(config_path=str(config_path))
    app.config["TESTING"] = True
    return app.test_client()


# ---------------------------------------------------------------------------
# gen1 model weights for training tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _ensure_gen1_weights():
    """Create v2 model weights in a temporary directory for training tests.

    Patches _derive_weight_path so rnn_v2.1 resolves to temp paths.
    This avoids modifying the real project directory while ensuring
    training tests can load model weights during _build_models().

    Scope: session — weight files are created once and reused.
    """
    import torch
    import tempfile
    from unittest.mock import patch

    from src.backend.ai.rnn_v2.model import GRUPolicyModelV2, GRUValueModelV2

    # Create temp gen1 directory with initialized model weights
    gen1_dir = tempfile.mkdtemp(prefix="test_gen1_")

    # Initialize and save policy model with random weights
    policy_model = GRUPolicyModelV2()
    torch.save(policy_model.state_dict(), os.path.join(gen1_dir, "policy.pth"))

    # Initialize and save value model with random weights
    value_model = GRUValueModelV2()
    torch.save(value_model.state_dict(), os.path.join(gen1_dir, "value.pth"))

    # Monkeypatch _derive_weight_path to return temp paths for rnn_v2 gen1
    import src.backend.ai.registry as registry_mod

    _original_derive = registry_mod._derive_weight_path

    def _patched_derive(name: str, gen: int, suffix: str) -> str:
        if name == "rnn_v2" and gen == 1:
            return os.path.join(gen1_dir, f"{suffix}.pth")
        return _original_derive(name, gen, suffix)

    with patch.object(registry_mod, "_derive_weight_path", side_effect=_patched_derive):
        yield

    # Cleanup temp directory after session
    import shutil
    shutil.rmtree(gen1_dir, ignore_errors=True)
