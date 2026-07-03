"""Shared fixtures for all backend tests.

Fixtures defined here are auto-discovered by pytest and available
to all test files in the tests/ directory tree without explicit imports.
"""

import json

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
