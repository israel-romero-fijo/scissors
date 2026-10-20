from pathlib import Path

import pytest

from scissors import Scissors


def pytest_addoption(parser):
    parser.addoption(
        "--quick",
        action="store_true",
        help="Run local checks without integration tests that launch processes or backends",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--quick"):
        return
    selected, deselected = [], []
    for item in items:
        (deselected if item.get_closest_marker("integration") else selected).append(item)
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)


@pytest.fixture
def app(tmp_path):
    with Scissors(tmp_path / "workspace") as service:
        yield service


@pytest.fixture
def twist_path():
    return Path(__file__).resolve().parents[1] / "plugins/examples/twist_mesh"


@pytest.fixture(autouse=True)
def isolated_plugin_store(tmp_path_factory, monkeypatch):
    """Keep tests and child processes away from the user's application plugins."""
    monkeypatch.setenv("SCISSORS_PLUGINS_DIR", str(tmp_path_factory.mktemp("application_plugins")))
