import subprocess
import sys

from scissors import Scissors, Settings
from scissors.api.service import Scissors as Service
from scissors.config import Settings as Configuration


def test_public_imports_keep_class_identity():
    assert Scissors is Service
    assert Settings is Configuration


def test_plugin_import_and_cli_help_do_not_load_engine():
    # Use a fresh interpreter: this pytest process already imported the engine.
    code = """
import sys
import scissors
from scissors.plugins.context import PluginContext
assert 'Scissors' in dir(scissors)
assert 'Settings' in dir(scissors)
assert 'scissors.api.service' not in sys.modules
assert 'trimesh' not in sys.modules
try:
    scissors.unknown_attribute
except AttributeError:
    pass
else:
    raise AssertionError('Unknown public attribute should raise AttributeError')
from scissors.cli import main
sys.argv = ['scissors', '--help']
try:
    main()
except SystemExit as exc:
    assert exc.code == 0
else:
    raise AssertionError('Help should exit')
assert 'scissors.api.service' not in sys.modules
assert 'trimesh' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr
    assert "usage: scissors" in result.stdout
