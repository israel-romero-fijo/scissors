"""Bounded OpenGL startup retries, failure expiry and concurrent discovery."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from scissors.core import rendering_fast as rendering


@pytest.fixture
def cache(monkeypatch):
    monkeypatch.setattr(rendering, "_vtk_probe_state", rendering._ProbeState())
    clock = [0.0]
    monkeypatch.setattr(rendering, "_probe_clock", lambda: clock[0])
    return clock


def test_transient_failure_retries_and_only_success_is_cached(cache, monkeypatch):
    responses = iter([(False, {"reason": "temporary"}), (True, {"device": "GPU"})])
    calls = []

    def probe():
        calls.append(1)
        return next(responses)

    monkeypatch.setattr(rendering, "_probe_vtk_once", probe)
    result = rendering.vtk_capability()
    assert result == (True, {"device": "GPU"})
    result[1]["device"] = "caller change"
    assert rendering.vtk_capability() == (True, {"device": "GPU"})
    assert len(calls) == 2


def test_persistent_failure_is_cached_briefly_then_rechecked(cache, monkeypatch):
    calls = []

    def probe():
        calls.append(1)
        return (False, {"reason": "offline"}) if cache[0] < 30 else (True, {"device": "GPU"})

    monkeypatch.setattr(rendering, "_probe_vtk_once", probe)
    assert not rendering.vtk_capability()[0]
    assert len(calls) == 2
    cache[0] = 29
    assert not rendering.vtk_capability()[0]
    assert len(calls) == 2
    cache[0] = 30
    assert rendering.vtk_capability()[0]
    assert len(calls) == 3


def test_concurrent_requests_share_one_successful_probe(cache, monkeypatch):
    calls = []

    def probe():
        calls.append(1)
        return True, {"device": "GPU"}

    monkeypatch.setattr(rendering, "_probe_vtk_once", probe)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: rendering.vtk_capability(), range(8)))
    assert all(result[0] for result in results)
    assert len(calls) == 1


def test_child_probe_remains_bounded_and_reports_malformed_output(monkeypatch):
    calls = []

    def run(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(returncode=0, stdout="[]")

    monkeypatch.setattr(rendering.subprocess, "run", run)
    assert not rendering._probe_vtk_once()[0]
    assert calls[0]["timeout"] == 10
    assert calls[0]["capture_output"]
