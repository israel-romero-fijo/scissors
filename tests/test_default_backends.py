import importlib.metadata
import importlib.util
import tomllib
from pathlib import Path

from packaging.requirements import Requirement

from scissors import Scissors


def test_package_installs_all_geometry_backends_by_default(app):
    names = {"manifold3d", "xatlas", "vtk", "ufbx"}
    declared = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    direct = {
        Requirement(value).name: Requirement(value) for value in declared["project"]["dependencies"]
    }
    assert names <= direct.keys()
    installed = {
        Requirement(value).name
        for value in importlib.metadata.requires("scissors")
        if Requirement(value).marker is None
    }
    assert names <= installed
    assert not declared["project"].get("optional-dependencies")
    assert not importlib.metadata.metadata("scissors").get_all("Provides-Extra")
    info = app.call("modeler.info")
    assert all(info["backends"].values())
    assert {
        "mesh.boolean",
        "mesh.simplify",
        "mesh.organic_union",
        "mesh.remesh",
        "uv.generate",
        "uv.unwrap",
    } <= app.registry.operations.keys()
    methods = app.registry.describe_operation("uv.generate")["parameters"]["properties"]["method"]
    assert "xatlas" in methods["enum"]


def test_boolean_catalog_and_errors_in_incomplete_installation(tmp_path, monkeypatch):
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: None if name == "manifold3d" else find_spec(name)
    )
    with Scissors(tmp_path / "workspace") as app:
        assert "mesh.boolean" in app.registry.operations
        assert app.call("modeler.info")["backends"]["boolean"] is False
        scene = app.scenes.create()
        first = app.meshes.create_cube(scene_id=scene.id)
        second = app.meshes.create_cube(scene_id=scene.id)
        before = app.engine.snapshot()
        failed = app.registry.execute(
            "mesh.boolean", {"scene_id": scene.id, "object_ids": [first.id, second.id]}
        )
        assert failed["error"]["code"] == "BACKEND_UNAVAILABLE"
        assert failed["error"]["details"]["requirement"] == "Scissors standard installation"
        assert app.engine.snapshot() == before
