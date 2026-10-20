"""Damaged inputs, reconstructed solids and atomic publication through the registry."""

import numpy as np
import pytest
import trimesh

from scissors import Scissors
from scissors.api.errors import ScissorsError


def overlapping_boxes():
    first = trimesh.creation.box()
    second = trimesh.creation.box()
    second.apply_translation([0.4, 0.3, 0.2])
    return trimesh.util.concatenate([first, second])


def repair_data(app, mesh, **options):
    scene = app.scenes.create()
    return app.call(
        "mesh.repair_topology_data",
        scene_id=scene.id,
        vertices=mesh.vertices.tolist(),
        faces=mesh.faces.tolist(),
        voxel_size=0.2,
        **options,
    )


@pytest.mark.integration
def test_resolve_overlapping_solids_reopen_and_keep_identity(app):
    scene = app.scenes.create()
    mesh = overlapping_boxes()
    source = app.call(
        "mesh.create", scene_id=scene.id, vertices=mesh.vertices.tolist(), faces=mesh.faces.tolist()
    )
    oid = source["object_id"]
    app.call("mesh.translate", object_id=oid, x=4, z=2)
    material = app.call("material.create", scene_id=scene.id, name="Retained")
    app.call("material.assign", object_id=oid, material_id=material["id"])
    app.call("uv.planar", object_id=oid)
    transform = app.engine.object(oid)[1].transform
    assert app.call("mesh.self_intersect", object_id=oid)["self_intersects"]
    before = app.engine.object(oid)[1].geometry_revision
    result = app.call("mesh.repair_topology", object_id=oid, voxel_size=0.2)
    assert result["object_id"] == oid
    assert result["geometry_revision"] != before
    assert result["validation"]["valid"]
    props = result["validation"]["properties"]
    assert props["self_intersections"]["status"] == "passed"
    assert props["non_manifold_edges"] == 0
    assert props["vertex_links_manifold"]
    assert result["watertight"]
    assert result["material_id"] == material["id"]
    assert not result["has_uv"]
    assert app.engine.object(oid)[1].transform == transform
    assert 3 < result["bounds"][0][0] < 4
    assert 1 < result["bounds"][0][2] < 2
    # The result represents union occupancy, not a convex hull or two retained shells.
    assert 1.1 < result["volume"] < 1.9
    output = app.engine.object(oid)[1].geometry()
    path = app.engine.root / "repaired.stl"
    app.export.stl(scene_id=scene.id, output="repaired.stl")
    reopened = trimesh.load_mesh(path)
    assert reopened.is_volume
    root = app.engine.root
    app.close()
    with Scissors(root) as again:
        restored = again.engine.object(oid)[1]
        np.testing.assert_allclose(restored.vertices, output.vertices)
        assert restored.geometry().is_volume


@pytest.mark.integration
def test_nonmanifold_soup_never_published_and_duplicate_cleanup(app):
    mesh = overlapping_boxes()
    # An internal face shares existing edges with a closed box: >2 incident faces.
    faces = np.vstack([mesh.faces, mesh.faces[0], [0, 1, 6], [0, 0, 0]])
    mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=faces, process=False)
    result = repair_data(app, mesh)
    assert result["validation"]["valid"]
    assert result["removed_input_faces"] == 2
    assert len(app.engine.object(result["object_id"])[0].objects) == 1


@pytest.mark.integration
def test_open_crossed_panels_shell(app):
    vertices = [
        [-1, -1, 0],
        [1, -1, 0],
        [1, 1, 0],
        [-1, 1, 0],
        [0, -1, -1],
        [0, 1, -1],
        [0, 1, 1],
        [0, -1, 1],
    ]
    mesh = trimesh.Trimesh(
        vertices=vertices, faces=[[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], process=False
    )
    result = repair_data(app, mesh, mode="shell", shell_thickness=0.6)
    assert result["watertight"]
    assert result["validation"]["properties"]["surface_components"] == 1
    assert result["volume"] > 3


@pytest.mark.parametrize(
    "options",
    [
        {"mode": "shell"},
        {"mode": "solid", "shell_thickness": 1},
        {"mode": "shell", "shell_thickness": 0.2},
    ],
)
def test_invalid_policy_is_atomic(app, options):
    mesh = trimesh.creation.box()
    with pytest.raises(ScissorsError):
        repair_data(app, mesh, **options)
    assert all(not scene.objects for scene in app.engine.scenes.values())


@pytest.mark.parametrize("damage", ["index", "fractional", "nonfinite", "empty"])
def test_bad_arrays_rejected_without_publication(app, damage):
    mesh = trimesh.creation.box()
    vertices, faces = mesh.vertices.tolist(), mesh.faces.tolist()
    if damage == "index":
        faces[0][0] = 1000
    elif damage == "fractional":
        faces[0][0] = 1.5
    elif damage == "nonfinite":
        vertices[0][0] = float("inf")
    else:
        faces = []
    scene = app.scenes.create()
    before = (app.engine.root / "state.json").read_bytes()
    with pytest.raises(ScissorsError):
        app.call(
            "mesh.repair_topology_data",
            scene_id=scene.id,
            vertices=vertices,
            faces=faces,
            voxel_size=0.2,
        )
    assert not app.engine.scene(scene.id).objects
    assert (app.engine.root / "state.json").read_bytes() == before


@pytest.mark.parametrize("limit", ["pairs", "memory", "vertices"])
def test_resource_limits_restore_state(app, monkeypatch, limit):
    mesh = trimesh.creation.box()
    scene = app.scenes.create()
    before = (app.engine.root / "state.json").read_bytes()
    setting, value = {
        "pairs": ("max_geometry_pairs", 1),
        "memory": ("max_memory_bytes", 1000),
        "vertices": ("max_vertices", 3),
    }[limit]
    monkeypatch.setattr(app.engine.settings, setting, value)
    with pytest.raises(ScissorsError) as error:
        app.call(
            "mesh.repair_topology_data",
            scene_id=scene.id,
            vertices=mesh.vertices.tolist(),
            faces=mesh.faces.tolist(),
            voxel_size=0.2,
        )
    assert error.value.code == "RESOURCE_LIMIT"
    assert not app.engine.scene(scene.id).objects
    assert (app.engine.root / "state.json").read_bytes() == before


@pytest.mark.integration
def test_open_plane_solid_fails_with_actionable_error(app):
    mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]], process=False
    )
    with pytest.raises(ScissorsError) as error:
        repair_data(app, mesh)
    assert error.value.details["error"] == "empty_interior"
    assert "shell" in error.value.details["hint"]
    assert all(not scene.objects for scene in app.engine.scenes.values())


@pytest.mark.integration
def test_nested_cavity_keeps_inward_wall(app):
    outer = trimesh.creation.box(extents=[2, 2, 2])
    inner = trimesh.creation.box()
    inner.invert()
    result = repair_data(app, trimesh.util.concatenate([outer, inner]))
    props = result["validation"]["properties"]
    assert props["surface_components"] == 2
    assert min(props["component_signed_volumes"]) < 0
    assert 6 < result["volume"] < 8


@pytest.mark.integration
def test_backend_failure_restores_existing_mesh(app, monkeypatch):
    import scissors.core.topology_repair as repair

    scene = app.scenes.create()
    obj = app.meshes.create_cube(scene_id=scene.id)
    original = app.engine.object(obj.id)[1].model_dump()
    checkpoint = (app.engine.root / "state.json").read_bytes()

    def failure(*args):
        raise ScissorsError("OPERATION_TIMEOUT", "worker timeout")

    monkeypatch.setattr(repair, "run_backend", failure)
    with pytest.raises(ScissorsError) as error:
        app.call("mesh.repair_topology", object_id=obj.id, voxel_size=0.2)
    assert error.value.code == "OPERATION_TIMEOUT"
    assert app.engine.object(obj.id)[1].model_dump() == original
    assert (app.engine.root / "state.json").read_bytes() == checkpoint


def test_verifier_rejects_intersections_and_pinched_vertices(app):
    from scissors.core.topology_repair import verify_surface

    with pytest.raises(ScissorsError, match="self-intersects"):
        verify_surface(overlapping_boxes(), app.engine.settings)
    first = trimesh.creation.box()
    second = trimesh.creation.box()
    second.apply_translation([1, 1, 1])
    touching = trimesh.util.concatenate([first, second])
    with pytest.raises(ScissorsError, match="pinched vertex"):
        verify_surface(touching, app.engine.settings)


def test_catalog_exposes_structured_inputs_and_scene_permissions(app):
    for name in ("mesh.repair_topology", "mesh.repair_topology_data"):
        operation = app.registry.operations[name]
        assert operation.permissions == ["scene.write"]
        assert operation.side_effects
        assert operation.parameters["properties"]["mode"]["enum"] == ["solid", "shell"]
    arrays = app.registry.operations["mesh.repair_topology_data"].parameters["properties"]
    assert arrays["faces"]["items"]["items"]["type"] == "integer"


@pytest.mark.integration
def test_transaction_rolls_back_successful_reconstruction(app):
    scene = app.scenes.create()
    obj = app.meshes.create_cube(scene_id=scene.id)
    before = app.engine.object(obj.id)[1].model_dump()
    checkpoint = (app.engine.root / "state.json").read_bytes()
    with pytest.raises(RuntimeError, match="abort"):
        with app.transaction():
            app.call("mesh.repair_topology", object_id=obj.id, voxel_size=0.2)
            assert app.engine.object(obj.id)[1].geometry_revision != (
                f"{before['metadata']['_geometry_revision']}:"
                f"{before['metadata']['_geometry_signature']}"
            )
            raise RuntimeError("abort")
    assert app.engine.object(obj.id)[1].model_dump() == before
    assert (app.engine.root / "state.json").read_bytes() == checkpoint


@pytest.mark.integration
def test_output_geometry_budget_is_reported_and_does_not_publish(app, monkeypatch):
    scene = app.scenes.create()
    mesh = trimesh.creation.box()
    monkeypatch.setattr(app.engine.settings, "max_faces", 20)
    with pytest.raises(ScissorsError) as error:
        app.call(
            "mesh.repair_topology_data",
            scene_id=scene.id,
            vertices=mesh.vertices.tolist(),
            faces=mesh.faces.tolist(),
            voxel_size=0.2,
        )
    assert error.value.code == "RESOURCE_LIMIT"
    assert not app.engine.scene(scene.id).objects
