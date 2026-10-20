"""Closed-solid edge treatment, containment and attribute preservation."""

import manifold3d as manifold
import numpy as np
import pytest

from scissors.api.errors import ScissorsError
from scissors.core.mesh import topology_mesh
from scissors.core.simplification import topology_signature


@pytest.mark.parametrize("operation", ["mesh.bevel", "mesh.chamfer", "mesh.fillet"])
def test_existing_cube_edges_keep_bounds_and_material(app, operation):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    material = app.call("material.create", scene_id=scene.id)
    app.call("material.assign", object_id=cube.id, material_id=material["id"])
    kwargs = {"radius": 0.1} if operation == "mesh.fillet" else {"width": 0.1}
    result = app.call(operation, object_id=cube.id, **kwargs)
    assert result["watertight"] and 0 < result["volume"] < 1
    assert result["material_id"] == material["id"]
    np.testing.assert_allclose(result["bounds"], [[-0.5] * 3, [0.5] * 3], atol=1e-8)
    obj = app.engine.object(cube.id)[1]
    assert topology_signature(obj) == (2, 1, 0)
    assert topology_mesh(obj.geometry()).is_winding_consistent


def test_existing_prism_chamfer_uses_world_geometry_and_retains_source(app):
    scene = app.scenes.create()
    prism = app.call("mesh.create_prism", scene_id=scene.id, sides=5)
    obj = app.engine.object(prism["object_id"])[1]
    obj.transform = np.diag([2.0, 2.0, 2.0, 1.0]).tolist()
    before = obj.model_dump()
    result = app.call("mesh.chamfer", object_id=obj.id, width=0.03, replace=False)
    candidate = app.engine.object(result["object_id"])[1]
    assert obj.model_dump() == before
    assert candidate.id != obj.id and candidate.transform == obj.transform
    assert 0 < result["volume"] < prism["volume"]


def test_edge_projection_preserves_uv_and_vertex_color_seams(app):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    app.call("uv.generate", object_id=cube.id, method="box")
    obj = app.engine.object(cube.id)[1]
    obj.vertex_colors = [[255, 0, 0, 255]] * len(obj.vertices)
    result = app.call("mesh.fillet", object_id=obj.id, radius=0.1, segments=3)
    candidate = app.engine.object(obj.id)[1]
    assert result["uv_method"] == "nearest_source"
    assert len(candidate.uv) == len(candidate.vertices) == len(candidate.vertex_colors)
    assert np.isfinite(candidate.uv).all()
    assert all(color == [255, 0, 0, 255] for color in candidate.vertex_colors)


def test_open_mesh_and_excessive_width_restore_state(app):
    scene = app.scenes.create()
    plane = app.meshes.create_plane(scene_id=scene.id)
    cube = app.meshes.create_cube(scene_id=scene.id)
    before = app.engine.snapshot()
    for object_id, width in [(plane.id, 0.1), (cube.id, 2)]:
        with pytest.raises(ScissorsError):
            app.call("mesh.chamfer", object_id=object_id, width=width)
        assert app.engine.snapshot() == before


def test_edge_pair_budget_rejects_without_committing(app):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    app.engine.settings.max_geometry_pairs = 1
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError) as failure:
        app.call("mesh.chamfer", object_id=cube.id)
    assert failure.value.code == "RESOURCE_LIMIT"
    assert app.engine.snapshot() == before


def test_concave_mode_fills_reentrant_corner_and_keeps_genus(app):
    scene = app.scenes.create()
    profile = app.call(
        "mesh.create_profile",
        scene_id=scene.id,
        points=[[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]],
    )
    solid = app.call("mesh.extrude", object_id=profile["object_id"], distance=0.5)
    result = app.call(
        "mesh.fillet",
        object_id=solid["object_id"],
        radius=0.05,
        segments=3,
        mode="concave",
    )
    assert result["watertight"] and result["volume"] > solid["volume"]


@pytest.mark.parametrize("sides", [3, 5, 7])
@pytest.mark.parametrize("mode", ["convex", "concave"])
def test_asymmetric_kernel_respects_source_containment(app, sides, mode):
    scene = app.scenes.create()
    prism = app.call("mesh.create_prism", scene_id=scene.id, sides=sides)
    app.meshes.rotate(object_id=prism["object_id"], axis="z", angle_degrees=17)
    app.meshes.translate(object_id=prism["object_id"], x=2, y=-3)
    result = app.call(
        "mesh.chamfer", object_id=prism["object_id"], width=0.03, mode=mode, replace=False
    )

    def solid(object_id):
        mesh = topology_mesh(app.engine.object(object_id)[1].geometry())
        return manifold.Manifold(
            manifold.Mesh64(np.asarray(mesh.vertices), np.asarray(mesh.faces, dtype=np.uint64))
        )

    source, treated = solid(prism["object_id"]), solid(result["object_id"])
    assert source.status() == treated.status() == manifold.Error.NoError
    difference = treated - source if mode == "convex" else source - treated
    assert abs(difference.volume()) < source.volume() * 1e-10
    if mode == "convex":
        assert 0 < treated.volume() < source.volume()
