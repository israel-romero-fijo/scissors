"""Surface-edit regression cases. Unrun under the no-tests instruction."""

import numpy as np
import pytest

from scissors.api.errors import ScissorsError
from scissors.core.mesh import topology_mesh


def test_shell_plane_closes_boundaries_preserves_uv_and_material(app):
    scene = app.scenes.create()
    plane = app.meshes.create_plane(scene_id=scene.id, size=2)
    app.uv.planar(object_id=plane.id)
    material = app.call("material.create", scene_id=scene.id)
    app.call("material.assign", object_id=plane.id, material_id=material["id"])
    result = app.call("mesh.shell", object_id=plane.id, thickness=0.1)
    obj = app.engine.object(plane.id)[1]
    assert result["watertight"] and result["boundary_walls"] == 4
    assert result["volume"] == pytest.approx(0.4)
    assert obj.material_id == material["id"] and len(obj.uv) == len(obj.vertices)
    np.testing.assert_allclose(result["bounds"], [[-1, -1, -0.05], [1, 1, 0.05]])
    assert topology_mesh(obj.geometry()).is_winding_consistent


def test_miter_offset_changes_face_distance_and_failure_restores_state(app):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    result = app.call("mesh.offset", object_id=cube.id, distance=0.1)
    np.testing.assert_allclose(result["bounds"], [[-0.6, -0.6, -0.6], [0.6, 0.6, 0.6]])
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError):
        app.call("mesh.offset", object_id=cube.id, distance=-2)
    assert app.engine.snapshot() == before


def test_closed_cube_shell_has_inner_volume_and_opposite_winding(app):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    result = app.call("mesh.shell", object_id=cube.id, thickness=0.1, offset=-1)
    assert result["watertight"] and result["boundary_walls"] == 0
    assert result["volume"] == pytest.approx(1 - 0.8**3)


@pytest.mark.parametrize("normal", [[1, 0, 0], [1, 1, 0]])
def test_capped_clip_is_closed_and_retains_uv_material_and_diagnostic_transform(app, normal):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    app.call("uv.generate", object_id=cube.id, method="box")
    material = app.call("material.create", scene_id=scene.id)
    app.call("material.assign", object_id=cube.id, material_id=material["id"])
    transform = app.engine.object(cube.id)[1].transform
    result = app.call("mesh.clip", object_id=cube.id, normal=normal)
    obj = app.engine.object(cube.id)[1]
    assert result["watertight"] and result["cap_faces"] > 0
    assert result["volume"] == pytest.approx(0.5)
    assert obj.material_id == material["id"] and obj.transform == transform
    assert len(obj.uv) == len(obj.vertices)
    assert topology_mesh(obj.geometry()).is_winding_consistent


def test_capped_plate_cut_keeps_through_hole(app):
    scene = app.scenes.create()
    plate = app.call(
        "mesh.create_plate",
        scene_id=scene.id,
        size=[2, 2],
        thickness=0.2,
        holes=[{"center": [0, 0], "radius": 0.3}],
    )
    result = app.call(
        "mesh.clip", object_id=plate["object_id"], normal=[0, 0, 1], point=[0, 0, 0.1]
    )
    assert result["watertight"] and result["volume"] == pytest.approx(plate["volume"] / 2)
    obj = app.engine.object(plate["object_id"])[1]
    faces = obj.geometry().triangles
    cap = faces[np.all(np.isclose(faces[:, :, 2], 0.1), axis=1)]
    assert len(cap) and np.min(np.linalg.norm(cap.mean(axis=1)[:, :2], axis=1)) > 0.25


def test_slice_returns_two_closed_parts_and_keeps_source_by_default(app):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    before = app.engine.object(cube.id)[1].vertices
    result = app.call("mesh.slice", object_id=cube.id, normal=[0, 1, 0])
    assert not result["source_replaced"]
    for key in ["positive", "negative"]:
        assert result[key]["object_id"] != cube.id
        assert result[key]["watertight"] and result[key]["volume"] == pytest.approx(0.5)
    assert app.engine.object(cube.id)[1].vertices == before


def test_open_surface_clip_can_remain_uncapped_and_empty_cut_rolls_back(app):
    scene = app.scenes.create()
    plane = app.meshes.create_plane(scene_id=scene.id, size=1.0)
    result = app.call("mesh.clip", object_id=plane.id, normal=[1, 0, 0], capped=False)
    assert not result["watertight"] and result["area"] == pytest.approx(0.5)
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError):
        app.call("mesh.clip", object_id=plane.id, normal=[1, 0, 0], point=[2, 0, 0])
    assert app.engine.snapshot() == before


def test_capped_clip_on_a_hollow_shell_reports_boundary_details(app):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id, size=1.0)
    app.call("mesh.shell", object_id=cube.id, thickness=0.45, offset=0)
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError) as failure:
        app.call(
            "mesh.clip",
            object_id=cube.id,
            normal=[1, 1, 1],
            point=[0, 0, 0],
            capped=True,
            replace=False,
            name="cut",
        )
    error = failure.value
    assert error.code == "GEOMETRY_ERROR"
    assert str(error) == "Capped cut did not remain watertight"
    details = error.details
    assert details["non_manifold_edges"] >= 1
    assert details["boundary_edges"] == 0
    assert details["loops_found"] == 0
    assert details["open_loops"] == 0
    assert details["cap_faces"] > 0
    assert "capped=false" in details["suggestion"]
    assert app.engine.snapshot() == before


def test_boundary_report_counts_boundary_loops():
    import trimesh

    from scissors.api.surfaces import _boundary_report

    cube = trimesh.creation.box(extents=[1, 1, 1])
    open_cube = trimesh.Trimesh(vertices=cube.vertices, faces=cube.faces[1:], process=False)
    report = _boundary_report(open_cube, {"cap_faces": 0})
    assert report["boundary_edges"] == 3
    assert report["loops_found"] == 1
    assert report["closed_loops"] == 1
    assert report["open_loops"] == 0
    assert report["non_manifold_edges"] == 0
