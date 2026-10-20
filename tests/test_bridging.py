"""Boundary bridge geometry, public discovery, rollback and persistence."""

import numpy as np
import pytest
import trimesh

from scissors import Scissors
from scissors.api.errors import ScissorsError
from scissors.core.mesh import face_components, topology_mesh


def disks(app, counts=(4, 4)):
    scene = app.scenes.create()
    ids = []
    for z, count in enumerate(counts):
        angle = np.arange(count) * 2 * np.pi / count
        vertices = [[0, 0, z], *np.column_stack((np.cos(angle), np.sin(angle), np.full(count, z)))]
        faces = [[0, i + 1, (i + 1) % count + 1] for i in range(count)]
        if z == 0:
            faces = [face[::-1] for face in faces]
        result = app.engine.add_mesh(
            scene.id, trimesh.Trimesh(vertices=vertices, faces=faces, process=False), "Disk", "test"
        )
        ids.append(result["object_id"])
    return ids


def selection(app, object_id):
    result = app.call("mesh.boundaries", object_id=object_id)
    assert len(result["loops"]) == 1
    return {key: result[key] for key in ("object_id", "geometry_revision")} | {"loop_index": 0}


@pytest.mark.parametrize("counts", [(4, 4), (4, 7), (9, 3)])
def test_bridge_unequal_loops_are_closed_oriented_and_preserve_sources(app, counts):
    ids = disks(app, counts)
    before = [app.engine.object(oid)[1].model_dump() for oid in ids]
    result = app.call("mesh.bridge", first=selection(app, ids[0]), second=selection(app, ids[1]))
    obj = app.engine.object(result["object_id"])[1]
    topology = topology_mesh(obj.geometry())
    assert topology.is_watertight and topology.is_winding_consistent
    assert topology.volume > 0
    assert len(result["bridge_face_indices"]) == sum(counts)
    assert len(face_components(len(topology.faces), topology.face_adjacency)) == 1
    assert [app.engine.object(oid)[1].model_dump() for oid in ids] == before
    np.testing.assert_allclose(obj.vertices, [*before[0]["vertices"], *before[1]["vertices"]])


def test_bridge_attributes_and_restart(app):
    ids = disks(app, (5, 8))
    material = app.call("material.create", scene_id=app.engine.object(ids[0])[0].id)["id"]
    for oid in ids:
        app.call("uv.planar", object_id=oid)
        app.call("material.assign", object_id=oid, material_id=material)
        obj = app.engine.object(oid)[1]
        obj.vertex_colors = [[10, 20, 30, 255]] * len(obj.vertices)
    source = [app.engine.object(oid)[1].model_dump() for oid in ids]
    result = app.call(
        "mesh.bridge", first=selection(app, ids[0]), second=selection(app, ids[1]), replace=True
    )
    obj = app.engine.object(result["object_id"])[1]
    assert obj.uv == [*source[0]["uv"], *source[1]["uv"]]
    assert obj.vertex_colors == [*source[0]["vertex_colors"], *source[1]["vertex_colors"]]
    assert obj.material_id == material
    workspace = app.engine.root
    saved = obj.model_dump()
    app.close()
    with Scissors(workspace) as reopened:
        assert reopened.engine.object(obj.id)[1].model_dump() == saved
        assert all(oid not in reopened.engine.object(obj.id)[0].objects for oid in ids)


def test_stale_invalid_and_resource_failures_restore_state(app):
    ids = disks(app)
    first, second = [selection(app, oid) for oid in ids]
    app.call("mesh.translate", object_id=ids[0], z=-1)
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError, match="stale"):
        app.call("mesh.bridge", first=first, second=second)
    assert app.engine.snapshot() == before
    first = selection(app, ids[0])
    for invalid in [dict(second, loop_index=99), dict(second, loop_index=-1)]:
        with pytest.raises(ScissorsError):
            app.call("mesh.bridge", first=first, second=invalid)
        assert app.engine.snapshot() == before
    app.engine.settings.max_faces = 10
    with pytest.raises(ScissorsError) as error:
        app.call("mesh.bridge", first=first, second=second)
    assert error.value.code == "RESOURCE_LIMIT"
    assert app.engine.snapshot() == before


def test_boundary_query_is_read_only_and_closed_mesh_has_no_loops(app):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    before = app.engine.snapshot()
    assert app.call("mesh.boundaries", object_id=cube.id)["loops"] == []
    assert app.engine.snapshot() == before
    described = app.registry.describe_operation("mesh.boundaries")
    assert not described["side_effects"]
    assert described["permissions"] == ["scene.read"]
    bridge = app.registry.describe_operation("mesh.bridge")
    assert bridge["side_effects"] and bridge["permissions"] == ["scene.write"]
    assert bridge["parameters"]["$defs"]["BoundarySelection"]["required"] == [
        "object_id",
        "geometry_revision",
        "loop_index",
    ]


def test_mixed_uv_and_different_materials_need_explicit_policy(app):
    ids = disks(app)
    app.call("uv.planar", object_id=ids[0])
    first, second = [selection(app, oid) for oid in ids]
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError, match="Mixed UV"):
        app.call("mesh.bridge", first=first, second=second)
    assert app.engine.snapshot() == before
    material = app.call("material.create", scene_id=app.engine.object(ids[0])[0].id)["id"]
    app.call("material.assign", object_id=ids[0], material_id=material)
    with pytest.raises(ScissorsError, match="Different source materials"):
        app.call("mesh.bridge", first=first, second=second, uv_policy="discard")
    result = app.call(
        "mesh.bridge", first=first, second=second, uv_policy="discard", material_id=material
    )
    assert not result["has_uv"] and result["material_id"] == material


def test_bridge_render_seams_and_baked_transforms(app):
    ids = disks(app, (4, 6))
    for oid in ids:
        app.call("mesh.facet", object_id=oid)
        app.call("mesh.translate", object_id=oid, x=10)
    sources = [app.engine.object(oid)[1].model_dump() for oid in ids]
    result = app.call("mesh.bridge", first=selection(app, ids[0]), second=selection(app, ids[1]))
    obj = app.engine.object(result["object_id"])[1]
    assert result["watertight"]
    assert len(obj.vertices) == sum(len(source["vertices"]) for source in sources)
    np.testing.assert_array_equal(obj.transform, np.eye(4))
    np.testing.assert_array_equal(obj.vertices, [*sources[0]["vertices"], *sources[1]["vertices"]])


def test_bridge_leaves_unselected_opening(app):
    ids = disks(app)
    obj = app.engine.object(ids[0])[1]
    mesh = obj.geometry()
    other = mesh.copy()
    other.apply_translation([5, 0, 0])
    obj.replace(trimesh.util.concatenate([mesh, other]))
    query = app.call("mesh.boundaries", object_id=obj.id)
    assert len(query["loops"]) == 2
    first = {key: query[key] for key in ("object_id", "geometry_revision")} | {"loop_index": 0}
    result = app.call("mesh.bridge", first=first, second=selection(app, ids[1]))
    assert len(app.call("mesh.boundaries", object_id=result["object_id"])["loops"]) == 1


def test_invalid_offsets_same_mesh_and_coincident_contours_fail_atomically(app):
    ids = disks(app)
    first, second = [selection(app, oid) for oid in ids]
    before = app.engine.snapshot()
    for kwargs in [dict(second=first), dict(second=second, second_offset=100)]:
        with pytest.raises(ScissorsError):
            app.call("mesh.bridge", first=first, **kwargs)
        assert app.engine.snapshot() == before
    app.call("mesh.translate", object_id=ids[1], z=-1)
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError) as error:
        app.call("mesh.bridge", first=first, second=selection(app, ids[1]))
    assert error.value.code == "GEOMETRY_ERROR"
    assert app.engine.snapshot() == before


def test_failure_after_output_insertion_restores_sources(app, monkeypatch):
    ids = disks(app)
    first, second = [selection(app, oid) for oid in ids]
    before = app.engine.snapshot()
    checked = app.engine.checked
    calls = 0

    def fail_final_check(obj, **kwargs):
        nonlocal calls
        if obj.metadata.get("primitive") == "bridge":
            calls += 1
            if calls == 2:
                raise ScissorsError("GEOMETRY_ERROR", "Rejected output attributes")
        return checked(obj, **kwargs)

    monkeypatch.setattr(app.engine, "checked", fail_final_check)
    with pytest.raises(ScissorsError, match="Rejected output"):
        app.call("mesh.bridge", first=first, second=second, replace=True)
    assert calls == 2
    assert app.engine.snapshot() == before


def test_branched_boundary_is_rejected(app):
    ids = disks(app)
    obj = app.engine.object(ids[0])[1]
    # Two triangles meet only at a vertex. Their boundary cannot be one contour.
    obj.replace(
        trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]],
            faces=[[0, 1, 2], [0, 3, 4]],
            process=False,
        )
    )
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError, match="branched"):
        app.call("mesh.boundaries", object_id=obj.id)
    assert app.engine.snapshot() == before
