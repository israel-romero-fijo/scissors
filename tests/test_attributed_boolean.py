"""Attributed boolean regression cases. Unrun under the no-tests instruction."""

import numpy as np
import pytest

from scissors.api.errors import ScissorsError
from scissors.core.mesh import topology_mesh


def operands(app):
    scene = app.scenes.create()
    body = app.meshes.create_cube(scene_id=scene.id)
    app.call("uv.generate", object_id=body.id, method="box")
    cutter = app.meshes.create_cylinder(scene_id=scene.id, radius=0.2, height=2, sections=16)
    material = app.call("material.create", scene_id=scene.id)
    app.call("material.assign", object_id=body.id, material_id=material["id"])
    return scene.id, body.id, cutter.id, material["id"]


def test_difference_retains_first_material_uv_seams_and_normal_arrays(app):
    scene, body, cutter, material = operands(app)
    before = app.engine.object(body)[1].model_copy(deep=True)
    result = app.call("mesh.boolean", scene_id=scene, object_ids=[body, cutter], mode="difference")
    obj = app.engine.object(result["object_id"])[1]
    assert result["watertight"] and 0 < result["volume"] < 1
    assert obj.material_id == material and len(obj.uv) == len(obj.vertices)
    assert result["faces_by_source"][body] > 0 and result["faces_by_source"][cutter] > 0
    np.testing.assert_allclose(np.linalg.norm(obj.normals, axis=1), 1, atol=1e-7)
    assert topology_mesh(obj.geometry()).is_watertight
    assert app.engine.object(body)[1].model_dump() == before.model_dump()


def test_source_material_policy_returns_group_with_combined_closed_geometry(app):
    scene = app.scenes.create()
    a = app.meshes.create_cube(scene_id=scene.id)
    b = app.meshes.create_cube(scene_id=scene.id)
    app.meshes.translate(object_id=b.id, x=0.5)
    colors = [[1, 0, 0, 1], [0, 1, 0, 1]]
    materials = []
    for obj, color in zip([a, b], colors, strict=True):
        mat = app.call("material.create", scene_id=scene.id, base_color=color)
        app.call("material.assign", object_id=obj.id, material_id=mat["id"])
        materials.append(mat["id"])
    result = app.call(
        "mesh.boolean", scene_id=scene.id, object_ids=[a.id, b.id], material_policy="source"
    )
    root = app.engine.object(result["object_id"])[1]
    assert root.kind == "group" and result["combined"]["watertight"]
    assert result["combined"]["volume"] == pytest.approx(1.5)
    children = [app.engine.object(oid)[1] for oid in result["object_ids"]]
    assert {obj.material_id for obj in children} == set(materials)
    assert all(obj.parent_id == root.id for obj in children)


def test_cut_material_and_cutter_uv_policy(app):
    scene, body, cutter, original = operands(app)
    app.call("uv.generate", object_id=cutter, method="cylindrical")
    cut = app.call("material.create", scene_id=scene, name="Cut face")
    result = app.call(
        "mesh.boolean",
        scene_id=scene,
        object_ids=[body, cutter],
        mode="difference",
        cut_material_id=cut["id"],
        cut_uv="cutter",
    )
    materials = {app.engine.object(oid)[1].material_id for oid in result["object_ids"]}
    assert materials == {original, cut["id"]}
    assert result["cut_uv"] == "cutter"


def test_geometry_only_opt_out_still_accepts_uv_seamed_closed_inputs(app):
    scene, body, cutter, _ = operands(app)
    result = app.call(
        "mesh.boolean",
        scene_id=scene,
        object_ids=[body, cutter],
        mode="difference",
        preserve_attributes=False,
    )
    assert result["watertight"] and not result["has_uv"] and result["material_id"] is None


def test_boolean_failures_restore_workspace_and_validate_ownership(app):
    scene, body, _, _ = operands(app)
    other = app.scenes.create()
    foreign = app.meshes.create_cube(scene_id=other.id)
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError):
        app.call("mesh.boolean", scene_id=scene, object_ids=[body, foreign.id])
    assert app.engine.snapshot() == before
    far = app.meshes.create_cube(scene_id=scene)
    app.meshes.translate(object_id=far.id, x=10)
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError):
        app.call("mesh.boolean", scene_id=scene, object_ids=[body, far.id], mode="intersection")
    assert app.engine.snapshot() == before


def test_boolean_deletes_inputs_only_when_requested(app):
    scene = app.scenes.create()
    body = app.meshes.create_cube(scene_id=scene.id, size=2.0)
    cutters = [
        app.meshes.create_cylinder(scene_id=scene.id, radius=0.2, height=3, sections=16).id
        for _ in range(3)
    ]
    inputs = [body.id, *cutters]
    kept = app.call("mesh.boolean", scene_id=scene.id, object_ids=inputs, mode="difference")
    assert "deleted_object_ids" not in kept
    assert set(inputs) <= set(app.engine.scene(scene.id).objects)

    consumed = app.call(
        "mesh.boolean",
        scene_id=scene.id,
        object_ids=inputs,
        mode="difference",
        delete_inputs=True,
    )
    assert set(consumed["deleted_object_ids"]) == set(inputs)
    remaining = set(app.engine.scene(scene.id).objects)
    assert not set(inputs) & remaining
    assert set(consumed["object_ids"]) <= remaining
