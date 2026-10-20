import numpy as np
import pytest


def test_scene_cube_transform_data_validation(app):
    scene = app.scenes.create(name="Test")
    cube = app.meshes.create_cube(scene_id=scene.id, size=2)
    moved = app.meshes.translate(object_id=cube.id, x=3)
    np.testing.assert_allclose(moved.bounds, [[2, -1, -1], [4, 1, 1]])
    app.meshes.rotate(object_id=cube.id, axis="x", angle_degrees=90)
    scaled = app.meshes.scale(object_id=cube.id, z=2)
    assert scaled.vertex_count == 8 and scaled.face_count == 12
    data = app.meshes.data(object_id=cube.id)
    assert len(data.vertices) == 8 and len(data.faces) == 12
    assert app.meshes.validate(object_id=cube.id, require_closed=True).valid
    assert app.scenes.inspect(scene_id=scene.id).objects[0]["id"] == cube.id


@pytest.mark.parametrize("primitive", ["sphere", "cylinder", "cone", "torus", "plane"])
def test_primitives_uv(app, primitive):
    scene = app.scenes.create(name=primitive)
    obj = getattr(app.meshes, f"create_{primitive}")(scene_id=scene.id)
    app.uv.generate(object_id=obj.id)
    assert app.meshes.validate(object_id=obj.id).valid
    assert app.uv.inspect(object_id=obj.id).has_uv
    if primitive == "plane":
        report = app.meshes.validate(object_id=obj.id, require_closed=True)
        assert not report.valid and "NOT_CLOSED" in report.errors
    else:
        assert app.meshes.validate(object_id=obj.id, require_closed=True).valid


def test_duplicate_merge_subdivide(app):
    scene = app.scenes.create(name="edit")
    obj = app.meshes.create_cube(scene_id=scene.id)
    duplicate = app.meshes.duplicate(object_id=obj.id)
    assert duplicate.id != obj.id
    app.meshes.translate(object_id=duplicate.id, x=2)
    merged = app.meshes.merge(scene_id=scene.id, object_ids=[obj.id, duplicate.id])
    assert merged.face_count == 24
    subdivided = app.meshes.subdivide(object_id=obj.id)
    assert subdivided.face_count == 48
    assert app.meshes.validate(object_id=obj.id, require_closed=True).valid


def test_validation_detects_corrupt_geometry(app):
    from scissors.core.mesh import MeshObject
    from scissors.core.validation import validate_mesh

    settings = app.engine.settings
    bad = MeshObject(id="bad", name="bad", vertices=[[0, 0, 0]], faces=[[0, 1, 2]])
    assert "INDEX_OUT_OF_RANGE" in validate_mesh(bad, settings)["errors"]
    empty = MeshObject(id="empty", name="empty", vertices=[], faces=[])
    assert "EMPTY_GEOMETRY" in validate_mesh(empty, settings)["errors"]
    bad.faces = [[0, 0, 0]]
    assert "DEGENERATE_FACES" in validate_mesh(bad, settings)["errors"]
