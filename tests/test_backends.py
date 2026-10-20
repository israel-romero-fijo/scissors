import importlib.util

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    importlib.util.find_spec("manifold3d") is None, reason="required manifold3d backend absent"
)
def test_manifold_boolean(app):
    scene = app.scenes.create(name="Boolean")
    a = app.meshes.create_cube(scene_id=scene.id, size=2)
    b = app.meshes.create_cube(scene_id=scene.id, size=2)
    app.meshes.translate(object_id=b.id, x=1)
    union = app.call("mesh.boolean", scene_id=scene.id, object_ids=[a.id, b.id])
    assert abs(union["volume"] - 12) < 1e-6
    assert app.meshes.validate(object_id=union["id"], require_closed=True).valid


@pytest.mark.skipif(
    importlib.util.find_spec("xatlas") is None, reason="required xatlas backend absent"
)
def test_xatlas_uv(app):
    scene = app.scenes.create(name="Atlas")
    obj = app.meshes.create_cube(scene_id=scene.id)
    app.uv.generate(object_id=obj.id, method="xatlas")
    uv = app.uv.inspect(object_id=obj.id)
    assert uv.has_uv and uv.validation["valid"]
    assert all(0 <= value <= 1 for row in uv.bounds for value in row)
