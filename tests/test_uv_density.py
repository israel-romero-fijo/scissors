"""Physical density targets, shared atlas capacity, and transactional UV changes."""

import numpy as np
import pytest

from scissors.api.errors import ScissorsError


def plane(app, sid, size=1):
    return app.call("mesh.create_plane", scene_id=sid, size=size)["id"]


@pytest.mark.parametrize("units,size", [("meters", 1), ("centimeters", 100), ("millimeters", 1000)])
def test_density_uses_physical_units_and_non_square_texture(app, units, size):
    sid = app.call("scene.create", units=units)["id"]
    oid = plane(app, sid, size)
    original = np.asarray(app.engine.object(oid)[1].vertices).copy()
    result = app.call(
        "uv.generate",
        object_id=oid,
        pixels_per_meter=64,
        texture_resolution=[512, 256],
        padding=4,
        atlas_region=[0.1, 0.1, 0.9, 0.9],
    )
    density = result["density"]["objects"][0]
    assert density["pixels_per_meter"] == pytest.approx(64)
    assert density["min_pixels_per_meter"] == pytest.approx(64)
    assert density["covered_area_fraction"] == 1
    # A square world surface stays square in pixel space, not normalized UV space.
    assert density["max_anisotropy"] == pytest.approx(1)
    obj = app.engine.object(oid)[1]
    np.testing.assert_allclose(obj.vertices, original)
    uv = np.asarray(obj.uv)
    assert np.all(uv >= 0.1) and np.all(uv <= 0.9)
    inspected = app.call("uv.inspect", object_id=oid, texture_resolution=[512, 256])
    assert inspected["density"]["pixels_per_meter"] == pytest.approx(64)


def test_multiple_objects_share_atlas_density_and_disjoint_rectangles(app):
    sid = app.call("scene.create")["id"]
    ids = [plane(app, sid, size) for size in [1, 2]]
    app.call("mesh.translate", object_id=ids[1], x=5)
    result = app.call(
        "uv.generate", object_ids=ids, pixels_per_meter=32, texture_resolution=[256, 256], padding=3
    )
    for report in result["density"]["objects"]:
        assert report["pixels_per_meter"] == pytest.approx(32)
    bounds = [np.array(app.engine.object(oid)[1].uv) for oid in ids]
    assert (
        bounds[0][:, 0].max() < bounds[1][:, 0].min()
        or (bounds[1][:, 0].max() < bounds[0][:, 0].min())
        or bounds[0][:, 1].max() < bounds[1][:, 1].min()
        or (bounds[1][:, 1].max() < bounds[0][:, 1].min())
    )


def test_repeat_target_can_exceed_one_tile(app):
    sid = app.call("scene.create")["id"]
    oid = plane(app, sid)
    app.call(
        "uv.generate",
        object_id=oid,
        pixels_per_meter=512,
        texture_resolution=[64, 64],
        density_mode="repeat",
    )
    obj = app.engine.object(oid)[1]
    assert np.ptp(np.asarray(obj.uv)[:, 0]) == pytest.approx(8)
    assert app.call("uv.inspect", object_id=oid, texture_resolution=[64, 64])["density"][
        "pixels_per_meter"
    ] == pytest.approx(512)


def test_capacity_failure_restores_every_object_and_persisted_state(app):
    sid = app.call("scene.create")["id"]
    ids = [plane(app, sid, 1), plane(app, sid, 20)]
    before = app.engine.snapshot()
    state = (app.engine.root / "state.json").read_bytes()
    with pytest.raises(ScissorsError) as failure:
        app.call("uv.generate", object_ids=ids, pixels_per_meter=100, texture_resolution=[128, 128])
    assert failure.value.details["reason"] == "ATLAS_CAPACITY"
    assert app.engine.snapshot() == before
    assert (app.engine.root / "state.json").read_bytes() == state


def test_preserve_uv_retains_material_colors_and_world_geometry(app):
    sid = app.call("scene.create")["id"]
    oid = app.call("mesh.create_cube", scene_id=sid)["id"]
    app.call("uv.generate", object_id=oid)
    obj = app.engine.object(oid)[1]
    obj.vertex_colors = [[128, 64, 32, 255] for _ in obj.vertices]
    before = {tuple(v) for v in obj.vertices}
    result = app.call(
        "uv.generate",
        object_id=oid,
        preserve_uv=True,
        pixels_per_meter=16,
        texture_resolution=[256, 256],
        padding=2,
    )
    after = app.engine.object(oid)[1]
    assert {tuple(v) for v in after.vertices} == before
    assert all(c == [128, 64, 32, 255] for c in after.vertex_colors)
    assert result["density"]["objects"][0]["island_count"] == 6
    assert result["density"]["objects"][0]["pixels_per_meter"] == pytest.approx(16)


@pytest.mark.parametrize(
    "arguments",
    [
        {"object_ids": []},
        {"pixels_per_meter": 0},
        {"pixels_per_meter": 10},
        {"pixels_per_meter": 10, "texture_resolution": [64, 64], "atlas_region": [0.8, 0, 0.2, 1]},
        {
            "pixels_per_meter": 10,
            "texture_resolution": [64, 64],
            "density_mode": "repeat",
            "padding": 1,
        },
    ],
)
def test_invalid_density_parameters_leave_state_unchanged(app, arguments):
    sid = app.call("scene.create")["id"]
    oid = plane(app, sid)
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError):
        app.call("uv.generate", **{"object_id": oid, **arguments})
    assert app.engine.snapshot() == before


def test_density_rejects_cross_scene_and_degenerate_projection(app):
    a, b = [app.call("scene.create")["id"] for _ in range(2)]
    ids = [plane(app, a), plane(app, b)]
    with pytest.raises(ScissorsError):
        app.call("uv.generate", object_ids=ids, pixels_per_meter=16, texture_resolution=[256, 256])
    cube = app.call("mesh.create_cube", scene_id=a)["id"]
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError):
        app.call(
            "uv.generate",
            object_id=cube,
            method="planar",
            pixels_per_meter=16,
            texture_resolution=[256, 256],
        )
    assert app.engine.snapshot() == before


def test_atlas_rejects_folded_chart_but_repeat_allows_overlap(app):
    sid = app.call("scene.create")["id"]
    angles = np.linspace(0, 2 * np.pi, 6)
    vertices = [[0, 0, 0], *[[float(np.cos(a)), float(np.sin(a)), 0] for a in angles]]
    oid = app.call(
        "mesh.create", scene_id=sid, vertices=vertices, faces=[[0, i, i + 1] for i in range(1, 6)]
    )["id"]
    # A connected UV fan wraps 450 degrees over a 360-degree world fan.
    obj = app.engine.object(oid)[1]
    obj.uv = [
        [0, 0],
        *[[float(np.cos(a)), float(np.sin(a))] for a in np.linspace(0, 2.5 * np.pi, 6)],
    ]
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError) as failure:
        app.call(
            "uv.generate",
            object_id=oid,
            preserve_uv=True,
            pixels_per_meter=8,
            texture_resolution=[128, 128],
        )
    assert failure.value.details["reason"] == "UV_OVERLAP"
    assert app.engine.snapshot() == before
    result = app.call(
        "uv.generate",
        object_id=oid,
        preserve_uv=True,
        pixels_per_meter=8,
        texture_resolution=[128, 128],
        density_mode="repeat",
    )
    assert result["density"]["objects"][0]["pixels_per_meter"] == pytest.approx(8)


def test_density_memory_limit_and_duplicate_ids_are_atomic(app):
    sid = app.call("scene.create")["id"]
    oid = plane(app, sid)
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError):
        app.call(
            "uv.generate", object_ids=[oid, oid], pixels_per_meter=16, texture_resolution=[256, 256]
        )
    assert app.engine.snapshot() == before

    app.engine.settings.max_memory_bytes = 1024
    with pytest.raises(ScissorsError) as failure:
        app.call("uv.generate", object_id=oid, pixels_per_meter=16, texture_resolution=[256, 256])
    assert failure.value.code == "RESOURCE_LIMIT"
    assert app.engine.snapshot() == before


def test_density_schema_and_atomic_scene_batch(app):
    from jsonschema import Draft202012Validator

    operation = app.registry.describe_operation("uv.generate")
    schema = operation["parameters"]
    for example in operation["examples"]:
        Draft202012Validator(schema).validate(example)
    assert operation["permissions"] == ["scene.write"]
    assert app.registry.describe_operation("uv.inspect")["permissions"] == ["scene.read"]
    assert not Draft202012Validator(schema).is_valid({})
    assert not Draft202012Validator(schema).is_valid({"object_ids": ["a"]})
    sid = app.call("scene.create")["id"]
    oid = plane(app, sid)
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError):
        app.call(
            "scene.edit",
            operations=[
                {
                    "operation": "uv.generate",
                    "arguments": {
                        "object_id": oid,
                        "pixels_per_meter": 16,
                        "texture_resolution": [256, 256],
                    },
                },
                {"operation": "mesh.translate", "arguments": {"object_id": "missing"}},
            ],
        )
    assert app.engine.snapshot() == before


def test_density_overlap_pair_budget_is_shared_by_charts(app):
    sid = app.call("scene.create")["id"]
    oid = app.call("mesh.create_cube", scene_id=sid)["id"]
    before = app.engine.snapshot()
    app.engine.settings.max_geometry_pairs = 1
    with pytest.raises(ScissorsError) as failure:
        app.call("uv.generate", object_id=oid, pixels_per_meter=16, texture_resolution=[256, 256])
    assert failure.value.code == "RESOURCE_LIMIT"
    assert app.engine.snapshot() == before


def test_density_fitting_timeout_is_atomic(app, monkeypatch):
    import scissors.core.uv_density as density

    sid = app.call("scene.create")["id"]
    oid = plane(app, sid)
    before = app.engine.snapshot()
    clock = iter([0, 100])
    monkeypatch.setattr(density.time, "monotonic", lambda: next(clock, 100))
    with pytest.raises(ScissorsError) as failure:
        app.call("uv.generate", object_id=oid, pixels_per_meter=16, texture_resolution=[256, 256])
    assert failure.value.code == "OPERATION_TIMEOUT"
    assert app.engine.snapshot() == before
