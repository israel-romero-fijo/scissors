"""Visibility contract regression cases, retained for future test runs."""

import copy

import pytest

from scissors.api.errors import ScissorsError


def obstacle(app):
    scene = app.scenes.create()
    obj = app.meshes.create_cube(scene_id=scene.id)
    app.meshes.translate(object_id=obj.id, x=-3)
    args = dict(
        scene_id=scene.id,
        eyes=[{"name": "player", "position": [0, 0, 0]}],
        horizontal_range=[-30, 30],
        vertical_range=[0, 0],
        sampling={"type": "steps", "horizontal": 15, "vertical": 5},
        max_distance=5,
        detail="samples",
    )
    return obj.id, args


def test_nearest_hit_and_read_only_determinism(app):
    oid, args = obstacle(app)
    state = app.engine.snapshot()
    first = app.call("scene.visibility_report", **args)
    assert first == app.call("scene.visibility_report", **args)
    assert app.engine.snapshot() == state
    assert first["coverage"]["clear"] == 4
    assert first["coverage"]["blocked"] == 1
    assert first["coverage"]["unresolved"] == 0
    sample = first["eyes"][0]["samples"][2]
    assert sample["object_id"] == oid
    assert sample["distance"] == pytest.approx(2.5)


def test_comparison_requires_matching_measurement_configuration(app):
    oid, args = obstacle(app)
    before = app.call("scene.visibility_report", **args)
    app.meshes.translate(object_id=oid, z=3)
    after = app.call("scene.visibility_report", **args, baseline=before)
    assert after["comparison"]["eyes"][0]["clear_delta"] == 1
    assert after["scene_revision"] != before["scene_revision"]
    assert after["configuration_fingerprint"] == before["configuration_fingerprint"]
    with pytest.raises(ScissorsError, match="same scene and measurement configuration"):
        app.call("scene.visibility_report", **{**args, "max_distance": 6}, baseline=before)
    invalid = copy.deepcopy(before)
    invalid["eyes"][0]["clear"] = 99
    with pytest.raises(ScissorsError, match="sample totals"):
        app.call("scene.visibility_report", **args, baseline=invalid)


def test_filters_clipping_and_geometric_transparency_policy(app):
    oid, args = obstacle(app)
    material = app.call("material.create", scene_id=args["scene_id"], name="Glass", alpha=0)
    app.call("material.assign", object_id=oid, material_id=material["id"])
    assert app.call("scene.visibility_report", **args)["coverage"]["blocked"] == 1
    assert (
        app.call("scene.visibility_report", **args, exclude_ids=[oid])["coverage"]["blocked"] == 0
    )
    assert (
        app.call("scene.visibility_report", **{**args, "max_distance": 2})["coverage"]["blocked"]
        == 0
    )
    assert app.call("scene.visibility_report", **args, near=4)["coverage"]["blocked"] == 0


def test_head_offsets_use_each_eye_origin(app):
    _, args = obstacle(app)
    result = app.call(
        "scene.visibility_report",
        **args,
        head_offsets=[
            {"name": "low", "offset": [0, 0, -2]},
            {"name": "neutral", "offset": [0, 0, 0]},
        ],
    )
    by_name = {eye["name"]: eye for eye in result["eyes"]}
    assert by_name["player@low"]["blocked"] == 0
    assert by_name["player@neutral"]["blocked"] == 1
    assert by_name["player@low"]["position"] == [0, 0, -2]


def test_resource_exhaustion_is_not_a_clear_result(app):
    _, args = obstacle(app)
    limit = app.engine.settings.max_geometry_pairs
    state = app.engine.snapshot()
    app.engine.settings.max_geometry_pairs = 1
    try:
        result = app.call("scene.visibility_report", **args)
        assert result["status"] == "uncompleted"
        assert result["coverage"]["unresolved"] > 0
        assert result["work"]["stopped_reason"] == "RESOURCE_LIMIT"
        assert app.engine.snapshot() == state
    finally:
        app.engine.settings.max_geometry_pairs = limit


def test_coplanar_surface_contact_is_unresolved(app):
    scene = app.scenes.create()
    app.call(
        "mesh.create",
        scene_id=scene.id,
        vertices=[[-2, -1, -1], [-2, 1, -1], [-2, 0, 1]],
        faces=[[0, 1, 2]],
    )
    result = app.call(
        "scene.visibility_report",
        scene_id=scene.id,
        eyes=[{"name": "edge", "position": [-2, 0, 0]}],
        forward=[0, 1, 0],
        horizontal_range=[0, 0],
        vertical_range=[0, 0],
        sampling={"type": "counts", "horizontal": 1, "vertical": 1},
    )
    assert result["coverage"]["unresolved"] == 1
    assert result["eyes"][0]["unresolved_reasons"]["coplanar_or_near_parallel_contact"] == 1


def test_step_grid_stays_within_bounds_and_resolves_zero(app):
    scene = app.scenes.create()
    args = dict(
        scene_id=scene.id,
        eyes=[{"name": "player", "position": [0, 0, 0]}],
        vertical_range=[0, 0],
        sampling={"type": "steps", "horizontal": 0.1, "vertical": 1},
    )
    result = app.call("scene.visibility_report", **args, horizontal_range=[-0.3, 0.3])
    assert result["eyes"][0]["central_horizontal_span_degrees"] == [-0.3, 0.3]
    full_axis = app.call("scene.visibility_report", **args, horizontal_range=[0, 102.35])
    assert full_axis["grid"]["horizontal_count"] == 1024
    assert full_axis["grid"]["sampled_horizontal_span"][1] <= 102.35


@pytest.mark.parametrize(
    "changes",
    [
        {"up": [-1, 0, 0]},
        {"forward": [0, 0, 0]},
        {"horizontal_range": [20, -20]},
        {"near": 5},
        {"eyes": [{"name": "same", "position": [0, 0, 0]}] * 2},
        {
            "horizontal_range": [-180, 180],
            "vertical_range": [-90, 90],
            "sampling": {"type": "counts", "horizontal": 1024, "vertical": 1024},
        },
    ],
)
def test_invalid_requests_fail_without_changes(app, changes):
    _, args = obstacle(app)
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError):
        app.call("scene.visibility_report", **{**args, **changes})
    assert before == app.engine.snapshot()
