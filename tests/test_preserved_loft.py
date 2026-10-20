"""Preserved loft contracts. Run through the normal test suite when requested."""

import hashlib
import json

import numpy as np
import pytest

from scissors.api.errors import ScissorsError
from scissors.core.loft import prepare_preserved, profile_source

RECTANGLE = [[0, 0], [2, 0], [2, 1], [0, 1]]


def sections(app, holes=None):
    sid = app.call("scene.create")["id"]
    ids = []
    for z in [0, 1, 2]:
        oid = app.call("mesh.create_profile", scene_id=sid, points=RECTANGLE, holes=holes)["id"]
        app.call("mesh.translate", object_id=oid, z=z)
        ids.append(oid)
    return ids


def assert_source_points(app, result):
    details = result["loft"]
    output = np.asarray(app.engine.object(result["id"])[1].vertices)
    mapping = np.asarray(details["render_vertex_to_geometry_vertex"])
    for section in details["sections"]:
        source = np.asarray(app.engine.object(section["object_id"])[1].vertices)
        for ring in section["rings"]:
            for source_index, output_index in zip(
                ring["source_indices"], ring["geometry_vertex_indices"], strict=True
            ):
                vertices = output[mapping == output_index]
                assert len(vertices)
                np.testing.assert_array_equal(
                    vertices, np.repeat(source[source_index][None], len(vertices), axis=0)
                )


@pytest.mark.parametrize("smooth", [False, True])
def test_preserved_loft_retains_corners_and_sources(app, smooth):
    ids = sections(app)
    before = [app.engine.object(oid)[1].model_dump() for oid in ids]
    result = app.call("mesh.loft", object_ids=ids, preserve_vertices=True, smooth=smooth)
    assert result["watertight"] and result["volume"] == pytest.approx(4)
    assert result["loft"]["self_intersection_check"]["status"] == "passed"
    assert result["loft"]["geometry_revision"] == result["geometry_revision"]
    assert_source_points(app, result)
    assert before == [app.engine.object(oid)[1].model_dump() for oid in ids]
    assert [section["layer"] for section in result["loft"]["sections"]] == [
        i * (4 if smooth else 1) for i in range(3)
    ]


def test_preserved_loft_uses_explicit_start_and_winding(app):
    ids = sections(app)
    obj = app.engine.object(ids[1])[1]
    with app.transaction():
        ring = obj.metadata["boundary_rings"][0]
        obj.metadata["boundary_rings"] = [(ring[2:] + ring[:2])[::-1]]
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError):
        app.call("mesh.loft", object_ids=ids, preserve_vertices=True)
    assert app.engine.snapshot() == before
    result = app.call(
        "mesh.loft",
        object_ids=ids,
        preserve_vertices=True,
        correspondence=[
            {
                "object_id": obj.id,
                "geometry_revision": obj.geometry_revision,
                "rings": [{"start_index": 1, "reverse": True}],
            }
        ],
    )
    assert result["watertight"] and result["volume"] == pytest.approx(4)
    assert_source_points(app, result)


@pytest.mark.parametrize(
    "extra",
    [
        {"resample": 8},
        {"closed": True},
        {
            "correspondence": [
                {"object_id": "unrelated", "geometry_revision": "stale", "rings": [{}]}
            ]
        },
    ],
)
def test_preserved_loft_bad_options_roll_back(app, extra):
    ids = sections(app)[:2]
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError):
        app.call("mesh.loft", object_ids=ids, preserve_vertices=True, **extra)
    assert app.engine.snapshot() == before


def test_preserved_loft_rejects_unequal_counts(app):
    ids = sections(app)
    sid = app.engine.object(ids[0])[0].id
    other = app.call("mesh.create_profile", scene_id=sid, points=[[0, 0], [2, 0], [0, 1]])["id"]
    app.call("mesh.translate", object_id=other, z=3)
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError, match="equal corresponding vertex counts"):
        app.call("mesh.loft", object_ids=[ids[0], other], preserve_vertices=True)
    assert app.engine.snapshot() == before


def test_preserved_loft_requires_complete_revision_bound_hole_map(app):
    ids = sections(app, holes=[[[0.5, 0.2], [0.5, 0.8], [1.5, 0.8], [1.5, 0.2]]])
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError):
        app.call("mesh.loft", object_ids=ids, preserve_vertices=True)
    assert app.engine.snapshot() == before
    maps = [
        {
            "object_id": oid,
            "geometry_revision": app.engine.object(oid)[1].geometry_revision,
            "rings": [{"source_ring": 0}, {"source_ring": 1}],
        }
        for oid in ids
    ]
    result = app.call("mesh.loft", object_ids=ids, preserve_vertices=True, correspondence=maps)
    assert result["watertight"] and result["volume"] == pytest.approx(2.8)
    assert_source_points(app, result)
    maps[0]["geometry_revision"] = "stale"
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError, match="stale source revision"):
        app.call("mesh.loft", object_ids=ids, preserve_vertices=True, correspondence=maps)
    assert app.engine.snapshot() == before


@pytest.mark.parametrize(
    ("options", "digest"),
    [
        ({}, "d097875f9593013774e9d94c7aeb4da21dd2ec38327a4f756ef950f5c6594dac"),
        ({"resample": 8}, "e99f72202fd9ea78a6c66a10dfda1d05f61c6bbebfb084e1ed076d6c4eebe6d8"),
        ({"smooth": True}, "65e72334af620bf22eb1ee47c23d80f0e93f06244ae11b2b1492a8083cbd49a8"),
        ({"caps": False}, "115d08ec5e749f77e4cdb477834135d687b6ab4b88f1690eb65bcad6e7d619fb"),
    ],
)
def test_loft_default_arrays_match_reference(app, options, digest):
    result = app.call("mesh.loft", object_ids=sections(app), **options)
    obj = app.engine.object(result["id"])[1]
    data = {key: getattr(obj, key) for key in ["vertices", "faces", "normals", "uv"]}
    assert hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest() == digest


def test_preserved_loft_reports_bounded_surface_check(app):
    ids = sections(app)
    app.engine.settings.max_geometry_pairs = 20
    result = app.call("mesh.loft", object_ids=ids, preserve_vertices=True)
    assert result["loft"]["self_intersection_check"]["status"] == "unresolved"
    assert result["loft"]["warnings"]


@pytest.mark.parametrize("mode", ["repeated", "short_edge", "crossed"])
def test_preserved_section_validation_rejects_invalid_contours(app, mode):
    ids = sections(app)
    objects = [profile_source(app.engine.object(oid)[1], True) for oid in ids]
    obj = objects[0]
    ring = obj.metadata["boundary_rings"][0]
    if mode == "repeated":
        obj.vertices[ring[1]] = obj.vertices[ring[0]][:]
    elif mode == "short_edge":
        obj.vertices[ring[-1]] = [obj.vertices[ring[0]][0] + 1e-13, *obj.vertices[ring[0]][1:]]
    else:
        obj.vertices[ring[1]], obj.vertices[ring[2]] = obj.vertices[ring[2]], obj.vertices[ring[1]]
    with pytest.raises(ScissorsError) as error:
        prepare_preserved(objects, [], app.engine.settings)
    assert error.value.code == "GEOMETRY_ERROR"
