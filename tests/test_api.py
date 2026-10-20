import pytest
from fastapi.testclient import TestClient

from scissors.api.errors import ScissorsError
from scissors.http.server import create_app


def test_validation_errors_name_the_failing_field(app):
    scene = app.scenes.create(name="Errors")
    with pytest.raises(ScissorsError) as failure:
        app.call("material.create", scene_id=scene.id, base_color=[2.0, 0.0, 0.0, 1.0])
    assert failure.value.code == "INVALID_PARAMETER"
    assert "base_color[0]" in str(failure.value)
    assert failure.value.details["parameter"] == ["base_color", 0]

    with pytest.raises(ScissorsError) as failure:
        app.call("mesh.create_torus", scene_id=scene.id, major_radius=0.2, minor_radius=0.3)
    assert "minor_radius" in str(failure.value)

    envelope = app.registry.execute("mesh.create_cube", {"scene_id": scene.id, "size": -1})
    assert envelope["error"]["code"] == "INVALID_PARAMETER"
    assert "size" in envelope["error"]["message"]


def test_registry_discovery_validation(app):
    names = {op["name"] for op in app.registry.list_operations()}
    assert "mesh.create_cube" in names
    description = app.registry.describe_operation("mesh.create_cube")
    assert "scene_id" in description["parameters"]["required"]
    scene = app.scenes.create(name="API")
    assert app.registry.execute("mesh.create_cube", {"scene_id": scene.id})["success"]
    response = app.registry.execute("mesh.create_cube", {"scene_id": scene.id, "size": -1})
    assert response["error"]["code"] == "INVALID_PARAMETER"
    assert app.registry.execute("missing", {})["error"]["code"] == "UNKNOWN_OPERATION"
    assert (
        app.registry.execute("mesh.create_cube", {"scene_id": scene.id, "typo": 1})["error"]["code"]
        == "INVALID_PARAMETER"
    )


def test_delete_catalog_advertises_the_batch_parameter(app):
    operation = app.registry.describe_operation("mesh.delete")
    properties = operation["parameters"]["properties"]
    assert {"object_id", "object_ids"} <= set(properties)
    # Agents discover the fast path from the description, so it must name the batch.
    assert "object_ids" in operation["description"]
    assert "batch" in operation["description"].lower()


def test_http_shares_registry_and_state(app):
    with TestClient(create_app(app)) as client:
        assert client.get("/api/v1/info").status_code == 200
        assert (
            client.get("/api/v1/operations/mesh.create_cube").json()["name"] == "mesh.create_cube"
        )
        response = client.post(
            "/api/v1/operations/invoke",
            json={"name": "scene.create", "arguments": {"name": "REST"}},
        )
        sid = response.json()["result"]["id"]
        assert app.scenes.inspect(scene_id=sid).name == "REST"
        assert client.get(f"/api/v1/scenes/{sid}").json()["name"] == "REST"
        assert client.get("/openapi.json").status_code == 200
        assert client.get("/api/v1/plugins").json()["plugins"] == []
        assert client.get("/api/v1/scenes/missing").status_code == 404


def test_http_authentication_and_origin(app):
    with TestClient(create_app(app, token="local-token")) as client:
        assert client.get("/api/v1/info").status_code == 401
        headers = {"Authorization": "Bearer local-token"}
        assert client.get("/api/v1/info", headers=headers).status_code == 200
        assert (
            client.post(
                "/api/v1/operations/invoke",
                headers={**headers, "Origin": "http://evil.test"},
                json={"name": "scene.create"},
            ).status_code
            == 403
        )
        assert (
            client.get("/api/v1/info", headers={**headers, "Host": "evil.test"}).status_code == 403
        )


def test_validation_error_reports_field_expected_and_received(app):
    scene = app.scenes.create(name="Errors")
    with pytest.raises(ScissorsError) as failure:
        app.call("scene.preview", scene_id=scene.id, width=32, height=32, background=[16, 18, 24])
    error = failure.value
    assert error.code == "INVALID_PARAMETER"
    assert "background[0]" in str(error)
    assert "expected <= 1" in str(error)
    assert "received 16" in str(error)
    assert error.details["parameter"] == ["background", 0]
    assert error.details["expected"] == "<= 1"
    assert error.details["received"] == 16
    assert str(error.details["reason"]) in str(error)

    envelope = app.registry.execute(
        "scene.preview",
        {"scene_id": scene.id, "width": 32, "height": 32, "background": [16, 18, 24]},
    )
    assert envelope["error"]["details"]["expected"] == "<= 1"
    assert "received 16" in envelope["error"]["message"]
