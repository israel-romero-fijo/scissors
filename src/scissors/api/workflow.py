"""Scene copy workflow contracts shared by every transport."""

import copy
from typing import Annotated, Any

from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field

from scissors.api.errors import ScissorsError
from scissors.api.registry import OperationRegistry
from scissors.core import checkpoints
from scissors.core.engine import Engine, identifier
from scissors.core.scene import Scene

CheckpointName = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")]
Finite = Annotated[float, Field(allow_inf_nan=False)]
Vector = Annotated[list[Finite], Field(min_length=3, max_length=3)]


class Placement(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    position: Vector | None = None
    rotation: Vector | None = None
    scale: (
        Annotated[list[Annotated[float, Field(gt=0)]], Field(min_length=3, max_length=3)] | None
    ) = None
    matrix: (
        Annotated[
            list[Annotated[list[Finite], Field(min_length=4, max_length=4)]],
            Field(min_length=4, max_length=4),
        ]
        | None
    ) = None


class FigureRow(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    figure: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    name: str | None = None
    transform: Placement | None = None
    material_id: str | None = None
    parent_id: str | None = None
    repeat: Annotated[int, Field(ge=1, le=4096)] = 1


MAX_TABLE_OBJECTS = 20_000


def register_workflow(registry: OperationRegistry, engine: Engine) -> None:
    def create_table(
        scene_id: str,
        rows: Annotated[list[FigureRow], Field(min_length=1)],
    ) -> dict[str, Any]:
        engine.scene(scene_id)
        parsed = [FigureRow.model_validate(row) for row in rows]
        catalog = registry.execute("mesh.figures")
        if not catalog["success"]:
            raise ScissorsError("UNKNOWN_OPERATION", "Native figure discovery is unavailable")
        allowed = {entry["operation"] for entry in catalog["result"]["figures"]}
        prepared = []
        for index, row in enumerate(parsed):
            operation = (
                row.figure if row.figure.startswith("mesh.create_") else f"mesh.create_{row.figure}"
            )
            op = registry.operations.get(operation)
            if operation not in allowed or op is None or op.plugin is not None:
                raise ScissorsError(
                    "INVALID_PARAMETER",
                    "Table row must use a native figure",
                    {"row": index, "figure": row.figure},
                )
            if any(
                permission not in {"scene.read", "scene.write", "filesystem.read"}
                for permission in op.permissions
            ):
                raise ValueError("Table figures may only change scene state")
            if "scene_id" in row.parameters:
                raise ValueError("Set the table scene_id once, outside row parameters")
            arguments = {**row.parameters, "scene_id": scene_id}
            if row.name is not None:
                arguments["name"] = row.name
            Draft202012Validator(op.parameters).validate(arguments)
            if row.material_id and engine.material(row.material_id)[0].id != scene_id:
                raise ValueError("Table material belongs to another scene")
            if row.parent_id and engine.object(row.parent_id)[0].id != scene_id:
                raise ValueError("Table parent belongs to another scene")
            if (
                row.transform
                and row.transform.matrix is not None
                and any(
                    getattr(row.transform, key) is not None
                    for key in ["position", "rotation", "scale"]
                )
            ):
                raise ValueError("Choose a placement matrix or position/rotation/scale")
            prepared.append((operation, arguments, row))

        def invoke(operation: str, arguments: dict[str, Any], index: int) -> dict[str, Any]:
            result = registry.execute(operation, arguments)
            if not result["success"]:
                raise ScissorsError(
                    "BATCH_FAILED",
                    "Figure table rolled back",
                    {"row": index, "operation": operation, "error": result["error"]},
                )
            return result["result"]

        created = []
        total = sum(row.repeat for _, _, row in prepared)
        if total > MAX_TABLE_OBJECTS:
            raise ScissorsError(
                "RESOURCE_LIMIT",
                "Figure table exceeds the object budget",
                {"requested": total, "limit": MAX_TABLE_OBJECTS},
            )
        # The outer registry call owns rollback and commit. Inner calls still validate schemas.
        registry.batch_depth += 1
        try:
            for index, (operation, arguments, row) in enumerate(prepared):
                for copy_index in range(row.repeat):
                    obj = invoke(operation, arguments, index)
                    oid = obj["object_id"]
                    if row.transform is not None:
                        placement = row.transform.model_dump(exclude_none=True)
                        method = "mesh.set_transform" if "matrix" in placement else "mesh.place"
                        invoke(method, {"object_id": oid, **placement}, index)
                    if row.material_id:
                        invoke(
                            "material.assign",
                            {"object_id": oid, "material_id": row.material_id},
                            index,
                        )
                    if row.parent_id:
                        invoke(
                            "object.parent", {"object_id": oid, "parent_id": row.parent_id}, index
                        )
                    created.append(
                        {
                            "row": index,
                            "repeat_index": copy_index,
                            "object_id": oid,
                            "figure": row.figure,
                            "name": obj["name"],
                        }
                    )
        finally:
            registry.batch_depth -= 1
        return {
            "scene_id": scene_id,
            "created_count": len(created),
            "objects": created,
            "atomic": True,
        }

    registry.add_function(
        "scene.create_table",
        create_table,
        "Create native figures atomically with typed placement/material/name rows. Each row "
        "accepts repeat to emit several copies; the call is bounded by the object budget.",
        permissions=["scene.write", "filesystem.read"],
    )

    def checkpoint(scene_id: str, name: CheckpointName) -> dict[str, Any]:
        return checkpoints.save(engine, scene_id, name)

    def revert_to(scene_id: str, name: CheckpointName) -> dict[str, Any]:
        return checkpoints.revert(engine, scene_id, name)

    def compare_snapshot(scene_id: str, name: CheckpointName) -> dict[str, Any]:
        return checkpoints.compare(engine, scene_id, name)

    registry.add_function(
        "scene.checkpoint",
        checkpoint,
        "Atomically save or replace a named scene checkpoint with stable IDs.",
        side_effects=False,
        permissions=["scene.read", "filesystem.write"],
    )
    registry.add_function(
        "scene.revert_to",
        revert_to,
        "Restore a named checkpoint for this scene, preserving stable IDs.",
        permissions=["scene.write", "filesystem.read"],
    )
    registry.add_function(
        "scene.compare_snapshot",
        compare_snapshot,
        "Compare current scene objects/materials against a named checkpoint by stable ID.",
        side_effects=False,
        permissions=["scene.read", "filesystem.read"],
    )

    def copy_objects(
        from_scene: str,
        to_scene: str,
        object_ids: Annotated[list[str], Field(min_length=1, max_length=4096)],
    ) -> dict[str, Any]:
        return engine.copy_objects(from_scene, to_scene, object_ids)

    def clone(scene_id: str, name: str | None = None) -> dict[str, Any]:
        original = engine.scene(scene_id)
        duplicate = Scene(
            id=identifier("scene"),
            name=name or f"{original.name} copy",
            units=original.units,
            coordinates=original.coordinates,
            metadata=copy.deepcopy(original.metadata),
        )
        engine.scenes[duplicate.id] = duplicate
        result = engine.copy_objects(
            scene_id, duplicate.id, list(original.objects), include_unused_materials=True
        )
        return {**duplicate.summary(), **result}

    registry.add_function(
        "scene.clone",
        clone,
        "Copy a scene with fresh IDs, independent materials and shared workspace textures.",
    )
    registry.add_function(
        "scene.copy_objects",
        copy_objects,
        "Copy selected subtrees in world space, remapping parents and material ownership.",
    )
