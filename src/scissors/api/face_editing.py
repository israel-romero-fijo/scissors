"""Local editing operations through the common operation registry."""

from typing import Annotated, Any, Literal

from pydantic import Field

from scissors.api.inspection import FaceSelection
from scissors.api.registry import OperationRegistry
from scissors.core.engine import Engine
from scissors.core.face_editing import edit_faces

Finite = Annotated[float, Field(allow_inf_nan=False)]
Vector = Annotated[list[Finite], Field(min_length=3, max_length=3)]


def register_face_editing(registry: OperationRegistry, engine: Engine) -> None:
    def move_faces(
        object_id: str, selection: FaceSelection, offset: Vector, detach: bool = True
    ) -> dict[str, Any]:
        return edit_faces(
            engine,
            object_id,
            FaceSelection.model_validate(selection).model_dump(),
            "move",
            offset=offset,
            detach=detach,
        )

    def delete_faces(
        object_id: str,
        selection: FaceSelection,
        empty_policy: Literal["error", "delete_object"] = "error",
    ) -> dict[str, Any]:
        return edit_faces(
            engine,
            object_id,
            FaceSelection.model_validate(selection).model_dump(),
            "delete",
            empty_policy=empty_policy,
        )

    def separate_faces(
        object_id: str, selection: FaceSelection, name: str | None = None, keep_source: bool = False
    ) -> dict[str, Any]:
        return edit_faces(
            engine,
            object_id,
            FaceSelection.model_validate(selection).model_dump(),
            "separate",
            name=name,
            copy_faces=keep_source,
        )

    def extrude_faces(object_id: str, selection: FaceSelection, offset: Vector) -> dict[str, Any]:
        return edit_faces(
            engine,
            object_id,
            FaceSelection.model_validate(selection).model_dump(),
            "extrude",
            offset=offset,
        )

    registry.add_function(
        "mesh.move_faces",
        move_faces,
        "Move selected faces by a world offset; detach from neighbors by default. "
        "With detach=false, move all render copies at selected positions, deforming neighbors.",
    )
    registry.add_function(
        "mesh.delete_faces",
        delete_faces,
        "Delete selected faces and compact attributes. Deleting every face requires "
        "empty_policy=delete_object; children are reparented.",
    )
    registry.add_function(
        "mesh.separate_faces",
        separate_faces,
        "Extract selected faces to a new mesh in world coordinates, preserving attributes. "
        "Remove them from the source unless keep_source=true. Delete an empty source.",
    )
    registry.add_function(
        "mesh.extrude_faces",
        extrude_faces,
        "Extrude selected face regions by a world offset: replace caps and add boundary walls. "
        "Preserve source attributes; walls use per-edge unit UV rectangles. "
        "Each selected component needs a consistently wound manifold open boundary.",
    )
