"""Bounded binary FBX documents for lossless rig/animation retention.

The document remains in source axes and units. Scissors edits a normalized bind-pose
view. Export rewrites only supported mesh reductions; bone transforms, bind matrices,
weights and animation curves never pass through the static Trimesh writer.
"""

import base64
import hashlib
import struct
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import trimesh
from pydantic import BaseModel, ConfigDict, Field, model_validator

from scissors.api.errors import ScissorsError
from scissors.core.fbx import Node, binary_file, node
from scissors.core.persistence import atomic_write, json_bytes

if TYPE_CHECKING:
    from scissors.core.engine import Engine
    from scissors.core.mesh import MeshObject
    from scissors.core.scene import Scene


class FbxBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    geometry_id: int
    control_points: list[int]
    world_vertices: list[list[float]]
    uv: list[list[float]] | None = None
    world_matrix: list[list[float]]
    content_hash: str

    @model_validator(mode="after")
    def check_binding(self) -> "FbxBinding":
        count = len(self.control_points)
        vertices, matrix = np.asarray(self.world_vertices), np.asarray(self.world_matrix)
        if (
            vertices.shape != (count, 3)
            or matrix.shape != (4, 4)
            or not np.isfinite(vertices).all()
            or not np.isfinite(matrix).all()
            or abs(float(np.linalg.det(matrix))) < 1e-12
            or any(index < 0 for index in self.control_points)
            or (self.uv is not None and np.asarray(self.uv).shape != (count, 2))
        ):
            raise ValueError("Invalid retained FBX control-point binding")
        return self


class FbxDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format_version: Literal[1] = 1
    path: str
    sha256: str
    structure_hash: str
    bindings: dict[str, FbxBinding] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)


def read_binary(data: bytes, limit: int, *, memory_limit: int | None = None) -> list[Node]:
    """Read only the bounded FBX binary grammar; never resolve external resources."""
    if len(data) > limit or not data.startswith(b"Kaydara FBX Binary  \x00\x1a\x00"):
        raise ValueError("Rig retention requires a binary FBX within the file budget")
    version = struct.unpack_from("<I", data, 23)[0]
    fmt, null = ("<QQQB", 25) if version >= 7500 else ("<IIIB", 13)
    allocated = 0

    def read(offset: int, depth: int) -> tuple[Node, int]:
        nonlocal allocated
        if depth > 256:
            raise ValueError("FBX document nesting exceeds limit")
        end, count, byte_count, name_length = struct.unpack_from(fmt, data, offset)
        if not offset + null <= end <= len(data) or count > limit:
            raise ValueError("Invalid FBX node bounds")
        cursor = offset + null
        name = data[cursor : cursor + name_length].decode("ascii")
        cursor += name_length
        start = cursor
        props: list[Any] = []
        for _ in range(count):
            kind = chr(data[cursor])
            cursor += 1
            if kind in "YCFDIL":
                code = {"Y": "<h", "C": "<?", "F": "<f", "D": "<d", "I": "<i", "L": "<q"}[kind]
                value = struct.unpack_from(code, data, cursor)[0]
                props.append(np.int32(value) if kind == "I" else value)
                cursor += struct.calcsize(code)
            elif kind in "SR":
                size = struct.unpack_from("<I", data, cursor)[0]
                cursor += 4
                if cursor + size > end:
                    raise ValueError("Invalid FBX property size")
                value = data[cursor : cursor + size]
                cursor += size
                props.append(value.decode("utf-8") if kind == "S" else value)
            elif kind in "fdilbc":
                size, compression, encoded = struct.unpack_from("<III", data, cursor)
                cursor += 12
                dtype = np.dtype(
                    {"f": "<f4", "d": "<f8", "i": "<i4", "l": "<i8", "b": "u1", "c": "i1"}[kind]
                )
                needed = size * dtype.itemsize
                allocated += needed
                if (
                    allocated > min(limit * 4, memory_limit or limit * 4)
                    or encoded + cursor > end
                    or compression not in (0, 1)
                ):
                    raise ValueError("FBX array exceeds document budget")
                payload = data[cursor : cursor + encoded]
                cursor += encoded
                if compression:
                    decoder = zlib.decompressobj()
                    payload = decoder.decompress(payload, needed + 1)
                    if not decoder.eof or decoder.unused_data:
                        raise ValueError("Invalid compressed FBX array")
                if len(payload) != needed:
                    raise ValueError("Invalid FBX array length")
                props.append(np.frombuffer(payload, dtype=dtype))
            else:
                raise ValueError("Unsupported FBX property type")
        if cursor - start != byte_count:
            raise ValueError("Invalid FBX property block")
        children = []
        while cursor < end:
            if data[cursor : cursor + null] == bytes(null):
                cursor += null
                break
            item, cursor = read(cursor, depth + 1)
            children.append(item)
        if cursor != end:
            raise ValueError("Invalid FBX child bounds")
        return Node(name, props, children), end

    nodes, cursor = [], 27
    while cursor + null <= len(data) and data[cursor : cursor + null] != bytes(null):
        item, cursor = read(cursor, 0)
        nodes.append(item)
    return nodes


def child(parent: Node, name: str) -> Node:
    return next(item for item in parent.children if item.name == name)


def document_bytes(nodes: list[Node], settings: Any) -> bytes:
    from scissors.core.exchange import ExchangeContext

    # All output is 7.4; typed int64 time arrays must remain int64.
    header = next((n for n in nodes if n.name == "FBXHeaderExtension"), None)
    if header:
        child(header, "FBXVersion").properties = [np.int32(7400)]
    context = ExchangeContext(trimesh.Scene(), settings, "retained")
    pending = list(nodes)
    size = 0
    while pending:
        item = pending.pop()
        size += 128
        for value in item.properties:
            size += (
                value.nbytes
                if isinstance(value, np.ndarray)
                else (len(value) if isinstance(value, (str, bytes)) else 32)
            )
        pending.extend(item.children)
    context.geometry_bytes = size
    return binary_file(nodes, context)


def retain_document(source: Path, resolver: Any, settings: Any) -> dict[str, Any]:
    nodes = read_binary(
        source.read_bytes(), settings.max_file_bytes, memory_limit=settings.max_memory_bytes // 4
    )
    objects = child(Node("root", children=nodes), "Objects").children
    connections = child(Node("root", children=nodes), "Connections")
    video_ids = {o.properties[0] for o in objects if o.name == "Video"}
    video_for_texture = {
        c.properties[2]: c.properties[1]
        for c in connections.children
        if c.properties[0] == "OO" and c.properties[1] in video_ids
    }
    labels = {}
    # Embed original images so the retained asset never depends on external paths.
    for item in sorted(objects, key=lambda o: o.name != "Video"):
        if item.name not in {"Video", "Texture"}:
            continue
        relative = next((n for n in item.children if n.name == "RelativeFilename"), None)
        filename = next((n for n in item.children if n.name in {"Filename", "FileName"}), None)
        content = next((n for n in item.children if n.name == "Content"), None)
        if item.name == "Video" and (content is None or not content.properties[0]):
            path = relative or filename
            if path is None:
                raise ValueError("FBX video has no image path")
            encoded = resolver.get(str(path.properties[0]))
            if content is None:
                item.children.append(node("Content", encoded))
            else:
                content.properties = [encoded]
        display_path = relative or filename
        source_name = str(display_path.properties[0]) if display_path is not None else ""
        suffix = Path(source_name.replace("\\", "/")).suffix or ".png"
        if item.name == "Video":
            label = f"embedded_{item.properties[0]}{suffix}"
            labels[item.properties[0]] = label
        elif item.properties[0] in video_for_texture:
            label = labels[video_for_texture[item.properties[0]]]
        elif source_name and (content is None or not content.properties[0]):
            raise ValueError("Retained FBX file textures require Video connections")
        else:
            continue
        # Only the display filename changes; Texture/Video connections stay exact.
        for field in [relative, filename]:
            if field is not None:
                field.properties = [label]
    bones = [
        o.properties[1].split("\x00")[0]
        for o in objects
        if o.name == "Model" and o.properties[2] == "LimbNode"
    ]
    clips = []
    for item in objects:
        if item.name == "AnimationStack":
            props = next((n for n in item.children if n.name == "Properties70"), Node("empty"))
            values = {p.properties[0]: p.properties[-1] for p in props.children if p.name == "P"}
            clips.append(
                {
                    "name": item.properties[1].split("\x00")[0],
                    "start_seconds": int(values.get("LocalStart", 0)) / 46186158000,
                    "end_seconds": int(values.get("LocalStop", 0)) / 46186158000,
                }
            )
    encoded_document = document_bytes(nodes, settings)
    return {
        "data": base64.b64encode(encoded_document).decode("ascii"),
        "summary": {
            "bone_count": len(bones),
            "bone_names": bones,
            "skin_count": sum(o.name == "Deformer" and o.properties[2] == "Skin" for o in objects),
            "animation_clips": clips,
            "mode": "preserved_fbx_document",
        },
    }


def canonical_hash(value: Any) -> str:
    def canonical(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: canonical(item[key]) for key in sorted(item)}
        if isinstance(item, list):
            return [canonical(v) for v in item]
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            return float(item) if item != 0 else 0.0
        return item

    return hashlib.sha256(json_bytes(canonical(value))).hexdigest()


def content_hash(obj: "MeshObject") -> str:
    return canonical_hash(
        {key: getattr(obj, key) for key in ("vertices", "faces", "normals", "uv", "vertex_colors")}
    )


def structure_hash(scene: "Scene") -> str:
    fields = ("id", "name", "kind", "parent_id", "hidden", "pivot", "transform", "material_id")
    value = {
        "units": scene.units,
        "objects": {
            oid: {key: getattr(obj, key) for key in fields} for oid, obj in scene.objects.items()
        },
        "materials": {mid: m.model_dump() for mid, m in scene.materials.items()},
    }
    return canonical_hash(value)


def export_document(
    engine: "Engine", scene: "Scene", output: Path, format: str, transform: Any
) -> dict[str, Any]:
    from scissors.core.export import artifact

    documents = scene.fbx_documents
    if format != "fbx" or transform is not None or len(documents) != 1:
        raise ScissorsError(
            "UNSUPPORTED_OPERATION",
            "Retained rigs require a complete FBX export without export_transform",
        )
    doc = documents[0]
    if structure_hash(scene) != doc.structure_hash:
        raise ScissorsError(
            "UNSUPPORTED_OPERATION",
            "Rig export requires unchanged hierarchy, transforms, materials and scene membership",
        )
    path = engine.path(doc.path)
    if path.stat().st_size > engine.settings.max_file_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Retained FBX exceeds file budget")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != doc.sha256:
        raise ScissorsError("INCOMPATIBLE_FORMAT", "Retained FBX integrity check failed")
    nodes = read_binary(
        raw, engine.settings.max_file_bytes, memory_limit=engine.settings.max_memory_bytes // 4
    )
    objects = child(Node("root", children=nodes), "Objects").children
    changed = [
        (oid, binding)
        for oid, binding in doc.bindings.items()
        if content_hash(scene.objects[oid]) != binding.content_hash
    ]
    if changed:
        # Multiple material parts/instances need a shared control-point edit plan.
        ids = [b.geometry_id for b in doc.bindings.values()]
        if len(ids) != len(set(ids)):
            raise ScissorsError(
                "UNSUPPORTED_OPERATION", "Rig mesh edits require one object per source geometry"
            )
        if any(
            o.name == "Deformer" and o.properties[2] not in {"Skin", "Cluster"} for o in objects
        ):
            raise ScissorsError(
                "UNSUPPORTED_OPERATION",
                "Mesh edits with blend shapes or additional deformers are unsupported",
            )
        for oid, binding in changed:
            rewrite_mesh(
                objects,
                child(Node("root", children=nodes), "Connections"),
                scene.objects[oid],
                binding,
            )
        raw = document_bytes(nodes, engine.settings)
    atomic_write(output, raw, engine.settings.max_file_bytes)
    return {
        **artifact(
            output,
            "fbx",
            [
                "Original FBX rig, bind matrices, animation curves and embedded media retained.",
                "FBX retention keeps source axes and units; the Scissors bind-pose view is Z-up.",
            ],
        ),
        "files": [str(output)],
        "rig": doc.summary,
    }


def rewrite_mesh(
    objects: list[Node], connections: Node, obj: "MeshObject", binding: FbxBinding
) -> None:
    original = next(
        (o for o in objects if o.name == "Geometry" and o.properties[0] == binding.geometry_id),
        None,
    )
    if original is None:
        raise ScissorsError("INCOMPATIBLE_FORMAT", "Retained FBX geometry binding is missing")
    lookup: dict[tuple, int] = {}
    for i, position in enumerate(binding.world_vertices):
        key = tuple([*position, *(binding.uv[i] if binding.uv is not None else [])])
        cp = binding.control_points[i]
        if key in lookup and lookup[key] != cp:
            raise ScissorsError(
                "UNSUPPORTED_OPERATION", "Coincident control points have ambiguous skin weights"
            )
        lookup[key] = cp
    try:
        cp_map = np.array(
            [
                lookup[tuple([*p, *(obj.uv[i] if obj.uv is not None else [])])]
                for i, p in enumerate(obj.vertices)
            ],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise ScissorsError(
            "UNSUPPORTED_OPERATION",
            "Rig mesh edits must retain original positions and UV; weight transfer is unsupported",
        ) from exc
    # Surviving render vertices inherit the exact original control-point weights.
    source_positions = child(original, "Vertices").properties[0].reshape(-1, 3)
    if len(cp_map) and int(cp_map.max()) >= len(source_positions):
        raise ScissorsError("INCOMPATIBLE_FORMAT", "Invalid retained FBX control point")
    positions = source_positions[cp_map]
    faces = np.asarray(obj.faces, dtype=np.int32)
    corners = faces.ravel().copy()
    corners[2::3] = -corners[2::3] - 1
    world = np.asarray(binding.world_matrix)
    normal = np.asarray(obj.normals) @ world[:3, :3]
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-15)
    layer_nodes = []
    layers = []

    def layer(
        kind: str, values: list[Node], mapping: str = "ByVertice", reference: str = "Direct"
    ) -> None:
        layer_nodes.append(
            node(
                "LayerElement" + kind,
                np.int32(0),
                children=[
                    node("Version", np.int32(101)),
                    node("Name", "UVMap" if kind == "UV" else ""),
                    node("MappingInformationType", mapping),
                    node("ReferenceInformationType", reference),
                    *values,
                ],
            )
        )
        layers.append(
            node(
                "LayerElement",
                children=[node("Type", "LayerElement" + kind), node("TypedIndex", np.int32(0))],
            )
        )

    layer("Normal", [node("Normals", normal)])
    if obj.uv is not None:
        layer("UV", [node("UV", np.asarray(obj.uv))])
    if obj.vertex_colors is not None:
        layer("Color", [node("Colors", np.asarray(obj.vertex_colors) / 255)])
    layer(
        "Material", [node("Materials", np.array([0], dtype=np.int32))], "AllSame", "IndexToDirect"
    )
    original.children = [
        node("GeometryVersion", np.int32(124)),
        node("Vertices", positions),
        node("PolygonVertexIndex", corners),
        *layer_nodes,
        node("Layer", np.int32(0), children=[node("Version", np.int32(100)), *layers]),
    ]
    parent = {
        int(c.properties[1]): int(c.properties[2])
        for c in connections.children
        if c.properties[0] == "OO"
    }
    skins = {
        o.properties[0]
        for o in objects
        if o.name == "Deformer"
        and o.properties[2] == "Skin"
        and parent.get(o.properties[0]) == binding.geometry_id
    }
    for skin in objects:
        if skin.name == "Deformer" and skin.properties[0] in skins:
            kind = next((n for n in skin.children if n.name == "SkinningType"), None)
            if kind is not None and kind.properties[0] not in {"Linear", "Rigid"}:
                raise ScissorsError(
                    "UNSUPPORTED_OPERATION", "Rig mesh edits require linear or rigid skinning"
                )
    for o in objects:
        if (
            o.name == "Deformer"
            and o.properties[2] == "Cluster"
            and parent.get(o.properties[0]) in skins
        ):
            index_node = next((n for n in o.children if n.name == "Indexes"), None)
            weight_node = next((n for n in o.children if n.name == "Weights"), None)
            if index_node is None and weight_node is None:
                # End bones may have a bind matrix but no weighted vertices.
                continue
            if index_node is None or weight_node is None:
                raise ScissorsError("GEOMETRY_ERROR", "Source FBX cluster has incomplete weights")
            indices = np.asarray(index_node.properties[0])
            weights = np.asarray(weight_node.properties[0])
            if len(indices) != len(weights):
                raise ScissorsError("GEOMETRY_ERROR", "Source FBX cluster weight counts differ")
            table = dict(zip(indices.tolist(), weights.tolist(), strict=True))
            values = np.array([table.get(int(cp), 0.0) for cp in cp_map])
            if not np.isfinite(values).all():
                raise ScissorsError("GEOMETRY_ERROR", "Source FBX skin weights must be finite")
            used = np.flatnonzero(values != 0)
            child(o, "Indexes").properties = [used.astype(np.int32)]
            child(o, "Weights").properties = [values[used]]
