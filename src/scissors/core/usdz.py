"""OpenUSD ASCII geometry/shaders packaged as aligned, uncompressed USDZ."""

import io
import json
import struct
import zipfile
from typing import Any

import numpy as np
import trimesh

from scissors.api.errors import ScissorsError
from scissors.config import Settings
from scissors.core.exchange import (
    ExchangeContext,
    mesh_colors,
    mesh_uv,
    safe_identifier,
    unit_scale,
)


def quoted(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def tuples(context: ExchangeContext, values: Any) -> str:
    rows = np.asarray(values)
    chunks = []
    for start in range(0, len(rows), 1024):
        chunk = ", ".join(
            "(" + ", ".join(format(float(value), ".17g") for value in row) + ")"
            for row in rows[start : start + 1024]
        )
        chunks.append(context.text(chunk))
    return "[" + ", ".join(chunks) + "]"


def package_usdz(files: dict[str, bytes], settings: Settings) -> bytes:
    """The first member is the default layer; every member's data begins on a 64-byte boundary."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
        for name, data in files.items():
            if "/" in name or "\\" in name or name in {".", ".."} or not name.isascii():
                raise ValueError("USDZ member names must be safe root-level ASCII names")
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            offset = buffer.tell() + 30 + len(name.encode("ascii"))
            if offset % 64:
                padding = (-(offset + 4)) % 64
                info.extra = struct.pack("<HH", 0x1986, padding) + bytes(padding)
            archive.writestr(info, data)
            if buffer.tell() > settings.max_file_bytes:
                raise ScissorsError("RESOURCE_LIMIT", "USDZ package exceeds file budget")
    result = buffer.getvalue()
    if len(result) > settings.max_file_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "USDZ directory exceeds file budget")
    return result


def export_usdz(
    scene: trimesh.Scene, settings: Settings, prefix: str, units: str
) -> tuple[bytes, dict[str, bytes], list[str]]:
    meters = unit_scale(units)
    context = ExchangeContext(scene, settings, prefix)
    roots = context.children.get(context.base, [])
    # Reuse an existing export/group root so root_name remains the package's default prim.
    existing_root = len(roots) == 1 and roots[0][1].get("geometry") is None
    root_name = safe_identifier(roots[0][0]) if existing_root else "Asset"
    root_path = f"/{root_name}"
    looks_name = "__ScissorsMaterials"
    while looks_name in scene.graph.nodes:
        looks_name += "_"
    looks_path = f"{root_path}/{looks_name}"
    text = [
        "#usda 1.0\n(\n",
        f'    defaultPrim = "{root_name}"\n    upAxis = "Z"\n',
        f"    metersPerUnit = {meters:.17g}\n)\n",
        f'def Xform "{root_name}" {{\n',
    ]

    def add(value: str) -> None:
        text.append(context.text(value))

    def transform(matrix: np.ndarray) -> None:
        rows = ", ".join("(" + context.numbers(row, ", ") + ")" for row in matrix.T)
        add(f"matrix4d xformOp:transform = ({rows})\n")
        add('uniform token[] xformOpOrder = ["xformOp:transform"]\n')

    if existing_root:
        transform(np.asarray(roots[0][1]["matrix"]))
    add(f'def Scope "{looks_name}" {{\n')
    for index, material in enumerate(context.materials):
        material_path = f"{looks_path}/Material{index}"
        add(f'def Material "Material{index}" {{\n')
        add(f"token outputs:surface.connect = <{material_path}/Surface.outputs:surface>\n")
        add('def Shader "UV" {\nuniform token info:id = "UsdPrimvarReader_float2"\n')
        add('string inputs:varname = "st"\nfloat2 outputs:result\n}\n')
        readers: list[tuple[str, str, np.ndarray]] = [
            ("Color", "float3", material.color[:3]),
            ("Opacity", "float", np.asarray([material.color[3]])),
        ]
        for name, role, fallback in readers:
            add(f'def Shader "{name}" {{\nuniform token info:id = "UsdPrimvarReader_{role}"\n')
            add(f'string inputs:varname = "display{name}"\n')
            formatted = (
                "(" + context.numbers(fallback, ", ") + ")"
                if role == "float3"
                else str(float(fallback[0]))
            )
            add(f"{role} inputs:fallback = {formatted}\n{role} outputs:result\n}}\n")
        paths = {}
        for slot, image in material.images.items():
            paths[slot] = f"{material_path}/{slot}"
            image_name = context.image(image)
            add(f'def Shader "{slot}" {{\nuniform token info:id = "UsdUVTexture"\n')
            add(f"asset inputs:file = @{image_name}@\n")
            add(f"float2 inputs:st.connect = <{material_path}/UV.outputs:result>\n")
            add('token inputs:wrapS = "repeat"\ntoken inputs:wrapT = "repeat"\n')
            color_space = "sRGB" if slot in {"base", "emissive"} else "raw"
            add(f'token inputs:sourceColorSpace = "{color_space}"\n')
            scale = np.ones(4)
            if slot == "base":
                scale = material.color
            elif slot == "emissive":
                scale[:3] = material.emissive
            elif slot == "orm":
                scale[1:3] = [material.roughness, material.metallic]
            elif slot == "normal":
                scale[:3] = 2
                add("float4 inputs:bias = (-1, -1, -1, 0)\n")
            add("float4 inputs:scale = (" + context.numbers(scale, ", ") + ")\n")
            add(
                "float3 outputs:rgb\nfloat outputs:r\nfloat outputs:g\n"
                "float outputs:b\nfloat outputs:a\n}\n"
            )
        add('def Shader "Surface" {\nuniform token info:id = "UsdPreviewSurface"\n')
        add("int inputs:useSpecularWorkflow = 0\n")
        if "base" in paths:
            add(f"color3f inputs:diffuseColor.connect = <{paths['base']}.outputs:rgb>\n")
            add(f"float inputs:opacity.connect = <{paths['base']}.outputs:a>\n")
        else:
            add(f"color3f inputs:diffuseColor.connect = <{material_path}/Color.outputs:result>\n")
            add(f"float inputs:opacity.connect = <{material_path}/Opacity.outputs:result>\n")
        if "orm" in paths:
            add(f"float inputs:roughness.connect = <{paths['orm']}.outputs:g>\n")
            add(f"float inputs:metallic.connect = <{paths['orm']}.outputs:b>\n")
        else:
            add(f"float inputs:roughness = {material.roughness:.17g}\n")
            add(f"float inputs:metallic = {material.metallic:.17g}\n")
        for slot, target, output, role in [
            ("normal", "normal", "rgb", "normal3f"),
            ("emissive", "emissiveColor", "rgb", "color3f"),
            ("occlusion", "occlusion", "r", "float"),
        ]:
            if slot in paths:
                add(f"{role} inputs:{target}.connect = <{paths[slot]}.outputs:{output}>\n")
            elif slot == "emissive":
                add(
                    "color3f inputs:emissiveColor = ("
                    + context.numbers(material.emissive, ", ")
                    + ")\n"
                )
        add("token outputs:surface\n}\n}\n")
    add("}\n")

    def nodes(parent: str, parent_path: str) -> None:
        taken: set[str] = set()
        for raw_name, data in context.children.get(parent, []):
            name = safe_identifier(raw_name)
            while name in taken or name == looks_name:
                name += "_"
            taken.add(name)
            path = f"{parent_path}/{name}"
            key = data.get("geometry")
            kind = "Mesh" if key is not None else "Xform"
            add(f'def {kind} "{name}"')
            if key is not None:
                add(' (prepend apiSchemas = ["MaterialBindingAPI"])')
            add(" {\n")
            add(f"custom string scissors:nodeName = {quoted(str(raw_name))}\n")
            transform(np.asarray(data["matrix"]))
            if key is not None:
                mesh = scene.geometry[key]
                material_index = context.mesh_materials[key]
                material = context.materials[material_index]
                add(
                    'uniform token subdivisionScheme = "none"\n'
                    'uniform token orientation = "rightHanded"\n'
                )
                add("point3f[] points = " + tuples(context, mesh.vertices) + "\n")
                add(
                    "int[] faceVertexCounts = ["
                    + context.numbers(np.full(len(mesh.faces), 3), ", ")
                    + "]\n"
                )
                add("int[] faceVertexIndices = [" + context.numbers(mesh.faces, ", ") + "]\n")
                add(
                    "normal3f[] normals = "
                    + tuples(context, mesh.vertex_normals)
                    + ' (interpolation = "vertex")\n'
                )
                uv = mesh_uv(mesh)
                if uv is not None:
                    add(
                        "texCoord2f[] primvars:st = "
                        + tuples(context, uv)
                        + ' (interpolation = "vertex")\n'
                    )
                colors = mesh_colors(mesh)
                if colors is not None:
                    colors = colors * material.color
                    add(
                        "color3f[] primvars:displayColor = "
                        + tuples(context, colors[:, :3])
                        + ' (interpolation = "vertex")\n'
                    )
                    add(
                        "float[] primvars:displayOpacity = ["
                        + context.numbers(colors[:, 3], ", ")
                        + '] (interpolation = "vertex")\n'
                    )
                add(f"bool doubleSided = {'true' if material.double_sided else 'false'}\n")
                add(f"rel material:binding = <{looks_path}/Material{material_index}>\n")
            nodes(raw_name, path)
            add("}\n")

    nodes(roots[0][0] if existing_root else context.base, root_path)
    add("}\n")
    layer = "".join(text).encode("utf-8")
    data = package_usdz({"defaultLayer.usda": layer, **context.assets}, settings)
    return (
        data,
        {},
        [
            "USDZ contains USDA first, uncompressed PNG assets and 64-byte-aligned data.",
            "Z-up, scene units, hierarchy, affine matrices and USD Preview Surface PBR retained.",
            "Vertex colors remain primvars; textures do not multiply vertex color into shading.",
            "This is OpenUSD USDZ with ASCII USDA; Apple viewers requiring USDC may reject it.",
            "Mesh data is repeated per instance; skeletal animation and rigging are not exported.",
        ],
    )
