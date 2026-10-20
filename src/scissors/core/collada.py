"""COLLADA 1.4.1 XML writer with real geometry, hierarchy and managed PNG assets."""

import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
import trimesh

from scissors.config import Settings
from scissors.core.exchange import ExchangeContext, mesh_colors, mesh_uv, unit_scale

NS = "http://www.collada.org/2005/11/COLLADASchema"


def element(parent: ET.Element, tag: str, text: str | None = None, **attributes: str) -> ET.Element:
    child = ET.SubElement(parent, f"{{{NS}}}{tag}", attributes)
    child.text = text
    return child


def export_dae(
    scene: trimesh.Scene, settings: Settings, prefix: str, units: str
) -> tuple[bytes, dict[str, bytes], list[str]]:
    meters = unit_scale(units)
    context = ExchangeContext(scene, settings, prefix)
    root = ET.Element(f"{{{NS}}}COLLADA", {"version": "1.4.1"})
    asset = element(root, "asset")
    contributor = element(asset, "contributor")
    element(contributor, "authoring_tool", "Scissors")
    # Fixed metadata makes repeated exports deterministic; this is not a source-file timestamp.
    element(asset, "created", "1970-01-01T00:00:00Z")
    element(asset, "modified", "1970-01-01T00:00:00Z")
    element(asset, "unit", name=units, meter=str(meters))
    element(asset, "up_axis", "Z_UP")
    images = element(root, "library_images")
    effects = element(root, "library_effects")
    materials = element(root, "library_materials")
    for index, material in enumerate(context.materials):
        context.check()
        effect = element(effects, "effect", id=f"effect{index}")
        profile = element(effect, "profile_COMMON")
        slots: dict[str, str] = {}
        for slot, image in material.images.items():
            factor = (
                material.color
                if slot == "base"
                else material.emissive
                if slot == "emissive"
                else None
            )
            path = context.image(image, factor)
            image_id = f"image{index}_{slot}"
            element(element(images, "image", id=image_id), "init_from", path)
            surface = element(
                element(profile, "newparam", sid=f"{slot}_surface"), "surface", type="2D"
            )
            element(surface, "init_from", image_id)
            sampler = element(element(profile, "newparam", sid=f"{slot}_sampler"), "sampler2D")
            element(sampler, "source", f"{slot}_surface")
            slots[slot] = f"{slot}_sampler"
        phong = element(element(profile, "technique", sid="common"), "phong")
        emission = element(phong, "emission")
        if "emissive" in slots:
            element(emission, "texture", texture=slots["emissive"], texcoord="UVSET0")
        else:
            element(emission, "color", context.numbers([*material.emissive, 1]))
        element(element(phong, "ambient"), "color", "0 0 0 1")
        diffuse = element(phong, "diffuse")
        if "base" in slots:
            element(diffuse, "texture", texture=slots["base"], texcoord="UVSET0")
        else:
            element(diffuse, "color", context.numbers(material.color))
        specular = (1 - material.metallic) * 0.04 + material.metallic * material.color[:3]
        element(element(phong, "specular"), "color", context.numbers([*specular, 1]))
        exponent = min(128, max(0, 2 / max(material.roughness, 0.01) ** 4 - 2))
        element(element(phong, "shininess"), "float", str(exponent))
        transparent = element(phong, "transparent", opaque="A_ONE")
        if "base" in slots:
            element(transparent, "texture", texture=slots["base"], texcoord="UVSET0")
        else:
            element(transparent, "color", context.numbers([1, 1, 1, material.color[3]]))
        element(element(phong, "transparency"), "float", "1")
        extra = element(element(effect, "extra"), "technique", profile="SCISSORS")
        element(extra, "metallic", str(material.metallic))
        element(extra, "roughness", str(material.roughness))
        element(extra, "double_sided", "1" if material.double_sided else "0")
        for slot in slots:
            if slot not in {"base", "emissive"}:
                element(extra, f"{slot}_image", f"image{index}_{slot}")
        readable_name = "".join(
            char if (char in "\t\n\r" or ord(char) >= 32) else "_" for char in material.name
        )
        material_node = element(materials, "material", id=f"material{index}", name=readable_name)
        element(material_node, "instance_effect", url=f"#effect{index}")
    geometries = element(root, "library_geometries")
    geometry_ids = {}
    for index, (key, mesh) in enumerate(scene.geometry.items()):
        context.check()
        geometry_id = geometry_ids[key] = f"geometry{index}"
        geometry = element(geometries, "geometry", id=geometry_id, name=str(key))
        container = element(geometry, "mesh")
        sources: list[tuple[str, Any, list[str]]] = [
            ("POSITION", mesh.vertices, ["X", "Y", "Z"]),
            ("NORMAL", mesh.vertex_normals, ["X", "Y", "Z"]),
        ]
        uv, colors = mesh_uv(mesh), mesh_colors(mesh)
        if uv is not None:
            sources.append(("TEXCOORD", uv, ["S", "T"]))
        if colors is not None:
            sources.append(("COLOR", colors, ["R", "G", "B", "A"]))
        for semantic, values, names in sources:
            source_id = f"{geometry_id}_{semantic}"
            source = element(container, "source", id=source_id)
            element(
                source,
                "float_array",
                context.numbers(values),
                id=f"{source_id}_array",
                count=str(np.size(values)),
            )
            accessor = element(
                element(source, "technique_common"),
                "accessor",
                source=f"#{source_id}_array",
                count=str(len(values)),
                stride=str(len(names)),
            )
            for name in names:
                element(accessor, "param", name=name, type="float")
        vertices_id = f"{geometry_id}_vertices"
        element(
            element(container, "vertices", id=vertices_id),
            "input",
            semantic="POSITION",
            source=f"#{geometry_id}_POSITION",
        )
        triangles = element(container, "triangles", count=str(len(mesh.faces)), material="material")
        for offset, (semantic, _, _) in enumerate(sources):
            attributes = {"set": "0"} if semantic in {"TEXCOORD", "COLOR"} else {}
            element(
                triangles,
                "input",
                semantic="VERTEX" if semantic == "POSITION" else semantic,
                source=f"#{vertices_id}"
                if semantic == "POSITION"
                else f"#{geometry_id}_{semantic}",
                offset=str(offset),
                **attributes,
            )
        element(
            triangles,
            "p",
            context.numbers(np.repeat(np.asarray(mesh.faces).reshape(-1), len(sources))),
        )
    visual = element(
        element(root, "library_visual_scenes"), "visual_scene", id="scene", name="Scene"
    )
    queue: list[tuple[str, ET.Element]] = [(context.base, visual)]
    node_index = 0
    while queue:
        parent, container = queue.pop(0)
        for name, data in context.children.get(parent, []):
            context.check()
            node = element(container, "node", id=f"node{node_index}", name=str(name), type="NODE")
            node_index += 1
            # COLLADA uses column-vector matrices written in row-major text order.
            element(node, "matrix", context.numbers(data["matrix"]), sid="transform")
            key = data.get("geometry")
            if key is not None:
                instance = element(node, "instance_geometry", url=f"#{geometry_ids[key]}")
                binding = element(element(instance, "bind_material"), "technique_common")
                binding_node = element(
                    binding,
                    "instance_material",
                    symbol="material",
                    target=f"#material{context.mesh_materials[key]}",
                )
                if mesh_uv(scene.geometry[key]) is not None:
                    element(
                        binding_node,
                        "bind_vertex_input",
                        semantic="UVSET0",
                        input_semantic="TEXCOORD",
                        input_set="0",
                    )
            queue.append((name, node))
    element(element(root, "scene"), "instance_visual_scene", url="#scene")
    if not len(images):
        root.remove(images)
    data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    if len(data) + context.asset_bytes > settings.max_file_bytes:
        from scissors.api.errors import ScissorsError

        raise ScissorsError("RESOURCE_LIMIT", "COLLADA export exceeds file budget")
    return (
        data,
        context.assets,
        [
            "COLLADA 1.4.1 retains hierarchy, matrices, normals, UV and vertex colors; Z-up.",
            "PBR uses Phong approximation. Normal/ORM/occlusion images are custom extras.",
            "Vertex-color and double-sided rendering depend on the receiving COLLADA importer.",
        ],
    )
