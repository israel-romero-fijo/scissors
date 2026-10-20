"""Bounded scene, image and text data shared by native exchange writers."""

import hashlib
import io
import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import trimesh
from PIL import Image

from scissors.api.errors import ScissorsError
from scissors.config import Settings

METERS = {"meters": 1.0, "centimeters": 0.01, "millimeters": 0.001}


def unit_scale(units: str) -> float:
    if units not in METERS:
        raise ValueError("Exchange export requires meters, centimeters or millimeters scene units")
    return METERS[units]


def safe_identifier(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", value) or "Asset"
    return name if name[0].isalpha() or name[0] == "_" else f"_{name}"


@dataclass
class ExchangeMaterial:
    name: str
    color: np.ndarray
    metallic: float = 0
    roughness: float = 0.7
    emissive: np.ndarray = field(default_factory=lambda: np.zeros(3))
    double_sided: bool = False
    images: dict[str, Image.Image] = field(default_factory=dict)


def mesh_material(mesh: trimesh.Trimesh) -> ExchangeMaterial:
    if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        material = mesh.visual.material
        if not isinstance(material, trimesh.visual.material.PBRMaterial):
            material = material.to_pbr()
        factor = material.baseColorFactor
        result = ExchangeMaterial(
            name=material.name or "Material",
            color=np.asarray(factor, dtype=float) / 255 if factor is not None else np.ones(4),
            metallic=float(material.metallicFactor or 0),
            roughness=float(
                material.roughnessFactor if material.roughnessFactor is not None else 1
            ),
            emissive=np.asarray(
                material.emissiveFactor if material.emissiveFactor is not None else [0, 0, 0],
                dtype=float,
            ),
            double_sided=bool(material.doubleSided),
        )
        for key, attribute in {
            "base": "baseColorTexture",
            "normal": "normalTexture",
            "orm": "metallicRoughnessTexture",
            "emissive": "emissiveTexture",
            "occlusion": "occlusionTexture",
        }.items():
            image = getattr(material, attribute)
            if image is not None:
                result.images[key] = image
        return result
    return ExchangeMaterial("VertexColor", np.ones(4))


def mesh_colors(mesh: trimesh.Trimesh) -> np.ndarray | None:
    if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        colors = mesh.visual.vertex_attributes.get("color")
        return np.asarray(colors, dtype=float) / 255 if colors is not None else None
    if isinstance(mesh.visual, trimesh.visual.color.ColorVisuals):
        return np.asarray(mesh.visual.vertex_colors, dtype=float) / 255
    return None


def mesh_uv(mesh: trimesh.Trimesh) -> np.ndarray | None:
    if (
        isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals)
        and mesh.visual.uv is not None
    ):
        return np.asarray(mesh.visual.uv)
    return None


class ExchangeContext:
    def __init__(self, scene: trimesh.Scene, settings: Settings, prefix: str):
        self.scene, self.settings = scene, settings
        self.base = str(scene.graph.base_frame)
        self.prefix = safe_identifier(prefix)[:64]
        self.deadline = time.monotonic() + settings.operation_timeout
        self.assets: dict[str, bytes] = {}
        self.image_cache: dict[tuple, str] = {}
        self.asset_bytes = 0
        self.text_bytes = 0
        self.geometry_bytes = sum(
            len(mesh.vertices) * 384 + len(mesh.faces) * 192 for mesh in scene.geometry.values()
        )
        self.children: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        self.edges = scene.graph.to_edgelist()
        for parent, child, data in self.edges:
            self.children.setdefault(parent, []).append((child, data))
        # XML and USD both represent nested objects. Bound depth before recursive serializers.
        pending = [(self.base, 0)]
        while pending:
            parent, depth = pending.pop()
            if depth > 128:
                raise ScissorsError("RESOURCE_LIMIT", "Exchange hierarchy exceeds depth limit")
            pending.extend((child, depth + 1) for child, _ in self.children.get(parent, []))
        self.materials: list[ExchangeMaterial] = []
        self.mesh_materials: dict[str, int] = {}
        self.cached_image_bytes = 0
        cached_images: set[int] = set()
        signatures: dict[tuple, int] = {}
        for key, mesh in scene.geometry.items():
            self.check()
            material = mesh_material(mesh)
            image_keys = []
            for slot, image in material.images.items():
                if id(image) not in cached_images:
                    self.cached_image_bytes += image.width * image.height * len(image.getbands())
                    cached_images.add(id(image))
                self.check(image.width * image.height * len(image.getbands()))
                # Pixel buffers were already bounded by the shared scene exporter.
                image_keys.append(
                    (slot, image.mode, image.size, hashlib.sha256(image.tobytes()).digest())
                )
            signature = (
                material.color.tobytes(),
                material.metallic,
                material.roughness,
                material.emissive.tobytes(),
                material.double_sided,
                tuple(image_keys),
            )
            if signature not in signatures:
                signatures[signature] = len(self.materials)
                self.materials.append(material)
            self.mesh_materials[key] = signatures[signature]

    def check(self, working_bytes: int = 0) -> None:
        if time.monotonic() > self.deadline:
            raise ScissorsError("RESOURCE_LIMIT", "Exchange serialization timed out")
        if self.text_bytes + self.asset_bytes > self.settings.max_file_bytes:
            raise ScissorsError("RESOURCE_LIMIT", "Exchange data exceeds file budget")
        if (
            self.geometry_bytes
            + self.cached_image_bytes
            + 4 * self.text_bytes
            + 5 * self.asset_bytes
            + working_bytes
            > self.settings.max_memory_bytes
        ):
            raise ScissorsError("RESOURCE_LIMIT", "Exchange buffers exceed memory budget")

    def text(self, value: str) -> str:
        self.text_bytes += len(value.encode("utf-8"))
        self.check()
        return value

    def numbers(self, values: Any, separator: str = " ") -> str:
        flat = np.asarray(values).reshape(-1)
        if not np.isfinite(flat).all():
            raise ValueError("Exchange arrays must be finite")
        integer = flat.dtype.kind in "iu"
        chunks = []
        for start in range(0, len(flat), 4096):
            chunk = separator.join(
                str(int(value)) if integer else format(float(value), ".17g")
                for value in flat[start : start + 4096]
            )
            chunks.append(self.text(chunk))
        return separator.join(chunks)

    def image(self, image: Image.Image, factor: np.ndarray | None = None) -> str:
        # Cache encoded results, not unbounded decoded copies. Reserve hashing,
        # PNG encoding and BytesIO copies as well as resident source images.
        pixels = image.width * image.height
        self.check(pixels * len(image.getbands()))
        effective = np.ones(4) if factor is None else np.asarray(factor, dtype=float)
        if effective.shape == (3,):
            effective = np.r_[effective, 1.0]
        signature = (
            image.mode,
            image.size,
            hashlib.sha256(image.tobytes()).digest(),
            tuple(effective.tolist()),
        )
        if signature in self.image_cache:
            return self.image_cache[signature]
        self.check(pixels * 20 + 65536)
        if factor is not None:
            image = tinted_image(
                image,
                factor,
                self.settings,
                self.geometry_bytes
                + self.cached_image_bytes
                + 4 * self.text_bytes
                + 5 * self.asset_bytes
                + pixels * 20
                + 65536,
            )
        self.check(pixels * 24 + 65536)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data = buffer.getvalue()
        key = f"{self.prefix}_{hashlib.sha256(data).hexdigest()}.png"
        if key not in self.assets:
            self.assets[key] = data
            self.asset_bytes += len(data)
        self.check()
        self.image_cache[signature] = key
        return key


def tinted_image(
    image: Image.Image, factor: np.ndarray, settings: Settings, reserved_bytes: int = 0
) -> Image.Image:
    """Bake a linear factor into an sRGB image for legacy single-texture material slots."""
    factor = np.asarray(factor, dtype=float)
    if factor.shape not in {(3,), (4,)} or not np.isfinite(factor).all() or np.any(factor < 0):
        raise ValueError("Texture tint needs three or four finite nonnegative factors")
    if reserved_bytes > settings.max_memory_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Texture tint exceeds memory budget")
    if np.all(factor == 1):
        return image
    pixels = image.width * image.height
    available = settings.max_memory_bytes - reserved_bytes - pixels * 4
    block_pixels = min(65536, available // 192)
    if block_pixels < 1:
        raise ScissorsError("RESOURCE_LIMIT", "Texture tint output exceeds memory budget")
    width = min(image.width, 256, block_pixels)
    height = min(image.height, max(1, block_pixels // width))
    output = Image.new("RGBA", image.size)
    deadline = time.monotonic() + settings.operation_timeout
    for y in range(0, image.height, height):
        for x in range(0, image.width, width):
            if time.monotonic() > deadline:
                raise ScissorsError("RESOURCE_LIMIT", "Texture tint timed out")
            rectangle = (x, y, min(x + width, image.width), min(y + height, image.height))
            rgba = np.asarray(image.crop(rectangle).convert("RGBA"), dtype=float) / 255
            rgb = rgba[:, :, :3]
            linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
            linear *= factor[:3]
            rgba[:, :, :3] = np.where(
                linear <= 0.0031308,
                linear * 12.92,
                1.055 * np.maximum(linear, 0) ** (1 / 2.4) - 0.055,
            )
            if len(factor) == 4:
                rgba[:, :, 3] *= factor[3]
            output.paste(
                Image.fromarray(np.clip(np.rint(rgba * 255), 0, 255).astype(np.uint8)), (x, y)
            )
            del rgba, rgb, linear
    return output
