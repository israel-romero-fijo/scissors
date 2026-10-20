"""Font outlines sampled into true extruded text geometry in the VTK/Manifold worker."""

import bisect
import hashlib
import struct
from pathlib import Path
from typing import Any

import numpy as np


def font_characters(path: Path, characters: set[int], limit: int) -> str:
    """Validate requested glyph IDs in bounded SFNT Unicode cmap tables."""
    if path.stat().st_size > limit:
        raise ValueError("Font exceeds file byte limit")
    raw = path.read_bytes()

    def number(offset: int, format: str = ">H") -> int:
        size = struct.calcsize(format)
        if offset < 0 or offset + size > len(raw):
            raise ValueError("Font table exceeds file bounds")
        return int(struct.unpack_from(format, raw, offset)[0])

    if raw[:4] not in {b"\x00\x01\x00\x00", b"OTTO", b"true"}:
        raise ValueError("Font must be a single-face TTF/OTF SFNT file")
    tables = {}
    for index in range(number(4)):
        offset = 12 + 16 * index
        tag = raw[offset : offset + 4]
        start, length = number(offset + 8, ">I"), number(offset + 12, ">I")
        if start + length > len(raw):
            raise ValueError("Font table exceeds file bounds")
        tables[tag] = (start, length)
    if b"cmap" not in tables or b"maxp" not in tables:
        raise ValueError("Font lacks Unicode cmap/glyph metadata")
    cmap, cmap_size = tables[b"cmap"]
    glyph_count = number(tables[b"maxp"][0] + 4)
    supported: set[int] = set()
    for index in range(number(cmap + 2)):
        record = cmap + 4 + 8 * index
        platform, encoding = number(record), number(record + 2)
        if platform != 0 and not (platform == 3 and encoding in {1, 10}):
            continue
        start = cmap + number(record + 4, ">I")
        kind = number(start)
        if kind not in {4, 12, 13}:
            continue
        length = number(start + 2) if kind == 4 else number(start + 4, ">I")
        if start + length > cmap + cmap_size:
            raise ValueError("Unicode cmap exceeds its table bounds")
        for code in characters - supported:
            glyph = 0
            if kind in {12, 13}:
                count, low, high = number(start + 12, ">I"), 0, number(start + 12, ">I")
                if 16 + 12 * count > length:
                    raise ValueError("Invalid Unicode cmap group count")
                while low < high:
                    mid = (low + high) // 2
                    group = start + 16 + 12 * mid
                    first, last = number(group, ">I"), number(group + 4, ">I")
                    if code < first:
                        high = mid
                    elif code > last:
                        low = mid + 1
                    else:
                        glyph = number(group + 8, ">I") + (code - first if kind == 12 else 0)
                        break
            elif code <= 65535:
                count = number(start + 6) // 2
                if 16 + 8 * count > length:
                    raise ValueError("Invalid Unicode cmap segment count")
                ends = np.frombuffer(raw, dtype=">u2", count=count, offset=start + 14)
                segment = bisect.bisect_left(ends, code)
                if segment < count:
                    first = number(start + 16 + 2 * count + 2 * segment)
                    delta = number(start + 16 + 4 * count + 2 * segment)
                    address = start + 16 + 6 * count + 2 * segment
                    relative = number(address)
                    if code >= first:
                        if relative:
                            target = address + relative + 2 * (code - first)
                            if target + 2 > start + length:
                                raise ValueError("Glyph index exceeds cmap bounds")
                            glyph = number(target)
                            glyph = (glyph + delta) & 65535 if glyph else 0
                        else:
                            glyph = (code + delta) & 65535
            if 0 < glyph < glyph_count:
                supported.add(code)
    missing = characters - supported
    if missing:
        raise ValueError(
            "Font lacks requested glyphs: " + ", ".join(f"U+{code:04X}" for code in sorted(missing))
        )
    return hashlib.sha256(raw).hexdigest()


def text_geometry(value: dict[str, Any]) -> dict[str, Any]:
    """Headless outline extraction; no raster image or display server is involved."""
    import manifold3d as m
    from vtkmodules.vtkCommonDataModel import vtkPath
    from vtkmodules.vtkRenderingCore import vtkTextProperty
    from vtkmodules.vtkRenderingFreeType import vtkFreeTypeTools

    from scissors.core.boolean_attributes import planar_uv
    from scissors.core.persistence import contained

    prop = vtkTextProperty()
    prop.SetFontSize(256)
    prop.SetJustificationToLeft()
    prop.SetVerticalJustificationToBottom()
    prop.SetBold(int(value["bold"]))
    prop.SetItalic(int(value["italic"]))
    prop.SetLineSpacing(value["line_spacing"])
    tools = vtkFreeTypeTools.GetInstance()
    tools.ForceCompiledFontsOn()
    tools.SetMaximumNumberOfBytes(max(1, min(value["max_memory_bytes"] // 8, 32 * 1024 * 1024)))
    font_hash = None
    if value["font_path"]:
        source = contained(Path(value["root"]), value["font_path"])
        font_hash = font_characters(
            source, {ord(char) for char in value["text"] if char != "\n"}, value["max_file_bytes"]
        )
        prop.SetFontFamilyAsString("File")
        if prop.GetFontFamilyAsString() != "File":
            raise ValueError("VTK font-file family is unavailable")
        prop.SetFontFile(str(source))
        # Pillow validates the same file independently so VTK cannot silently substitute Arial.
        from PIL import ImageFont

        ImageFont.truetype(str(source), size=16)
    else:
        prop.SetFontFamilyAsString(value["font"])
    path = vtkPath()
    if not tools.StringToPath(prop, value["text"], 72, path):
        raise ValueError("Font outline extraction failed")
    if path.GetNumberOfPoints() > value["max_vertices"]:
        raise ValueError("Raw font path exceeds geometry budget")
    points = np.array([path.GetPoint(i)[:2] for i in range(path.GetNumberOfPoints())], dtype=float)
    codes = [path.GetCodes().GetValue(i) for i in range(path.GetNumberOfPoints())]
    segments = value["segments"]
    contours: list[np.ndarray] = []
    current: list[np.ndarray] = []
    index = 0
    sampled = 0

    def finish() -> None:
        if len(current) < 3:
            return
        ring = np.asarray(current)
        if np.array_equal(ring[0], ring[-1]):
            ring = ring[:-1]
        keep = np.concatenate([[True], np.linalg.norm(np.diff(ring, axis=0), axis=1) > 1e-12])
        ring = ring[keep]
        if len(ring) >= 3:
            contours.append(ring * value["size"] / 256)

    while index < len(codes):
        code = codes[index]
        if code == vtkPath.MOVE_TO:
            finish()
            current = [points[index]]
            index += 1
        elif code == vtkPath.LINE_TO:
            if not current:
                raise ValueError("Line contour lacks a start point")
            current.append(points[index])
            index += 1
        elif code in {vtkPath.CONIC_CURVE, vtkPath.CUBIC_CURVE}:
            degree = 2 if code == vtkPath.CONIC_CURVE else 3
            if (
                not current
                or index + degree > len(points)
                or any(codes[i] != code for i in range(index, index + degree))
            ):
                raise ValueError("Font curve has invalid control-point grouping")
            control = np.vstack([current[-1], points[index : index + degree]])
            for t in np.linspace(0, 1, segments + 1)[1:]:
                work = control.copy()
                for remaining in range(degree, 0, -1):
                    work[:remaining] = (1 - t) * work[:remaining] + t * work[1 : remaining + 1]
                current.append(work[0].copy())
            index += degree
        else:
            raise ValueError("Unsupported font path code")
        sampled += 1
        if sum(len(ring) for ring in contours) + len(current) > value["max_vertices"] // 12:
            raise ValueError("Text outline exceeds geometry budget; reduce text or segments")
    finish()
    if not contours:
        raise ValueError("Text contains no visible font outlines")
    section = m.CrossSection(contours, m.FillRule.NonZero)
    solid = section.extrude(value["depth"])
    if solid.status() != m.Error.NoError or solid.is_empty():
        raise ValueError("Text extrusion failed")
    mesh = solid.to_mesh64()
    vertices, faces = np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts)
    if not value["capped"]:
        z = vertices[faces][:, :, 2]
        caps = np.all(np.isclose(z, 0, rtol=0, atol=value["depth"] * 1e-9), axis=1) | np.all(
            np.isclose(z, value["depth"], rtol=0, atol=value["depth"] * 1e-9), axis=1
        )
        faces = faces[~caps]
    triangles = vertices[faces]
    if len(faces) * 3 > value["max_vertices"] or len(faces) > value["max_faces"]:
        raise ValueError("Text extrusion exceeds geometry budget")
    uv = planar_uv(triangles, value["size"]).reshape(-1, 2)
    return {
        "vertices": triangles.reshape(-1, 3).tolist(),
        "faces": np.arange(len(faces) * 3).reshape(-1, 3).tolist(),
        "uv": uv.tolist(),
        "contours": len(contours),
        "sampled_path_commands": sampled,
        "font_sha256": font_hash,
    }
