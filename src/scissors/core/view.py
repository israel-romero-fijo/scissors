"""Camera framing shared by the preview rasterizers.

The VTK offscreen renderer and Pillow painter share camera placement and
projection arithmetic, including explicit world cameras and orthographic scale.
"""

from dataclasses import dataclass

import numpy as np

from scissors.core.export import view_matrix

BACKGROUND = (242 / 255, 244 / 255, 248 / 255)


def normalized(value: np.ndarray) -> np.ndarray:
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-15)


@dataclass(frozen=True)
class Frame:
    """Camera placement plus the affine screen mapping used by every backend."""

    width: int
    height: int
    eye: np.ndarray
    target: np.ndarray
    rotation: np.ndarray
    projection: str
    fov: float
    focal: float
    ortho_scale: float
    center: np.ndarray
    near: float
    far: float
    offset_x: float
    offset_y: float
    low: np.ndarray
    high: np.ndarray

    @property
    def up(self) -> np.ndarray:
        """World direction drawn towards the top of the image."""
        return self.rotation[1]

    @property
    def light(self) -> np.ndarray:
        """Default light direction, from the target towards the camera."""
        return self.rotation[2]

    def scaled(self, factor: int) -> "Frame":
        """Same view at a larger pixel grid, used by supersampled painters."""
        return Frame(
            width=self.width * factor,
            height=self.height * factor,
            eye=self.eye,
            target=self.target,
            rotation=self.rotation,
            projection=self.projection,
            fov=self.fov,
            focal=self.focal * factor,
            ortho_scale=self.ortho_scale * factor,
            center=self.center,
            near=self.near,
            far=self.far,
            offset_x=self.offset_x,
            offset_y=self.offset_y,
            low=self.low,
            high=self.high,
        )


def build_frame(
    points: np.ndarray,
    *,
    width: int,
    height: int,
    azimuth_degrees: float,
    elevation_degrees: float,
    zoom: float,
    offset_x: float,
    offset_y: float,
    projection: str,
    camera_position: list[float] | None,
    camera_target: list[float] | None,
    camera_up: list[float] | None,
    fov: float,
    near: float,
    far: float | None,
    orthographic_scale: float | None = None,
) -> Frame:
    """Place the camera exactly as the documented preview contract describes."""
    points = np.asarray(points, dtype=float)
    if orthographic_scale is not None:
        if (
            projection != "orthographic"
            or not np.isfinite(orthographic_scale)
            or orthographic_scale <= 0
        ):
            raise ValueError("orthographic_scale must be positive and use orthographic projection")
    low, high = points.min(axis=0), points.max(axis=0)
    target = np.asarray(
        camera_target if camera_target is not None else (low + high) / 2, dtype=float
    )
    radius = max(float(np.linalg.norm(high - low) / 2), near * 2)
    rotation = view_matrix(azimuth_degrees, elevation_degrees)
    half_angle = np.arctan(np.tan(np.radians(fov) / 2) * min(width / height, 1))
    distance = radius / max(np.sin(half_angle), 1e-6) * 1.15 + near
    eye = (
        np.asarray(camera_position, dtype=float)
        if camera_position is not None
        else target + rotation[2] * distance
    )
    if camera_position is not None or camera_up is not None:
        back = eye - target
        if not np.isfinite(np.linalg.norm(back)) or np.linalg.norm(back) < 1e-12:
            raise ValueError("Camera position and target must differ")
        back = normalized(back)
        up = np.asarray(camera_up if camera_up is not None else [0, 0, 1], dtype=float)
        if camera_up is None and abs(float(back @ up)) > 1 - 1e-9:
            up = np.array([0, -1, 0])
        right = np.cross(up, back)
        if not np.isfinite(np.linalg.norm(right)) or np.linalg.norm(right) < 1e-12:
            raise ValueError("Camera up must not be parallel to its view direction")
        right = normalized(right)
        rotation = np.vstack([right, np.cross(back, right), back])
    projected = (points - eye) @ rotation.T
    depths = -projected[:, 2]
    far = float(far) if far is not None else max(near + radius * 2, float(depths.max()) + radius)
    if far <= near:
        raise ValueError("Camera far distance must exceed near distance")
    extent = np.ptp(projected[:, :2], axis=0)
    ortho_scale = (
        min(max(width - 40, 1) / max(extent[0], 1e-12), max(height - 40, 1) / max(extent[1], 1e-12))
        * zoom
    )
    if orthographic_scale is not None:
        ortho_scale = height / orthographic_scale * zoom
    center = (
        (projected[:, :2].min(axis=0) + projected[:, :2].max(axis=0)) / 2
        if camera_position is None
        else np.zeros(2)
    )
    focal = height / (2 * np.tan(np.radians(fov) / 2)) * zoom
    if (
        not np.isfinite(
            np.concatenate([eye, target, rotation.ravel(), center, [near, far, focal, ortho_scale]])
        ).all()
        or focal <= 0
        or ortho_scale <= 0
    ):
        raise ValueError("Camera arithmetic exceeds the finite numerical range")
    return Frame(
        width=width,
        height=height,
        eye=eye,
        target=target,
        rotation=rotation,
        projection=projection,
        fov=fov,
        focal=focal,
        ortho_scale=ortho_scale,
        center=center,
        near=near,
        far=far,
        offset_x=offset_x,
        offset_y=offset_y,
        low=low,
        high=high,
    )


def project(frame: Frame, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map world points to screen pixels and positive camera-forward distance."""
    camera = (np.asarray(points, dtype=float) - frame.eye) @ frame.rotation.T
    depth = -camera[:, 2]
    if frame.projection == "perspective":
        xy = camera[:, :2] / np.maximum(depth, frame.near)[:, None] * frame.focal
    else:
        xy = (camera[:, :2] - frame.center) * frame.ortho_scale
    screen = xy * np.array([1.0, -1.0]) + np.array(
        [
            frame.width / 2 + frame.offset_x * frame.width,
            frame.height / 2 + frame.offset_y * frame.height,
        ]
    )
    return screen, depth


def clipped_face_mask(
    vertices: np.ndarray,
    faces: np.ndarray,
    clip_axis: str | None,
    clip_position: float | None,
    clip_keep: str,
) -> np.ndarray:
    """Hide whole triangles on the discarded side of a world-axis section plane."""
    if clip_axis is None or clip_position is None:
        return np.ones(len(faces), dtype=bool)
    axis = "xyz".index(clip_axis)
    along = np.asarray(vertices)[faces][:, :, axis].mean(axis=1)
    return along >= clip_position if clip_keep == "positive" else along <= clip_position
