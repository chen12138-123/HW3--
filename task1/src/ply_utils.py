from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


C0 = 0.28209479177387814


@dataclass
class PointCloud:
    points: np.ndarray
    colors: np.ndarray

    def copy(self) -> "PointCloud":
        return PointCloud(self.points.copy(), self.colors.copy())


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def clamp_colors(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors, dtype=np.float32)
    if colors.max(initial=0) <= 1.0:
        colors = colors * 255.0
    return np.clip(colors, 0, 255).astype(np.uint8)


def save_xyz_rgb_ply(path: str | Path, points: np.ndarray, colors: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    points = np.asarray(points, dtype=np.float32)
    colors = clamp_colors(colors)

    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def save_3dgs_compatible_ply(path: str | Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write a minimal 3DGS-style PLY with SH degree 0 and default splat params."""
    try:
        from plyfile import PlyData, PlyElement
    except Exception:
        save_xyz_rgb_ply(path, points, colors)
        return

    path = Path(path)
    ensure_dir(path.parent)
    points = np.asarray(points, dtype=np.float32)
    rgb01 = clamp_colors(colors).astype(np.float32) / 255.0
    f_dc = (rgb01 - 0.5) / C0
    n = len(points)

    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
    ]
    dtype += [(f"f_rest_{i}", "f4") for i in range(45)]
    dtype += [
        ("opacity", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
    ]

    vertices = np.empty(n, dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertices["nx"], vertices["ny"], vertices["nz"] = 0.0, 0.0, 0.0
    vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    for i in range(45):
        vertices[f"f_rest_{i}"] = 0.0
    vertices["opacity"] = 4.59512
    vertices["scale_0"], vertices["scale_1"], vertices["scale_2"] = -5.4, -5.4, -5.4
    vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"] = 1.0, 0.0, 0.0, 0.0

    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(path))


def load_ply(path: str | Path) -> PointCloud:
    path = Path(path)
    try:
        from plyfile import PlyData

        ply = PlyData.read(str(path))
        vertex = ply["vertex"]
        points = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32)
        names = vertex.data.dtype.names or ()
        if {"red", "green", "blue"}.issubset(names):
            colors = np.column_stack([vertex["red"], vertex["green"], vertex["blue"]]).astype(np.uint8)
        elif {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
            sh = np.column_stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]]).astype(np.float32)
            colors = np.clip((sh * C0 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        else:
            colors = np.full((len(points), 3), 180, dtype=np.uint8)
        return PointCloud(points=points, colors=colors)
    except Exception:
        return _load_ascii_xyz_rgb(path)


def _load_ascii_xyz_rgb(path: Path) -> PointCloud:
    with path.open("r", encoding="ascii", errors="ignore") as f:
        lines = f.readlines()
    header_end = 0
    for i, line in enumerate(lines):
        if line.strip() == "end_header":
            header_end = i + 1
            break

    pts = []
    colors = []
    for line in lines[header_end:]:
        parts = line.strip().split()
        if len(parts) >= 6:
            pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
            colors.append([float(parts[3]), float(parts[4]), float(parts[5])])
    if not pts:
        return PointCloud(np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8))
    return PointCloud(np.asarray(pts, dtype=np.float32), clamp_colors(np.asarray(colors)))


def transform_cloud(
    cloud: PointCloud,
    scale: float | Iterable[float] = 1.0,
    translation: Iterable[float] = (0.0, 0.0, 0.0),
    rotation_z_deg: float = 0.0,
) -> PointCloud:
    pts = cloud.points.astype(np.float32).copy()
    scale_arr = np.asarray(scale if np.iterable(scale) else [scale, scale, scale], dtype=np.float32)
    pts *= scale_arr
    theta = np.deg2rad(rotation_z_deg)
    rot = np.array(
        [[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    pts = pts @ rot.T
    pts += np.asarray(translation, dtype=np.float32)
    return PointCloud(pts, cloud.colors.copy())


def merge_clouds(clouds: Iterable[PointCloud]) -> PointCloud:
    clouds = list(clouds)
    if not clouds:
        return PointCloud(np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8))
    return PointCloud(
        np.vstack([cloud.points for cloud in clouds]).astype(np.float32),
        np.vstack([cloud.colors for cloud in clouds]).astype(np.uint8),
    )


def _property_names(ply_data) -> tuple[str, ...]:
    return tuple(p.name for p in ply_data["vertex"].properties)


def load_vertex_array(path: str | Path) -> np.ndarray:
    """Load the raw vertex structured array from a PLY file."""
    try:
        from plyfile import PlyData
    except Exception as exc:  # pragma: no cover - dependency should be installed in this project
        raise RuntimeError("plyfile is required for raw 3DGS PLY operations") from exc
    ply = PlyData.read(str(path))
    return ply["vertex"].data.copy()


def save_vertex_array(path: str | Path, vertices: np.ndarray) -> None:
    """Write a raw structured vertex array back to a binary PLY file."""
    try:
        from plyfile import PlyData, PlyElement
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("plyfile is required for raw 3DGS PLY operations") from exc
    path = Path(path)
    ensure_dir(path.parent)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(path))


def transform_3dgs_vertices(
    vertices: np.ndarray,
    *,
    scale: float | Iterable[float] = 1.0,
    translation: Iterable[float] = (0.0, 0.0, 0.0),
    rotation_z_deg: float = 0.0,
) -> np.ndarray:
    """Transform a 3DGS-compatible structured vertex array in-place-safe."""
    out = vertices.copy()
    names = out.dtype.names or ()
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("PLY vertex array must contain x/y/z fields")

    scale_arr = np.asarray(scale if np.iterable(scale) else [scale, scale, scale], dtype=np.float32)
    theta = np.deg2rad(rotation_z_deg)
    rot = np.array(
        [[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    pts = np.column_stack([out["x"], out["y"], out["z"]]).astype(np.float32)
    pts *= scale_arr
    pts = pts @ rot.T
    pts += np.asarray(translation, dtype=np.float32)
    out["x"], out["y"], out["z"] = pts[:, 0], pts[:, 1], pts[:, 2]

    scale_fields = [name for name in ("scale_0", "scale_1", "scale_2") if name in names]
    if scale_fields:
        log_scale = np.log(float(np.mean(scale_arr)))
        for field in scale_fields:
            out[field] = out[field] + log_scale
    return out


def downsample_vertices(vertices: np.ndarray, max_points: int, seed: int = 20260623) -> np.ndarray:
    if max_points <= 0 or len(vertices) <= max_points:
        return vertices.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(vertices), size=max_points, replace=False)
    idx.sort()
    return vertices[idx].copy()


def filter_vertices_by_color(
    vertices: np.ndarray,
    *,
    saturation_min: float = 0.12,
    value_min: float = 0.08,
    keep_neutral_bright: bool = False,
) -> np.ndarray:
    """Remove low-saturation background splats from generated assets."""
    cloud = vertex_array_to_point_cloud(vertices)
    rgb = cloud.colors.astype(np.float32) / 255.0
    maxc = rgb.max(axis=1)
    minc = rgb.min(axis=1)
    saturation = np.zeros_like(maxc)
    valid = maxc > 1e-6
    saturation[valid] = (maxc[valid] - minc[valid]) / maxc[valid]
    keep = (saturation >= saturation_min) & (maxc >= value_min)
    if keep_neutral_bright:
        neutral_bright = (saturation < saturation_min) & (maxc > 0.72) & (minc > 0.52)
        keep |= neutral_bright
    if keep.sum() < max(1000, len(vertices) * 0.08):
        return vertices.copy()
    return vertices[keep].copy()


def merge_3dgs_vertex_arrays(arrays: Iterable[np.ndarray]) -> np.ndarray:
    arrays = [arr for arr in arrays if len(arr) > 0]
    if not arrays:
        raise ValueError("No vertices to merge")
    first_dtype = arrays[0].dtype
    for arr in arrays[1:]:
        if arr.dtype != first_dtype:
            raise ValueError("All 3DGS vertex arrays must share the same dtype before merge")
    return np.concatenate(arrays)


def vertex_array_to_point_cloud(vertices: np.ndarray) -> PointCloud:
    points = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float32)
    names = vertices.dtype.names or ()
    if {"red", "green", "blue"}.issubset(names):
        colors = np.column_stack([vertices["red"], vertices["green"], vertices["blue"]]).astype(np.uint8)
    elif {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        sh = np.column_stack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]]).astype(np.float32)
        colors = np.clip((sh * C0 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    else:
        colors = np.full((len(points), 3), 180, dtype=np.uint8)
    return PointCloud(points, colors)


def cloud_stats(path: str | Path) -> dict[str, float]:
    cloud = load_ply(path)
    if len(cloud.points) == 0:
        return {"points": 0, "extent_x": 0.0, "extent_y": 0.0, "extent_z": 0.0}
    mins = cloud.points.min(axis=0)
    maxs = cloud.points.max(axis=0)
    ext = maxs - mins
    return {
        "points": int(len(cloud.points)),
        "extent_x": float(ext[0]),
        "extent_y": float(ext[1]),
        "extent_z": float(ext[2]),
    }
