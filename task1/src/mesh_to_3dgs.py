from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh

try:
    from .ply_utils import save_3dgs_compatible_ply, save_xyz_rgb_ply
except ImportError:
    from ply_utils import save_3dgs_compatible_ply, save_xyz_rgb_ply


def sample_mesh_colors(mesh: trimesh.Trimesh, face_indices: np.ndarray, points: np.ndarray) -> np.ndarray:
    visual = mesh.visual
    if getattr(visual, "kind", None) == "texture":
        try:
            color_visual = visual.to_color()
            vertex_colors = np.asarray(color_visual.vertex_colors[:, :3], dtype=np.float32)
            face_vertices = mesh.faces[face_indices]
            return vertex_colors[face_vertices].mean(axis=1)
        except Exception:
            pass
    if getattr(visual, "kind", None) == "vertex" and hasattr(visual, "vertex_colors"):
        face_vertices = mesh.faces[face_indices]
        vertex_colors = np.asarray(visual.vertex_colors[:, :3], dtype=np.float32)
        return vertex_colors[face_vertices].mean(axis=1)
    if getattr(visual, "kind", None) == "face" and hasattr(visual, "face_colors"):
        return np.asarray(visual.face_colors[face_indices, :3], dtype=np.float32)
    if getattr(visual, "material", None) is not None and hasattr(visual.material, "baseColorFactor"):
        base = np.asarray(visual.material.baseColorFactor[:3], dtype=np.float32)
        return np.tile(base * 255.0 if base.max() <= 1 else base, (len(points), 1))
    return np.full((len(points), 3), 170, dtype=np.float32)


def convert_mesh_to_3dgs(mesh_path: Path, out_path: Path, num_points: int, xyzrgb: bool = False) -> None:
    mesh = trimesh.load(str(mesh_path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    points, face_indices = trimesh.sample.sample_surface_even(mesh, num_points)
    colors = sample_mesh_colors(mesh, face_indices, points)
    if xyzrgb:
        save_xyz_rgb_ply(out_path, points, colors)
    else:
        save_3dgs_compatible_ply(out_path, points, colors)
    print(f"Converted {mesh_path} -> {out_path} with {len(points)} sampled splats.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a textured mesh into a 3DGS-compatible PLY.")
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--points", type=int, default=100000)
    parser.add_argument("--xyzrgb", action="store_true", help="Write a simple XYZRGB PLY instead of 3DGS SH properties.")
    args = parser.parse_args()
    convert_mesh_to_3dgs(args.mesh, args.out, args.points, args.xyzrgb)


if __name__ == "__main__":
    main()
