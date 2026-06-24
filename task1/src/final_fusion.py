from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .demo_pipeline import configure_matplotlib, make_video, plot_cloud, render_views
    from .ply_utils import (
        downsample_vertices,
        filter_vertices_by_color,
        load_ply,
        load_vertex_array,
        merge_3dgs_vertex_arrays,
        save_vertex_array,
        save_xyz_rgb_ply,
        transform_3dgs_vertices,
        transform_cloud,
        vertex_array_to_point_cloud,
    )
except ImportError:
    from demo_pipeline import configure_matplotlib, make_video, plot_cloud, render_views
    from ply_utils import (
        downsample_vertices,
        filter_vertices_by_color,
        load_ply,
        load_vertex_array,
        merge_3dgs_vertex_arrays,
        save_vertex_array,
        save_xyz_rgb_ply,
        transform_3dgs_vertices,
        transform_cloud,
        vertex_array_to_point_cloud,
    )


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
OUT = ROOT / "output" / "final_fused"
LOG = ROOT / "output" / "logs"


def stats(vertices: np.ndarray) -> dict[str, Any]:
    pts = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float32)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    return {
        "points": int(len(vertices)),
        "min": mins.tolist(),
        "max": maxs.tolist(),
        "extent": (maxs - mins).tolist(),
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    configure_matplotlib()
    FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)

    paths = {
        "background": ROOT / "output" / "real_3dgs_bg" / "point_cloud" / "iteration_7000" / "point_cloud.ply",
        "object_a": ROOT / "output" / "object_a_official_3dgs" / "point_cloud" / "iteration_7000" / "point_cloud.ply",
        "object_b": ROOT / "output" / "final_assets" / "object_b_final" / "model.ply",
        "object_c": ROOT / "output" / "final_assets" / "object_c_final" / "model.ply",
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    bg = downsample_vertices(load_vertex_array(paths["background"]), 180000, seed=11)
    obj_a = downsample_vertices(load_vertex_array(paths["object_a"]), 90000, seed=17)
    obj_b = filter_vertices_by_color(load_vertex_array(paths["object_b"]), saturation_min=0.55, keep_neutral_bright=False)
    obj_c = filter_vertices_by_color(load_vertex_array(paths["object_c"]), saturation_min=0.35, keep_neutral_bright=False)

    obj_a = transform_3dgs_vertices(obj_a, scale=0.010, translation=(-1.55, -0.78, 0.58), rotation_z_deg=22)
    obj_b = transform_3dgs_vertices(obj_b, scale=1.15, translation=(0.80, -0.45, 0.56), rotation_z_deg=-14)
    obj_c = transform_3dgs_vertices(obj_c, scale=1.25, translation=(1.65, 0.95, 0.50), rotation_z_deg=18)

    fused = merge_3dgs_vertex_arrays([bg, obj_a, obj_b, obj_c])
    model_path = OUT / "model.ply"
    save_vertex_array(model_path, fused)
    save_vertex_array(ROOT / "output" / "fused" / "model.ply", fused)

    fused_cloud = vertex_array_to_point_cloud(fused)
    save_xyz_rgb_ply(OUT / "preview_xyzrgb.ply", fused_cloud.points, fused_cloud.colors)
    core_mask = (
        (fused_cloud.points[:, 0] > -4.5)
        & (fused_cloud.points[:, 0] < 4.2)
        & (fused_cloud.points[:, 1] > -3.2)
        & (fused_cloud.points[:, 1] < 3.0)
        & (fused_cloud.points[:, 2] > -1.1)
        & (fused_cloud.points[:, 2] < 3.2)
    )
    core_cloud = type(fused_cloud)(fused_cloud.points[core_mask], fused_cloud.colors[core_mask])
    save_xyz_rgb_ply(OUT / "preview_core_xyzrgb.ply", core_cloud.points, core_cloud.colors)
    plot_cloud(core_cloud, FIG / "final_fused_scene.png", "Final fused 3D scene: real 3DGS background + A/B/C assets", elev=23, azim=-42, point_size=0.75, max_points=90000)
    plot_cloud(core_cloud, FIG / "final_fused_scene_top.png", "Final fused scene top view", elev=78, azim=-85, point_size=0.75, max_points=90000)
    render_views(core_cloud, FIG, "final_fused_scene", elev=24)
    make_video(core_cloud, OUT / "roaming_video.mp4", OUT / "video_frames", frames=72)

    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "Raw 3DGS-compatible PLY merge after 3D coordinate transforms; no 2D screenshot compositing.",
        "inputs": {name: str(path) for name, path in paths.items()},
        "transforms": {
            "object_a": {"scale": 0.010, "translation": [-1.55, -0.78, 0.58], "rotation_z_deg": 22},
            "object_b": {"scale": 1.15, "translation": [0.80, -0.45, 0.56], "rotation_z_deg": -14, "color_filter": "saturation >= 0.55"},
            "object_c": {"scale": 1.25, "translation": [1.65, 0.95, 0.50], "rotation_z_deg": 18, "color_filter": "saturation >= 0.35"},
        },
        "components": {
            "background": stats(bg),
            "object_a": stats(obj_a),
            "object_b": stats(obj_b),
            "object_c": stats(obj_c),
            "fused": stats(fused),
            "preview_core_points": int(len(core_cloud.points)),
        },
        "outputs": {
            "model_ply": str(model_path),
            "preview_xyzrgb": str(OUT / "preview_xyzrgb.ply"),
            "video": str(OUT / "roaming_video.mp4"),
            "figures": [
                str(FIG / "final_fused_scene.png"),
                str(FIG / "final_fused_scene_top.png"),
                str(FIG / "final_fused_scene_view_1.png"),
                str(FIG / "final_fused_scene_view_2.png"),
                str(FIG / "final_fused_scene_view_3.png"),
                str(FIG / "final_fused_scene_view_4.png"),
            ],
        },
    }
    write_json(OUT / "fusion_metadata.json", meta)
    write_json(LOG / "final_fusion_metadata.json", meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
