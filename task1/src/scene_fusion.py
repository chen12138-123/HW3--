from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .demo_pipeline import make_video, plot_cloud, render_views
    from .ply_utils import load_ply, merge_clouds, save_xyz_rgb_ply, transform_cloud
except ImportError:
    from demo_pipeline import make_video, plot_cloud, render_views
    from ply_utils import load_ply, merge_clouds, save_xyz_rgb_ply, transform_cloud


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse background and A/B/C assets into a unified 3DGS-style scene.")
    parser.add_argument("--bg", type=Path, required=True, help="Background model directory containing model.ply.")
    parser.add_argument("--obj_a", type=Path, required=True, help="Object A model directory containing model.ply.")
    parser.add_argument("--obj_b", type=Path, required=True, help="Object B model directory containing model.ply.")
    parser.add_argument("--obj_c", type=Path, required=True, help="Object C model directory containing model.ply.")
    parser.add_argument("--output_render", type=Path, required=True, help="Output MP4 path. A PLY is written next to it.")
    parser.add_argument("--skip-video", action="store_true", help="Only write PLY and figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bg = load_ply(args.bg / "model.ply")
    obj_a = transform_cloud(load_ply(args.obj_a / "model.ply"), scale=0.62, translation=(-2.35, -1.05, 0.88), rotation_z_deg=20)
    obj_b = transform_cloud(load_ply(args.obj_b / "model.ply"), scale=0.58, translation=(1.9, -0.55, 0.25), rotation_z_deg=-9)
    obj_c = transform_cloud(load_ply(args.obj_c / "model.ply"), scale=0.72, translation=(0.2, 2.15, 0.12), rotation_z_deg=12)
    fused = merge_clouds([bg, obj_a, obj_b, obj_c])

    args.output_render.parent.mkdir(parents=True, exist_ok=True)
    out_ply = args.output_render.with_suffix(".ply")
    save_xyz_rgb_ply(out_ply, fused.points, fused.colors)
    save_xyz_rgb_ply(args.output_render.parent / "fused" / "model.ply", fused.points, fused.colors)
    plot_cloud(fused, Path("figures/fused_scene.png"), "Fused scene: background + A/B/C assets", elev=22, azim=-48, point_size=0.55)
    plot_cloud(fused, Path("figures/fused_scene_top.png"), "Fused scene top view", elev=78, azim=-90, point_size=0.5)
    render_views(fused, Path("figures"), "fused_scene")

    if not args.skip_video:
        make_video(fused, args.output_render, args.output_render.parent / "video_frames")
    print(f"Merged PLY saved to {out_ply}")
    if not args.skip_video:
        print(f"Roaming video saved to {args.output_render}")


if __name__ == "__main__":
    main()
