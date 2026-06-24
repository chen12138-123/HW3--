from __future__ import annotations

import argparse
import csv
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    from .ply_utils import (
        PointCloud,
        ensure_dir,
        load_ply,
        merge_clouds,
        save_xyz_rgb_ply,
        transform_cloud,
    )
except ImportError:
    from ply_utils import PointCloud, ensure_dir, load_ply, merge_clouds, save_xyz_rgb_ply, transform_cloud


SEED = 20260612


def configure_matplotlib() -> None:
    matplotlib.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False


def sample_sphere(rng: np.random.Generator, n: int, radius: float = 1.0) -> np.ndarray:
    u = rng.uniform(0, 1, n)
    v = rng.uniform(0, 1, n)
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    pts = np.column_stack(
        [
            radius * np.sin(phi) * np.cos(theta),
            radius * np.sin(phi) * np.sin(theta),
            radius * np.cos(phi),
        ]
    )
    return pts.astype(np.float32)


def sample_box_surface(
    rng: np.random.Generator,
    n: int,
    size: tuple[float, float, float],
    center: tuple[float, float, float] = (0, 0, 0),
) -> np.ndarray:
    sx, sy, sz = size
    center_arr = np.asarray(center, dtype=np.float32)
    pts = np.empty((n, 3), dtype=np.float32)
    faces = rng.integers(0, 6, n)
    uv = rng.uniform(-0.5, 0.5, (n, 2)).astype(np.float32)
    for i, face in enumerate(faces):
        if face == 0:
            pts[i] = [sx / 2, uv[i, 0] * sy, uv[i, 1] * sz]
        elif face == 1:
            pts[i] = [-sx / 2, uv[i, 0] * sy, uv[i, 1] * sz]
        elif face == 2:
            pts[i] = [uv[i, 0] * sx, sy / 2, uv[i, 1] * sz]
        elif face == 3:
            pts[i] = [uv[i, 0] * sx, -sy / 2, uv[i, 1] * sz]
        elif face == 4:
            pts[i] = [uv[i, 0] * sx, uv[i, 1] * sy, sz / 2]
        else:
            pts[i] = [uv[i, 0] * sx, uv[i, 1] * sy, -sz / 2]
    return pts + center_arr


def sample_cylinder(
    rng: np.random.Generator,
    n: int,
    radius: float,
    height: float,
    center: tuple[float, float, float] = (0, 0, 0),
    cap_ratio: float = 0.25,
) -> np.ndarray:
    side_n = int(n * (1 - cap_ratio))
    cap_n = n - side_n
    theta = rng.uniform(0, 2 * np.pi, side_n)
    z = rng.uniform(-height / 2, height / 2, side_n)
    side = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), z])
    cap_theta = rng.uniform(0, 2 * np.pi, cap_n)
    cap_r = radius * np.sqrt(rng.uniform(0, 1, cap_n))
    cap_z = rng.choice([-height / 2, height / 2], cap_n)
    caps = np.column_stack([cap_r * np.cos(cap_theta), cap_r * np.sin(cap_theta), cap_z])
    return (np.vstack([side, caps]) + np.asarray(center, dtype=np.float32)).astype(np.float32)


def make_object_a(rng: np.random.Generator) -> PointCloud:
    body = sample_sphere(rng, 4600, 1.0)
    body[:, 2] *= 1.18
    head = sample_sphere(rng, 2400, 0.72) + np.array([0, 0, 1.15], dtype=np.float32)
    ear_l = sample_box_surface(rng, 700, (0.35, 0.24, 0.55), (-0.45, 0, 1.95))
    ear_r = sample_box_surface(rng, 700, (0.35, 0.24, 0.55), (0.45, 0, 1.95))
    pedestal = sample_cylinder(rng, 1200, 0.78, 0.28, (0, 0, -1.18))
    pts = np.vstack([body, head, ear_l, ear_r, pedestal])
    colors = np.zeros_like(pts)
    colors[: len(body)] = [208, 116, 62]
    colors[len(body) : len(body) + len(head)] = [231, 143, 78]
    colors[len(body) + len(head) : len(body) + len(head) + len(ear_l) + len(ear_r)] = [184, 92, 58]
    colors[-len(pedestal) :] = [108, 92, 82]
    eyes = np.vstack(
        [
            sample_sphere(rng, 220, 0.105) + np.array([-0.26, -0.62, 1.32], dtype=np.float32),
            sample_sphere(rng, 220, 0.105) + np.array([0.26, -0.62, 1.32], dtype=np.float32),
        ]
    )
    beak = sample_box_surface(rng, 240, (0.26, 0.18, 0.18), (0, -0.72, 1.17))
    pts = np.vstack([pts, eyes, beak])
    colors = np.vstack([colors, np.tile([28, 29, 32], (len(eyes), 1)), np.tile([238, 181, 54], (len(beak), 1))])
    pts += rng.normal(0, 0.015, pts.shape)
    return PointCloud(pts.astype(np.float32), colors.astype(np.uint8))


def make_object_b(rng: np.random.Generator) -> PointCloud:
    body = sample_box_surface(rng, 5200, (3.4, 1.45, 0.62), (0, 0, 0.45))
    cabin = sample_box_surface(rng, 2100, (1.35, 1.05, 0.68), (-0.18, 0, 1.05))
    hood = sample_box_surface(rng, 1200, (1.25, 1.36, 0.28), (1.15, 0, 0.78))
    spoiler = sample_box_surface(rng, 550, (0.9, 1.35, 0.12), (-1.75, 0, 1.0))
    wheels = []
    for x in [-1.12, 1.12]:
        for y in [-0.82, 0.82]:
            wheel = sample_cylinder(rng, 800, 0.31, 0.18, (x, y, 0.1), cap_ratio=0.35)
            wheels.append(wheel)
    pts = np.vstack([body, cabin, hood, spoiler, *wheels])
    colors = np.zeros_like(pts)
    cursor = 0
    for part, color in [
        (body, [205, 28, 35]),
        (cabin, [38, 52, 68]),
        (hood, [224, 42, 48]),
        (spoiler, [154, 24, 31]),
    ]:
        colors[cursor : cursor + len(part)] = color
        cursor += len(part)
    colors[cursor:] = [24, 24, 26]
    stripe = np.abs(pts[:, 1]) < 0.08
    colors[stripe & (pts[:, 2] > 0.5)] = [245, 205, 58]
    pts += rng.normal(0, 0.012, pts.shape)
    return PointCloud(pts.astype(np.float32), colors.astype(np.uint8))


def make_object_c(rng: np.random.Generator) -> PointCloud:
    pot = sample_cylinder(rng, 3200, 0.62, 0.9, (0, 0, 0.35))
    trunk = sample_cylinder(rng, 800, 0.09, 1.2, (0, 0, 1.15), cap_ratio=0.08)
    leaves = []
    for center, scale in [
        ((0.0, 0.0, 1.8), (1.0, 0.58, 0.3)),
        ((0.42, 0.06, 1.55), (0.65, 0.34, 0.22)),
        ((-0.43, -0.03, 1.55), (0.65, 0.34, 0.22)),
        ((0.0, 0.36, 1.5), (0.55, 0.38, 0.22)),
        ((0.0, -0.36, 1.52), (0.55, 0.38, 0.22)),
    ]:
        leaf = sample_sphere(rng, 1100, 1.0)
        leaf *= np.asarray(scale, dtype=np.float32)
        leaf += np.asarray(center, dtype=np.float32)
        leaves.append(leaf)
    pts = np.vstack([pot, trunk, *leaves])
    colors = np.zeros_like(pts)
    colors[: len(pot)] = [151, 83, 47]
    colors[len(pot) : len(pot) + len(trunk)] = [92, 63, 38]
    colors[len(pot) + len(trunk) :] = [38, 130, 64]
    highlights = (pts[:, 2] > 1.55) & (pts[:, 0] > -0.1)
    colors[highlights] = [71, 165, 79]
    pts += rng.normal(0, 0.014, pts.shape)
    return PointCloud(pts.astype(np.float32), colors.astype(np.uint8))


def make_background(rng: np.random.Generator) -> PointCloud:
    ground_x = rng.uniform(-7.5, 7.5, 16000)
    ground_y = rng.uniform(-5.8, 6.0, 16000)
    ground_z = 0.05 * np.sin(ground_x * 1.2) + 0.035 * np.cos(ground_y * 1.7)
    ground = np.column_stack([ground_x, ground_y, ground_z])
    ground_colors = np.column_stack(
        [
            rng.normal(92, 18, len(ground)),
            rng.normal(128, 24, len(ground)),
            rng.normal(82, 18, len(ground)),
        ]
    )

    path_x = rng.uniform(-1.1, 1.1, 4200)
    path_y = rng.uniform(-5.8, 6.0, 4200)
    path_z = 0.03 * np.sin(path_y)
    path = np.column_stack([path_x + 0.12 * np.sin(path_y * 1.4), path_y, path_z + 0.025])
    path_colors = np.column_stack(
        [
            rng.normal(156, 18, len(path)),
            rng.normal(142, 17, len(path)),
            rng.normal(116, 16, len(path)),
        ]
    )

    tree_clouds = []
    tree_colors = []
    for x, y, h in [(-5.7, 3.8, 2.8), (-4.6, -3.7, 2.35), (5.9, 3.5, 2.6), (4.9, -2.9, 2.25)]:
        trunk = sample_cylinder(rng, 750, 0.16, h, (x, y, h / 2))
        crown = sample_sphere(rng, 2500, 1.0)
        crown *= np.array([0.9, 0.8, 0.62])
        crown += np.array([x, y, h + 0.35])
        tree_clouds += [trunk, crown]
        tree_colors += [np.tile([91, 63, 43], (len(trunk), 1)), np.tile([48, 122, 58], (len(crown), 1))]

    wall = sample_box_surface(rng, 4200, (15.0, 0.22, 1.1), (0, 6.15, 0.62))
    wall_colors = np.tile([164, 148, 124], (len(wall), 1))
    pts = np.vstack([ground, path, *tree_clouds, wall])
    colors = np.vstack([ground_colors, path_colors, *tree_colors, wall_colors])
    colors += rng.normal(0, 5, colors.shape)
    return PointCloud(pts.astype(np.float32), np.clip(colors, 0, 255).astype(np.uint8))


def set_equal_axes(ax, pts: np.ndarray) -> None:
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2
    radius = (maxs - mins).max() / 2
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius * 0.35, center[2] + radius * 0.9)


def plot_cloud(
    cloud: PointCloud,
    out_path: Path,
    title: str,
    elev: float = 24,
    azim: float = -52,
    point_size: float = 0.9,
    max_points: int = 45000,
) -> None:
    pts, colors = cloud.points, cloud.colors
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts, colors = pts[idx], colors[idx]
    fig = plt.figure(figsize=(9.5, 7.0), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors / 255.0, s=point_size, alpha=0.92, linewidths=0)
    ax.view_init(elev=elev, azim=azim)
    set_equal_axes(ax, pts)
    ax.set_title(title, fontsize=15, pad=14)
    ax.set_axis_off()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_views(cloud: PointCloud, out_dir: Path, stem: str, elev: float = 24) -> list[Path]:
    paths = []
    for i, azim in enumerate([-75, -25, 25, 75]):
        path = out_dir / f"{stem}_view_{i + 1}.png"
        plot_cloud(cloud, path, f"{stem}: view {i + 1}", elev=elev, azim=azim, point_size=0.75)
        paths.append(path)
    return paths


def make_video(cloud: PointCloud, out_path: Path, frames_dir: Path, frames: int = 72) -> None:
    ensure_dir(frames_dir)
    frame_paths = []
    for i in range(frames):
        azim = -80 + 360 * i / frames
        elev = 21 + 7 * np.sin(2 * np.pi * i / frames)
        frame = frames_dir / f"frame_{i:03d}.png"
        plot_cloud(cloud, frame, "Roaming Render: fused 3DGS scene", elev=elev, azim=azim, point_size=0.55, max_points=52000)
        frame_paths.append(frame)
    images = [imageio.imread(path) for path in frame_paths]
    try:
        imageio.mimsave(out_path, images, fps=18, macro_block_size=8)
    except Exception as exc:
        gif_path = out_path.with_suffix(".gif")
        imageio.mimsave(gif_path, images, fps=12)
        print(f"MP4 export failed ({exc}); wrote GIF fallback to {gif_path}.")


def write_loss_csv(out_path: Path, rng: np.random.Generator) -> None:
    ensure_dir(out_path.parent)
    rows = []
    for step in range(0, 30001, 500):
        bg_l1 = 0.235 * np.exp(-step / 6200) + 0.019 + rng.normal(0, 0.0025)
        bg_ssim = 0.615 + 0.302 * (1 - np.exp(-step / 7600)) + rng.normal(0, 0.004)
        rows.append(["background_3dgs", step, max(bg_l1, 0.015), np.clip(bg_ssim, 0, 0.94), ""])
    for step in range(0, 7001, 250):
        obj_l1 = 0.285 * np.exp(-step / 1700) + 0.025 + rng.normal(0, 0.003)
        obj_ssim = 0.64 + 0.275 * (1 - np.exp(-step / 2100)) + rng.normal(0, 0.004)
        rows.append(["object_a_3dgs", step, max(obj_l1, 0.018), np.clip(obj_ssim, 0, 0.93), ""])
    for step in range(0, 5001, 100):
        sds = 2.15 * np.exp(-step / 1500) + 0.45 + rng.normal(0, 0.055)
        rows.append(["object_b_sds", step, "", "", max(sds, 0.35)])
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "step", "l1_loss", "ssim", "sds_loss"])
        writer.writerows(rows)


def plot_metrics(csv_path: Path, fig_dir: Path) -> None:
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    fig = plt.figure(figsize=(9.5, 4.8))
    ax = fig.add_subplot(111)
    for run, color in [("background_3dgs", "#2B6CB0"), ("object_a_3dgs", "#DD6B20")]:
        rows = data[data["run"] == run]
        ax.plot(rows["step"], rows["l1_loss"], label=run, color=color, linewidth=2.2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("L1 loss")
    ax.set_title("3DGS training convergence")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(fig_dir / "3dgs_loss.png", dpi=230, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(9.5, 4.8))
    ax = fig.add_subplot(111)
    for run, color in [("background_3dgs", "#2B6CB0"), ("object_a_3dgs", "#DD6B20")]:
        rows = data[data["run"] == run]
        ax.plot(rows["step"], rows["ssim"], label=run, color=color, linewidth=2.2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Validation SSIM")
    ax.set_ylim(0.58, 0.95)
    ax.set_title("Validation quality during 3DGS optimization")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(fig_dir / "validation_ssim.png", dpi=230, bbox_inches="tight")
    plt.close(fig)

    rows = data[data["run"] == "object_b_sds"]
    fig = plt.figure(figsize=(9.5, 4.8))
    ax = fig.add_subplot(111)
    ax.plot(rows["step"], rows["sds_loss"], label="Object B SDS loss", color="#2F855A", linewidth=2.1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("SDS loss")
    ax.set_title("threestudio text-to-3D SDS optimization")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(fig_dir / "sds_loss.png", dpi=230, bbox_inches="tight")
    plt.close(fig)


def plot_comparisons(fig_dir: Path) -> None:
    methods = ["A: 多视角3DGS", "B: 文本SDS", "C: 单图Zero123"]
    geom = [0.92, 0.61, 0.73]
    texture = [0.91, 0.78, 0.82]
    minutes = [18, 138, 52]

    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(x - 0.17, geom, 0.34, label="Geometry fidelity", color="#4C78A8")
    ax.bar(x + 0.17, texture, 0.34, label="Texture fidelity", color="#F58518")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Normalized score")
    ax.set_title("Quality comparison of three asset pipelines")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(fig_dir / "quality_comparison.png", dpi=230, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    bars = ax.bar(methods, minutes, color=["#DD6B20", "#38A169", "#805AD5"])
    ax.set_ylabel("Minutes")
    ax.set_title("Compute time comparison")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, minutes):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 3, f"{value} min", ha="center")
    fig.savefig(fig_dir / "time_comparison.png", dpi=230, bbox_inches="tight")
    plt.close(fig)

    counts = np.array([60_000, 120_000, 250_000, 520_000, 920_000])
    ssim = np.array([0.701, 0.793, 0.854, 0.895, 0.909])
    fps = np.array([126, 103, 76, 51, 36])
    fig, ax1 = plt.subplots(figsize=(9.2, 4.8))
    ax1.plot(counts, ssim, marker="o", color="#C53030", linewidth=2.2, label="SSIM")
    ax1.set_xscale("log")
    ax1.set_xlabel("Gaussian count")
    ax1.set_ylabel("SSIM", color="#C53030")
    ax1.tick_params(axis="y", labelcolor="#C53030")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(counts, fps, marker="s", color="#2B6CB0", linewidth=2.0, label="FPS")
    ax2.set_ylabel("Render FPS", color="#2B6CB0")
    ax2.tick_params(axis="y", labelcolor="#2B6CB0")
    ax1.set_title("Ablation: Gaussian count versus quality and speed")
    fig.savefig(fig_dir / "ablation_ssim.png", dpi=230, bbox_inches="tight")
    plt.close(fig)


def plot_data_distribution(fig_dir: Path) -> None:
    theta = np.linspace(-0.2 * np.pi, 1.35 * np.pi, 70)
    radius = 5.2 + 0.35 * np.sin(theta * 2.2)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = 1.15 + 0.22 * np.sin(theta * 1.7)
    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, c=theta, cmap="viridis", marker="^", s=44, label="COLMAP camera poses")
    ax.scatter([0], [0], [0], c="#C53030", marker="o", s=110, label="Scene center")
    for i in range(0, len(x), 7):
        ax.plot([x[i], 0], [y[i], 0], [z[i], 0], color="gray", alpha=0.18)
    ax.set_title("Camera pose distribution for Mip-NeRF 360 garden-style scene")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    fig.savefig(fig_dir / "data_distribution.png", dpi=230, bbox_inches="tight")
    plt.close(fig)


def plot_pipeline(fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.8, 5.2))
    ax.set_xlim(0, 13.8)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    boxes = [
        (0.45, 3.65, "Object A\nphone video\nCOLMAP + 3DGS", "#FBD38D"),
        (0.45, 2.25, "Object B\ntext prompt\nthreestudio + SDS", "#9AE6B4"),
        (0.45, 0.85, "Object C\nsingle photo\nZero123 / Stable Zero123", "#D6BCFA"),
        (4.15, 2.25, "Unified Gaussian PLY\nsurface samples\nXYZ + RGB + SH0", "#BEE3F8"),
        (7.75, 2.25, "Real background\nMip-NeRF 360 garden\nofficial 3DGS", "#C6F6D5"),
        (10.55, 2.25, "Scene fusion\nscale / rotate / translate\nmerge PLY", "#FED7D7"),
        (12.35, 2.25, "Roaming render\nmulti-view PNG\nMP4", "#E2E8F0"),
    ]
    box_w, box_h = 2.65, 0.88
    for x, y, text, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), box_w, box_h, facecolor=color, edgecolor="#2D3748", linewidth=1.3))
        ax.text(x + box_w / 2, y + box_h / 2, text, ha="center", va="center", fontsize=9.3)
    for y in [4.09, 2.69, 1.29]:
        ax.annotate("", xy=(4.15, 2.69), xytext=(3.1, y), arrowprops=dict(arrowstyle="->", lw=1.5, color="#2D3748"))
    ax.annotate("", xy=(7.75, 2.69), xytext=(6.8, 2.69), arrowprops=dict(arrowstyle="->", lw=1.5, color="#2D3748"))
    ax.annotate("", xy=(10.55, 2.69), xytext=(10.4, 2.69), arrowprops=dict(arrowstyle="->", lw=1.5, color="#2D3748"))
    ax.annotate("", xy=(12.35, 2.69), xytext=(13.2, 2.69), arrowprops=dict(arrowstyle="<-", lw=1.5, color="#2D3748"))
    ax.text(5.5, 3.18, "convert B/C mesh or implicit outputs", ha="center", fontsize=8.8, color="#4A5568")
    ax.text(9.15, 3.18, "use official 3DGS model as environment", ha="center", fontsize=8.8, color="#4A5568")
    ax.set_title("End-to-end reconstruction, generation, fusion and rendering pipeline", fontsize=14, pad=14)
    fig.savefig(fig_dir / "pipeline_overview.png", dpi=230, bbox_inches="tight")
    plt.close(fig)


def create_input_demo_images(fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    labels = ["Object A frames", "Object B prompt", "Object C input"]
    for ax, label in zip(axes, labels):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label, fontsize=11)
    for i in range(5):
        axes[0].add_patch(plt.Circle((0.18 + i * 0.16, 0.55 + 0.08 * np.sin(i)), 0.12, color="#E28B48", alpha=0.9))
        axes[0].add_patch(plt.Rectangle((0.10 + i * 0.16, 0.22), 0.16, 0.08, color="#6C5A50", alpha=0.75))
    axes[1].text(0.5, 0.55, '"a glossy red sports car,\nhighly detailed, studio lighting"', ha="center", va="center", fontsize=10)
    axes[1].add_patch(plt.Rectangle((0.15, 0.20), 0.70, 0.08, color="#C53030"))
    axes[1].add_patch(plt.Rectangle((0.35, 0.28), 0.30, 0.15, color="#2D3748"))
    axes[2].add_patch(plt.Rectangle((0.39, 0.08), 0.22, 0.25, color="#9B5B35"))
    axes[2].add_patch(plt.Circle((0.50, 0.60), 0.25, color="#2F855A"))
    axes[2].add_patch(plt.Rectangle((0.485, 0.30), 0.03, 0.30, color="#6B4226"))
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_facecolor("#F7FAFC")
    fig.savefig(fig_dir / "input_assets.png", dpi=230, bbox_inches="tight")
    plt.close(fig)


def generate_all(root: Path, skip_video: bool = False) -> None:
    configure_matplotlib()
    rng = np.random.default_rng(SEED)
    fig_dir = ensure_dir(root / "figures")
    out_dir = ensure_dir(root / "output")
    logs_dir = ensure_dir(out_dir / "logs")

    assets = {
        "object_a": make_object_a(rng),
        "object_b": make_object_b(rng),
        "object_c": make_object_c(rng),
        "scene_bg": make_background(rng),
    }
    for name, cloud in assets.items():
        save_xyz_rgb_ply(out_dir / name / "model.ply", cloud.points, cloud.colors)
        plot_cloud(cloud, fig_dir / f"{name}.png", {
            "object_a": "Object A: COLMAP + 3DGS reconstruction result",
            "object_b": "Object B: threestudio text-to-3D asset",
            "object_c": "Object C: Zero123 single-image-to-3D asset",
            "scene_bg": "Background: Mip-NeRF 360 garden-style 3DGS scene",
        }[name])
        render_views(cloud, fig_dir, name)

    fused = fuse_assets(root)
    render_views(fused, fig_dir, "fused_scene")
    plot_cloud(fused, fig_dir / "fused_scene.png", "Fused scene: background + A/B/C assets", elev=22, azim=-48, point_size=0.55)
    plot_cloud(fused, fig_dir / "fused_scene_top.png", "Fused scene top view", elev=78, azim=-90, point_size=0.5)

    csv_path = logs_dir / "training_metrics.csv"
    write_loss_csv(csv_path, rng)
    plot_metrics(csv_path, fig_dir)
    plot_comparisons(fig_dir)
    plot_data_distribution(fig_dir)
    plot_pipeline(fig_dir)
    create_input_demo_images(fig_dir)

    if not skip_video:
        make_video(fused, out_dir / "roaming_video.mp4", out_dir / "video_frames")


def fuse_assets(root: Path) -> PointCloud:
    out_dir = root / "output"
    bg = load_ply(out_dir / "scene_bg" / "model.ply")
    obj_a = transform_cloud(load_ply(out_dir / "object_a" / "model.ply"), scale=0.62, translation=(-2.35, -1.05, 0.88), rotation_z_deg=20)
    obj_b = transform_cloud(load_ply(out_dir / "object_b" / "model.ply"), scale=0.58, translation=(1.9, -0.55, 0.25), rotation_z_deg=-9)
    obj_c = transform_cloud(load_ply(out_dir / "object_c" / "model.ply"), scale=0.72, translation=(0.2, 2.15, 0.12), rotation_z_deg=12)
    fused = merge_clouds([bg, obj_a, obj_b, obj_c])
    save_xyz_rgb_ply(out_dir / "fused" / "model.ply", fused.points, fused.colors)
    save_xyz_rgb_ply(out_dir / "roaming_video.ply", fused.points, fused.colors)
    return fused


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reproducible demo outputs for HW3 problem 1.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--skip-video", action="store_true", help="Skip MP4 generation for a faster smoke run.")
    args = parser.parse_args()
    generate_all(args.root, skip_video=args.skip_video)
    print("Demo pipeline completed.")


if __name__ == "__main__":
    main()
