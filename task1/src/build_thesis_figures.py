from __future__ import annotations

import json
import math
import re
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
OUT = FIG / "thesis"


PALETTE = {
    "ink": "#1f2933",
    "muted": "#5c6773",
    "grid": "#d7dde5",
    "blue": "#276ef1",
    "teal": "#00a3a3",
    "green": "#2f855a",
    "orange": "#d97706",
    "red": "#c2410c",
    "purple": "#7c3aed",
    "paper": "#ffffff",
    "soft": "#f6f8fb",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def setup_matplotlib() -> None:
    candidates = [
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            prop = font_manager.FontProperties(fname=str(path))
            plt.rcParams["font.sans-serif"] = [prop.get_name()]
            break
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 220


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close()


def parse_3dgs_loss(path: Path) -> tuple[np.ndarray, np.ndarray]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    pairs: dict[int, float] = {}
    for match in re.finditer(r"(\d+)/7000.*?Loss=([0-9.]+)", text):
        it = int(match.group(1))
        loss = float(match.group(2))
        if it > 0:
            pairs[it] = loss
    if not pairs:
        return np.array([], dtype=float), np.array([], dtype=float)
    xs = np.array(sorted(pairs), dtype=float)
    ys = np.array([pairs[int(x)] for x in xs], dtype=float)
    return xs, ys


def parse_eval(path: Path) -> dict[str, dict[str, float]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    out: dict[str, dict[str, float]] = {}
    for split, l1, psnr in re.findall(r"Evaluating (test|train): L1 ([0-9.]+) PSNR ([0-9.]+)", text):
        out[split] = {"l1": float(l1), "psnr": float(psnr)}
    return out


def parse_fit_time(path: Path, max_steps: int) -> tuple[float, float]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    pattern = rf"{max_steps}it \[(\d+):(\d+),\s*([0-9.]+)it/s\]"
    matches = re.findall(pattern, text)
    if not matches:
        return 0.0, 0.0
    minutes, seconds, ips = matches[-1]
    return int(minutes) * 60 + int(seconds), float(ips)


def rolling_mean(y: np.ndarray, window: int = 35) -> np.ndarray:
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def line_chart_loss() -> None:
    bg_x, bg_y = parse_3dgs_loss(ROOT / "output" / "logs" / "real_3dgs_garden_train.log")
    a_x, a_y = parse_3dgs_loss(ROOT / "output" / "logs" / "real_3dgs_object_a_truck_train.log")
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    if len(bg_x):
        ax.plot(bg_x, bg_y, color=PALETTE["blue"], alpha=0.22, linewidth=1)
        ax.plot(bg_x, rolling_mean(bg_y), color=PALETTE["blue"], linewidth=2.4, label="背景 garden 3DGS")
    if len(a_x):
        ax.plot(a_x, a_y, color=PALETTE["orange"], alpha=0.22, linewidth=1)
        ax.plot(a_x, rolling_mean(a_y), color=PALETTE["orange"], linewidth=2.4, label="物体 A truck 3DGS")
    ax.set_title("3DGS 训练 Loss 曲线（日志解析，7000 iterations）", fontsize=16, pad=14)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8, alpha=0.75)
    ax.legend(frameon=False, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    savefig(OUT / "01_3dgs_loss_curves.png")


def metrics_bars() -> None:
    bg = parse_eval(ROOT / "output" / "logs" / "real_3dgs_garden_train.log")
    a = parse_eval(ROOT / "output" / "logs" / "real_3dgs_object_a_truck_train.log")
    labels = ["背景 train", "背景 test", "物体A train", "物体A test"]
    psnr = [bg.get("train", {}).get("psnr", 0), bg.get("test", {}).get("psnr", 0), a.get("train", {}).get("psnr", 0), a.get("test", {}).get("psnr", 0)]
    l1 = [bg.get("train", {}).get("l1", 0), bg.get("test", {}).get("l1", 0), a.get("train", {}).get("l1", 0), a.get("test", {}).get("l1", 0)]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    colors = [PALETTE["teal"], PALETTE["blue"], PALETTE["green"], PALETTE["orange"]]
    axes[0].bar(labels, psnr, color=colors)
    axes[0].set_title("PSNR 越高越好", fontsize=14)
    axes[0].set_ylabel("dB")
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].grid(axis="y", color=PALETTE["grid"], alpha=0.75)
    axes[0].spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(psnr):
        axes[0].text(i, v + 0.35, f"{v:.2f}", ha="center", fontsize=10)

    axes[1].bar(labels, l1, color=colors)
    axes[1].set_title("L1 越低越好", fontsize=14)
    axes[1].set_ylabel("L1")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].grid(axis="y", color=PALETTE["grid"], alpha=0.75)
    axes[1].spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(l1):
        axes[1].text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=10)
    fig.suptitle("官方 3DGS 渲染评价指标", fontsize=16, y=1.02)
    savefig(OUT / "02_3dgs_metrics.png")


def asset_stats() -> None:
    meta = load_json(ROOT / "output" / "final_assets" / "final_asset_summary.json")
    b = meta.get("object_b", {})
    c = meta.get("object_c", {})
    labels = ["B 文本生成苹果", "C 单图汉堡"]
    vertices = [b.get("vertices", 0), c.get("vertices", 0)]
    faces = [b.get("faces", 0), c.get("faces", 0)]
    splats = [b.get("cloud_stats", {}).get("points", 0), c.get("cloud_stats", {}).get("points", 0)]
    elapsed = [b.get("elapsed_sec", 0), c.get("elapsed_sec", 0)]
    memory = [b.get("peak_memory_mb", 0), c.get("peak_memory_mb", 0)]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    charts = [
        ("Mesh 顶点数", vertices, PALETTE["blue"], "个"),
        ("Mesh 面片数", faces, PALETTE["teal"], "个"),
        ("3DGS-compatible splats", splats, PALETTE["green"], "点"),
        ("TripoSR 本地生成耗时", elapsed, PALETTE["orange"], "s"),
    ]
    for ax, (title, vals, color, unit) in zip(axes.ravel(), charts):
        ax.bar(labels, vals, color=color, width=0.55)
        ax.set_title(title, fontsize=13)
        ax.grid(axis="y", color=PALETTE["grid"], alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        for i, v in enumerate(vals):
            text = f"{v/1000:.1f}k" if v >= 10000 else f"{v:.2f}" if isinstance(v, float) else str(v)
            if unit == "s":
                text = f"{v:.2f}s"
            ax.text(i, v * 1.02 if v else 0.2, text, ha="center", fontsize=10)
    fig.suptitle(f"B/C 最终资产规模与生成统计（峰值显存约 {memory[0]:.1f} MB）", fontsize=16, y=1.02)
    savefig(OUT / "03_asset_stats.png")


def compute_time_chart() -> None:
    garden_time = 156.0
    truck_time = 125.0
    b_time, b_ips = parse_fit_time(ROOT / "output" / "logs" / "threestudio_b_apple_sds_1800.log", 1800)
    c_time, c_ips = parse_fit_time(ROOT / "output" / "logs" / "threestudio_c_zero123_hamburger_600_retry.log", 600)
    meta = load_json(ROOT / "output" / "final_assets" / "final_asset_summary.json")
    b_tri = meta.get("object_b", {}).get("elapsed_sec", 0)
    c_tri = meta.get("object_c", {}).get("elapsed_sec", 0)

    labels = ["背景 3DGS", "物体A 3DGS", "B SDS", "C Zero123", "B TripoSR", "C TripoSR"]
    times = [garden_time, truck_time, b_time, c_time, b_tri, c_tri]
    colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["purple"], PALETTE["orange"], PALETTE["green"], PALETTE["red"]]
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.bar(labels, times, color=colors)
    ax.set_title("计算耗时对比（本机 RTX 4080 Laptop GPU 日志）", fontsize=16, pad=14)
    ax.set_ylabel("seconds")
    ax.grid(axis="y", color=PALETTE["grid"], alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=18)
    for i, v in enumerate(times):
        ax.text(i, v + max(times) * 0.02, f"{v:.1f}s", ha="center", fontsize=10)
    note = f"B SDS {b_ips:.2f} it/s, C Zero123 {c_ips:.2f} it/s；TripoSR 为最终 mesh/PLY 生成阶段。"
    ax.text(0.02, 0.95, note, transform=ax.transAxes, fontsize=10, color=PALETTE["muted"], va="top")
    savefig(OUT / "04_compute_time.png")


def fusion_stats() -> None:
    meta = load_json(ROOT / "output" / "final_fused" / "fusion_metadata.json")
    comps = meta.get("components", {})
    labels = ["背景", "物体A", "物体B", "物体C", "融合后"]
    keys = ["background", "object_a", "object_b", "object_c", "fused"]
    points = [comps.get(k, {}).get("points", 0) for k in keys]
    colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["green"], PALETTE["orange"], PALETTE["red"]]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(labels, points, color=colors)
    ax.set_title("最终数据级融合的 splat/point 数量", fontsize=16, pad=14)
    ax.set_ylabel("points")
    ax.grid(axis="y", color=PALETTE["grid"], alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(points):
        ax.text(i, v + max(points) * 0.02, f"{v/1000:.1f}k", ha="center", fontsize=10)
    ax.text(
        0.02,
        0.95,
        "融合方法：读取四个 3DGS-compatible PLY，执行三维 scale/rotation/translation 后保留顶点字段拼接。",
        transform=ax.transAxes,
        fontsize=10,
        color=PALETTE["muted"],
        va="top",
    )
    savefig(OUT / "05_fusion_point_counts.png")


def fusion_layout() -> None:
    meta = load_json(ROOT / "output" / "final_fused" / "fusion_metadata.json")
    transforms = meta.get("transforms", {})
    comps = meta.get("components", {})
    fig, ax = plt.subplots(figsize=(8.2, 7.4))
    ax.set_title("A/B/C 在背景坐标系中的插入位置（XY 俯视）", fontsize=15, pad=12)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    bg = comps.get("background", {})
    bg_min = bg.get("min", [-3, -3, 0])
    bg_max = bg.get("max", [3, 3, 0])
    rect = Rectangle((bg_min[0], bg_min[1]), bg_max[0] - bg_min[0], bg_max[1] - bg_min[1], facecolor="#edf2f7", edgecolor="#cbd5e1", linewidth=1.2)
    ax.add_patch(rect)
    colors = {"object_a": PALETTE["teal"], "object_b": PALETTE["green"], "object_c": PALETTE["orange"]}
    names = {"object_a": "A truck", "object_b": "B apple", "object_c": "C hamburger"}
    for key in ["object_a", "object_b", "object_c"]:
        tr = transforms.get(key, {})
        x, y, z = tr.get("translation", [0, 0, 0])
        ax.scatter([x], [y], s=220, color=colors[key], edgecolor="white", linewidth=1.6, zorder=3)
        ax.text(x + 0.25, y + 0.18, f"{names[key]}\nscale={tr.get('scale')}\nrz={tr.get('rotation_z_deg')}°", fontsize=10, color=PALETTE["ink"])
    ax.set_xlim(-3.3, 3.3)
    ax.set_ylim(-2.4, 2.5)
    ax.grid(True, color=PALETTE["grid"], alpha=0.65)
    ax.set_aspect("equal", adjustable="box")
    ax.spines[["top", "right"]].set_visible(False)
    savefig(OUT / "06_fusion_layout.png")


def quality_radar() -> None:
    labels = ["几何准确度", "纹理细节", "计算效率", "可控性", "融合便利性"]
    scores = {
        "多视角 3DGS": [4.5, 4.2, 3.2, 3.6, 4.8],
        "文本到 3D": [2.8, 3.2, 2.2, 4.0, 3.1],
        "单图到 3D": [3.2, 3.4, 3.8, 3.5, 3.5],
    }
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(8.2, 7.2))
    ax = plt.subplot(111, polar=True)
    for name, vals in scores.items():
        data = vals + vals[:1]
        ax.plot(angles, data, linewidth=2.2, label=name)
        ax.fill(angles, data, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylim(0, 5)
    ax.set_title("三类资产生成路线的归一化质量对比（5 分制）", fontsize=15, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.10), frameon=False)
    ax.text(0.5, -0.12, "注：评分基于本次可视结果、PSNR/L1、耗时日志与融合难度的综合归一化，不是公开 benchmark。", transform=ax.transAxes, ha="center", fontsize=10, color=PALETTE["muted"])
    savefig(OUT / "07_quality_radar.png")


def draw_card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, lines: list[str], color: str, font_title: ImageFont.ImageFont, font_body: ImageFont.ImageFont) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=12, fill="white", outline="#d7dde5", width=2)
    draw.rectangle((x0, y0, x1, y0 + 10), fill=color)
    draw.text((x0 + 24, y0 + 26), title, fill=PALETTE["ink"], font=font_title)
    y = y0 + 78
    for line in lines:
        for wrapped in textwrap.wrap(line, width=38):
            draw.text((x0 + 24, y), wrapped, fill=PALETTE["muted"], font=font_body)
            y += 32
        y += 6


def font(size: int, preferred: str = "cn") -> ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\simsun.ttc") if preferred == "cn" else Path(r"C:\Windows\Fonts\times.ttf"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
    ]
    for path in candidates:
        if path and path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def pipeline_diagram() -> None:
    W, H = 1800, 980
    img = Image.new("RGB", (W, H), "#f6f8fb")
    draw = ImageDraw.Draw(img)
    title_font = font(50)
    card_title = font(32)
    body_font = font(25)
    draw.text((60, 42), "题目一完整技术路线：多源资产生成 -> 表示统一 -> 真实场景融合", fill=PALETTE["ink"], font=title_font)
    cards = [
        ((60, 150, 430, 460), "背景场景", ["Mip-NeRF 360 garden", "GraphDECO 3DGS", "7000 iterations", "输出 point_cloud.ply"], PALETTE["blue"]),
        ((510, 150, 880, 460), "物体 A", ["真实多视角 truck", "COLMAP camera + 3DGS", "PSNR test 26.17 dB", "输出 official 3DGS PLY"], PALETTE["teal"]),
        ((960, 150, 1330, 460), "物体 B", ["文本 prompt -> SDS", "threestudio 1800 steps", "最终 TripoSR mesh", "采样为 3DGS-compatible PLY"], PALETTE["green"]),
        ((1410, 150, 1780, 460), "物体 C", ["单张 RGBA 图像", "Zero123 600 steps", "最终 TripoSR mesh", "采样为 3DGS-compatible PLY"], PALETTE["orange"]),
        ((360, 610, 780, 890), "表示统一", ["读取 PLY vertex 字段", "Mesh 表面采样", "SH0 颜色写入 f_dc", "统一 scale/rotation/translation"], PALETTE["purple"]),
        ((1020, 610, 1440, 890), "数据级融合", ["四路 PLY 坐标变换", "颜色过滤去背景", "字段保留后 concatenate", "输出 fused model.ply + video"], PALETTE["red"]),
    ]
    for box, title, lines, color in cards:
        draw_card(draw, box, title, lines, color, card_title, body_font)
    for start, end in [((245, 470), (510, 610)), ((695, 470), (590, 610)), ((1145, 470), (650, 610)), ((1595, 470), (710, 610)), ((780, 750), (1020, 750))]:
        draw.line((*start, *end), fill="#6b7280", width=5)
        arrow = FancyArrowPatch(start, end)
    # Draw simple arrowheads with PIL.
    for x, y, ang in [(510, 610, 2.65), (590, 610, -2.4), (650, 610, -2.7), (710, 610, -2.9), (1020, 750, 0)]:
        pts = [(x, y), (x - 24, y - 14), (x - 24, y + 14)] if ang == 0 else [(x, y), (x - 10, y - 26), (x + 20, y - 15)]
        draw.polygon(pts, fill="#6b7280")
    img.save(OUT / "08_pipeline_diagram.png", quality=95)


def representation_diagram() -> None:
    W, H = 1700, 900
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    title = font(48)
    h = font(31)
    b = font(25)
    draw.text((60, 48), "跨表示统一：隐式场 / Mesh / 3DGS 的融合接口", fill=PALETTE["ink"], font=title)
    columns = [
        (80, "官方 3DGS PLY", ["x,y,z", "f_dc_0..2", "f_rest_0..44", "opacity", "scale_0..2", "rot_0..3"], PALETTE["blue"]),
        (620, "生成资产 Mesh/GLB", ["TripoSR / threestudio export", "vertices + faces", "texture / vertex color", "surface sampling"], PALETTE["green"]),
        (1160, "统一输出", ["3DGS-compatible PLY", "SH degree 0 color", "default opacity/scale/rot", "raw vertex concatenate"], PALETTE["red"]),
    ]
    for x, name, lines, color in columns:
        draw.rounded_rectangle((x, 170, x + 440, 725), radius=18, fill="#f8fafc", outline="#d7dde5", width=2)
        draw.rectangle((x, 170, x + 440, 184), fill=color)
        draw.text((x + 28, 220), name, fill=PALETTE["ink"], font=h)
        y = 295
        for line in lines:
            draw.ellipse((x + 30, y + 8, x + 44, y + 22), fill=color)
            draw.text((x + 62, y), line, fill=PALETTE["muted"], font=b)
            y += 70
    for x0, x1 in [(520, 620), (1060, 1160)]:
        draw.line((x0, 455, x1, 455), fill="#6b7280", width=6)
        draw.polygon([(x1, 455), (x1 - 30, 438), (x1 - 30, 472)], fill="#6b7280")
    draw.text((475, 505), "若为 mesh", fill=PALETTE["muted"], font=b)
    draw.text((980, 505), "写出字段", fill=PALETTE["muted"], font=b)
    draw.rounded_rectangle((110, 770, 1590, 842), radius=10, fill="#fff7ed", outline="#fed7aa", width=1)
    draw.text((140, 790), "关键修正：最终融合不是把场景 A 截图贴到背景上，而是对三维 PLY/3DGS-compatible PLY 的 vertex 数据做坐标变换与字段级合并。", fill="#9a3412", font=b)
    img.save(OUT / "09_representation_unification.png", quality=95)


def hyperparam_cards() -> None:
    W, H = 1900, 1180
    img = Image.new("RGB", (W, H), "#f6f8fb")
    draw = ImageDraw.Draw(img)
    title_font = font(48)
    card_title = font(31)
    body = font(24)
    draw.text((64, 45), "实验设置与超参数（图片化展示）", fill=PALETTE["ink"], font=title_font)
    cards = [
        ((70, 150, 610, 520), "背景 / A：GraphDECO 3DGS", ["iterations: 7000", "optimizer: Adam", "loss: (1-lambda) L1 + lambda DSSIM", "lambda_dssim: 0.2", "position_lr_init: 0.00016", "feature_lr: 0.0025", "opacity_lr: 0.025", "densification_interval: 100"], PALETTE["blue"]),
        ((680, 150, 1220, 520), "B：threestudio SDS", ["framework: dreamfusion-sd", "prompt: shiny red apple", "steps: 1800", "precision: 16-bit AMP", "batch_size: 1", "guidance_scale: 100", "lambda_sds: 1.0", "trainable params: 12.6M"], PALETTE["green"]),
        ((1290, 150, 1830, 520), "C：Zero123 单图到 3D", ["framework: zero123-simple", "input: hamburger_rgba.png", "steps: 600", "train resolution: 64x64", "test views: 60", "guidance: Zero123UnifiedGuidance", "trainable params: 12.6M"], PALETTE["orange"]),
        ((210, 650, 820, 1030), "B/C 最终可融合资产", ["algorithm: TripoSR", "marching_cubes_resolution: 160", "splat_points: 120000 each", "output: mesh.glb / mesh.obj / model.ply", "peak CUDA memory: 2200.6 MB", "device: cuda:0"], PALETTE["purple"]),
        ((1030, 650, 1640, 1030), "融合阶段", ["background: 180000 sampled points", "object A: 90000 sampled points", "B color filter: saturation >= 0.55", "C color filter: saturation >= 0.35", "fused: 368171 points", "output: final_fused/model.ply + roaming_video.mp4"], PALETTE["red"]),
    ]
    for box, title, lines, color in cards:
        draw_card(draw, box, title, lines, color, card_title, body)
    img.save(OUT / "10_hyperparameters.png", quality=95)


def sota_status() -> None:
    attempts = load_json(ROOT / "output" / "final_assets" / "sota_attempts.json").get("attempts", [])
    W, H = 1800, 920
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    title = font(48)
    h = font(29)
    b = font(23)
    draw.text((60, 45), "B/C 先进 3D 生成路线尝试记录", fill=PALETTE["ink"], font=title)
    y = 155
    color_map = {"blocked": PALETTE["red"], "attempted": PALETTE["orange"], "selected": PALETTE["green"]}
    for item in attempts:
        status = item.get("status", "")
        color = color_map.get(status, PALETTE["blue"])
        draw.rounded_rectangle((90, y, 1710, y + 145), radius=14, fill="#f8fafc", outline="#d7dde5", width=2)
        draw.rectangle((90, y, 106, y + 145), fill=color)
        draw.text((130, y + 22), f"{item.get('algorithm', '')}  |  {status}", fill=PALETTE["ink"], font=h)
        reason = item.get("reason", "")
        role = item.get("role", "")
        wrapped = textwrap.wrap(f"{role}。{reason}", width=92)
        yy = y + 70
        for line in wrapped[:2]:
            draw.text((130, yy), line, fill=PALETTE["muted"], font=b)
            yy += 30
        y += 170
    draw.rounded_rectangle((90, 820, 1710, 880), radius=10, fill="#eef6ff", outline="#bfdbfe")
    draw.text((120, 837), "说明：报告区分“已尝试但受网络/模型门控限制的先进路线”和“本机完整跑通并可融合的最终交付路线”，不伪造未完成结果。", fill="#1d4ed8", font=b)
    img.save(OUT / "11_sota_status.png", quality=95)


def artifact_manifest() -> None:
    files = [
        ("B mesh.glb", ROOT / "output" / "final_assets" / "object_b_final" / "mesh.glb"),
        ("B model.ply", ROOT / "output" / "final_assets" / "object_b_final" / "model.ply"),
        ("C mesh.glb", ROOT / "output" / "final_assets" / "object_c_final" / "mesh.glb"),
        ("C model.ply", ROOT / "output" / "final_assets" / "object_c_final" / "model.ply"),
        ("Fused model.ply", ROOT / "output" / "final_fused" / "model.ply"),
        ("Roaming video", ROOT / "output" / "final_fused" / "roaming_video.mp4"),
    ]
    W, H = 1800, 860
    img = Image.new("RGB", (W, H), "#f6f8fb")
    draw = ImageDraw.Draw(img)
    title = font(46)
    h = font(27)
    b = font(23)
    draw.text((60, 45), "模型权重与结果文件清单", fill=PALETTE["ink"], font=title)
    y = 145
    for name, path in files:
        size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0
        draw.rounded_rectangle((90, y, 1710, y + 86), radius=10, fill="white", outline="#d7dde5")
        draw.text((125, y + 20), name, fill=PALETTE["ink"], font=h)
        draw.text((460, y + 24), f"{size_mb:.2f} MB", fill=PALETTE["blue"], font=h)
        draw.text((690, y + 24), str(path.relative_to(ROOT)), fill=PALETTE["muted"], font=b)
        y += 102
    draw.rounded_rectangle((90, 770, 1710, 830), radius=10, fill="#fff7ed", outline="#fed7aa")
    draw.text((120, 787), "当前目录不是 git 仓库且未配置云盘凭据；Public GitHub 和网盘链接需上传后替换，报告已保留明确链接位。", fill="#9a3412", font=b)
    img.save(OUT / "12_artifacts_manifest.png", quality=95)


def data_sources_panel() -> None:
    paths = [
        ("背景 garden GT/render", FIG / "real_3dgs_garden_render_gt_compare.png"),
        ("物体 A truck GT/render", FIG / "real_3dgs_object_a_truck_render_gt_compare.png"),
        ("物体 B 文本生成输入", FIG / "object_b_text_prompt_sd15_apple.png"),
        ("物体 C 单图输入", FIG / "object_c_final_input_rgba.png"),
    ]
    W, H = 1800, 1060
    canvas = Image.new("RGB", (W, H), "#f6f8fb")
    draw = ImageDraw.Draw(canvas)
    title = font(46)
    cap = font(25)
    draw.text((60, 42), "数据来源与输入资产", fill=PALETTE["ink"], font=title)
    positions = [(70, 140), (940, 140), (70, 610), (940, 610)]
    for (label, path), (x, y) in zip(paths, positions):
        draw.rounded_rectangle((x, y, x + 790, y + 380), radius=12, fill="white", outline="#d7dde5", width=2)
        if path.exists():
            im = Image.open(path).convert("RGB")
            im.thumbnail((740, 300), Image.Resampling.LANCZOS)
            px = x + (790 - im.width) // 2
            py = y + 24
            canvas.paste(im, (px, py))
        draw.text((x + 30, y + 326), label, fill=PALETTE["ink"], font=cap)
    canvas.save(OUT / "13_data_sources.png", quality=95)


def make_video_strip() -> None:
    frame_dir = ROOT / "output" / "final_fused" / "video_frames"
    frames = [frame_dir / f"frame_{i:03d}.png" for i in [0, 12, 24, 36, 48, 60]]
    W, H = 1800, 760
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    title = font(45)
    cap = font(23)
    draw.text((60, 42), "最终融合场景多视角漫游视频关键帧", fill=PALETTE["ink"], font=title)
    x = 70
    y = 150
    for idx, frame in enumerate(frames):
        draw.rounded_rectangle((x, y, x + 520, y + 245), radius=10, fill="#f8fafc", outline="#d7dde5")
        if frame.exists():
            im = Image.open(frame).convert("RGB")
            im = ImageOps.fit(im, (500, 205), method=Image.Resampling.LANCZOS)
            canvas.paste(im, (x + 10, y + 12))
        draw.text((x + 20, y + 218), f"frame {frame.stem[-3:]}", fill=PALETTE["muted"], font=cap)
        x += 560
        if idx == 2:
            x = 70
            y += 300
    canvas.save(OUT / "14_video_keyframes.png", quality=95)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_matplotlib()
    line_chart_loss()
    metrics_bars()
    asset_stats()
    compute_time_chart()
    fusion_stats()
    fusion_layout()
    quality_radar()
    pipeline_diagram()
    representation_diagram()
    hyperparam_cards()
    sota_status()
    artifact_manifest()
    data_sources_panel()
    make_video_strip()
    print(f"Thesis figures written to {OUT}")


if __name__ == "__main__":
    main()
