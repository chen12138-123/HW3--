from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
GS_REPO = ROOT / "submodules" / "gaussian-splatting"


def run(cmd: list[str], cwd: Path = GS_REPO, log_path: Path | None = None) -> None:
    print("Running:", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    if log_path is None:
        subprocess.run(cmd, cwd=str(cwd), env=env, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.buffer.write(line.encode("utf-8", errors="replace"))
            sys.stdout.buffer.flush()
            log.write(line)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, cmd)


def check_extensions() -> None:
    code = (
        "import torch, torchvision, diff_gaussian_rasterization, simple_knn._C; "
        "print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torchvision.__version__)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def latest_iteration(model_path: Path) -> int:
    pc_dir = model_path / "point_cloud"
    iterations = []
    if pc_dir.exists():
        for child in pc_dir.iterdir():
            if child.is_dir() and child.name.startswith("iteration_"):
                try:
                    iterations.append(int(child.name.split("_")[-1]))
                except ValueError:
                    pass
    if not iterations:
        raise FileNotFoundError(f"No point_cloud/iteration_* found under {model_path}")
    return max(iterations)


def train_scene(
    source: Path,
    model_path: Path,
    iterations: int,
    images: str,
    resolution: int,
    eval_split: bool,
    log_path: Path,
) -> list[str]:
    check_extensions()
    model_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "train.py",
        "-s",
        str(source.resolve()),
        "-m",
        str(model_path.resolve()),
        "-i",
        images,
        "-r",
        str(resolution),
        "--iterations",
        str(iterations),
        "--disable_viewer",
        "--data_device",
        "cpu",
    ]
    if eval_split:
        cmd.append("--eval")
    run(cmd, log_path=log_path)
    return cmd


def render_scene(
    source: Path,
    model_path: Path,
    images: str,
    log_path: Path,
    skip_train: bool = False,
    skip_test: bool = False,
) -> tuple[int, list[str]]:
    iteration = latest_iteration(model_path)
    cmd = [
        sys.executable,
        "render.py",
        "-m",
        str(model_path.resolve()),
        "-s",
        str(source.resolve()),
        "-i",
        images,
        "--iteration",
        str(iteration),
    ]
    if skip_train:
        cmd.append("--skip_train")
    if skip_test:
        cmd.append("--skip_test")
    run(cmd, log_path=log_path)
    return iteration, cmd


def copy_render_outputs(model_path: Path, scene_name: str, iteration: int, limit: int = 6) -> list[Path]:
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    candidates = [
        model_path / "test" / f"ours_{iteration}" / "renders",
        model_path / "train" / f"ours_{iteration}" / "renders",
    ]
    render_dir = next((path for path in candidates if path.exists()), None)
    if render_dir is None:
        raise FileNotFoundError(f"No render output found for iteration {iteration} in {model_path}")
    copied = []
    for idx, src in enumerate(sorted(render_dir.glob("*.png"))[:limit]):
        dst = figures / f"real_3dgs_{scene_name}_render_{idx + 1}.png"
        shutil.copy2(src, dst)
        copied.append(dst)
    gt_dir = render_dir.parent / "gt"
    if gt_dir.exists():
        for idx, src in enumerate(sorted(gt_dir.glob("*.png"))[: min(limit, 3)]):
            shutil.copy2(src, figures / f"real_3dgs_{scene_name}_gt_{idx + 1}.png")
    if copied:
        make_contact_sheet(copied, figures / f"real_3dgs_{scene_name}_contact.png", f"Official 3DGS renders: {scene_name}")
        make_render_gt_comparison(model_path, scene_name, iteration, limit=min(limit, 4))
    return copied


def make_contact_sheet(paths: list[Path], out_path: Path, title: str) -> None:
    thumbs = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        img.thumbnail((420, 260), Image.LANCZOS)
        canvas = Image.new("RGB", (420, 260), "white")
        canvas.paste(img, ((420 - img.width) // 2, (260 - img.height) // 2))
        thumbs.append(canvas)
    cols = 3
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 420, rows * 260 + 42), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\simhei.ttf", 24)
    except Exception:
        font = None
    draw.text((18, 10), title, fill=(25, 35, 45), font=font)
    for i, thumb in enumerate(thumbs):
        x = (i % cols) * 420
        y = 42 + (i // cols) * 260
        sheet.paste(thumb, (x, y))
    sheet.save(out_path)


def make_render_gt_comparison(model_path: Path, scene_name: str, iteration: int, limit: int = 4) -> Path | None:
    figures = ROOT / "figures"
    render_dir = None
    for candidate in [
        model_path / "test" / f"ours_{iteration}" / "renders",
        model_path / "train" / f"ours_{iteration}" / "renders",
    ]:
        if candidate.exists():
            render_dir = candidate
            break
    if render_dir is None:
        return None
    gt_dir = render_dir.parent / "gt"
    if not gt_dir.exists():
        return None
    render_paths = sorted(render_dir.glob("*.png"))[:limit]
    gt_paths = sorted(gt_dir.glob("*.png"))[:limit]
    if not render_paths or not gt_paths:
        return None

    cell_w, cell_h = 360, 230
    label_h = 34
    rows = min(len(render_paths), len(gt_paths))
    sheet = Image.new("RGB", (2 * cell_w, rows * (cell_h + label_h) + label_h), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\simhei.ttf", 22)
        small = ImageFont.truetype(r"C:\Windows\Fonts\simhei.ttf", 18)
    except Exception:
        font = None
        small = None
    draw.text((cell_w // 2 - 36, 7), "3DGS render", fill=(25, 35, 45), font=font)
    draw.text((cell_w + cell_w // 2 - 20, 7), "GT", fill=(25, 35, 45), font=font)
    for row, (render_path, gt_path) in enumerate(zip(render_paths, gt_paths)):
        y0 = label_h + row * (cell_h + label_h)
        for col, src in enumerate([render_path, gt_path]):
            img = Image.open(src).convert("RGB")
            img.thumbnail((cell_w, cell_h), Image.LANCZOS)
            canvas = Image.new("RGB", (cell_w, cell_h), "white")
            canvas.paste(img, ((cell_w - img.width) // 2, (cell_h - img.height) // 2))
            sheet.paste(canvas, (col * cell_w, y0))
        draw.text((10, y0 + cell_h + 7), f"camera {row + 1:02d}", fill=(70, 80, 90), font=small)
    out = figures / f"real_3dgs_{scene_name}_render_gt_compare.png"
    sheet.save(out)
    return out


def write_metadata(
    source: Path,
    model_path: Path,
    scene_name: str,
    images: str,
    resolution: int,
    iterations: int,
    eval_split: bool,
    train_cmd: list[str] | None,
    render_cmd: list[str] | None,
    rendered_iteration: int | None,
) -> Path:
    try:
        import torch
        import torchvision

        torch_info = {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            "torchvision": torchvision.__version__,
        }
    except Exception as exc:
        torch_info = {"error": str(exc)}

    latest = None
    ply_path = None
    try:
        latest = latest_iteration(model_path)
        ply_path = str((model_path / "point_cloud" / f"iteration_{latest}" / "point_cloud.ply").resolve())
    except Exception:
        pass

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": "Mip-NeRF 360",
        "scene": scene_name,
        "source_path": str(source.resolve()),
        "image_folder": images,
        "model_path": str(model_path.resolve()),
        "point_cloud_ply": ply_path,
        "requested_iterations": iterations,
        "latest_iteration": latest,
        "rendered_iteration": rendered_iteration,
        "resolution_flag": resolution,
        "eval_split": eval_split,
        "official_3dgs_repo": str(GS_REPO.resolve()),
        "official_3dgs_github": "https://github.com/graphdeco-inria/gaussian-splatting",
        "mipnerf360_url": "https://jonbarron.info/mipnerf360/",
        "train_command": train_cmd,
        "render_command": render_cmd,
        "environment": torch_info,
    }
    out = model_path / "real_3dgs_metadata.json"
    out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.copy2(out, ROOT / "figures" / f"real_3dgs_{scene_name}_metadata.json")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official 3DGS train/render and collect rasterized outputs.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, default=ROOT / "output" / "real_3dgs_bg")
    parser.add_argument("--scene_name", type=str, default="garden")
    parser.add_argument("--images", type=str, default="images_4")
    parser.add_argument("--iterations", type=int, default=7000)
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_render", action="store_true")
    args = parser.parse_args()

    logs = ROOT / "output" / "logs"
    train_cmd = None
    render_cmd = None
    iteration = None
    if not args.skip_train:
        train_cmd = train_scene(
            args.source,
            args.model_path,
            args.iterations,
            args.images,
            args.resolution,
            args.eval,
            logs / f"real_3dgs_{args.scene_name}_train.log",
        )
    if not args.skip_render:
        iteration, render_cmd = render_scene(
            args.source,
            args.model_path,
            args.images,
            logs / f"real_3dgs_{args.scene_name}_render.log",
        )
        copy_render_outputs(args.model_path, args.scene_name, iteration)
        print(f"Collected official 3DGS rasterized renders for {args.scene_name}.")
    write_metadata(
        args.source,
        args.model_path,
        args.scene_name,
        args.images,
        args.resolution,
        args.iterations,
        args.eval,
        train_cmd,
        render_cmd,
        iteration,
    )


if __name__ == "__main__":
    main()
