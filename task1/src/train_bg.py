from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    from .demo_pipeline import make_background, plot_cloud
    from .ply_utils import save_xyz_rgb_ply
except ImportError:
    from demo_pipeline import make_background, plot_cloud
    from ply_utils import save_xyz_rgb_ply


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Background scene reconstruction using official 3D Gaussian Splatting.")
    parser.add_argument("--source", type=Path, required=True, help="Mip-NeRF 360 / COLMAP source dataset path.")
    parser.add_argument("--model_path", type=Path, required=True, help="Output directory for the background model.")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--repo", type=Path, default=Path("submodules/gaussian-splatting"))
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--eval", action="store_true", help="Use Mip-NeRF 360 train/test split.")
    parser.add_argument("--demo", action="store_true", help="Generate the lightweight reproducible demo background.")
    return parser.parse_args()


def run(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run_official_3dgs(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    source = args.source.resolve()
    model_path = args.model_path.resolve()
    if not repo.exists():
        raise FileNotFoundError(f"3DGS repo not found: {repo}")
    model_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "train.py",
        "-s",
        str(source),
        "-m",
        str(model_path),
        "--iterations",
        str(args.iterations),
        "-r",
        str(args.resolution),
        "--data_device",
        "cpu",
    ]
    if args.eval:
        cmd.append("--eval")
    run(cmd, repo)
    render_cmd = [sys.executable, "render.py", "-m", str(model_path), "-s", str(source)]
    run(render_cmd, repo)
    if args.eval:
        run([sys.executable, "metrics.py", "-m", str(model_path)], repo)


def run_demo(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(20260612)
    cloud = make_background(rng)
    args.model_path.mkdir(parents=True, exist_ok=True)
    save_xyz_rgb_ply(args.model_path / "model.ply", cloud.points, cloud.colors)
    plot_cloud(cloud, Path("figures/scene_bg.png"), "Background: Mip-NeRF 360 garden-style 3DGS scene")
    print(f"Demo background saved to {args.model_path / 'model.ply'}")


def main() -> None:
    args = parse_args()
    if args.demo:
        run_demo(args)
        return
    run_official_3dgs(args)


if __name__ == "__main__":
    main()
