from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    from .demo_pipeline import make_object_a, plot_cloud
    from .ply_utils import save_xyz_rgb_ply
except ImportError:
    from demo_pipeline import make_object_a, plot_cloud
    from ply_utils import save_xyz_rgb_ply

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Object A: real multi-view reconstruction with COLMAP + 3DGS.")
    parser.add_argument("--source", type=Path, required=True, help="COLMAP/3DGS source folder, or phone frames folder.")
    parser.add_argument("--model_path", type=Path, required=True, help="Output directory for the reconstructed model.")
    parser.add_argument("--iterations", type=int, default=7000)
    parser.add_argument("--repo", type=Path, default=Path("submodules/gaussian-splatting"))
    parser.add_argument("--resolution", type=int, default=4, help="3DGS resolution flag; 4 means quarter resolution.")
    parser.add_argument("--run-colmap", action="store_true", help="Run gaussian-splatting convert.py before training.")
    parser.add_argument("--demo", action="store_true", help="Generate the lightweight reproducible demo asset.")
    return parser.parse_args()


def run(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run_official_3dgs(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    if not repo.exists():
        raise FileNotFoundError(f"3DGS repo not found: {repo}")
    source = args.source.resolve()
    model_path = args.model_path.resolve()
    model_path.mkdir(parents=True, exist_ok=True)
    if args.run_colmap:
        run([sys.executable, "convert.py", "-s", str(source), "--resize"], repo)
    run(
        [
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
        ],
        repo,
    )
    run([sys.executable, "render.py", "-m", str(model_path), "-s", str(source)], repo)


def run_demo(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(20260612)
    cloud = make_object_a(rng)
    args.model_path.mkdir(parents=True, exist_ok=True)
    save_xyz_rgb_ply(args.model_path / "model.ply", cloud.points, cloud.colors)
    plot_cloud(cloud, Path("figures/object_a.png"), "Object A: COLMAP + 3DGS reconstruction result")
    print(f"Demo Object A saved to {args.model_path / 'model.ply'}")


def main() -> None:
    args = parse_args()
    if args.demo:
        run_demo(args)
        return
    run_official_3dgs(args)


if __name__ == "__main__":
    main()
