from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    from .demo_pipeline import make_object_c, plot_cloud
    from .ply_utils import save_xyz_rgb_ply
except ImportError:
    from demo_pipeline import make_object_c, plot_cloud
    from ply_utils import save_xyz_rgb_ply


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Object C: single-image-to-3D generation with Zero123 / Stable Zero123.")
    parser.add_argument("--image", type=Path, required=True, help="Foreground RGBA input image.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument("--repo", type=Path, default=Path("submodules/threestudio/threestudio-main"))
    parser.add_argument("--config", type=str, default="configs/stable-zero123.yaml")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--demo", action="store_true", help="Generate the lightweight reproducible demo asset.")
    return parser.parse_args()


def run(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run_zero123(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    if not repo.exists():
        raise FileNotFoundError(f"threestudio repo not found: {repo}")
    args.out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "launch.py",
        "--config",
        args.config,
        "--train",
        "--gpu",
        args.gpu,
        f"data.image_path={args.image.resolve()}",
        f"trainer.max_steps={args.max_steps}",
        "system.loggers.wandb.enable=false",
    ]
    run(cmd, repo)
    print("After training, export the mesh with threestudio's export stage, then run src/mesh_to_3dgs.py.")


def run_demo(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(20260612)
    cloud = make_object_c(rng)
    args.out.mkdir(parents=True, exist_ok=True)
    save_xyz_rgb_ply(args.out / "model.ply", cloud.points, cloud.colors)
    plot_cloud(cloud, Path("figures/object_c.png"), "Object C: Zero123 single-image-to-3D asset")
    print(f"Demo Object C saved to {args.out / 'model.ply'}")


def main() -> None:
    args = parse_args()
    if args.demo:
        run_demo(args)
        return
    run_zero123(args)


if __name__ == "__main__":
    main()
