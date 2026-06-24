from __future__ import annotations

import json
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SF3D_ROOT = Path(r"C:\sf3d_ascii")
ASSETS = ROOT / "output" / "assets_sota"
ASSETS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SF3D_ROOT))

from sf3d.system import SF3D  # noqa: E402
from sf3d.utils import resize_foreground  # noqa: E402


def load_foreground_image(path: Path, foreground_ratio: float = 0.85) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    return resize_foreground(image, foreground_ratio)


def run_one(
    name: str,
    image_path: Path,
    *,
    texture_resolution: int = 1024,
    remesh: str = "none",
    vertex_count: int = -1,
    foreground_ratio: float = 0.85,
) -> None:
    out_dir = ASSETS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared = load_foreground_image(image_path, foreground_ratio)
    prepared.save(out_dir / "input_prepared.png")
    meta = {
        "algorithm": "Stable Fast 3D local official code",
        "repository": "https://github.com/Stability-AI/stable-fast-3d",
        "model": "stabilityai/stable-fast-3d",
        "input": str(image_path),
        "texture_resolution": texture_resolution,
        "remesh": remesh,
        "vertex_count": vertex_count,
        "foreground_ratio": foreground_ratio,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SF3D.from_pretrained(
        "stabilityai/stable-fast-3d",
        config_name="config.yaml",
        weight_name="model.safetensors",
    )
    model.to(device)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        autocast = torch.autocast(device_type=device, dtype=torch.bfloat16) if device == "cuda" else nullcontext()
        with autocast:
            mesh, _ = model.run_image(
                [prepared],
                bake_resolution=texture_resolution,
                remesh=remesh,
                vertex_count=vertex_count,
            )
    glb_path = out_dir / "mesh.glb"
    mesh.export(str(glb_path), include_normals=True)
    final_path = ASSETS / f"{name}.glb"
    shutil.copy2(glb_path, final_path)
    meta["glb"] = str(final_path)
    meta["elapsed_sec"] = round(time.time() - t0, 2)
    if torch.cuda.is_available():
        meta["peak_memory_mb"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
    (ASSETS / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(final_path)


def main() -> None:
    run_one(
        "object_c_sf3d_local_hamburger",
        ROOT / "submodules" / "threestudio" / "threestudio-main" / "load" / "images" / "hamburger_rgba.png",
    )


if __name__ == "__main__":
    main()
