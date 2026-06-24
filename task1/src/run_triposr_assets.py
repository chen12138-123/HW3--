from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
TRIPOSR_ROOT = Path(r"C:\TripoSR")
ASSET = ROOT / "output" / "assets_sota"
ASSET.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(TRIPOSR_ROOT))

from tsr.system import TSR  # noqa: E402
from tsr.utils import resize_foreground  # noqa: E402


def prepare_image(path: Path, out_path: Path, foreground_ratio: float = 0.85) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    image = resize_foreground(image, foreground_ratio)
    arr = np.asarray(image).astype(np.float32) / 255.0
    rgb = arr[:, :, :3] * arr[:, :, 3:4] + (1.0 - arr[:, :, 3:4]) * 0.5
    prepared = Image.fromarray((rgb * 255.0).astype(np.uint8))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.save(out_path)
    return prepared


def run_one(model: TSR, name: str, image_path: Path, resolution: int = 192) -> None:
    out_dir = ASSET / name
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_image(image_path, out_dir / "input_prepared.png")
    meta = {
        "algorithm": "TripoSR official code with skimage marching-cubes fallback",
        "repository": "https://github.com/VAST-AI-Research/TripoSR",
        "model": "stabilityai/TripoSR",
        "input": str(image_path),
        "mc_resolution": resolution,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(f"[{name}] start", flush=True)
    t0 = time.time()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        print(f"[{name}] forward", flush=True)
        scene_codes = model([prepared], device=device)
        print(f"[{name}] render views", flush=True)
        renders = model.render(scene_codes, n_views=8, return_type="pil")
        print(f"[{name}] extract mesh resolution={resolution}", flush=True)
        meshes = model.extract_mesh(scene_codes, True, resolution=resolution)
    for i, image in enumerate(renders[0]):
        image.save(out_dir / f"render_{i:03d}.png")
    mesh = meshes[0]
    obj_path = out_dir / "mesh.obj"
    glb_path = out_dir / "mesh.glb"
    mesh.export(obj_path)
    mesh.export(glb_path)
    shutil.copy2(glb_path, ASSET / f"{name}.glb")
    shutil.copy2(obj_path, ASSET / f"{name}.obj")
    meta.update(
        {
            "obj": str(ASSET / f"{name}.obj"),
            "glb": str(ASSET / f"{name}.glb"),
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "bounds": mesh.bounds.tolist() if mesh.bounds is not None else None,
            "elapsed_sec": round(time.time() - t0, 2),
        }
    )
    if torch.cuda.is_available():
        meta["peak_memory_mb"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
    (ASSET / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[main] loading TripoSR on {device}", flush=True)
    model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(8192)
    model.to(device)
    model.eval()
    print("[main] model loaded", flush=True)
    run_one(model, "object_b_triposr_robot_planter", ASSET / "object_b_text_prompt_sd15_robot_planter.png", resolution=128)
    run_one(
        model,
        "object_c_triposr_hamburger",
        ROOT / "submodules" / "threestudio" / "threestudio-main" / "load" / "images" / "hamburger_rgba.png",
        resolution=128,
    )


if __name__ == "__main__":
    main()
