from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from PIL import Image, ImageOps

try:
    import rembg
except Exception:
    rembg = None

try:
    from .mesh_to_3dgs import convert_mesh_to_3dgs
    from .ply_utils import cloud_stats, ensure_dir, load_ply
except ImportError:
    from mesh_to_3dgs import convert_mesh_to_3dgs
    from ply_utils import cloud_stats, ensure_dir, load_ply


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "output" / "final_assets"
FIG = ROOT / "figures"
LOG = ROOT / "output" / "logs"
TRIPOSR_ROOT = Path(r"C:\TripoSR")
SF3D_ROOT = Path(r"C:\sf3d_ascii")
SEED = 20260623


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_probe(name: str, cmd: list[str], timeout: int = 120) -> dict[str, Any]:
    t0 = time.time()
    meta: dict[str, Any] = {
        "name": name,
        "command": cmd,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        meta.update(
            {
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-4000:],
                "stderr_tail": proc.stderr[-4000:],
            }
        )
    except Exception as exc:
        meta["error"] = f"{type(exc).__name__}: {exc}"
    meta["elapsed_sec"] = round(time.time() - t0, 2)
    return meta


def record_sota_attempts() -> None:
    attempts = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notes": (
            "The preferred current SOTA routes were attempted first. They are recorded here "
            "so the final report distinguishes attempted algorithms from completed results."
        ),
        "attempts": [
            {
                "algorithm": "Tencent Hunyuan3D-2.1",
                "role": "preferred text/image-to-3D route for B/C",
                "repository": "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1",
                "status": "blocked",
                "reason": "git clone was reset by the network before the working tree was downloaded; HF Space also timed out earlier.",
            },
            {
                "algorithm": "Microsoft TRELLIS / TRELLIS.2",
                "role": "preferred image-to-3D route with Gaussian/mesh output",
                "repository": "https://github.com/microsoft/TRELLIS",
                "status": "blocked",
                "reason": "HF Space API requests timed out from this environment; local Windows dependencies and 12GB VRAM are not sufficient for a reliable full run.",
            },
            {
                "algorithm": "Stability AI Stable Fast 3D",
                "role": "fast feed-forward image-to-3D fallback",
                "repository": "https://github.com/Stability-AI/stable-fast-3d",
                "status": "attempted",
                "reason": "local code is available; the gated model/HF Space failed previously, so TripoSR is used as the guaranteed local completion path.",
            },
            {
                "algorithm": "TripoSR",
                "role": "completed local single-image-to-3D reconstruction for B/C",
                "repository": "https://github.com/VAST-AI-Research/TripoSR",
                "status": "selected",
                "reason": "pretrained weights are cached locally and the method completes on the available RTX 4080 Laptop GPU.",
            },
        ],
    }
    write_json(ASSETS / "sota_attempts.json", attempts)


def remove_background(image: Image.Image) -> Image.Image:
    image = image.convert("RGBA")
    if rembg is not None:
        try:
            session = rembg.new_session()
            image = rembg.remove(image, session=session).convert("RGBA")
        except Exception:
            pass

    arr = np.asarray(image).astype(np.int16)
    corners = np.concatenate(
        [
            arr[:24, :24, :3].reshape(-1, 3),
            arr[:24, -24:, :3].reshape(-1, 3),
            arr[-24:, :24, :3].reshape(-1, 3),
            arr[-24:, -24:, :3].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(corners, axis=0)
    diff = np.abs(arr[:, :, :3] - bg).mean(axis=2)
    bg_alpha = np.clip((diff - 14) * 10, 0, 255).astype(np.uint8)
    old_alpha = arr[:, :, 3].astype(np.uint8)
    alpha = np.minimum(old_alpha, bg_alpha)
    alpha[old_alpha < 24] = 0
    out = arr.astype(np.uint8)
    out[:, :, 3] = np.maximum(out[:, :, 3], alpha)
    out[:, :, 3] = alpha
    return Image.fromarray(out, "RGBA")


def crop_and_center_rgba(src: Path, dst: Path, foreground_ratio: float = 0.82, size: int = 512) -> Image.Image:
    image = remove_background(Image.open(src))
    alpha = image.getchannel("A")
    alpha = alpha.point(lambda p: 0 if p < 32 else min(255, int((p - 32) * 1.35)))
    image.putalpha(alpha)
    bbox = image.getbbox()
    if bbox is None:
        raise RuntimeError(f"No foreground detected in {src}")
    image = image.crop(bbox)
    max_side = max(image.size)
    canvas_side = max(1, int(max_side / foreground_ratio))
    canvas = Image.new("RGBA", (canvas_side, canvas_side), (0, 0, 0, 0))
    canvas.alpha_composite(image, ((canvas_side - image.width) // 2, (canvas_side - image.height) // 2))
    canvas = canvas.resize((size, size), Image.Resampling.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst)
    return canvas


def crop_b_subject(src: Path, dst: Path) -> Image.Image:
    """Crop the generated planter image to the actual object before background removal."""
    image = Image.open(src).convert("RGB")
    w, h = image.size
    # The SD output placed the object on the right side and included a wooden tabletop.
    crop = image.crop((int(w * 0.43), int(h * 0.17), int(w * 0.98), int(h * 0.93)))
    tmp = ASSETS / "object_b" / "subject_crop.png"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    crop.save(tmp)
    rgba = remove_background(crop)
    arr = np.asarray(rgba).copy()
    rgb = arr[:, :, :3].astype(np.int16)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    green_score = g - np.maximum(r, b)
    green_leaf = (g > 95) & (green_score > 8) & (g > b + 18)
    bright_pot = (r > 175) & (g > 165) & (b > 150) & (np.abs(r - g) < 55) & (np.abs(g - b) < 65)
    yellow_leaf = (r > 145) & (g > 125) & (b < 105)
    soil = (r > 75) & (r < 170) & (g > 55) & (g < 145) & (b > 25) & (b < 115)
    mask = green_leaf | bright_pot | yellow_leaf
    yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
    center_mask = ((xx - mask.shape[1] * 0.56) / (mask.shape[1] * 0.45)) ** 2 + (
        (yy - mask.shape[0] * 0.55) / (mask.shape[0] * 0.55)
    ) ** 2 < 1.0
    pot_top = ((xx - mask.shape[1] * 0.43) / (mask.shape[1] * 0.26)) ** 2 + (
        (yy - mask.shape[0] * 0.72) / (mask.shape[0] * 0.13)
    ) ** 2 < 1.0
    mask = (mask & center_mask) | (soil & pot_top)
    alpha = np.where(mask, 255, 0).astype(np.uint8)
    # Smooth only the edge; keep the interior opaque so TripoSR sees a clean object.
    alpha_img = Image.fromarray(alpha, "L").filter(ImageFilter.GaussianBlur(1.2)) if False else Image.fromarray(alpha, "L")
    rgba.putalpha(alpha_img)
    bbox = rgba.getbbox()
    if bbox is None:
        raise RuntimeError("Object B subject mask is empty")
    rgba = rgba.crop(bbox)
    canvas_side = max(1, int(max(rgba.size) / 0.78))
    canvas = Image.new("RGBA", (canvas_side, canvas_side), (0, 0, 0, 0))
    canvas.alpha_composite(rgba, ((canvas_side - rgba.width) // 2, (canvas_side - rgba.height) // 2))
    canvas = canvas.resize((512, 512), Image.Resampling.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst)
    return canvas


def flatten_rgba(image: Image.Image, background=(255, 255, 255)) -> Image.Image:
    canvas = Image.new("RGBA", image.size, background + (255,))
    canvas.alpha_composite(image)
    return canvas.convert("RGB")


def create_object_b_input() -> Path:
    src = generate_object_b_text_image()
    out_dir = ensure_dir(ASSETS / "object_b")
    rgba = crop_b_apple(src, out_dir / "input_rgba.png")
    rgb = flatten_rgba(rgba, background=(245, 245, 245))
    rgb_path = out_dir / "input_rgb.png"
    rgb.save(rgb_path)
    shutil.copy2(rgba.filename if getattr(rgba, "filename", None) else out_dir / "input_rgba.png", FIG / "object_b_final_input_rgba.png")
    rgb.save(FIG / "object_b_final_input_rgb.png")
    return rgb_path


def crop_b_apple(src: Path, dst: Path) -> Image.Image:
    image = Image.open(src).convert("RGB")
    arr = np.asarray(image).astype(np.int16)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    apple = (r > 135) & (r > g * 1.45) & (r > b * 1.35)
    leaf = (g > 55) & (r > 50) & (r > g * 1.05) & (r > b * 1.25) & (g > b * 1.05)
    highlight = (r > 220) & (g > 210) & (b > 200)
    yy, xx = np.ogrid[: apple.shape[0], : apple.shape[1]]
    roi = ((xx - apple.shape[1] * 0.50) / (apple.shape[1] * 0.24)) ** 2 + (
        (yy - apple.shape[0] * 0.56) / (apple.shape[0] * 0.26)
    ) ** 2 < 1.0
    leaf_roi = ((xx - apple.shape[1] * 0.52) / (apple.shape[1] * 0.15)) ** 2 + (
        (yy - apple.shape[0] * 0.35) / (apple.shape[0] * 0.13)
    ) ** 2 < 1.0
    mask = (apple & roi) | (leaf & leaf_roi) | (highlight & roi)
    if mask.sum() < 500:
        raise RuntimeError("Object B apple mask is unexpectedly small")
    alpha = np.where(mask, 255, 0).astype(np.uint8)
    rgba = Image.fromarray(np.dstack([arr.astype(np.uint8), alpha]), "RGBA")
    bbox = rgba.getbbox()
    if bbox is None:
        raise RuntimeError("Object B apple mask is empty")
    rgba = rgba.crop(bbox)
    side = max(1, int(max(rgba.size) / 0.72))
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.alpha_composite(rgba, ((side - rgba.width) // 2, (side - rgba.height) // 2))
    canvas = canvas.resize((512, 512), Image.Resampling.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst)
    return canvas


def generate_object_b_text_image() -> Path:
    out_dir = ensure_dir(ASSETS / "object_b")
    image_path = out_dir / "text_prompt_sd15_apple.png"
    meta_path = out_dir / "text_prompt_sd15_apple.json"
    if image_path.exists():
        return image_path
    from diffusers import StableDiffusionPipeline

    prompt = (
        "a glossy red apple with one small green leaf, single centered object, "
        "isolated on pure white background, product render, high detail, no table"
    )
    negative_prompt = (
        "multiple objects, plate, bowl, basket, table, floor, wall, scene, text, "
        "watermark, cropped, blurry, low quality, deformed"
    )
    seed = SEED + 17
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
        local_files_only=True,
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=40,
        guidance_scale=8.5,
        generator=generator,
    ).images[0]
    image.save(image_path)
    shutil.copy2(image_path, FIG / "object_b_text_prompt_sd15_apple.png")
    write_json(
        meta_path,
        {
            "algorithm": "Stable Diffusion v1.5 text-to-image frontend for text-to-3D",
            "model": "runwayml/stable-diffusion-v1-5",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "num_inference_steps": 40,
            "guidance_scale": 8.5,
            "output": str(image_path),
        },
    )
    return image_path


def create_object_c_input() -> Path:
    src = ROOT / "submodules" / "threestudio" / "threestudio-main" / "load" / "images" / "hamburger_rgba.png"
    out_dir = ensure_dir(ASSETS / "object_c")
    rgba = crop_and_center_rgba(src, out_dir / "input_rgba.png", foreground_ratio=0.82)
    rgb = flatten_rgba(rgba, background=(245, 245, 245))
    rgb_path = out_dir / "input_rgb.png"
    rgb.save(rgb_path)
    shutil.copy2(out_dir / "input_rgba.png", FIG / "object_c_final_input_rgba.png")
    rgb.save(FIG / "object_c_final_input_rgb.png")
    return rgb_path


def load_triposr_model():
    if not TRIPOSR_ROOT.exists():
        raise FileNotFoundError(f"TripoSR root not found: {TRIPOSR_ROOT}")
    sys.path.insert(0, str(TRIPOSR_ROOT))
    from tsr.system import TSR  # type: ignore

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    local_model = ROOT / "data" / "models" / "TripoSR"
    model_id = str(local_model) if local_model.exists() else "stabilityai/TripoSR"
    model = TSR.from_pretrained(model_id, config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(8192)
    model.to(device)
    model.eval()
    return model, device, model_id


def render_grid(images: list[Image.Image], out_path: Path, label: str) -> None:
    thumbs = [img.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS) for img in images]
    cols = 4
    rows = int(np.ceil(len(thumbs) / cols))
    grid = Image.new("RGB", (cols * 256, rows * 256), (255, 255, 255))
    for i, img in enumerate(thumbs):
        grid.paste(img, ((i % cols) * 256, (i // cols) * 256))
    grid.save(out_path)


def run_triposr_asset(model, device: str, model_id: str, name: str, image_path: Path, resolution: int, splat_points: int) -> dict[str, Any]:
    out_dir = ensure_dir(ASSETS / name)
    meta: dict[str, Any] = {
        "name": name,
        "algorithm": "TripoSR",
        "repository": "https://github.com/VAST-AI-Research/TripoSR",
        "model": model_id,
        "input_image": str(image_path),
        "marching_cubes_resolution": resolution,
        "splat_points": splat_points,
        "device": device,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    t0 = time.time()
    image = Image.open(image_path).convert("RGB")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        scene_codes = model([image], device=device)
        renders = model.render(scene_codes, n_views=8, elevation_deg=0, height=256, width=256, return_type="pil")[0]
        meshes = model.extract_mesh(scene_codes, True, resolution=resolution)
    mesh = meshes[0]
    obj_path = out_dir / "mesh.obj"
    glb_path = out_dir / "mesh.glb"
    mesh.export(obj_path)
    mesh.export(glb_path)
    for i, render in enumerate(renders):
        render.save(out_dir / f"render_{i:03d}.png")
    render_grid(renders, FIG / f"{name}_triposr_views_grid.png", name)

    ply_path = out_dir / "model.ply"
    convert_mesh_to_3dgs(obj_path, ply_path, splat_points)
    shutil.copy2(ply_path, ROOT / "output" / name / "model.ply")
    shutil.copy2(glb_path, ROOT / "output" / name / "mesh.glb")
    shutil.copy2(obj_path, ROOT / "output" / name / "mesh.obj")

    mesh_loaded = trimesh.load(str(obj_path), force="mesh")
    meta.update(
        {
            "obj": str(obj_path),
            "glb": str(glb_path),
            "ply": str(ply_path),
            "vertices": int(len(mesh_loaded.vertices)),
            "faces": int(len(mesh_loaded.faces)),
            "bounds": mesh_loaded.bounds.tolist() if mesh_loaded.bounds is not None else None,
            "elapsed_sec": round(time.time() - t0, 2),
            "cloud_stats": cloud_stats(ply_path),
        }
    )
    if torch.cuda.is_available():
        meta["peak_memory_mb"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
    write_json(out_dir / "metadata.json", meta)
    write_json(ASSETS / f"{name}.json", meta)
    return meta


def main() -> None:
    ensure_dir(ASSETS)
    ensure_dir(FIG)
    ensure_dir(LOG)
    for name in ["object_b_final", "object_c_final"]:
        ensure_dir(ROOT / "output" / name)

    record_sota_attempts()
    b_input = create_object_b_input()
    c_input = create_object_c_input()
    model, device, model_id = load_triposr_model()
    results = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "object_b": run_triposr_asset(model, device, model_id, "object_b_final", b_input, resolution=160, splat_points=120000),
        "object_c": run_triposr_asset(model, device, model_id, "object_c_final", c_input, resolution=160, splat_points=120000),
    }
    write_json(ASSETS / "final_asset_summary.json", results)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
