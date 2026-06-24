from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from gradio_client import Client

try:
    from gradio_client import handle_file
except ImportError:
    def handle_file(path: str) -> str:
        return path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "assets_sota"
OUT.mkdir(parents=True, exist_ok=True)


def _copy_file(value: Any, target: Path) -> str | None:
    if value is None:
        return None
    path: str | None = None
    if isinstance(value, str):
        path = value
    elif isinstance(value, dict):
        path = value.get("path") or value.get("name")
    elif hasattr(value, "path"):
        path = str(value.path)
    if not path:
        return None
    src = Path(path)
    if not src.exists():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, target)
    return str(target)


def _serializable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(v) for v in value]
    if hasattr(value, "path"):
        return {"path": str(value.path)}
    return repr(value)


def write_meta(name: str, meta: dict[str, Any]) -> None:
    meta = dict(meta)
    meta["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (OUT / f"{name}.json").write_text(
        json.dumps(_serializable(meta), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_hunyuan_text_to_3d() -> None:
    name = "object_b_hunyuan3d2_text"
    prompt = (
        "a cute ceramic robot planter, small rounded white robot body, "
        "green succulent plant growing from the top, clean studio lighting, "
        "single centered object, high quality 3D asset"
    )
    params = {
        "text_prompt": prompt,
        "image": None,
        "front": None,
        "back": None,
        "left": None,
        "right": None,
        "inference_steps": 50,
        "guidance_scale": 5.5,
        "seed": 20260623,
        "octree_resolution": 256,
        "remove_background": True,
        "number_of_chunks": 200000,
        "randomize_seed": False,
    }
    meta: dict[str, Any] = {
        "algorithm": "Tencent Hunyuan3D-2 official Hugging Face Space",
        "space": "tencent/Hunyuan3D-2",
        "endpoint": "/generation_all",
        "params": params,
    }
    try:
        client = Client("tencent/Hunyuan3D-2")
        result = client.predict(
            params["text_prompt"],
            params["image"],
            params["front"],
            params["back"],
            params["left"],
            params["right"],
            params["inference_steps"],
            params["guidance_scale"],
            params["seed"],
            params["octree_resolution"],
            params["remove_background"],
            params["number_of_chunks"],
            params["randomize_seed"],
            api_name="/generation_all",
        )
        meta["raw_result"] = result
        if isinstance(result, tuple):
            mesh_file = _copy_file(result[0], OUT / f"{name}_shape.glb")
            textured_file = _copy_file(result[1], OUT / f"{name}_textured.glb")
            meta["copied_files"] = {
                "shape": mesh_file,
                "textured": textured_file,
            }
            if len(result) > 2:
                meta["html_output"] = result[2]
            if len(result) > 3:
                meta["mesh_stats"] = result[3]
            if len(result) > 4:
                meta["used_seed"] = result[4]
        write_meta(name, meta)
    except Exception as exc:
        meta["error"] = f"{type(exc).__name__}: {exc}"
        write_meta(name, meta)
        raise


def run_sf3d_image_to_3d() -> None:
    name = "object_c_sf3d_hamburger"
    input_image = ROOT / "submodules" / "threestudio" / "threestudio-main" / "load" / "images" / "hamburger_rgba.png"
    params = {
        "input_image": str(input_image),
        "foreground_ratio": 0.85,
        "remesh_option": "Triangle",
        "vertex_count": 16000,
        "texture_size": 1024,
    }
    meta: dict[str, Any] = {
        "algorithm": "Stability AI Stable Fast 3D official Hugging Face Space",
        "space": "stabilityai/stable-fast-3d",
        "endpoint": "/run_button",
        "params": params,
    }
    try:
        client = Client("stabilityai/stable-fast-3d")
        result = client.predict(
            handle_file(str(input_image)),
            params["foreground_ratio"],
            params["remesh_option"],
            params["vertex_count"],
            params["texture_size"],
            api_name="/run_button",
        )
        meta["raw_result"] = result
        if isinstance(result, tuple):
            preview = _copy_file(result[0], OUT / f"{name}_foreground.png")
            model = _copy_file(result[1], OUT / f"{name}.glb")
            meta["copied_files"] = {"preview": preview, "model": model}
        write_meta(name, meta)
    except Exception as exc:
        meta["error"] = f"{type(exc).__name__}: {exc}"
        write_meta(name, meta)
        raise


def run_sf3d_image_to_3d_retry_rgb() -> None:
    name = "object_c_sf3d_hamburger_rgb_retry"
    input_image = OUT / "object_c_hamburger_rgb_for_sf3d.png"
    params = {
        "input_image": str(input_image),
        "foreground_ratio": 0.85,
        "remesh_option": "None",
        "vertex_count": -1,
        "texture_size": 1024,
    }
    meta: dict[str, Any] = {
        "algorithm": "Stability AI Stable Fast 3D official Hugging Face Space",
        "space": "stabilityai/stable-fast-3d",
        "endpoint": "/run_button",
        "params": params,
    }
    try:
        client = Client("stabilityai/stable-fast-3d")
        result = client.predict(
            handle_file(str(input_image)),
            params["foreground_ratio"],
            params["remesh_option"],
            params["vertex_count"],
            params["texture_size"],
            api_name="/run_button",
        )
        meta["raw_result"] = result
        if isinstance(result, tuple):
            preview = _copy_file(result[0], OUT / f"{name}_foreground.png")
            model = _copy_file(result[1], OUT / f"{name}.glb")
            meta["copied_files"] = {"preview": preview, "model": model}
        write_meta(name, meta)
    except Exception as exc:
        meta["error"] = f"{type(exc).__name__}: {exc}"
        write_meta(name, meta)
        raise


def main() -> None:
    run_hunyuan_text_to_3d()
    run_sf3d_image_to_3d()


if __name__ == "__main__":
    main()
