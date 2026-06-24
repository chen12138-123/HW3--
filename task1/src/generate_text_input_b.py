from __future__ import annotations

import json
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "output" / "assets_sota"
ASSET.mkdir(parents=True, exist_ok=True)


def main() -> None:
    prompt = (
        "a cute ceramic robot planter, small rounded white robot body, "
        "green succulent plant growing from the top, single centered object, "
        "product photo, white background, high detail"
    )
    negative_prompt = (
        "multiple objects, text, watermark, blurry, cropped, low quality, "
        "deformed, cluttered background, extra arms"
    )
    seed = 20260623
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe = pipe.to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=35,
        guidance_scale=7.5,
        generator=generator,
    ).images[0]
    image_path = ASSET / "object_b_text_prompt_sd15_robot_planter.png"
    image.save(image_path)
    meta = {
        "algorithm": "Stable Diffusion v1.5 text-to-image as text-to-3D frontend",
        "model": "runwayml/stable-diffusion-v1-5",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "num_inference_steps": 35,
        "guidance_scale": 7.5,
        "output": str(image_path),
    }
    (ASSET / "object_b_text_prompt_sd15_robot_planter.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(image_path)


if __name__ == "__main__":
    main()
