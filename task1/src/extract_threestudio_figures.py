from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
THREESTUDIO = ROOT / "submodules" / "threestudio" / "threestudio-main"


def numeric_key(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


def latest_trial(exp_name: str, glob_pattern: str) -> Path:
    exp_dir = THREESTUDIO / "outputs" / exp_name
    trials = sorted(exp_dir.glob(glob_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not trials:
        raise FileNotFoundError(f"No trial matched {exp_dir / glob_pattern}")
    return trials[0]


def latest_test_dir(save_dir: Path, requested_step: int | None) -> tuple[int, Path]:
    candidates: list[tuple[int, Path]] = []
    for p in save_dir.glob("it*-test"):
        match = re.fullmatch(r"it(\d+)-test", p.name)
        if match and p.is_dir():
            candidates.append((int(match.group(1)), p))
    if not candidates:
        raise FileNotFoundError(f"No it*-test directory under {save_dir}")
    if requested_step is not None:
        for step, path in candidates:
            if step == requested_step:
                return step, path
        raise FileNotFoundError(f"No it{requested_step}-test directory under {save_dir}")
    return max(candidates, key=lambda item: item[0])


def load_triplet(path: Path) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
    image = Image.open(path).convert("RGB")
    w, h = image.size
    third = w // 3
    return (
        image,
        image.crop((0, 0, third, h)),
        image.crop((third, 0, 2 * third, h)),
        image.crop((2 * third, 0, w, h)),
    )


def label(image: Image.Image, text: str) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    draw.rectangle((0, 0, x1 - x0 + 12, y1 - y0 + 9), fill=(255, 255, 255))
    draw.text((6, 4), text, fill=(0, 0, 0), font=font)
    return out


def grid(images: list[Image.Image], cols: int, pad: int = 10) -> Image.Image:
    if not images:
        raise ValueError("No images supplied")
    cell_w = max(im.width for im in images)
    cell_h = max(im.height for im in images)
    rows = (len(images) + cols - 1) // cols
    out = Image.new(
        "RGB",
        (cols * cell_w + (cols + 1) * pad, rows * cell_h + (rows + 1) * pad),
        (246, 247, 248),
    )
    for idx, im in enumerate(images):
        x = pad + (idx % cols) * (cell_w + pad) + (cell_w - im.width) // 2
        y = pad + (idx // cols) * (cell_h + pad) + (cell_h - im.height) // 2
        out.paste(im, (x, y))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True, help="threestudio output experiment name")
    parser.add_argument("--trial-glob", required=True, help="glob under outputs/<exp>")
    parser.add_argument("--prefix", required=True, help="figure filename prefix")
    parser.add_argument("--step", type=int)
    parser.add_argument("--views", default="0,8,16,24,32,40,48,56")
    parser.add_argument("--triplets", default="0,15,30,45")
    parser.add_argument("--progress", default="")
    parser.add_argument("--thumb", type=int, default=192)
    parser.add_argument("--cols", type=int, default=4)
    args = parser.parse_args()

    FIG.mkdir(exist_ok=True)
    trial = latest_trial(args.exp, args.trial_glob)
    save_dir = trial / "save"
    step, test_dir = latest_test_dir(save_dir, args.step)
    frames = sorted(test_dir.glob("*.png"), key=numeric_key)
    if not frames:
        raise FileNotFoundError(f"No PNG frames under {test_dir}")

    view_ids = [int(x) for x in args.views.split(",") if x.strip()]
    rgb_tiles: list[Image.Image] = []
    for idx in view_ids:
        _, rgb, _, _ = load_triplet(frames[idx % len(frames)])
        rgb = rgb.resize((args.thumb, args.thumb), Image.Resampling.LANCZOS)
        rgb_tiles.append(label(rgb, f"view {idx:02d}"))
    grid(rgb_tiles, args.cols).save(FIG / f"{args.prefix}_rgb_views_grid.png")

    triplet_ids = [int(x) for x in args.triplets.split(",") if x.strip()]
    for idx in triplet_ids:
        triplet, _, _, _ = load_triplet(frames[idx % len(frames)])
        triplet = triplet.resize((args.thumb * 3, args.thumb), Image.Resampling.LANCZOS)
        label(triplet, f"view {idx:02d}: RGB | normal | opacity").save(
            FIG / f"{args.prefix}_triplet_{idx:03d}.png"
        )

    progress_steps = [int(x) for x in args.progress.split(",") if x.strip()]
    progress_tiles: list[Image.Image] = []
    for progress_step in progress_steps:
        p = save_dir / f"it{progress_step}-0.png"
        if not p.exists():
            continue
        _, rgb, _, _ = load_triplet(p)
        rgb = rgb.resize((args.thumb, args.thumb), Image.Resampling.LANCZOS)
        progress_tiles.append(label(rgb, f"{progress_step} steps"))
    if progress_tiles:
        grid(progress_tiles, min(len(progress_tiles), args.cols)).save(
            FIG / f"{args.prefix}_progress_rgb.png"
        )

    print(f"trial={trial}")
    print(f"step={step}")
    print(f"frames={len(frames)}")


if __name__ == "__main__":
    main()
