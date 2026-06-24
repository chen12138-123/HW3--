from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "output" / "assets_sota"
ASSET.mkdir(parents=True, exist_ok=True)


def flatten_rgba(src: Path, dst: Path, bg=(255, 255, 255)) -> None:
    image = Image.open(src).convert("RGBA")
    canvas = Image.new("RGBA", image.size, bg + (255,))
    canvas.alpha_composite(image)
    canvas.convert("RGB").save(dst)


def main() -> None:
    c_src = ROOT / "submodules" / "threestudio" / "threestudio-main" / "load" / "images" / "hamburger_rgba.png"
    c_dst = ASSET / "object_c_hamburger_rgb_for_sf3d.png"
    flatten_rgba(c_src, c_dst)
    meta = {
        "object_c_rgb_input": str(c_dst),
        "source": str(c_src),
        "background": "white",
    }
    (ASSET / "prepared_inputs.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(c_dst)


if __name__ == "__main__":
    main()
