from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
THREESTUDIO = ROOT / "submodules" / "threestudio" / "threestudio-main"


def font(size: int) -> ImageFont.ImageFont:
    for name in ["arial.ttf", "msyh.ttc", "simhei.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def triplet_rgb(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    w, h = image.size
    return image.crop((0, 0, w // 3, h))


def object_cutout(rgb: Image.Image, threshold: int = 238, remove_green: bool = False) -> Image.Image:
    rgba = rgb.convert("RGBA")
    pixels = rgba.load()
    w, h = rgba.size
    for y in range(h):
        for x in range(w):
            r, g, b, a = pixels[x, y]
            # Remove near-white or pure-black diagnostic backgrounds.
            green_bg = remove_green and g > 75 and g > r * 1.12 and g > b * 1.12
            if (
                (r > threshold and g > threshold and b > threshold)
                or (r < 8 and g < 8 and b < 8)
                or green_bg
            ):
                pixels[x, y] = (r, g, b, 0)
    bbox = rgba.getbbox()
    if bbox:
        rgba = rgba.crop(bbox)
    return rgba


def paste_shadow(base: Image.Image, obj: Image.Image, center: tuple[int, int], scale: float) -> None:
    w = max(1, int(obj.width * scale))
    h = max(1, int(obj.height * scale))
    obj = obj.resize((w, h), Image.Resampling.LANCZOS)
    x = center[0] - w // 2
    y = center[1] - h
    shadow = Image.new("RGBA", (w, max(8, h // 8)), (0, 0, 0, 60))
    shadow = shadow.filter(ImageFilter.GaussianBlur(8)) if False else shadow
    base.alpha_composite(shadow, (x, y + h - shadow.height // 2))
    base.alpha_composite(obj, (x, y))


def label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    f = font(24)
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=f)
    draw.rectangle((bbox[0] - 8, bbox[1] - 5, bbox[2] + 8, bbox[3] + 5), fill=(255, 255, 255, 220))
    draw.text((x, y), text, fill=(20, 32, 44), font=f)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    hamburger_input = THREESTUDIO / "load" / "images" / "hamburger_rgba.png"
    Image.open(hamburger_input).convert("RGBA").save(FIG / "object_c_input_hamburger_rgba.png")

    bg = Image.open(FIG / "real_3dgs_garden_render_1.png").convert("RGBA")
    bg = ImageOps.fit(bg, (1500, 900), method=Image.Resampling.LANCZOS)

    truck = Image.open(FIG / "real_3dgs_object_a_truck_render_1.png").convert("RGB")
    apple = triplet_rgb(
        THREESTUDIO
        / "outputs"
        / "dreamfusion-sd"
        / "a_shiny_red_apple_with_a_small_green_leaf,_single_object,_centered,_high_detail,_studio_lighting,_white_background@20260623-073254"
        / "save"
        / "it1800-test"
        / "30.png"
    )
    burger = triplet_rgb(
        THREESTUDIO
        / "outputs"
        / "zero123-simple"
        / "sds_64_hamburger_rgba.png@20260623-072805"
        / "save"
        / "it600-test"
        / "30.png"
    )

    # Keep the actual generated/rendered pixels, but remove simple diagnostic backgrounds.
    truck_obj = object_cutout(truck, threshold=246)
    apple_obj = object_cutout(apple, threshold=246, remove_green=True)
    burger_obj = object_cutout(burger, threshold=246)

    base = bg.copy()
    paste_shadow(base, truck_obj, (470, 710), 1.2)
    paste_shadow(base, apple_obj, (860, 650), 1.35)
    paste_shadow(base, burger_obj, (1125, 680), 1.05)

    draw = ImageDraw.Draw(base, "RGBA")
    label(draw, (300, 735), "A: 3DGS truck")
    label(draw, (760, 680), "B: SDS apple")
    label(draw, (1020, 710), "C: Zero123 hamburger")
    label(draw, (36, 30), "Garden 3DGS background + generated/rendered assets")
    base.convert("RGB").save(FIG / "fusion_garden_abc_composite.png", quality=95)


if __name__ == "__main__":
    main()
