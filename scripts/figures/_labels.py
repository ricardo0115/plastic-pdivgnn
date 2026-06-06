from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from PIL import JpegImagePlugin  # noqa: F401 - register JPEG handler for PDF export

FONT_CANDIDATES: tuple[str, ...] = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def add_row_labels_to_png(
    src: Path,
    dst: Path,
    labels: tuple[str, ...],
    font_size: int = 44,
    gap: int = 40,
    side_pad: int = 24,
) -> None:
    img = Image.open(src).convert("RGB")
    w, h = img.size
    n = len(labels)

    font = _load_font(font_size)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    text_widths: list[int] = [
        probe.textbbox((0, 0), text, font=font)[2]
        - probe.textbbox((0, 0), text, font=font)[0]
        for text in labels
    ]
    max_text_w = max(text_widths)
    margin = max_text_w + 2 * side_pad

    total_left = margin + gap
    canvas = Image.new("RGB", (w + total_left, h), "white")
    canvas.paste(img, (total_left, 0))

    draw = ImageDraw.Draw(canvas)
    row_h = h / n
    for i, text in enumerate(labels):
        x = margin // 2
        y = int(row_h * (i + 0.5))
        draw.text((x, y), text, fill="black", font=font, anchor="mm")

    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst)
