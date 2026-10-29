"""Deployment photo of the ESP32-S3 board, prepared for the paper figure.

Source: ``figures/esp32.png``, the author's edited photo (4032x3024, RGBA).  It is
a hand-held shot with the board standing vertically, cut out on a transparent
background (91.8 % of the pixels are fully transparent, 0.2 % are the
anti-aliased edge).  ``figures/esp32.pdf`` carries the same photo re-encoded as a
JPEG, so the PNG is the better source: lossless, and the alpha channel already
gives a clean silhouette without touching the tones of the board.

The figure needs the board horizontal, cropped to that silhouette, on a white
background that blends with the page:

1. composite the RGBA image on white (the alpha edge stays anti-aliased);
2. crop to the alpha bounding box, with a small pad;
3. rotate 90 deg counter-clockwise (the brighter end, the USB ports, goes left);
4. resize to 1500 px wide and add a 2.5 % white margin, so that no dark pixel
   touches the figure boundary.

No tone mapping is applied: the board is genuinely dark (the solder mask is
black), and an autocontrast would only crush it further.  If the source has no
alpha channel, the background is removed with a flood fill from the border that
stops on strong gradients.

Usage: python make_fig_esp32.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "figures" / "esp32.png"
OUT = REPO / "figures" / "esp32_s3.png"
WIDTH = 1500
MARGIN = 0.025


def on_white(image: Image.Image) -> tuple[Image.Image, np.ndarray]:
    """Composite on white and return the image plus its opacity mask."""
    if image.mode == "RGBA":
        alpha = np.asarray(image)[:, :, 3].astype(float) / 255.0
        canvas = Image.new("RGB", image.size, (255, 255, 255))
        canvas.paste(image.convert("RGB"), (0, 0), image.split()[-1])
        return canvas, alpha
    return image.convert("RGB"), np.ones(image.size[::-1])


def whiten_background(image: Image.Image, work_width: int = 700) -> Image.Image:
    """Flood fill the dark background from the border, stopping on edges."""
    width, height = image.size
    small = image.resize((work_width, round(height * work_width / width)), Image.LANCZOS)
    gray = np.asarray(small.convert("L")).astype(float)
    gy, gx = np.gradient(gray)
    passable = (gray < 45) & (np.hypot(gx, gy) < 20)

    mask = np.zeros_like(passable)
    mask[0, :] |= passable[0, :]
    mask[-1, :] |= passable[-1, :]
    mask[:, 0] |= passable[:, 0]
    mask[:, -1] |= passable[:, -1]
    for _ in range(500):
        grown = mask.copy()
        grown[1:, :] |= mask[:-1, :]
        grown[:-1, :] |= mask[1:, :]
        grown[:, 1:] |= mask[:, :-1]
        grown[:, :-1] |= mask[:, 1:]
        grown &= passable
        if grown.sum() == mask.sum():
            break
        mask = grown

    full = np.asarray(
        Image.fromarray((mask * 255).astype(np.uint8)).resize((width, height), Image.NEAREST)
    ) > 127
    rgb = np.asarray(image).copy()
    rgb[full] = 255
    return Image.fromarray(rgb)


def main() -> None:
    source = Image.open(SRC)
    image, alpha = on_white(source)
    if alpha.min() > 0.99:                      # no cutout: remove the background
        image = whiten_background(image)
        alpha = np.asarray(image.convert("L")).astype(float)
        alpha = (255.0 - alpha) / 255.0          # dark = subject
        alpha[alpha < 0.02] = 0.0

    rows = np.where(alpha.mean(axis=1) > 0.005)[0]
    cols = np.where(alpha.mean(axis=0) > 0.005)[0]
    pad = round(0.02 * max(cols[-1] - cols[0], rows[-1] - rows[0]))
    box = (max(0, cols[0] - pad), max(0, rows[0] - pad),
           min(image.size[0], cols[-1] + 1 + pad), min(image.size[1], rows[-1] + 1 + pad))
    image = image.crop(box).transpose(Image.ROTATE_90)

    image = image.resize((WIDTH, round(image.size[1] * WIDTH / image.size[0])), Image.LANCZOS)
    border = round(MARGIN * WIDTH)
    canvas = Image.new("RGB", (WIDTH + 2 * border, image.size[1] + 2 * border), (255, 255, 255))
    canvas.paste(image, (border, border))
    canvas.save(OUT, optimize=True)
    print("saved", OUT.relative_to(REPO), canvas.size)


if __name__ == "__main__":
    main()
