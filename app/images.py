"""Opening uploaded label images safely and turning any image mode into plain 8-bit pixels.

Both readers go through ``flatten`` so a label exported with a transparent background, a 16-bit
scan or a CMYK file is read the same way as an ordinary RGB PNG.
"""

from __future__ import annotations

import io
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import MAX_IMAGE_PIXELS, MIN_IMAGE_SIDE


class ImageError(ValueError):
    """An image we cannot check, with a message the agent can act on."""


def open_image(data: bytes, name: str) -> Image.Image:
    """Decode an uploaded file, refusing non-images, decompression bombs and unreadably small images."""
    if not data:
        raise ImageError(f"'{name}' is empty.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(data))
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ImageError(f"'{name}' is {image.width} x {image.height} pixels, more than this tool accepts "
                                 f"({MAX_IMAGE_PIXELS // 1_000_000} megapixels). Please export the label at a smaller size.")
            image.load()
    except ImageError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ImageError(f"'{name}' is too large to open safely. Please export the label at a smaller size.") from None
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, EOFError):
        raise ImageError(f"'{name}' could not be read as an image. Please upload a PNG or JPG of the label.") from None
    if min(image.size) < MIN_IMAGE_SIDE:
        raise ImageError(f"'{name}' is only {image.width} x {image.height} pixels, too small to read. "
                         "Please upload the label at full size.")
    return image


def flatten(image: Image.Image) -> Image.Image:
    """Upright, 8-bit RGB or L pixels: EXIF rotation applied, transparency over white, deep images rescaled."""
    img = ImageOps.exif_transpose(image)
    if img.mode in ("I;16", "I;16L", "I;16B", "I;16N", "I", "F"):
        # Pillow clips these to 0-255 when converting to "L", which turns a 16-bit scan white.
        arr = np.asarray(img, dtype=np.float64)
        lo, hi = float(arr.min()), float(arr.max())
        scaled = (arr - lo) * (255.0 / (hi - lo)) if hi > lo else np.zeros_like(arr)
        return Image.fromarray(scaled.round().astype(np.uint8))
    if img.mode in ("RGBA", "LA", "PA", "La", "RGBa") or (img.mode == "P" and "transparency" in img.info):
        # Transparent pixels usually store black: without this, black text on a transparent
        # background becomes black on black.
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    if img.mode not in ("RGB", "L"):
        return img.convert("RGB")
    return img
