"""Local OCR with Tesseract 5 (the default reader; runs fully inside the agency network)."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from time import perf_counter

import numpy as np
import pytesseract
from PIL import Image, ImageFilter, ImageOps
from pytesseract import Output

from ..config import (TESSERACT_CMD, TESSERACT_MAX_PIXELS, TESSERACT_MAX_SIDE, TESSERACT_MAX_WIDTH,
                      TESSERACT_MIN_WIDTH, TESSERACT_PSM, TESSERACT_PSM_EXTRA, TESSERACT_TIMEOUT_S)
from ..images import flatten
from ..normalize import normalize_loose
from .base import LabelReading, OCRResult, OCRWord, ReaderError, View

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
# Tesseract's OpenMP threads fight each other (and our batch workers) for the CPU. One thread per
# process measured 0.66 s instead of 1.30 s median per label, and a 250-label batch with four workers
# in 44 s instead of 157 s, with identical verdicts. Set the variable yourself to override.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")


# Each pass is a separate tesseract process: passes of one label run side by side.
_PASSES = ThreadPoolExecutor(max_workers=int(os.getenv("TESSERACT_PARALLEL_PASSES", "4")), thread_name_prefix="ocr")


@lru_cache(maxsize=1)
def tesseract_version() -> str | None:
    try:
        return str(pytesseract.get_tesseract_version())
    except Exception:  # binary missing
        return None


def otsu_threshold(gray: np.ndarray) -> int:
    """Classic Otsu: the gray level that best separates ink from paper."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    sum_total = float(np.dot(np.arange(256), hist))
    sum_b = w_b = 0.0
    best, thr = 0.0, 127
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b, m_f = sum_b / w_b, (sum_total - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best:
            best, thr = var, t
    return thr


def ocr_scale(width: int, height: int, min_width: int = TESSERACT_MIN_WIDTH, max_width: int = TESSERACT_MAX_WIDTH) -> float:
    """Scale factor that brings the label into Tesseract's comfort zone without exploding its size.

    The width target alone is not enough: a 40 x 4000 strip would be upscaled to 1600 x 160000.
    """
    if width < min_width:
        scale = min_width / width
    elif width > max_width:
        scale = max_width / width
    else:
        scale = 1.0
    return min(scale, TESSERACT_MAX_SIDE / max(width, height), (TESSERACT_MAX_PIXELS / (width * height)) ** 0.5)


def principal_channel(rgb: Image.Image) -> Image.Image:
    """The single gray channel that best separates the label's colors (first principal component).

    Plain luminance can make orange text on green, or gold on purple, almost the same gray; the
    principal component keeps them apart. Oriented so that the background (the median) is light."""
    a = np.asarray(rgb.convert("RGB"), dtype=np.float32)
    flat = a.reshape(-1, 3)
    sample = flat[:: max(1, len(flat) // 200_000)]
    mean = sample.mean(0)
    _, _, vt = np.linalg.svd(sample - mean, full_matrices=False)
    p = (a - mean) @ vt[0]
    lo, hi = float(p.min()), float(p.max())
    p = (p - lo) * (255.0 / (hi - lo)) if hi > lo else np.zeros_like(p)
    img = Image.fromarray(p.astype(np.uint8))
    if np.median(p) < 128:
        img = ImageOps.invert(img)
    return ImageOps.autocontrast(img, cutoff=1)


def local_contrast(gray: Image.Image, radius: int = 24, gain: float = 2.2) -> Image.Image:
    """Subtract the local background so faint or small text stands out wherever it sits."""
    a = np.asarray(gray, dtype=np.float32)
    bg = np.asarray(gray.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
    return Image.fromarray(np.clip(128 + (a - bg) * gain, 0, 255).astype(np.uint8))


def ink_mask(gray: Image.Image) -> np.ndarray:
    arr = np.asarray(gray)
    ink = arr < otsu_threshold(arr)
    return ~ink if ink.mean() > 0.5 else ink


def preprocess(image: Image.Image, min_width: int = TESSERACT_MIN_WIDTH,
               max_width: int = TESSERACT_MAX_WIDTH) -> tuple[Image.Image, np.ndarray]:
    """Grayscale, scale into Tesseract's comfort zone, stretch contrast, and build an ink mask.

    Returns (image for OCR, boolean ink array in the same coordinates).
    """
    img = flatten(image).convert("L")
    scale = ocr_scale(img.width, img.height, min_width, max_width)
    if abs(scale - 1.0) > 1e-3:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    img = ImageOps.autocontrast(img, cutoff=1)
    arr = np.asarray(img)
    ink = arr < otsu_threshold(arr)
    if ink.mean() > 0.5:  # light text on a dark label: flip so Tesseract sees dark-on-light
        img = ImageOps.invert(img)
        ink = ~ink
    return img, ink


def _group_words(data: dict) -> tuple[list[str], list[OCRWord], float | None]:
    """Words grouped into lines, in Tesseract's reading order."""
    words: list[OCRWord] = []
    line_words: dict[tuple[int, int, int], list[OCRWord]] = {}
    for i, raw in enumerate(data["text"]):
        text = (raw or "").strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < 0:
            continue
        word = OCRWord(text=text, left=int(data["left"][i]), top=int(data["top"][i]), width=int(data["width"][i]),
                       height=int(data["height"][i]), conf=conf, line_index=-1)
        line_words.setdefault((data["block_num"][i], data["par_num"][i], data["line_num"][i]), []).append(word)
        words.append(word)
    lines: list[str] = []
    for idx, members in enumerate(line_words.values()):   # keys were created in reading order
        lines.append(" ".join(w.text for w in members))
        for w in members:
            w.line_index = idx
    return lines, words, (sum(w.conf for w in words) / len(words) if words else None)


def parse_psm(spec: str | int | None, default: int = TESSERACT_PSM) -> tuple[int, int | None]:
    """"4" -> (4, None); "4+11" -> (4, 11). Used by config and the benchmarking knob."""
    if spec is None or spec == "":
        return default, TESSERACT_PSM_EXTRA
    parts = [int(x) for x in str(spec).split("+")]
    return parts[0], (parts[1] if len(parts) > 1 else None)


class TesseractReader:
    """Runs Tesseract in one page-segmentation mode and, optionally, a second one.

    No single mode reads every label: the uniform-block modes drop oversized brand lines, the
    sparse mode occasionally misses short centred lines. Merging two passes (lines the first pass
    did not produce are appended) costs ~0.3 s and lifts recall; the guided matching does not mind
    the extra lines.
    """

    name = "tesseract"

    def __init__(self, psm: int = TESSERACT_PSM, extra_psm: int | None = TESSERACT_PSM_EXTRA):
        self.psm = psm
        self.extra_psm = extra_psm

    def _pass(self, img: Image.Image, psm: int):
        try:
            data = pytesseract.image_to_data(img, config=f"--psm {psm} --oem 1", output_type=Output.DICT,
                                             timeout=TESSERACT_TIMEOUT_S)
        except pytesseract.TesseractNotFoundError:
            raise ReaderError("Local OCR (Tesseract) is not installed on this server. "
                              "Please tell the administrator.") from None
        except RuntimeError as e:  # pytesseract reports its timeout as RuntimeError
            raise ReaderError(f"Reading the label took longer than {TESSERACT_TIMEOUT_S} s and was stopped. "
                              "Please try a smaller or cleaner image.") from e
        return _group_words(data)

    def read(self, image: Image.Image) -> LabelReading:
        t0 = perf_counter()
        img, ink = preprocess(image)
        second = None
        if self.extra_psm is not None and self.extra_psm != self.psm:
            second = _PASSES.submit(self._pass, img, self.extra_psm)
        lines, words, mean_conf = self._pass(img, self.psm)
        if second is not None:
            lines2, words2, _ = second.result()
            seen = {normalize_loose(l) for l in lines}
            moved: dict[int, int] = {}   # second-pass line index -> index in the merged list
            for idx, line in enumerate(lines2):
                key = normalize_loose(line)
                if not key or key in seen:
                    continue
                seen.add(key)
                moved[idx] = len(lines)
                lines.append(line)
            # Re-index in one go: re-indexing inside the loop above let a word whose new index equals a
            # later second-pass index be picked up again (wrong line, duplicated word).
            for w in words2:
                if w.line_index in moved:
                    w.line_index = moved[w.line_index]
                    words.append(w)
        mode = f"psm {self.psm}" + (f"+{self.extra_psm}" if self.extra_psm is not None else "")
        ocr = OCRResult(text="\n".join(lines), lines=lines, words=words, ink=ink, mean_conf=mean_conf,
                        engine=f"tesseract {tesseract_version() or '?'} ({mode})", ms=(perf_counter() - t0) * 1000,
                        views=[View(rot=0, inverted=False, ink=ink, size=img.size)], source=(img, image))
        return LabelReading(ocr=ocr)

    # Lines from the extra passes must look like text: reading horizontal print sideways produces
    # low-confidence fragments, and those must not be matched against the application.
    MIN_EXTRA_LINE_CONF = 55
    MIN_EXTRA_WORDS = 2

    def extend(self, reading: LabelReading) -> bool:
        """Read the label again turned 90 degrees each way (sideways text: warnings and bottler lines
        on cans and wine labels) and inverted (light text on dark panels). New lines are appended.
        Returns True when anything was added. Costs about three more passes; the pipeline only asks
        for it when the first read left something missing."""
        ocr = reading.ocr
        if ocr.extended or ocr.source is None:
            return False
        ocr.extended = True
        t0 = perf_counter()
        img, original = ocr.source
        base_ink = ocr.views[0].ink
        # Chosen by measurement on 20 real approved labels (scripts/real_labels.py): the two turned
        # views read sideways warnings and bottler lines, the color-aware local-contrast view reads
        # light or colored text on colored panels. Inverting the image added nothing (Tesseract
        # already handles light-on-dark text), and further views added nothing either.
        rgb = flatten(original).convert("RGB").resize(img.size, Image.LANCZOS)
        contrast = local_contrast(principal_channel(rgb))
        variants = [
            (90, False, img.rotate(90, expand=True), np.rot90(base_ink, 1)),
            (270, False, img.rotate(270, expand=True), np.rot90(base_ink, -1)),
            (0, False, contrast, ink_mask(contrast)),
        ]
        futures = [_PASSES.submit(self._pass, v[2], 11) for v in variants]
        seen = {normalize_loose(l) for l in ocr.lines}
        added = False
        for (rot, inverted, view_img, view_ink), fut in zip(variants, futures):
            lines2, words2, _ = fut.result()
            view_index = len(ocr.views)
            ocr.views.append(View(rot=rot, inverted=inverted, ink=view_ink, size=view_img.size))
            by_line: dict[int, list[OCRWord]] = {}
            for w in words2:
                by_line.setdefault(w.line_index, []).append(w)
            for idx, line in enumerate(lines2):
                members = by_line.get(idx, [])
                real = [w for w in members if sum(ch.isalpha() for ch in w.text) >= 3]
                conf = sum(w.conf for w in members) / len(members) if members else 0
                key = normalize_loose(line)
                if not key or key in seen or len(real) < self.MIN_EXTRA_WORDS or conf < self.MIN_EXTRA_LINE_CONF:
                    continue
                seen.add(key)
                new_index = len(ocr.lines)
                ocr.lines.append(line)
                for w in members:
                    w.line_index, w.view = new_index, view_index
                    ocr.words.append(w)
                added = True
        ocr.text = "\n".join(ocr.lines)
        ocr.ms += (perf_counter() - t0) * 1000
        ocr.engine += " + turned and contrast passes"
        return added
