"""Local OCR with Tesseract 5 (the default reader; runs fully inside the agency network)."""

from __future__ import annotations

from functools import lru_cache
from time import perf_counter

import numpy as np
import pytesseract
from PIL import Image, ImageOps
from pytesseract import Output

from ..config import TESSERACT_CMD, TESSERACT_MAX_WIDTH, TESSERACT_MIN_WIDTH, TESSERACT_PSM, TESSERACT_PSM_EXTRA
from ..normalize import normalize_loose
from .base import LabelReading, OCRResult, OCRWord

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


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


def preprocess(image: Image.Image, min_width: int = TESSERACT_MIN_WIDTH,
               max_width: int = TESSERACT_MAX_WIDTH) -> tuple[Image.Image, np.ndarray]:
    """Grayscale, scale into Tesseract's comfort zone, stretch contrast, and build an ink mask.

    Returns (image for OCR, boolean ink array in the same coordinates).
    """
    img = ImageOps.exif_transpose(image).convert("L")
    if img.width < min_width:
        scale = min_width / img.width
    elif img.width > max_width:
        scale = max_width / img.width
    else:
        scale = 1.0
    if scale != 1.0:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    img = ImageOps.autocontrast(img, cutoff=1)
    arr = np.asarray(img)
    ink = arr < otsu_threshold(arr)
    if ink.mean() > 0.5:  # light text on a dark label: flip so Tesseract sees dark-on-light
        img = ImageOps.invert(img)
        ink = ~ink
    return img, ink


def _group_words(data: dict) -> tuple[list[str], list[OCRWord], float | None]:
    lines: list[str] = []
    words: list[OCRWord] = []
    line_words: dict[tuple[int, int, int], list[int]] = {}
    confs: list[float] = []
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
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        if key not in line_words:
            line_words[key] = []
            lines.append("")
        line_index = len(lines) - 1 if not line_words[key] else _line_index_of(line_words, key)
        line_words[key].append(len(words))
        words.append(OCRWord(text=text, left=int(data["left"][i]), top=int(data["top"][i]),
                             width=int(data["width"][i]), height=int(data["height"][i]), conf=conf,
                             line_index=line_index))
        confs.append(conf)
    # Rebuild line strings from their words (keys were created in reading order).
    for idx, key in enumerate(line_words):
        lines[idx] = " ".join(words[w].text for w in line_words[key])
        for w in line_words[key]:
            words[w].line_index = idx
    return lines, words, (sum(confs) / len(confs) if confs else None)


def _line_index_of(line_words: dict, key: tuple) -> int:
    return list(line_words).index(key)


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
        data = pytesseract.image_to_data(img, config=f"--psm {psm} --oem 1", output_type=Output.DICT)
        return _group_words(data)

    def read(self, image: Image.Image) -> LabelReading:
        t0 = perf_counter()
        img, ink = preprocess(image)
        lines, words, mean_conf = self._pass(img, self.psm)
        if self.extra_psm is not None and self.extra_psm != self.psm:
            lines2, words2, _ = self._pass(img, self.extra_psm)
            seen = {normalize_loose(l) for l in lines}
            for idx, line in enumerate(lines2):
                key = normalize_loose(line)
                if not key or key in seen:
                    continue
                seen.add(key)
                new_index = len(lines)
                lines.append(line)
                for w in words2:
                    if w.line_index == idx:
                        w.line_index = new_index
                        words.append(w)
        mode = f"psm {self.psm}" + (f"+{self.extra_psm}" if self.extra_psm is not None else "")
        ocr = OCRResult(text="\n".join(lines), lines=lines, words=words, ink=ink, mean_conf=mean_conf,
                        engine=f"tesseract {tesseract_version() or '?'} ({mode})", ms=(perf_counter() - t0) * 1000)
        return LabelReading(ocr=ocr)
