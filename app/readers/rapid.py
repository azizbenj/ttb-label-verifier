"""Second local reader: RapidOCR (PP-OCRv4 text detection + recognition on ONNX Runtime, CPU only).

The models ship inside the wheel, so nothing is downloaded at runtime and the reader works inside a
locked-down network like Tesseract does. It reads display and curved typefaces, light text over
photographs and small print far better than Tesseract, at about one second per label on a laptop
CPU. It is used two ways:

  * as the escalation of the default reader (``add_rapid_view``): when Tesseract's passes leave a
    field NOT FOUND / MISMATCH or the warning short of PASS, the upright image is read once more
    with RapidOCR and its lines are appended as a further view, so the existing matching sees them;
  * as a reader of its own (``RapidOCRReader``, ``OCR_ENGINE=rapid``).

RapidOCR returns one quadrilateral box, text and confidence per text line. The pipeline needs word
boxes (pins, crops, the bold heuristic), so each line's box is split across its words in proportion
to their character counts, along the line's long axis. Everything else (scaling, straightening, the
ink mask) is shared with the Tesseract reader so the coordinates of both readers agree.
"""

from __future__ import annotations

import importlib.metadata
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from functools import lru_cache
from time import perf_counter

import numpy as np
from PIL import Image

from ..config import (RAPID_ESCALATION, RAPID_INPUT, RAPID_MIN_LINE_CONF, RAPID_THREADS, RAPID_TIMEOUT_S,
                      RAPID_WORKERS)
from ..images import flatten
from ..normalize import normalize_loose
from .base import LabelReading, OCRResult, OCRWord, ReaderError, View
from .tesseract import preprocess_with_skew

log = logging.getLogger("labelcheck")

ENGINE_KEY = "rapid"      # OCRWord.engine of every word this module produces

_engine = None
_engine_lock = threading.Lock()
# Reads run on these threads so a caller can give up after RAPID_TIMEOUT_S (the engine call itself
# cannot be interrupted: a read that is abandoned finishes on its thread and the pool size caps how
# many reads, abandoned or not, run at once). One engine instance serves every thread: ONNX Runtime
# sessions are safe to run concurrently and RapidOCR keeps no per-call state (measured: four
# concurrent reads of one engine return exactly the sequential results).
_POOL = ThreadPoolExecutor(max_workers=RAPID_WORKERS, thread_name_prefix="rapid")


@lru_cache(maxsize=1)
def rapid_version() -> str | None:
    """The installed rapidocr-onnxruntime version, or None when the package is missing."""
    try:
        return importlib.metadata.version("rapidocr-onnxruntime")
    except importlib.metadata.PackageNotFoundError:
        return None


def _onnxruntime_version() -> str:
    try:
        return importlib.metadata.version("onnxruntime")
    except importlib.metadata.PackageNotFoundError:
        return "?"


def engine():
    """The process-wide RapidOCR engine, created on first use (about 0.2 s, models loaded from the wheel)."""
    global _engine
    with _engine_lock:
        if _engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError:
                raise ReaderError("The RapidOCR reader is not installed on this server "
                                  "(pip install rapidocr-onnxruntime). Please tell the administrator.") from None
            kwargs = {"intra_op_num_threads": RAPID_THREADS} if RAPID_THREADS > 0 else {}
            eng = RapidOCR(**kwargs)
            # The first run of a session is several times slower than the rest (ONNX Runtime warms its
            # kernels); pay that on a small blank image here rather than on a label's clock.
            eng(np.full((64, 256), 255, dtype=np.uint8))
            _engine = eng
        return _engine


def rapid_input(img: Image.Image, original: Image.Image, skew: float, scaled_size: tuple[int, int],
                mode: str = RAPID_INPUT) -> np.ndarray:
    """The array handed to RapidOCR, in the coordinates of the preprocessed (scaled, straightened) image.

    ``gray`` is the same contrast-stretched grayscale Tesseract reads; ``color`` is the original colours
    scaled and straightened the same way (BGR, as RapidOCR expects a 3-channel array). Measured on the
    real labels both read about the same; ``gray`` is the default because it needs no second resize.
    Both sides are padded with white to a multiple of 32 pixels (the detector's own grid), which keeps
    the detector's internal resize at the image's own size: an OpenCV 5.0.0 resize of an unpadded
    1799-pixel-wide label crashed the process on macOS/arm64, and a fresh, padded array did not in 140
    reads. Padding on the right and bottom leaves every coordinate unchanged.
    """
    if mode == "color":
        rgb = flatten(original).convert("RGB").resize(scaled_size, Image.LANCZOS)
        if skew:
            rgb = rgb.rotate(skew, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
        arr = np.asarray(rgb)[:, :, ::-1]
    else:
        arr = np.asarray(img.convert("L"))
    h, w = arr.shape[:2]
    ph, pw = -(-h // 32) * 32, -(-w // 32) * 32
    out = np.full((ph, pw) + arr.shape[2:], 255, dtype=np.uint8)
    out[:h, :w] = arr
    return out


def _call(eng, arr: np.ndarray) -> list:
    res, _ = eng(arr)
    return res or []


def run_engine(arr: np.ndarray, timeout: float = RAPID_TIMEOUT_S) -> list:
    """RapidOCR's raw result: a list of [quad (4 points), text, confidence 0-1] per line, bounded in time
    (the one-off engine start-up is not counted against it)."""
    eng = engine()
    future = _POOL.submit(_call, eng, arr)
    try:
        return future.result(timeout=timeout)
    except FutureTimeout:
        raise ReaderError(f"Reading the label with RapidOCR took longer than {timeout:g} s and was stopped. "
                          "Please try a smaller or cleaner image.") from None
    except ReaderError:
        raise
    except Exception as e:  # a model or OpenCV failure must read as a plain message, never a traceback
        log.warning("RapidOCR failed: %s", e)
        raise ReaderError("RapidOCR could not read this label. Please try another image or the Tesseract reader.") from e


def lines_from_result(res: list, min_conf: float = 0.0) -> tuple[list[str], list[OCRWord], float | None]:
    """Lines, word boxes and mean confidence (0-100) from RapidOCR's raw result; lines scoring under
    ``min_conf`` (0-100) are dropped.

    Each line's quadrilateral becomes an upright box; the box is split across the line's words in
    proportion to their character counts along the line's long axis (vertical for text printed
    sideways), so pins, crops and the bold heuristic have something close to real word boxes.
    """
    lines: list[str] = []
    words: list[OCRWord] = []
    for quad, text, conf in res:
        text = " ".join(str(text).split())
        conf = float(conf)
        if not text or conf * 100 < min_conf or not any(ch.isalnum() for ch in text):
            continue
        xs = [float(p[0]) for p in quad]
        ys = [float(p[1]) for p in quad]
        left, top = min(xs), min(ys)
        width, height = max(1.0, max(xs) - left), max(1.0, max(ys) - top)
        vertical = height > width and len(text) >= 3
        line_index = len(lines)
        lines.append(text)
        pos = 0
        for tok in text.split():
            start, end = pos / len(text), (pos + len(tok)) / len(text)
            pos += len(tok) + 1
            if vertical:
                box = (left, top + start * height, width, (end - start) * height)
            else:
                box = (left + start * width, top, (end - start) * width, height)
            words.append(OCRWord(text=tok, left=int(round(box[0])), top=int(round(box[1])),
                                 width=max(1, int(round(box[2]))), height=max(1, int(round(box[3]))),
                                 conf=round(conf * 100, 1), line_index=line_index, engine=ENGINE_KEY))
    mean = sum(w.conf for w in words) / len(words) if words else None
    return lines, words, mean


def engine_label() -> str:
    return f"rapidocr {rapid_version() or '?'} (PP-OCRv4, onnxruntime {_onnxruntime_version()})"


class RapidOCRReader:
    """RapidOCR as the reader of a label (``OCR_ENGINE=rapid``), on its own."""

    name = "rapid"

    def __init__(self, input_mode: str = RAPID_INPUT, escalate_with_tesseract: bool = RAPID_ESCALATION):
        self.input_mode = input_mode
        self.escalate_with_tesseract = escalate_with_tesseract

    def read(self, image: Image.Image) -> LabelReading:
        t0 = perf_counter()
        img, ink, skew, scaled_size = preprocess_with_skew(image)
        res = run_engine(rapid_input(img, image, skew, scaled_size, self.input_mode))
        lines, words, mean_conf = lines_from_result(res)
        ocr = OCRResult(text="\n".join(lines), lines=lines, words=words, ink=ink, mean_conf=mean_conf,
                        engine=engine_label(), ms=(perf_counter() - t0) * 1000,
                        views=[View(rot=0, inverted=False, ink=ink, size=img.size, engine=ENGINE_KEY)],
                        source=(img, image, skew, scaled_size), skew=skew)
        return LabelReading(ocr=ocr)

    def extend(self, reading: LabelReading) -> bool:
        """Tesseract as the escalation of RapidOCR: its two upright passes are appended as a further
        view when RapidOCR left something missing (the reverse of the default arrangement; it follows
        the same RAPID_ESCALATION switch)."""
        from .tesseract import TesseractReader, tesseract_version
        ocr = reading.ocr
        if not self.escalate_with_tesseract or ocr.extended or ocr.source is None or tesseract_version() is None:
            return False
        ocr.extended = True
        t0 = perf_counter()
        img = ocr.source[0]
        try:
            lines2, words2, _ = TesseractReader().read_preprocessed(img)
        except ReaderError:
            return False
        added = append_view(ocr, lines2, words2, View(rot=0, inverted=False, ink=ocr.views[0].ink, size=img.size),
                            min_conf=TesseractReader.MIN_EXTRA_LINE_CONF)
        ocr.ms += (perf_counter() - t0) * 1000
        ocr.engine += " + tesseract pass"
        return added


def append_view(ocr: OCRResult, lines2: list[str], words2: list[OCRWord], view: View, *, min_conf: float = 0.0) -> bool:
    """Append another reading of the label as a view of its own. Lines are kept even when an earlier
    pass read the same text: the warning statement is assembled per view, and a view with holes where
    another pass happened to read a line identically would assemble nothing. Lines read with less
    than ``min_conf`` (0-100, the mean over the line's words) are left out. Returns True when a line
    was added."""
    view_index = len(ocr.views)
    ocr.views.append(view)
    by_line: dict[int, list[OCRWord]] = {}
    for w in words2:
        by_line.setdefault(w.line_index, []).append(w)
    added = False
    for idx, line in enumerate(lines2):
        members = by_line.get(idx, [])
        conf = sum(w.conf for w in members) / len(members) if members else 0.0
        if not normalize_loose(line) or conf < min_conf:
            continue
        new_index = len(ocr.lines)
        ocr.lines.append(line)
        for w in members:
            w.line_index, w.view = new_index, view_index
            ocr.words.append(w)
        added = True
    ocr.text = "\n".join(ocr.lines)
    return added


def add_rapid_view(ocr: OCRResult, input_mode: str = RAPID_INPUT) -> bool:
    """The escalation: read the upright image with RapidOCR and append its lines as a new view.

    Runs once per label, only when the pipeline still has something missing after Tesseract's own
    extra passes. A read that fails or runs past RAPID_TIMEOUT_S is skipped, never fatal. Returns
    True when lines were added."""
    if ocr.escalated or ocr.source is None or not ocr.views:
        return False
    ocr.escalated = True
    t0 = perf_counter()
    img, original, skew, scaled_size = ocr.source
    try:
        res = run_engine(rapid_input(img, original, skew, scaled_size, input_mode))
    except ReaderError as e:
        log.warning("RapidOCR escalation skipped: %s", e)
        return False
    lines2, words2, _ = lines_from_result(res, min_conf=RAPID_MIN_LINE_CONF)
    added = append_view(ocr, lines2, words2, View(rot=0, inverted=False, ink=ocr.views[0].ink, size=img.size,
                                                  engine=ENGINE_KEY))
    ocr.ms += (perf_counter() - t0) * 1000
    ocr.engine += " + rapidocr pass"
    return added
