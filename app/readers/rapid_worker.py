"""RapidOCR inside a worker process.

The engine runs native code (ONNX Runtime and OpenCV). A crash there, such as the segfault an
OpenCV 5.0.0 resize produced once on macOS/arm64, would otherwise take the whole web server down
with every batch job it holds in memory. In a worker process a crash fails only that read: the
parent sees a broken pool, starts a new one, and the label keeps the verdict Tesseract gave it.

This module imports nothing from the app, so a worker starts quickly and stays small.
"""

from __future__ import annotations

_engine = None


def init(threads: int) -> None:
    """Load the models once per worker and warm the session (its first run is several times slower)."""
    global _engine
    import numpy as np
    from rapidocr_onnxruntime import RapidOCR
    kwargs = {"intra_op_num_threads": threads} if threads > 0 else {}
    _engine = RapidOCR(**kwargs)
    _engine(np.full((64, 256), 255, dtype=np.uint8))


def read(arr) -> list:
    """RapidOCR's result as plain Python values (quad, text, confidence), so it pickles back cheaply."""
    res, _ = _engine(arr)
    return [[[[float(x), float(y)] for x, y in quad], str(text), float(conf)] for quad, text, conf in (res or [])]


def ping() -> bool:
    return _engine is not None
