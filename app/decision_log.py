"""The decision log: every answer from the review queue, one JSON object per line.

``DECISION_LOG`` names the file (off when unset). A decision (``pass``: the label is fine and our flag
was a false alarm; ``fail``: the label is wrong; ``skip``) and its undo (``clear``) are appended with
what the tool had concluded about the label: the verdict, every question it raised and which one the
queue asked first, each field's verdict and note, the warning checks, the reader and the timings. Never
the image, its preview or the text read from it. ``scripts/decisions_report.py`` turns the log into
pass rates per rule and a calibration CSV. Standard library only.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from . import config
from .decisions import prompts_for

if TYPE_CHECKING:  # batch.py imports this module, so its types are only named here
    from .batch import BatchItem, BatchJob

log = logging.getLogger("labelcheck")
_LOCK = threading.Lock()   # several workers and requests append to the one file


def decision_record(job: BatchJob, item: BatchItem, index: int, value: str) -> dict:
    """One log line as a dict: what was decided and what the tool had concluded about the label."""
    r = item.result
    prompts = prompts_for(r) if r else []
    w = r.warning if r else None
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "job": job.id,
        "source": job.source,
        "index": index,
        "application_id": item.application_id,
        "image": item.image_name,
        "decision": value,                      # pass, fail, skip, or clear (the decision was undone)
        "overall": item.status,                 # PASS, REVIEW, FAIL, or ERROR (nothing was compared)
        "asked": prompts[0].key if prompts else None,   # the question the review queue asked first
        "prompts": [{"key": p.key, "what": p.what, "verdict": p.verdict, "question": p.question,
                     "left": list(p.left), "right": list(p.right), "asked": i == 0}
                    for i, p in enumerate(prompts)],
        "fields": [{"key": f.key, "verdict": f.verdict.value, "expected": f.expected, "found": f.found,
                    "note": f.note} for f in r.fields] if r else [],
        "warning": {"present": w.present, "wording": w.wording.value, "heading_caps": w.heading_caps.value,
                    "heading_bold": w.heading_bold.value, "wording_score": w.wording_score} if w else None,
        "read_confidence": r.read_confidence if r else None,
        "words_read": r.words_read if r else None,
        "reader": r.reader if r else "",
        "timings": {"read_ms": r.timings.read_ms, "match_ms": r.timings.match_ms,
                    "total_ms": r.timings.total_ms} if r else None,
    }


def log_decision(job: BatchJob, item: BatchItem, index: int, value: str) -> bool:
    """Append the decision to ``DECISION_LOG``; True when a line was written. A failure to write is a
    warning in the server log, never an error for the request: the decision is still kept with the batch."""
    path = config.DECISION_LOG
    if not path:
        return False
    try:
        line = json.dumps(decision_record(job, item, index, value), ensure_ascii=False, separators=(",", ":"))
        with _LOCK, open(path, "a", encoding="utf-8", errors="replace") as f:
            f.write(line + "\n")
        return True
    except Exception as e:  # a log entry is never a reason to fail the request
        log.warning("decision log: could not write to %s (%s: %s)", path, type(e).__name__, e)
        return False
