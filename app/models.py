"""Pydantic models shared by the pipeline, the API and the templates."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Verdict(str, Enum):
    MATCH = "MATCH"
    NEAR_MATCH = "NEAR MATCH"
    MISMATCH = "MISMATCH"
    NOT_FOUND = "NOT FOUND"
    SKIPPED = "SKIPPED"


class Status(str, Enum):
    """Outcome of a check or of the whole label."""

    PASS = "PASS"
    REVIEW = "REVIEW"
    FAIL = "FAIL"
    ERROR = "ERROR"


class Application(BaseModel):
    """What the applicant told TTB. Blank optional fields are skipped."""

    brand_name: str
    class_type: str
    alcohol_content: str
    net_contents: str
    bottler_name_address: str = ""
    country_of_origin: str = ""
    application_id: str = ""

    @field_validator("*", mode="before")
    @classmethod
    def _strip(cls, v):
        return v.strip() if isinstance(v, str) else v


class FieldResult(BaseModel):
    key: str
    label: str
    expected: str
    found: str | None = None
    verdict: Verdict
    score: int = 0
    note: str = ""
    lines: list[int] | None = None   # OCR line range [first, last] the value was read from, when located
    box: list[float] | None = None   # where it sits on the label: [left, top, width, height] in % of the image


class DiffItem(BaseModel):
    expected: str
    found: str
    expected_before: str = ""   # a few words of context from the required text
    expected_after: str = ""
    found_before: str = ""      # the same context as read from the label
    found_after: str = ""


class WarningResult(BaseModel):
    present: bool
    found_text: str = ""
    wording: Status
    wording_score: int = 0
    wording_note: str = ""
    diff: list[DiffItem] = Field(default_factory=list)
    heading_caps: Status
    heading_caps_note: str = ""
    heading_bold: Status
    heading_bold_note: str = ""
    bold_ratio: float | None = None
    overall: Status
    box: list[float] | None = None   # where the statement sits on the label, in % of the image


class Timings(BaseModel):
    read_ms: float = 0.0
    match_ms: float = 0.0
    total_ms: float = 0.0


class VerificationResult(BaseModel):
    overall: Status
    summary: str
    fields: list[FieldResult]
    warning: WarningResult
    timings: Timings
    reader: str
    ocr_text: str = ""
    image_name: str = ""
    application_id: str = ""
    read_confidence: float | None = None   # mean OCR word confidence (0-100), when the reader reports one
    image_aspect: float | None = None      # height / width of the label image, for the evidence crops
    skew_deg: float = 0.0                  # the reader straightened the image by this much; previews must match
