"""The cloud reader's mapping from the model's structured answer to a LabelReading (SDK call mocked)."""

from types import SimpleNamespace

from PIL import Image

from app.models import Application, Status, Verdict
from app.pipeline import verify
from app.readers import claude_vision
from app.readers.claude_vision import ClaudeVisionReader, LabelExtraction


class FakeMessages:
    def __init__(self, parsed, stop_reason="end_turn"):
        self.parsed, self.stop_reason, self.calls = parsed, stop_reason, []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(parsed_output=self.parsed, stop_reason=self.stop_reason)


def make_reader(monkeypatch, parsed, stop_reason="end_turn"):
    fake = FakeMessages(parsed, stop_reason)
    monkeypatch.setattr(claude_vision.anthropic, "Anthropic", lambda: SimpleNamespace(messages=fake))
    return ClaudeVisionReader(model="test-model"), fake


WARNING = ("GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink alcoholic beverages "
           "during pregnancy because of the risk of birth defects. (2) Consumption of alcoholic beverages impairs "
           "your ability to drive a car or operate machinery, and may cause health problems.")


def test_reader_maps_fields_and_warning_hint(monkeypatch):
    parsed = LabelExtraction(
        transcription="OLD TOM DISTILLERY\nKentucky Straight Bourbon Whiskey\n45% Alc./Vol. (90 Proof)\n750 mL\n" + WARNING,
        brand_name="OLD TOM DISTILLERY", class_type="Kentucky Straight Bourbon Whiskey",
        alcohol_content="45% Alc./Vol. (90 Proof)", net_contents="750 mL", bottler_name_address=None,
        country_of_origin=None, government_warning_text=WARNING, warning_heading_is_bold=True)
    reader, fake = make_reader(monkeypatch, parsed)
    reading = reader.read(Image.new("RGB", (1200, 1600), "white"))
    assert reading.fields["brand_name"] == "OLD TOM DISTILLERY" and reading.fields["bottler_name_address"] is None
    assert reading.warning_hint.heading_bold is True and reading.ocr.lines[0] == "OLD TOM DISTILLERY"
    call = fake.calls[0]
    assert call["model"] == "test-model" and call["output_format"] is LabelExtraction
    assert call["messages"][0]["content"][0]["type"] == "image"

    app = Application(brand_name="OLD TOM DISTILLERY", class_type="Kentucky Straight Bourbon Whiskey",
                      alcohol_content="90 Proof", net_contents="750 mL")
    result = verify(app, Image.new("RGB", (10, 10), "white"), reader)
    assert result.overall == Status.PASS and result.warning.heading_bold == Status.PASS
    assert {f.verdict for f in result.fields if f.verdict != Verdict.SKIPPED} == {Verdict.MATCH}


def test_reader_appends_warning_when_missing_from_transcription(monkeypatch):
    parsed = LabelExtraction(transcription="OLD TOM DISTILLERY", brand_name="OLD TOM DISTILLERY", class_type=None,
                             alcohol_content=None, net_contents=None, bottler_name_address=None, country_of_origin=None,
                             government_warning_text=WARNING, warning_heading_is_bold=False)
    reader, _ = make_reader(monkeypatch, parsed)
    reading = reader.read(Image.new("RGB", (10, 10), "white"))
    assert any(l.startswith("GOVERNMENT WARNING") for l in reading.ocr.lines)


def test_refusal_raises_readable_error(monkeypatch):
    parsed = LabelExtraction(transcription="", brand_name=None, class_type=None, alcohol_content=None, net_contents=None,
                             bottler_name_address=None, country_of_origin=None, government_warning_text=None,
                             warning_heading_is_bold=None)
    reader, _ = make_reader(monkeypatch, parsed, stop_reason="refusal")
    try:
        reader.read(Image.new("RGB", (10, 10), "white"))
    except RuntimeError as e:
        assert "declined" in str(e)
    else:
        raise AssertionError("expected a RuntimeError")
