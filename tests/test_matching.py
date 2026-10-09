from app.matching import (
    compare_alcohol,
    compare_country,
    compare_text,
    compare_volume,
    locate_and_compare,
    locate_text,
)
from app.models import Verdict

LINES = [
    "OLD TOM DISTILLERY",
    "KENTUCKY STRAIGHT",
    "BOURBON WHISKEY",
    "45% ALC./VOL. (90 PROOF)",
    "750 mL",
    "Distilled and bottled by",
    "Old Tom Distillery, Bardstown, Kentucky 40004",
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women",
]


def test_locate_finds_exact_line():
    loc = locate_text("OLD TOM DISTILLERY", LINES)
    assert loc.text == "OLD TOM DISTILLERY" and loc.score == 100


def test_locate_spans_two_lines():
    loc = locate_text("Kentucky Straight Bourbon Whiskey", LINES)
    assert loc.text == "KENTUCKY STRAIGHT BOURBON WHISKEY" and loc.score == 100


def test_locate_extracts_substring_from_longer_line():
    loc = locate_text("Old Tom Distillery", ["Bottled by Old Tom Distillery, Bardstown, KY"])
    assert loc.text == "Old Tom Distillery"


def test_locate_returns_none_below_floor():
    assert locate_text("Riverbend Brewing Company", LINES) is None


def test_compare_text_exact():
    assert compare_text("brand_name", "OLD TOM DISTILLERY", "OLD TOM DISTILLERY").verdict == Verdict.MATCH


def test_brand_case_difference_needs_review():
    r = compare_text("brand_name", "STONE'S THROW", "Stone's Throw")
    assert r.verdict == Verdict.NEAR_MATCH and r.score == 100


def test_unicode_apostrophe_is_still_exact():
    assert compare_text("brand_name", "STONE'S THROW", "STONE’S THROW").verdict == Verdict.MATCH


def test_class_type_case_difference_is_match():
    r = compare_text("class_type", "Kentucky Straight Bourbon Whiskey", "KENTUCKY STRAIGHT BOURBON WHISKEY")
    assert r.verdict == Verdict.MATCH


def test_similar_text_is_near_match():
    r = compare_text("class_type", "Kentucky Straight Bourbon Whiskey", "Kentucky Straight Bourbon Whisky")
    assert r.verdict == Verdict.NEAR_MATCH and r.score >= 88


def test_different_text_is_mismatch():
    r = compare_text("brand_name", "OLD TOM DISTILLERY", "RIVER BEND BREWING")
    assert r.verdict == Verdict.MISMATCH


def test_not_found_when_nothing_located():
    r = locate_and_compare("brand_name", "RIVER BEND BREWING", LINES)
    assert r.verdict == Verdict.NOT_FOUND and r.found is None


def test_not_found_with_fallback_becomes_mismatch():
    r = locate_and_compare("brand_name", "RIVER BEND BREWING", LINES, fallback_found="OLD TOM DISTILLERY")
    assert r.verdict == Verdict.MISMATCH and r.found == "OLD TOM DISTILLERY"


def test_optional_blank_is_skipped():
    assert locate_and_compare("bottler_name_address", "", LINES).verdict == Verdict.SKIPPED


def test_multiline_address_matches():
    r = locate_and_compare("bottler_name_address", "Old Tom Distillery, Bardstown, Kentucky 40004", LINES)
    assert r.verdict == Verdict.MATCH, r


def test_address_with_prefix_on_label():
    r = locate_and_compare("bottler_name_address",
                           "Distilled and bottled by Old Tom Distillery, Bardstown, Kentucky 40004", LINES)
    assert r.verdict == Verdict.MATCH, r


def test_alcohol_match_percent_vs_percent_and_proof():
    r = compare_alcohol("45% Alc./Vol.", "\n".join(LINES))
    assert r.verdict == Verdict.MATCH and "90 PROOF" in r.found


def test_alcohol_proof_on_application_percent_on_label():
    assert compare_alcohol("90 proof", "ALC. 45% BY VOL.").verdict == Verdict.MATCH


def test_alcohol_proof_only_on_label():
    r = compare_alcohol("45% ABV", "Aged 4 years. 90 PROOF. 750 mL")
    assert r.verdict == Verdict.MATCH and "proof" in r.note.lower()


def test_label_proof_contradicting_its_percentage_needs_review():
    r = compare_alcohol("45%", "45% Alc./Vol. (80 Proof)")
    assert r.verdict == Verdict.NEAR_MATCH and "80" in r.note
    assert compare_alcohol("90 Proof", "45% Alc./Vol. (86 Proof)").verdict == Verdict.NEAR_MATCH
    assert compare_alcohol("45%", "45% Alc./Vol. (90 Proof)").verdict == Verdict.MATCH


def test_alcohol_mismatch():
    r = compare_alcohol("40% Alc./Vol.", "45% ALC./VOL.")
    assert r.verdict == Verdict.MISMATCH and "45%" in r.note


def test_alcohol_not_found():
    assert compare_alcohol("45% Alc./Vol.", "OLD TOM DISTILLERY 750 mL").verdict == Verdict.NOT_FOUND


def test_alcohol_bad_application_value():
    r = compare_alcohol("forty five", "45% ALC./VOL.")
    assert r.verdict == Verdict.NOT_FOUND and "application" in r.note


def test_volume_match_variants():
    assert compare_volume("750 mL", "750ml").verdict == Verdict.MATCH
    assert compare_volume("750 mL", "Net contents 75O ML").verdict == Verdict.MATCH
    assert compare_volume("355 mL", "12 FL. OZ.").verdict == Verdict.MATCH


def test_volume_mismatch_and_not_found():
    assert compare_volume("750 mL", "1 L").verdict == Verdict.MISMATCH
    assert compare_volume("750 mL", "OLD TOM").verdict == Verdict.NOT_FOUND


def test_country_found_and_skipped():
    lines = LINES + ["Product of Scotland"]
    assert compare_country("Scotland", lines).verdict == Verdict.MATCH
    assert compare_country("Product of Scotland", lines).verdict == Verdict.MATCH
    assert compare_country("", lines).verdict == Verdict.SKIPPED


def test_country_not_found():
    assert compare_country("France", LINES).verdict == Verdict.NOT_FOUND


def test_country_different_origin_statement_is_mismatch():
    r = compare_country("Scotland", LINES + ["Product of Ireland"])
    assert r.verdict == Verdict.MISMATCH and r.found == "Product of Ireland"


def test_country_match_shows_full_statement():
    r = compare_country("Scotland", LINES + ["Product of Scotland"])
    assert r.verdict == Verdict.MATCH and r.found == "Product of Scotland"


def test_similar_or_containing_country_names_are_never_a_silent_match():
    assert compare_country("Austria", ["Product of Australia"]).verdict == Verdict.NEAR_MATCH
    for app, label in [("Guinea", "Product of Equatorial Guinea"), ("Ireland", "Product of Northern Ireland"),
                       ("Mexico", "Distilled in New Mexico"), ("Dominica", "Product of Dominican Republic"),
                       ("Niger", "Product of Nigeria")]:
        r = compare_country(app, [label])
        assert r.verdict == Verdict.MISMATCH and r.found == label, (app, label, r)


def test_origin_statement_beats_country_name_elsewhere_on_label():
    r = compare_country("Jamaica", ["JAMAICA STYLE DARK RUM", "Product of Trinidad"])
    assert r.verdict == Verdict.MISMATCH and r.found == "Product of Trinidad"


def test_country_without_origin_statement():
    assert compare_country("Mexico", ["TEQUILA", "MEXICO"]).verdict == Verdict.MATCH
    assert compare_country("Mexico", ["Tequila from Mexico"]).verdict == Verdict.NEAR_MATCH


def test_country_statement_variants():
    r = compare_country("France", ["PRODUCT OF FRANCE IMPORTED BY ATLANTIC WINE & SPIRITS, MIAMI"])
    assert r.verdict == Verdict.MATCH and r.found == "PRODUCT OF FRANCE"
    assert compare_country("Saint Lucia", ["Product of St. Lucia"]).verdict == Verdict.MATCH
    assert compare_country("Netherlands", ["Product of The Netherlands"]).verdict == Verdict.MATCH
    assert compare_country("Scotland", ["Bottled in Bond", "Product of Scotland 700 mL"]).verdict == Verdict.MATCH
    assert compare_country("Scotland", ["Distilled in Scotland, bottled in Canada"]).verdict == Verdict.NEAR_MATCH


def test_trailing_period_does_not_break_exact_match():
    assert compare_text("brand_name", "RIVER BEND BREWING CO.", "RIVER BEND BREWING CO").verdict == Verdict.MATCH


def test_locate_prefers_case_exact_span_on_tie():
    lines = ["Distilled and bottled by Old Tom Distillery, Bardstown, KY", "OLD TOM DISTILLERY"]
    loc = locate_text("OLD TOM DISTILLERY", lines)
    assert loc.text == "OLD TOM DISTILLERY"


def test_volume_lost_decimal_point_is_near_match_not_match():
    r = compare_volume("1.5 L", "ALC. 14.5% BY VOL. L5L")
    assert r.verdict == Verdict.NEAR_MATCH and "decimal point" in r.note
    assert compare_volume("1.5 L", "15 L").verdict == Verdict.NEAR_MATCH
    assert compare_volume("750 mL", "75 mL").verdict == Verdict.MISMATCH


def test_locate_prefers_prominent_line_over_case_exact_mention():
    lines = ["HARBOR LANTERN RUM CO.", "Silver Rum", "Distilled and Bottled by Harbor Lantern Rum Co., Portland, OR"]
    loc = locate_text("Harbor Lantern Rum Co.", lines, preferred_line=0)
    assert loc.text == "HARBOR LANTERN RUM CO"  # edge punctuation is trimmed for display
    assert compare_text("brand_name", "Harbor Lantern Rum Co.", loc.text).verdict == Verdict.NEAR_MATCH


def test_mangled_litre_volume_asks_for_confirmation():
    r = compare_volume("1.5 L", "ALC. 15% BY VOL. LSL")
    assert r.verdict == Verdict.NEAR_MATCH and r.found == "15 L"


# --- the application value must be the whole phrase on the label --------------------------------
def test_value_inside_a_longer_phrase_is_never_a_silent_match():
    for key, app, line in [("class_type", "Rum", "SPICED RUM"), ("class_type", "Vodka", "GRAPEFRUIT FLAVORED VODKA"),
                           ("class_type", "Bourbon Whiskey", "STRAIGHT BOURBON WHISKEY"),
                           ("brand_name", "OLD TOM", "OLD TOM DISTILLERY")]:
        r = locate_and_compare(key, app, ["45% ALC./VOL.", line, "750 mL"])
        assert r.verdict == Verdict.NEAR_MATCH and r.found == line, (app, line, r)


def test_phrase_ending_at_punctuation_or_line_break_still_matches():
    assert locate_and_compare("class_type", "Silver Rum", ["SILVER RUM · AGED 2 YEARS"]).verdict == Verdict.MATCH
    assert locate_and_compare("class_type", "Kentucky Straight Bourbon Whiskey", LINES).verdict == Verdict.MATCH
    # The bottler statement is expected to carry a prefix ("Distilled and bottled by ...").
    r = locate_and_compare("bottler_name_address", "Old Tom Distillery, Bardstown, KY",
                           ["Bottled by Old Tom Distillery, Bardstown, KY"])
    assert r.verdict == Verdict.MATCH


def test_located_span_reports_its_own_lines():
    loc = locate_text("Product of Scotland", ["GLEN MORAR", "Single Malt Scotch Whisky", "Product of Scotland"])
    assert (loc.line_start, loc.line_end) == (2, 2)


def test_brand_found_only_in_small_print_needs_review():
    from app.models import Application
    from app.readers.base import OCRResult, OCRWord
    from app.readers.extract import extract_and_compare

    rows = [("HAZY DAZE", 90), ("India Pale Ale", 50), ("6.8% ALC./VOL. 12 FL. OZ.", 40), ("Brewed and Bottled by", 30),
            ("River Bend Brewing Co., Portland, OR 97209", 30)]
    words = [OCRWord(text=t, left=0, top=0, width=10, height=h, conf=90, line_index=i)
             for i, (line, h) in enumerate(rows) for t in line.split()]
    ocr = OCRResult(text="\n".join(r[0] for r in rows), lines=[r[0] for r in rows], words=words)
    app = Application(brand_name="River Bend Brewing Co.", class_type="India Pale Ale", alcohol_content="6.8%",
                      net_contents="12 fl oz")
    brand = extract_and_compare(app, ocr)[0]
    assert brand.verdict == Verdict.NEAR_MATCH and "small print" in brand.note and "HAZY DAZE" in brand.note
    app = app.model_copy(update={"brand_name": "HAZY DAZE"})
    assert extract_and_compare(app, ocr)[0].verdict == Verdict.MATCH


def test_volume_compound_and_thousands_compare_as_numbers():
    assert compare_volume("22 fl oz", "1 PINT 6 FL. OZ.").verdict == Verdict.MATCH
    assert compare_volume("1,000 mL", "1 L").verdict == Verdict.MATCH
    assert compare_volume("750 mL", "GROWN IN VOLCANIC SOIL 750 mL").verdict == Verdict.MATCH


def test_volume_reading_with_lost_decimal_beside_the_right_one_is_a_match():
    assert compare_volume("1.5 L", "1.5L\n15L").verdict == Verdict.MATCH


def test_volume_two_different_readings_ask_for_review():
    r = compare_volume("750 mL", "750 mL\n1 L")
    assert r.verdict == Verdict.NEAR_MATCH and "also reads" in r.note


def test_volume_misread_non_standard_size_is_near_match_but_real_wrong_size_is_mismatch():
    assert compare_volume("750 mL", "760ML").verdict == Verdict.NEAR_MATCH
    assert compare_volume("750 mL", "700 mL").verdict == Verdict.MISMATCH


def test_volume_us_customary_forms():
    assert compare_volume("16 fl oz", "DRINK FRESH ONE PINT 6.5% ALC/VOL").verdict == Verdict.MATCH
    assert compare_volume("568 mL", "1 PINT 3.2 FL. OZ. (568 mL)").verdict == Verdict.MATCH
    assert compare_volume("750 mL", "750 mL/25.4 OZ").verdict == Verdict.MATCH


def test_alcohol_proof_written_first_and_disagreeing_readings():
    assert compare_alcohol("51% Alc./Vol.", "PROOF 102\nALC/VOL 51%").verdict == Verdict.MATCH
    r = compare_alcohol("57.7% Alc./Vol.", "ALC 07.7% /VOL\nALC 57.7%/VOL")
    assert r.verdict == Verdict.NEAR_MATCH and "also reads" in r.note


def test_brand_letter_spaced_trademark_and_one_letter_misread():
    assert locate_and_compare("brand_name", "SOUTH COAST WINERY", ["S O U T H C O A S T W I N E R Y"]).verdict == Verdict.MATCH
    assert compare_text("brand_name", "TOBACCO BARN DISTILLERY", "TOBACCO BARN DISTILLERY®").verdict == Verdict.MATCH
    assert compare_text("brand_name", "CON PAZ", "CON FAZ").verdict == Verdict.NEAR_MATCH


def test_a_percentage_not_marked_as_alcohol_never_matches():
    # The alcohol statement was not read; the only percentages are a blend and a grain bill.
    r = compare_alcohol("13.5%", "MERLOT\nBlend: 13.5% Petit Verdot, 86.5% Merlot\n750 ml")
    assert r.verdict == Verdict.NEAR_MATCH and "not marked as alcohol" in r.note
    assert compare_alcohol("5%", "HAZY WHEAT ALE\nBrewed with 5% wheat malt\n12 FL OZ").verdict == Verdict.NEAR_MATCH
    r = compare_alcohol("14%", "CABERNET SAUVIGNON\n86% Cabernet Sauvignon, 14% Merlot")
    assert r.verdict in (Verdict.NEAR_MATCH, Verdict.NOT_FOUND)
    assert compare_alcohol("40%", "86% Cabernet Sauvignon").verdict == Verdict.NOT_FOUND


def test_alcoholic_and_volcanic_do_not_mark_a_percentage_as_alcohol():
    r = compare_alcohol("15%", "85% Syrah, 15% Grenache from volcanic soil\n12.5% ALC/VOL")
    assert r.verdict == Verdict.MISMATCH and "12.5" in r.note
    text = "15% Merlot. GOVERNMENT WARNING: (1) ... alcoholic beverages"
    assert compare_alcohol("15%", text).verdict != Verdict.MATCH


def test_a_vision_model_field_is_already_the_alcohol_statement():
    assert compare_alcohol("45%", "45%", statement=True).verdict == Verdict.MATCH


def test_metric_figures_must_agree_exactly_us_figures_may_be_rounded():
    assert compare_volume("750 mL", "753 mL").verdict != Verdict.MATCH
    assert compare_volume("750 mL", "750 mL / 25.4 FL OZ").verdict == Verdict.MATCH     # 751.2 mL, rounded
    assert compare_volume("12 fl oz", "12 FL OZ (355 mL)").verdict == Verdict.MATCH
    assert compare_volume("1.75 L", "1.75 L (59.2 FL OZ)").verdict == Verdict.MATCH


def test_a_second_reading_that_disagrees_is_shown_not_ignored():
    r = compare_volume("750 mL", "750 mL\n760 mL")
    assert r.verdict == Verdict.NEAR_MATCH and "760" in r.note
    assert compare_volume("1.5 L", "1.5 L\n15 L").verdict == Verdict.MATCH   # the same statement, point lost


# --- several readings of the same place --------------------------------------------------------
def _ocr_with_readings(readings: list[str]):
    """The same brand line read by several passes: every reading sits at the same place on the label."""
    from app.readers.base import OCRResult, OCRWord
    lines = list(readings) + ["Kentucky Straight Bourbon Whiskey", "45% ALC./VOL.", "750 mL"]
    words = []
    for i, line in enumerate(lines):
        top = 100 if i < len(readings) else 300 + 80 * i
        x = 100
        for t in line.split():
            words.append(OCRWord(text=t, left=x, top=top, width=40 * len(t), height=60, conf=90, line_index=i))
            x += 40 * len(t) + 30
    return OCRResult(text="\n".join(lines), lines=lines, words=words)


def test_a_reading_that_agrees_with_the_application_is_not_trusted_over_one_that_disagrees():
    from app.models import Application
    from app.readers.extract import extract_and_compare
    app = Application(brand_name="BARN BREW", class_type="Kentucky Straight Bourbon Whiskey",
                      alcohol_content="45%", net_contents="750 mL")
    # The label says BARK BREW; one pass misread it as the application's BARN BREW.
    brand = extract_and_compare(app, _ocr_with_readings(["BARK BREW", "BARN BREW"]))[0]
    assert brand.verdict == Verdict.NEAR_MATCH and "BARK BREW" in brand.note
    # Readings that differ only by OCR letter confusions or are cut short are noise, not disagreement.
    for other in ("BAm BREW", "BARN BRE"):
        app2 = app.model_copy(update={"brand_name": "BARN BREW"})
        assert extract_and_compare(app2, _ocr_with_readings(["BARN BREW", other]))[0].verdict == Verdict.MATCH, other


def test_rules_for_likely_misreads_never_excuse_a_real_difference():
    # 700 mL is a standard size: one digit from 750 mL, but a real difference, not a probable misread.
    assert compare_volume("750 mL", "700 mL").verdict == Verdict.MISMATCH
    assert compare_volume("750 mL", "760 mL").verdict == Verdict.NEAR_MATCH   # 760 is not a size anyone fills
    # A one-letter difference in the brand is a question for the agent, never a match.
    assert compare_text("brand_name", "BARN BREW", "BARK BREW").verdict == Verdict.NEAR_MATCH
    assert compare_text("brand_name", "CON PAZ", "CON FAZ").verdict == Verdict.NEAR_MATCH
