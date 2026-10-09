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
