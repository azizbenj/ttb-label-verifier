# Label Check

A prototype that verifies an alcohol beverage label image against the data in its label approval
application, built for a take-home exercise. It is an independent prototype, not affiliated with or
endorsed by any government agency. An agent types (or uploads a CSV of) the application values, adds
the label image(s), and gets a per-field verdict in a few seconds, plus a strict check of the
government health warning.

- **Live demo:** https://ttb-label-verifier-production-28e6.up.railway.app
- **Source:** https://github.com/azizbenj/ttb-label-verifier

| What you give it | What you get back |
|---|---|
| Brand name, class/type, alcohol content, net contents, bottler name & address, country of origin (imports) | One row per field: the application value, the text found on the label, and a verdict: **MATCH**, **NEAR MATCH** (agent decides), **MISMATCH** or **NOT FOUND**, with a one-line reason |
| The label image (PNG/JPG/TIFF/WEBP), or several (front, back, neck) for one application, or a CSV + 200-300 images for a batch | Government warning: present / exact wording (with a word diff) / heading in capitals / heading bold |
| | Timing for every label, and for batches a filterable table and a CSV export |

## Quick start (local)

Prerequisites: Python 3.12 and the Tesseract binary (`brew install tesseract`, `apt-get install tesseract-ocr tesseract-ocr-eng`, or `conda install -c conda-forge tesseract`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install --no-deps rapidocr-onnxruntime==1.4.4   # the second local reader; see "OCR choice" for why --no-deps
uvicorn app.main:app --reload --port 8000
# open http://localhost:8000
pytest -q
```

If Tesseract is not on your PATH, point the app at it with `TESSERACT_CMD=/path/to/tesseract`. Without the
`rapidocr-onnxruntime` package the app still runs; the RapidOCR escalation is then silently unavailable
(`/healthz` reports `"rapidocr": null`).

## Run with Docker

```bash
docker build -t label-check .
docker run --rm -p 8000:8000 label-check
```

The image is `python:3.12-slim` plus the Debian `tesseract-ocr` package (about 30 MB) and the RapidOCR
reader (ONNX Runtime, headless OpenCV and three bundled models, about 200 MB more; no apt package is
needed for them). It makes no network calls at runtime, so it can run inside the agency network. Docker
was not running on the machine this was written on, so the image has not been rebuilt since RapidOCR was
added: the wheels exist for Linux x86_64 / Python 3.12 and the Dockerfile imports them as a build check,
but do build it once before relying on it.

## Deploy (Railway)

```bash
railway login
railway init -n ttb-label-verifier
railway domain
./scripts/deploy.sh          # railway up, then waits until /healthz reports the new deployment id
```

Railway builds the Dockerfile on its side. The container runs as an unprivileged user, has a Docker
`HEALTHCHECK`, and must run as **one instance with one process**: batch jobs live in that process's memory.
`/healthz` returns 503 when the default reader cannot run (for example no Tesseract binary), so point the
platform's health check at it. Optional environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `OCR_ENGINE` | `tesseract` | Default reader (`tesseract`, `rapid` or `claude`) |
| `RAPID_ESCALATION` | `1` | Read a label once more with RapidOCR when Tesseract's passes leave a field missing or the warning short of PASS (`0` turns it off; with `OCR_ENGINE=rapid` the same switch lets Tesseract escalate RapidOCR) |
| `RAPID_INPUT` | `gray` | What RapidOCR reads: Tesseract's preprocessed grayscale, or `color` (the scaled, straightened original) |
| `RAPID_THREADS` / `RAPID_WORKERS` | `4` / `4` | ONNX Runtime threads per read; reads in flight at once |
| `LABEL_TIME_BUDGET_S` | `5` | Per-label time budget: the turned/contrast passes and the RapidOCR escalation run only while the time spent plus their expected cost fits (`0` = no budget) |
| `ANTHROPIC_API_KEY` | unset | When set, the UI shows a "Reader" toggle and the cloud reader can be selected per request |
| `CLAUDE_MODEL` | `claude-opus-5-5` | Model used by the cloud reader |
| `CLAUDE_TIMEOUT_S` | `30` | Per-attempt timeout of a cloud read (one retry) |
| `BATCH_WORKERS` | `min(4, CPUs)` | Parallel OCR workers for batch jobs |
| `DECISION_LOG` | unset | JSON-lines file that records every review decision with what the tool had concluded about the label, never the image (see "Learning from decisions") |
| `TESSERACT_PSM` | `4+11` | Tesseract page-segmentation mode(s); two modes joined with `+` are merged (see "OCR choice") |
| `TESSERACT_CMD` | unset | Path to the tesseract binary if it is not on PATH |

## How it works

```
label image ──> LabelReader ──> text + word boxes ──> locate each application value ──> verdict per field
                 (Tesseract,                           on the label (fuzzy search)
                  then RapidOCR                    └─> government warning checks ──────> PASS / REVIEW / FAIL
application ───── if needed, or Claude) ─────────────┘
```

1. **Read.** The image is flattened (EXIF rotation applied, transparent backgrounds put on white, 16-bit scans
   rescaled), converted to grayscale, scaled so the width is at least 1600 px (never above 10 megapixels), contrast-stretched and
   handed to Tesseract 5 (LSTM engine) twice: once in "single column" mode (PSM 4) and once in "sparse text" mode
   (PSM 11). No single mode reads every label (block modes drop oversized brand lines, sparse mode occasionally
   misses short centred lines), so lines the first pass did not produce are appended from the second. The two passes
   cost about 1 s together on Railway's shared CPU. We keep the word boxes and a binarized copy of the image for the
   bold check. OCR digit confusions are repaired in context (`75O` → `750`, `L.75 L` → `1.75 L`).
   Before reading, a scan tilted by 0.8° to 6° is straightened (the angle at which the rows of ink are most
   uneven); smaller tilts are left alone because Tesseract copes with them and resampling costs detail.
   When the first read leaves a field missing or different, or the warning is not a clean pass, the label is
   read again three more ways, in parallel: turned 90° each way (warnings and bottler lines printed sideways on
   cans and wine labels) and as a color-aware local-contrast image (light or colored text on colored panels).
   Only lines that look like real text are kept from these passes, and a pass that fails or takes longer than
   8 s is skipped rather than failing the label.
   **The figures get a second look.** A wrong number is the one error a compliance check cannot afford, and a
   whole-page pass reads a figure as one small thing among everything else: now and then it drops the decimal
   point or a digit ("1.5 L" read as "LSL" or "15L", "57.7%" as "07.7%"). So when the alcohol content or the net
   contents comes back missing, different or doubtful (two readings disagree, a decimal point was lost, a
   probable misread, proof against percentage), the lines that carry a figure are cut out of the page, scaled so
   the line is about 25 px tall and read again on their own in Tesseract's single-line mode, before the turned
   and contrast passes (`app/readers/numbers.py`). The size was measured rather than assumed: over every figure
   line of the batch set, the stress renderings and the real labels, Tesseract read the volume right on 338 of
   353 crops at 20-25 px against 322 for the page pass itself, and enlarging the crop, which one might expect to
   help, made it worse ("750 mL" became "790 mL" at 3×). Each second read is one more reading of that place on
   the label, with word boxes mapped back onto the page so the evidence crop still points at it; at most four
   crops of about 0.1 s each, each bounded by the same 8 s limit and skipped on failure. What a second read may
   change is a matching rule, described below.
   If that still leaves a field missing or different, or the warning short of a pass, the upright image is read
   once more with a second engine, **RapidOCR** (PP-OCRv4 text detection and recognition on ONNX Runtime, on the
   CPU, models bundled): it reads display and curved typefaces, light text over photographs and small print that
   Tesseract cannot. Its lines are appended as a further view of the label, with each line's box split across
   its words in proportion to their length, so the same matching, evidence pins and bold heuristic apply. This
   escalation costs about one more second on a laptop and runs only on labels that need it; clean labels never
   pay for it.
   **Time budget.** The turned/contrast passes and the RapidOCR escalation are optional, and each runs only
   while the time already spent on the label plus the pass's expected cost (a running average kept per
   process, seeded with 1.2 s and 1.5 s) fits inside `LABEL_TIME_BUDGET_S` (5 s). A pass that would not fit
   is skipped, the verdict as it stands is kept, and the result's reader line says so ("RapidOCR escalation
   skipped for time"). On a shared vCPU every pass costs two to three times what it costs here, so there the
   budget will skip the escalation on the hardest labels rather than overrun; raise the budget or set it to
   0 if accuracy matters more than the 5 seconds.
   Several images for one application (front, back, neck) are stacked into one before reading, because the
   warning is usually on the back.
2. **Find each field.** We know what we are looking for, so instead of parsing an arbitrary label we search for the
   span of consecutive label words (across up to three lines) that best matches each application value. This is far
   more robust than blind extraction: "Bottled by OLD TOM DISTILLERY, Bardstown" still yields "OLD TOM DISTILLERY".
   The other side of that coin is guarded: for the brand and the class/type, a value that is only part of a longer
   phrase ("Rum" on a label reading "SPICED RUM") or a brand found only in small print (inside "Bottled by ...") is
   a NEAR MATCH, never a silent MATCH.
   Alcohol content and net contents are picked up with patterns (`45% Alc./Vol.`, `90 Proof`, `750 mL`, `12 FL OZ`)
   and compared as numbers. If nothing resembles the brand name, the most prominent line on the label is shown as the
   mismatch so the agent sees what the label actually says.
3. **Compare** after normalization (next section) and classify with the thresholds in `app/config.py`.
4. **Government warning:** locate the statement, diff it word by word against 27 CFR 16.21, check the heading's
   capitalization from the OCR text, and estimate its boldness from the image.

Code map:

| Path | Role |
|---|---|
| `app/config.py` | Every threshold, the mandated warning text, field definitions, runtime settings |
| `app/images.py` | Safe image opening (size, pixel and decompression-bomb limits) and flattening of any image mode to plain pixels |
| `app/normalize.py` | Unicode/case/punctuation normalization, ABV-phrase unification, proof→ABV, volume→mL, OCR digit repair |
| `app/matching.py` | Guided fuzzy location of a value on the label, verdict rules, numeric comparison, country of origin |
| `app/warning.py` | Warning statement location, wording diff, heading caps, stroke-width bold heuristic |
| `app/readers/` | `base.py` (interface), `tesseract.py` (local OCR), `rapid.py` (RapidOCR: second local reader and the escalation), `claude_vision.py` (cloud), `extract.py` (rules over OCR) |
| `app/pipeline.py` | One label end to end with timings, the time budget for the optional passes, and the overall verdict |
| `app/evidence.py` | What the result page shows as evidence: crops of the label, unit conversions, misread explanations |
| `app/decisions.py` | The questions a flagged label asks the agent, one per field or warning check (review queue and single result) |
| `app/decision_log.py` | The decision log: one JSON line per review answer and undo when `DECISION_LOG` is set (never the image) |
| `app/batch.py` | CSV parsing, image/zip intake, thread-pool jobs, CSV export |
| `app/main.py` + `templates/` + `static/` | FastAPI routes, server-rendered UI (no build step, no CDN, no external assets); `review.html` is the review queue, `report.html` the printable reports |
| `design/` | The UI design spec: 23 boards, tokens, the implementation order |
| `scripts/generate_labels.py` | Synthetic label generator (Pillow) |
| `scripts/bench.py`, `scripts/stress_test.py` | Accuracy and timing on the synthetic sets; the same labels in other typefaces and degraded images |
| `scripts/fetch_registry_labels.py`, `scripts/real_labels.py`, `scripts/real_labels.csv` | Real approved labels: download, ground truth, field-level scoring |
| `scripts/decisions_report.py` | Pass rates per rule and a calibration CSV from the decision log |
| `data/samples/`, `data/batch/` | 15 sample labels (one per failure type) and a 250-label batch, each with its application CSV |
| `tests/` | Unit tests for every module plus end-to-end and API tests |

## OCR choice and trade-offs

The local default is **Tesseract 5**, with **RapidOCR** as a second local engine that reads a label only
when Tesseract's passes leave something unresolved. The readers sit behind a small interface
(`app/readers/base.py`); RapidOCR can also be the primary reader (`OCR_ENGINE=rapid`), and a **Claude
Vision** reader can be enabled with an environment variable for demos.

| | Tesseract 5 (default) | RapidOCR (escalation, shipped) | Claude Vision (opt-in) |
|---|---|---|---|
| What it is | LSTM OCR, Debian package | PaddleOCR's PP-OCRv4 detection + recognition models run on ONNX Runtime, CPU only; the `rapidocr-onnxruntime` wheel carries the models | Vision model over HTTPS |
| Runs inside a locked-down network | Yes, no downloads at runtime | Yes, no downloads at runtime | No, needs outbound HTTPS |
| Speed per label on this laptop | ~0.5-1.5 s (two passes) | ~0.7-1.3 s per read, 0.2 s to start; measured in "Measured results" | ~3-8 s (network + model) |
| Deployment weight | +30 MB apt package | +200 MB of wheels (onnxruntime 72 MB, headless OpenCV 119 MB, models 15 MB, measured installed on this Mac); no apt package | SDK only |
| Clean rendered labels (artwork, scans) | Very good | Very good | Excellent |
| Decorative/script fonts, curved text, light text on photos | Weak | Good | Best |
| Photos at an angle, glare | Poor (out of scope) | Fair | Good |
| Word boxes (needed for the bold heuristic and the pins) | Yes | Line boxes; split across the words by character count | Not needed: the model judges boldness itself |
| Spaces between words | Kept | Sometimes dropped (`GLENMORAR`, `IndiaPaleAle`), which is why it is not the primary reader | Kept |
| Data leaves the agency | No | No | Yes |

Why Tesseract first: the 5-second budget and the "cloud APIs may be blocked" constraint dominate. Tesseract is
fast on a small container, trivially deployable in an air-gapped network, and the application-guided matching
compensates for most of its OCR noise. Its weakness is stylized brand typography and text over photographs:
that is where RapidOCR pays, so it runs as the escalation, on the labels that need it, inside the time budget.
As a primary reader RapidOCR measured worse (see "Measured results"): it drops the space between words often
enough to fail the warning's word-for-word check, and the line boxes it returns make the bold heuristic
approximate. The cloud reader remains one flag away for demos.

Packaging note: the `rapidocr-onnxruntime` wheel declares a dependency on `opencv-python`, the GUI build of
OpenCV, whose Linux wheel needs `libGL` and pulls about 150 MB of Mesa from apt. The headless build is the same
`cv2` module without the window bindings, so `requirements.txt` pins `opencv-python-headless` together with
the rest of RapidOCR's dependencies and RapidOCR itself is installed with `pip install --no-deps` (Quick start
and the Dockerfile). One engine instance is shared by every thread (ONNX Runtime sessions are safe to run
concurrently; four concurrent reads returned exactly the sequential results), each read is bounded by
`RAPID_TIMEOUT_S` (10 s) and a read that fails or runs out of time is skipped, never fatal to the label.

Page-segmentation mode: the generated labels were benchmarked with PSM 3, 4, 6, 11 and merged pairs; see
"Measured results". `TESSERACT_PSM=4+11` is the default; a single mode (`TESSERACT_PSM=4`) halves the read time.

## Matching rules and thresholds

Normalization applied before any text comparison (`app/normalize.py`):

- Unicode NFKC; curly quotes and apostrophes → straight; dashes → `-`; non-breaking spaces → spaces
- Whitespace collapsed; case folded; punctuation dropped except inside numbers (`45.5`, `1,000`)
- `Alc./Vol.`, `Alc. by Vol.`, `Alcohol by Volume`, `ABV` → one token
- Proof → ABV (`90 Proof` = 45%); `mL`/`ml`/`ML`/`milliliters`, `L`/`liter(s)`, `cl`, `fl oz`/`oz`, pints, quarts
  and gallons → millilitres, including compound statements (`1 PINT 6 FL. OZ.` = 22 fl oz); `1,000 mL` is a
  thousands separator, `1,5 L` a decimal comma
- OCR digit repair inside numbers only: `75O mL` → `750 mL`, `l0` → `10`; a volume with some digits read as
  look-alike letters (`7S0 mL`, `L5L`) is repaired only when the label has no ordinary volume and never inside
  an address (`CHICAGO IL 60607`); letters alone (`LSL`, `LL`, `IL`) are never made into a figure, since "LLC"
  cut short would become 1 L (the second read of the line's crop reads the printed digits instead)

Verdicts (`app/matching.py`), all thresholds in `app/config.py`:

| Setting | Default | Meaning |
|---|---|---|
| strict equality | — | Identical after unicode/whitespace cleanup → **MATCH** |
| loose equality | — | Identical after full normalization → **MATCH**, except for fields in `case_review_fields` (`brand_name`) where a capitalization/punctuation-only difference is a **NEAR MATCH** ("STONE'S THROW" vs "Stone's Throw": trivial, but an agent should confirm) |
| `near_match` | 88 | rapidfuzz similarity (0-100) at or above this → **NEAR MATCH**, below → **MISMATCH** |
| `find_floor` | 60 | Best-matching span on the label scores below this → **NOT FOUND** |
| `locate_max_lines` | 3 | A value may wrap over up to this many OCR lines (addresses, long class names) |
| `whole_phrase_fields` | brand, class/type | A MATCH inside a longer phrase on its line ("Rum" in "SPICED RUM", no punctuation between) → **NEAR MATCH** showing the whole line |
| `brand_small_print_ratio` | 0.5 | A brand found only in text under half the height of the label's largest line → **NEAR MATCH** |
| `conflict_floor` | 70 | Another reading of the same place on the label (a second OCR pass or view) at least this similar to the value but saying something else turns a MATCH on the brand, class/type or bottler into a **NEAR MATCH** quoting both readings |
| `RAPID_TRUST_CONF` | 90 | Tesseract's word confidences and RapidOCR's line scores are not on one scale. For the disagreeing-readings rule only, a RapidOCR line scored at or above this (0-100) counts as certain: a disagreement it reads is never dismissed as the less sure reading, and a Tesseract reading of the same place never overrides a value it agrees with. Below it, its score competes as read |
| `abv_tolerance` | 0.05 pp | Alcohol content compared as numbers; any larger difference is a **MISMATCH** |
| `proof_tolerance` | 0.5 proof | A label whose proof disagrees with its own percentage by more than this ("45% (80 Proof)") → **NEAR MATCH** |
| `volume_tolerance_ml` | 0.5 mL | Net contents compared in millilitres (so `12 FL OZ` = `355 mL`). Between a US figure and a metric one 0.5% is allowed (`750 mL / 25.4 FL OZ`); within one system the figures must agree (`753 mL` is not `750 mL`). Same digits and unit but no decimal point on the label (`15 L` read for `1.5 L`) → NEAR MATCH, never a silent match |
| `warning_locate` | 75 | Similarity needed to recognise the "GOVERNMENT WARNING" line |
| `warning_near` | 97 | Warning wording at or above this (but not exact) → NEEDS REVIEW with a diff; below → FAIL |
| `meaning_conf` | 70 | A difference that changes the warning's meaning ("should drink", "can cause", "men") → **FAIL** instead of a look, only when every label word that makes it was read with at least this confidence (the "clear word" level of `unreadable_word_conf`) |
| `meaning_gap` | 1.2 | A missing "not" or modal fails only when its neighbours, on one line, are at most this many text heights apart: a wider gap is a word OCR dropped |
| `meaning_min_score` | 90 | Words are judged for meaning only when the statement as read is at least this similar to the required text |
| `meaning_missing` / `meaning_inserted` / `meaning_substitutes` | lists | The required words whose absence changes the statement (not, should, may); the words whose insertion does (never, only, sometimes, no...); and, per required word, the real words that change it when printed in its place (see "Government warning check") |
| `bold_ratio` | 1.30 | Heading stroke width ÷ body stroke width at or above this → "looks bold" (measured: bold headings 1.51-2.04, regular 1.03-1.14) |
| `bold_min_text_px` | 14 | Below this text height the stroke measurement is not attempted |
| `bold_failure_is_fail` | false | A "does not look bold" result asks for review instead of failing the label |
| `same_place_overlap` | 0.5 | Two OCR lines whose boxes overlap by this fraction of the smaller one's height and width are two readings of one printed line of the warning: only one is taken |
| `column_tolerance` / `column_min_lines` | 0.25 / 2 | A word whose box ends before the warning's left edge or starts after its right edge (allowing this fraction of the text height) is outside its column; the edge counts as a neighbouring column, and such words are set aside and quoted, only when they occur on at least this many of the statement's rows |
| `duplicate_sentence` | 90 | Lines outside the warning that match one of its sentences at least this well (partial similarity) are that sentence printed again → NEEDS REVIEW, quoting them; a complete second statement is not reported |
| `unreadable_min_words` / `unreadable_word_conf` | 8 / 70 | Fewer clear words than this, nothing read for any field (not even a disagreeing value) and no warning → "We couldn't read this label", no verdict |
| `number_reread_max_crops` | 4 | At most this many figure lines are cut out and read again when the alcohol content or net contents is missing, different or doubtful (lines that carry a figure outright first, then lines that only look as if they might: `L751`) |
| `number_reread_text_px` | 25 | Each crop is scaled so its line is this tall before the second read (measured: 20-25 px reads best; the page pass works at 40-60 px) |
| `number_reread_max_edits` | 1 | A page reading is set aside as a misread only when every closer read of its line agrees with the application and the page reading's digits are within this many edits of it (`15` for `1.5`, `790` for `750`, `077` for `577`); further away, both readings are named in a NEAR MATCH |

Real labels taught a few more rules, each covered by tests:

- **Every reading is considered.** A label often states the alcohol content or the volume more than once
  (`750 mL` on the front, `75 cl` on the back), and OCR may produce two readings of one statement. If any
  reading agrees with the application and another one disagrees, the result is a NEAR MATCH that lists both.
  A reading that only lost its decimal point is treated as the same statement, and US and metric figures within
  0.5% of each other (`16 FL OZ` and `473 mL`) agree. The same holds for the brand, class/type and bottler: a
  MATCH found in one reading while another reading of the same place says something else ("BARK BREW" printed,
  one pass reading "BARN BREW") is a NEAR MATCH. Readings that differ only by OCR's usual letter confusions
  (`rn`/`m`, `0`/`O`, `5`/`S`...) or are cut short do not count as disagreeing.
- **A closer read may correct a figure, never outvote the application.** The second read of a figure line
  (step 1) is one more reading, so the rule above still holds: a closer read that disagrees with the application
  makes a NEAR MATCH naming both, and a wrong number re-read as the same wrong number stays a MISMATCH ("A second
  read of that line says the same"). The one exception: when every closer read of a line agrees with the
  application and the page's reading of that line is within one edit of it (a dropped decimal point, one digit),
  the page reading is set aside as the misread it almost certainly is and the field is a MATCH whose note says
  what the page first read. A second read made for the net contents never reaches the alcohol content and the
  other way round, so a clean MATCH on one figure is never changed by a crop of the other, and no second read
  reaches the brand, class/type, bottler or country.
- **A percentage is only an alcohol content when the label says so.** It must sit next to `Alc./Vol.`, `ABV`,
  `alcohol by volume` or a proof figure (OCR slips such as `ALG/VGL` included; "alcoholic" in the warning and
  "volcanic" are not). A matching figure without such a word ("Blend: 13.5% Petit Verdot") is a NEAR MATCH,
  never a MATCH.
- **A likely misread is not a mismatch, and the application's value is not assumed.** A volume that is not a
  standard size and is one digit away from the application's is a NEAR MATCH. When the application's size is
  the only standard size one digit away (`1760 mL` for `1.75 L`) it is explained as a probable reading error;
  when other standard sizes are just as close (`760 mL` is one digit from 700, 710, 720 and 750 mL) the note
  names them and says the label could say any of them, and the question to the agent has no suggested answer.
- **The application's proof is checked too.** When the application states a proof with its percentage
  (`40% Alc./Vol. (80 Proof)`), a label proof that differs is a MISMATCH even though the percentage agrees; with
  no proof on the label, an application whose proof contradicts its own percentage is a NEAR MATCH.
- **Letter-spaced brands** (`B A R N  B R E W`) are matched with the spaces closed up; a single-letter
  difference in the brand is a NEAR MATCH, and trademark signs (® ™ ©) are ignored.
- **`U.S.` before a unit and words for numbers** (`ONE PINT`) are understood.

Blank optional fields on the application (bottler, country) are **SKIPPED**, not failed. Country of origin is
decided by the label's origin statements ("Product of X", "Imported from X", "Distilled in X"): only an identical
country name is a MATCH; a close spelling (an OCR slip, or Austria vs Australia) is a NEAR MATCH; a different
country, including one that merely contains the expected name ("Equatorial Guinea" for "Guinea"), is a MISMATCH
that shows what the label says. Without an origin statement, only the country name on a line of its own matches.

Overall label status: **FAIL** if any field is MISMATCH/NOT FOUND or the warning fails; **REVIEW** if any field is
NEAR MATCH or a warning check needs a look; otherwise **PASS**.

## Government warning check

Required text (27 CFR 16.21):

> GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink alcoholic beverages during
> pregnancy because of the risk of birth defects. (2) Consumption of alcoholic beverages impairs your ability to
> drive a car or operate machinery, and may cause health problems.

Four separate results are shown so the agent sees exactly what is wrong:

1. **Present**: the statement (or its body) was found on the label. The OCR lines are grouped by *frame*: the
   upright read and the contrast read of the same image share one frame (their boxes are in the same coordinates,
   so a heading read in one and a body read in the other still make one statement); each turned read is a frame of
   its own. Within a frame the lines are ordered top to bottom and the statement is grown line by line while that
   brings it closer to the required text, passing over up to four rows that belong to something else ("For sale
   only in Ohio"). The word boxes then decide what belongs to it:
   - *Two readings of one printed line.* The block pass and the sparse pass often read the same row differently
     (one across two columns, the other each column; one with a stray mark). Two lines whose boxes overlap by half
     their height and width are two readings of one row: the one closer to the required text is kept, never both
     (equal readings: the one OCR was surer of). A duplicate read can no longer add a sentence fragment twice.
   - *A neighbouring column.* The words of the provisional statement that align with the required text define the
     statement's own left and right extent across all its lines. A word whose box lies entirely beyond that extent
     is outside the column; when words lie beyond the same edge on at least two of the statement's rows, that
     edge borders another column (a keg collar's "ATTENTION-READ BEFORE TAPPING", a "12 FL OZ" to the left) and
     all words beyond it are set aside and quoted in the note without affecting the verdict. The gap between the
     columns plays no part (on a real keg collar it is a normal word gap); only the extent does. A single line with
     words sticking out is left alone and judged as wording: they may be words added to the statement. Stray
     marks with no letters beyond the extent ("~", "|", a "4" at the edge of a can) are dropped silently.
   - *No boxes* (the cloud reader, or a test): words at the start or end of a line that belong beside the
     statement are left out on wording alone, and the wording result then asks for a look, quoting them, because
     their position cannot tell a neighbouring column from words added to the statement.
   - A heading split over two lines ("GOVERNMENT" / "WARNING:") is one heading: both lines are the heading for the
     capitals check and both words are measured for the bold check.
2. **Wording**: compared word for word after normalization. Punctuation is ignored because OCR drops commas and
   periods unreliably, and a word the label hyphenates over a line break ("SUR-" / "GEON") is rejoined when the
   join spells a word of the required text. Any difference is listed as "required text says / label says". A
   difference with similarity ≥ 97 is flagged for **review** (it may be a misprint or an OCR error; the diff lets
   the agent decide in a second); anything larger (missing sentence, paraphrase) **fails**. The statement must
   carry each sentence once, in order: sentence (1) printed twice inside the statement, or (2) before (1), fails;
   a sentence printed again elsewhere on the label (sentence (1) on the front and again on the back, without (2))
   asks for a look and quotes the second copy, while a complete second statement is not a problem. When a
   discarded reading of a statement line disagrees on a word that is not in the required text and OCR was at
   least as sure of it, the wording asks for a look and names both readings: a misread that happens to agree
   with the required text must not hide a misprint the other reading saw.

   **A difference that changes what the warning says fails**, even when it is one word. Each difference is
   classified (`meaning_changes` in `app/warning.py`, word lists and thresholds in `app/config.py`):
   - *"not" or a modal left out*: "should drink" for "should not drink", "women not drink" for "women should not
     drink" (`meaning_missing`: not, should, may);
   - *a modal replaced*: "can cause" or "must not" (`meaning_substitutes` for should and may: can, could,
     might, must, will, would, shall, need, do, does, cannot, and should/may for each other);
   - *a negation, quantifier or hedge inserted*: "may never cause", "only", "sometimes", "no"
     (`meaning_inserted`);
   - *a key word replaced by another real word*: "men", "woman" or "people" for "women", "improves" for
     "impairs", "benefits" for "problems", "Attorney" for "Surgeon", "vehicle" for "car" (the per-word lists in
     `meaning_substitutes`).

   The real-word test is those lists: a word counts only when it is listed for the required word it replaces,
   so a garble never does. A word that becomes the required one once OCR's usual confusions are undone (the
   same map the field matcher uses: `rn`/`m`, `cl`/`d`, `vv`/`w`, `0`/`o`, `1`/`l`/`i`, `5`/`s`, `8`/`b`, so
   "wornen", "rnay", "n0t"), and a word split or run together ("no t", "shouldnot"), are misreads whatever the
   lists say. A change of form that leaves the meaning ("impair", "defect", "problem") is left out of the lists
   on purpose: it breaks the fixed text, so it asks for a look, but it does not reverse the warning. And the
   reading must be sure: every word of the label that makes the change must have been read with confidence of
   at least `meaning_conf` (70, the confidence this project already calls a clear word); for a missing word,
   both neighbours must be read that surely, on the same line, no further apart than `meaning_gap` (1.2) times
   the text height, because a gap wide enough to hold "not" is a word OCR dropped, not one the label left out.
   Across a line break the gap counts as closed only in justified type: the first line runs to the statement's
   right edge (as two other lines do) and the next starts at its left edge; ragged or centred lines cannot tell
   a word dropped at a line's end from one left out, and ask for a look. Only the meaning words themselves may
   be missing ("not", "should not"): a longer run missing is a line OCR did not read.
   An inserted word counts only between two words of its own line (one at a line's end may belong to a
   neighbouring column: "FOR SALE ONLY IN OHIO"). Words are judged where the label's words line up with the
   required ones (as many words, or one more or one fewer, the rest the same or misreads of them), and only
   when the statement as read is at least `meaning_min_score` (90) similar to the required text: a stretch
   that reads differently in some other way is other text read into the statement far more often than a
   rewritten warning, and it fails on similarity anyway. Without word boxes (the cloud reader) there is no
   confidence to check, so a meaning-changing difference asks for a look like any other; the same holds for a
   statement read by RapidOCR, whose score is the whole line's and whose word boxes are shares of the line's box.

   The note names the change in plain words ("The label says 'should drink' where the warning requires 'should
   not drink': this changes its meaning."), and the review question names the word at stake ("Does the label
   print "not" in "should not drink"?"). "Yes" is still the first button and still means the label is fine
   (the tool misread it), but it is not offered as the suggested answer. Every other difference keeps the rule
   above: a look at 97 or more, a failure below.
3. **Heading in capitals**: the OCR text of the heading must read `GOVERNMENT WARNING:`; title case fails, a missing
   colon asks for review, and capitals with a letter OCR could not read cleanly (`WARNlNG`, or `ERNMENT` cut at the
   image edge) ask for review. A different word that was read clearly (`HEALTH WARNING:`, `GOVT WARNING:`) fails.
4. **Heading bold (heuristic)**: from the word boxes, we take the heading words (read cleanly, glued to a
   neighbour such as `WARNING:(1)`, or misread the way the capitals check tolerates, `ERNMENT WARMING:`) and the
   body words of the statement, measured in the ink of the view that read the statement, and estimate each
   group's mean stroke width as 2 × ink area ÷ ink perimeter on the binarized image (for a stroke
   of width w and length L the area is wL and the perimeter about 2L, so the estimate does not depend on stroke
   orientation or letter case), normalized by cap height. If the heading's strokes are at least `bold_ratio`
   (1.30×) thicker than the body's it "looks bold". Ink polarity is decided per word, so a light-on-dark warning
   panel is measured correctly. Calibrated with `scripts/calibrate_bold.py`, which renders the 243 warning
   statements of the batch set twice, with a bold and a regular heading (both font families, with and without
   blur): bold headings measure 1.51-2.04, regular ones 1.03-1.14. It is still a heuristic that assumes the body
   is set in regular weight at a similar size; 27 CFR 16.22(a)(2) also forbids a bold body, and an all-bold
   statement measures about 1.0, so it asks for a look. Doubtful results ask for a look rather than failing the
   label. With the cloud reader the model answers the bold question directly.

## Batch mode

Upload a CSV and the label images (multi-select, or a zip, or both). Images are matched to rows by file name
(case-insensitive; the extension may be omitted in the CSV). Columns are matched case-insensitively and common
aliases are accepted (`brand`, `abv`, `volume`, `bottler`, `country`...):

| Column | Required | Example |
|---|---|---|
| `image` | yes | `label_0001.png` |
| `brand_name` | yes | `OLD TOM DISTILLERY` |
| `class_type` | yes | `Kentucky Straight Bourbon Whiskey` |
| `alcohol_content` | yes | `45% Alc./Vol.`, `90 Proof`, `45` |
| `net_contents` | yes | `750 mL`, `1.75 L`, `12 fl oz` |
| `bottler_name_address` | no | `Old Tom Distillery, Bardstown, Kentucky 40004` |
| `country_of_origin` | no (imports) | `Scotland` |
| `application_id` | no | `APP-0001` |

Several images for one application go in one cell, separated by `;` or `|` (`front.png; back.png`, up to six);
they are stacked and read as one label. Comma-, semicolon- and tab-separated files are accepted. A template is downloadable from the UI. Rows with blank
required values or an alcohol content / net contents that cannot be read, rows without an image, images without a
row, and file names shared by several different images (`lot1/label.png` and `lot2/label.png`: never guessed) are
reported in plain language and the rest of the batch still runs. Jobs run in a thread pool (`BATCH_WORKERS`) and
the page polls every second; results have Pass/Review/Fail filters, a search box, click-to-expand details, and a
CSV export with every field's verdict, found text, note, the four warning results and timings (cells that would
start a spreadsheet formula are prefixed with an apostrophe).

Limits: 500 labels and 1 GB of images per batch, 20 MB and 40 megapixels per image, zip-bomb guards. Jobs live in
memory; finished ones are kept for 24 hours (at most the 50 most recent) and disappear on restart. A finished
batch has a link of its own (`/batch/<id>`) that opens the page on the batch tab.

### Reviewing and exporting

- **Triage first.** The results table lists REVIEW rows first, then FAIL, ERROR and PASS; filters (`1` `2` `3`
  `4` `0`), search, sort, `J`/`K` to move, `X` to select, `Enter` for the full comparison. A row whose image
  could not be read is an ERROR (grey), never a FAIL: nothing is wrong with the label.
- **Review queue** (`R`, or "Review the N"): one label that needs a look at a time, and every question it
  raises in turn, each with its region outlined on the label: "Is this the same brand name?", "Does the label
  say 'should'?", "Is GOVERNMENT WARNING: printed in bold?". Every field or warning check that is not a clean
  match asks one, a MISMATCH or NOT FOUND included. `Y` means the label is fine (a reading error on our side),
  `N` that the label is wrong, `S` skips; an answer moves on to the label's next open question, and to the next
  label once every question on it has an answer. One answer settles one question, never the whole label: the
  label's decision is **pass** only when every question was answered yes, **fail** when any was answered no,
  **partial** while some are open. "Yes" is the suggested (green) button only when the evidence leans to it;
  when the tool read something different from the application (a MISMATCH, a warning wording difference, a
  misread one digit from several sizes) both answers look alike, so the application's value is never offered
  as the likely truth.
- **Fails to check** ("Check the N fails"): the same queue over the FAIL labels, so a misread ("1.75 L" read as
  "1L") can be recorded before anything is sent back. Answers never change a verdict; they are kept with the
  batch and go into the export as `decision` (the label's) and `answers` (one per question:
  `brand_name: yes; net_contents: open`) columns. The same prompts appear under each flagged row or warning
  check in the single result and in the full comparison, and print with the report. Both queues work without
  JavaScript: the answers are forms.
- **Export dialog**: which labels (all, shown now with the current filter and search, or the selected rows),
  what to include (per-field verdicts and reasons, the warning checks and diff, decisions, all text read,
  timings), and the format: CSV (UTF-8 with a BOM so Excel keeps accents) or printable reports, one page per
  label for the case file with a reviewer decision block. Quick exports: ready to approve (the passes plus the
  labels the agent cleared, `status=PASS,CLEARED`) and fails only. The file is named after the batch, the scope
  and the date.
- **Print**: "Print report" on any result prints a Letter page: the application, the verdict, the label with
  its numbered regions, the table, the warning checks, and a reviewer decision block (approve / return /
  second look, signature, notes).

#### Learning from decisions

Set `DECISION_LOG=/path/decisions.jsonl` and every answer from the review queue (and every undo) is appended
to that file as one JSON object per line, with what the tool had concluded about the label: the job, the
batch's source name, the application id and image name, the answer (`pass`: the label is fine and our flag
was a false alarm, `fail`: the label is wrong, `skip`, `clear`: undone), the question it answers (`key`), the
label's decision after it (`label_decision`), the overall verdict, every question the label raised (key, what,
verdict, question, the two readings) with the answered one marked,
each field's verdict / expected / found / note, the four warning statuses and the wording score, the read
confidence, the words read, the reader and the timings. Never the image, its preview or the text read from
it. Writes are append-only behind a lock (the server's workers share it; keep one process per log file), and
a write that fails (a missing directory, a full disk) is a warning in the server log, never an error for the
agent. Off when the variable is unset.

`python scripts/decisions_report.py decisions.jsonl` keeps the last answer per job, application and question
(an undone one drops out) and prints, per question (`brand_name`, `net_contents`, `warning_wording`,
`warning_bold`...), how many labels raised it, how many answers were about it, the pass / fail / skip
answers and the pass rate: pass ÷ (pass + fail), the share of that rule's flags the agents found to be false
alarms. A rule with a high pass rate costs agents time for nothing; one with a low pass rate catches real
problems. Then the ten most frequent notes behind `pass` answers and the ten behind `fail`. `--csv
calibration.csv` writes one row per label decided pass or fail in the layout of `scripts/real_labels.csv`
(the application id as `ttbid`, the application values, `<field>=review` in `expect` when the agent confirmed
a NEAR MATCH, nothing after a `fail`, the decision and the question in `notes`), ready for
`scripts/real_labels.py` once the images are in `data/real/`. The script needs only the standard library, so
it runs on a copy of the log anywhere.

### Error states

Errors are inline, where the result would have been, and say what happened in one line and what to do in the
next. Missing fields are checked in the browser before anything is sent, each input is marked and the
summary links to it; without JavaScript the server returns the same page with the typed values kept. The drop
zone itself becomes the error for a missing, unreadable or oversized image, with the file name quoted. A CSV
with the wrong headers gets a table of expected header vs the near-miss header in the file. An image that
opens but yields almost no text ("fewer than 8 confident words and nothing read for any field") gets a grey,
verdict-free card with what usually fixes it, never a FAIL; a label where a value was read and disagrees is a
normal FAIL however few words it has. The page works at 1280, 768 and 390 px without horizontal scrolling; on a
phone the batch table becomes a list.

Every screen also works without JavaScript: both tabs show one under the other, the forms post and come back as
whole pages, a running batch's page (`/batch/<id>`) reloads itself every 3 seconds until it is done, each row
links to its full comparison, and the review queue's answers are forms.

## Test data

`scripts/generate_labels.py` renders flat, clean labels with Pillow (DejaVu Sans/Serif, three layouts, four palettes).

```bash
python scripts/generate_labels.py samples               # data/samples: 15 labels + samples.csv
python scripts/generate_labels.py batch --count 250     # data/batch: 250 labels + applications.csv
```

Sample set: four clean bases (the README's OLD TOM DISTILLERY bourbon, a wine, a beer, an imported Scotch with
"Product of Scotland") and one variant per failure type: wrong ABV, missing warning, warning heading not in capitals,
warning heading not bold, brand capitalization differs, wrong net contents, different brand, one word of the warning
changed, warning cut short, net contents missing, wrong country. Every row carries the application values plus the
expected outcome, so the same files drive the end-to-end tests and the "Try a sample" menu.

Batch set: 250 random labels across spirits, wine, beer and imports: 202 clean and 48 with a planted defect (wrong
ABV 11, heading not in capitals 10, missing warning 7, brand capitalization 6, wrong net contents 5, missing net
contents 4, altered warning 3, heading not bold 2). Wrong brand and wrong country appear only in the sample set.
`expected_overall` is in the CSV so accuracy can be measured.

### Real approved labels

Synthetic labels only prove the tool reads its own generator. To test it on what agents actually see,
`scripts/real_labels.py` runs it on 20 approved labels taken from the public registry of approved label
applications: 6 domestic spirits, 4 imports, 5 wines and 5 beers, chosen to be hard (light text over photos,
sideways warnings, curved display type, a handwritten keg collar, colored panels). The images belong to the
brand owners, so they are not in the repository; `scripts/fetch_registry_labels.py` downloads them into the
gitignored `data/real/`. `scripts/real_labels.csv` records what each label really says, so every field should come
back MATCH unless its `expect` column says otherwise (4 fields are legitimate NEAR MATCHes, 3 are not on the
label). `--defects` also checks every label against a wrong alcohol content and a wrong net contents, which must
never come back MATCH.

```bash
python scripts/fetch_registry_labels.py $(cut -d, -f1 scripts/real_labels.csv | tail -n +2)
python scripts/real_labels.py --stitch       # each record's images into one file
python scripts/real_labels.py --defects
```

`scripts/stress_test.py` renders the 15 sample labels again in eleven other typefaces, six display faces for the
brand, and degraded images (JPEG quality 35, a half and a third of the resolution, tilted 1.5° and 4°, blur,
sensor noise). It only uses fonts already installed on the machine and writes nothing to the repository.

## Tests

```bash
pytest -q
```

- `test_normalize.py`: quotes, case, ABV phrases, proof→ABV, mL/L/cl/fl oz, OCR digit repair
- `test_matching.py`: every verdict boundary, the brand-name case rule, numeric tolerances, multi-line addresses, country of origin
- `test_warning.py`: exact/altered/truncated wording with diffs, heading capitalization, the bold heuristic on rendered bold vs regular text in both font families and two sizes
- `test_e2e.py`: every sample label through the real Tesseract pipeline must produce its expected outcome in under 5 s
- `test_api.py`: the HTTP surface, friendly errors (missing fields, bad values, non-image files, decompression bombs), a small batch with a missing image and a stray file, CSV export
- `test_batch_flow.py`: the single and batch HTTP flows with a fake reader, so templates and job handling are covered without Tesseract; CSV dialects, unreadable rows, same-named images, encrypted zips, export escaping, job retention
- `test_http.py`: concurrent checks do not queue behind each other, reader failures map to readable 502/500 errors, upload size guard, `/healthz` 503
- `test_reader.py`: transparent / 16-bit / palette / CMYK images, size and pixel limits, the two-pass merge bookkeeping, Tesseract timeouts, tilt estimation, straightened previews
- `test_boxes.py`: highlight boxes for each field, including text read from a turned view mapped back onto the upright image
- `test_formats.py`: JPG, TIFF, WEBP and BMP uploads and zips of them, single and batch
- `test_multi_image.py`: several images per application, in the form and in batch rows
- `test_errors.py`: every error state (missing fields with and without JavaScript, bad or oversized image, CSV headers, batch limits, reader failure with retry) and the unreadable-image card and ERROR row
- `test_decisions.py`: review prompts (Yes always means the label is fine), decisions kept with a batch, the review queue (with and without JavaScript), export scopes and column groups, printable reports, the shared batch link
- `test_decision_log.py`: the decision log (one compact line per answer and undo with the documented keys, never the image, concurrent writes, an unwritable path never fails the request) and the report script (the last decision wins, pass rates per question, the notes behind the answers, the calibration CSV in the `real_labels.csv` layout)
- `test_nojs.py`: the batch flow as a browser without JavaScript runs it: post, redirect, self-reloading progress page, row links, whole-page errors
- `test_claude_reader.py`: the cloud reader's mapping, error handling, refusal fallbacks and client limits, with the SDK mocked
- `test_rapid_reader.py`: RapidOCR on the Old Tom sample (lines, word boxes inside the image, confidences, the engine string; skipped when the package is missing), the line-to-word split and input padding, the escalation appending a view with a fake engine, the cross-engine confidence rule, time-outs, `get_reader("rapid")` and `/healthz`
- `test_budget.py`: the per-label time budget with a fake clock: passes that fit run, passes that do not are skipped and named in the reader string, and the process learns what its passes cost

OCR-dependent tests skip automatically when the Tesseract binary is absent. `.github/workflows/ci.yml` runs the whole
suite on Ubuntu with Tesseract installed, then benchmarks both sets and fails the build if a sample misses its
expected verdict, if batch accuracy drops below 95%, or if any label with a planted defect comes back PASS.
`scripts/bench.py` reports accuracy, missed defects and timing for the sample or batch set, locally (`-j N` to run
labels in parallel) or against a deployed URL (`--url`).

## Measured results

All numbers from `scripts/bench.py` against the Railway deployment (shared vCPU, one container), so they include
HTTP overhead; local runs on a laptop are faster. "Expected verdict" means the overall PASS / REVIEW / FAIL the
generator recorded for that label.

**Sample set (15 labels, one per failure type), reader `4+11`:** 15/15 expected verdicts, median 0.8 s per label,
p95 1.0 s. The README's OLD TOM DISTILLERY label passes every field and the warning in under a second.

**Batch set (250 labels: 202 clean, 48 with a planted defect):**

| Tesseract mode | Expected verdict | Median | p95 |
|---|---|---|---|
| `6` (uniform block) | 12/15 on the samples: drops oversized brand lines | 0.44 s | 0.46 s |
| `3` (auto) | 214/250 | 0.49 s | 0.71 s |
| `11` (sparse) | 230/250 | 0.57 s | 0.65 s |
| `4` (single column) | 229/250 | 0.57 s | 0.66 s |
| `4+11` merged (default) | **242/250 (97%)** | 0.81 s | 0.91 s |
| `3+11` merged | 241/250 | 1.07 s | 1.24 s |

(The single-mode and `3+11` rows were measured one build earlier, before the final bold-heuristic and digit-repair
changes; the default row is the final build. Timings vary with Railway's load: the same default reader measured
1.09 s median on a busier run.)

Of the 8 misses with the default reader, 7 are conservative: the tool asked for a look (REVIEW) on a label the
generator marked clean, because Tesseract dropped the decimal point in a litre volume ("1.5 L" read as "15L") or
misread a capital in the brand line and matched the title-case bottler mention instead. The remaining one is a clean
label failed by a genuine OCR error ("750 mL" read as "790 mL", reported as a MISMATCH that shows the misread text,
so an agent resolves it from the image in seconds). More importantly, no planted defect was reported as a PASS: every wrong
ABV, wrong or missing volume, missing or altered warning and non-capital or non-bold heading was caught, and every
brand-capitalization case was flagged for review (the wrong-brand and wrong-country cases are in the sample set,
where all 15 labels get their expected verdict).

Throughput: the 250-label sample batch completes in about 90 s through the UI with four workers, i.e. 0.35 s per
label of wall-clock time.

**After the code review (October 2026), measured locally** (Tesseract 5.5.3, 11-core Mac, `scripts/bench.py`, one
label at a time; the Railway figures for the current build are in the next table):

| Set | Expected verdict | Planted defects reported as PASS | Median | p95 |
|---|---|---|---|---|
| Samples, before the review | 15/15 | 0 | 1.30 s | 1.35 s |
| Samples, after | 15/15 | 0 | 0.67 s | 0.75 s |
| Batch, before the review | 243/250 | 0 | 1.24 s | 1.33 s |
| Batch, after | 243/250 (same seven labels) | 0 | 0.68 s | 0.74 s |

The review's fixes (country of origin, whole-phrase brand and class/type, label proof consistency, volume parsing,
warning heading, image intake) change no verdict on these two sets: their labels never exercise the cases that were
wrong, which is why each fix comes with its own tests. The speed-up comes from running Tesseract with one OpenMP
thread per process (`OMP_THREAD_LIMIT=1`); with four batch workers the 250-label batch takes 44 s instead of 157 s
on the same machine.

**On Railway (`scripts/bench.py --url ...`, one label at a time, so the figures include HTTP overhead and the
extra turned and contrast passes where they run).** Two deploys measured a day apart:

| Build | Set | Expected verdict | Planted defects reported as PASS | Median | p95 | Max |
|---|---|---|---|---|---|---|
| `4c88e85` (before the second review), 8 Oct 2026 | Samples (15) | 15/15 | 0 | 1.00 s | 1.13 s | 1.13 s |
| | Batch (250) | 242/250 | 0 | 0.45 s | 1.08 s | 1.27 s |
| `5e3e150` (after the second review), 9 Oct 2026 | Samples (15) | 15/15 | 0 | 2.5 s | 3.2 s | 3.6 s |
| | Batch (250) | 242/250 | 0 | 1.02 s | 2.8 s | 4.6 s |

The verdicts are identical between the two builds (the same eight conservative misses: a clean label asked for a
look because a litre volume was read without its decimal point, or the brand was matched in title case). The
times are not: on 9 October every label took about twice as long, including the four clean samples, whose two
Tesseract passes took 1.0 s instead of 0.45 s and which run none of the review's new logic. The review's changes
add no OCR pass (the extra passes run on the same 10 of 15 samples as before) and measured faster locally, so
the difference is the shared vCPU on that day; the README's earlier note that the same build measured 0.81 s and
1.09 s on two runs is the same effect. What the slow day does show: a label that needs the extra passes costs
2.4-3.5 s there, and the slowest batch label reached 4.6 s, close to the 5-second budget. If that recurs, the
options are a container with a guaranteed CPU, or fewer extra passes (`TESSERACT_PARALLEL_PASSES`, and which
views `extend` tries), which is a product trade-off rather than a bug. Compared with the 0.81 s median measured
on Railway before the first code review, `OMP_THREAD_LIMIT=1` (one OpenMP thread per Tesseract process) halved
the per-label time on the quiet day, so that setting stays.

**Real approved labels (October 2026, local, `scripts/real_labels.py --defects`).** The 20 labels carry 104 filled
application fields:

| | Before the real-label work | Now |
|---|---|---|
| Fields with the expected verdict | 54/104 | 57/104 |
| Flagged for review (NEAR MATCH where MATCH was expected) | not measured | 29 |
| Accepted without the look the ground truth expects (MATCH where NEAR MATCH was expected) | not measured | 2 |
| False alarms: MISMATCH or NOT FOUND for text that is on the label | 32 | 16 |
| Government warning: pass / review / fail (all 20 carry it) | 13 fail | 5 / 11 / 4, then 8 / 9 / 3 with the layout rules below |
| Planted wrong alcohol content or net contents reported as MATCH | 0 of 40 | 0 of 40 |
| Time per label: median / max | | 2.3 s / 4.3 s |

(Until the second code review the scorer counted a MATCH where the ground truth expects a look as "expected",
which read 59/104. The two such fields are the class "GIN" printed under "THE SPIRIT OF TENNESSEE" and the country
"MEXICO" read from "HECHO EN MEXICO"; both are defensible readings, so they are reported rather than changed.)

Read this honestly: no real label passes untouched. 8 come back REVIEW and 12 FAIL, so on real artwork the tool is a
fast first pass that points the agent at what to look at, not an unattended approver. What it does not do is let a
wrong value through: all 40 planted defects were caught. The remaining false alarms come from brand names in
display or curved typefaces, light text over photographs, a handwritten keg collar, and single-digit misreads
(`57.7%` read as `07.7%`); each shows the misread text and its place on the label, so the agent settles it from the
image. Real labels take longer than the synthetic ones (median 2.4 s against under 1 s) because most of them need the
extra turned and contrast passes; the slowest was 4.3 s, inside the 5-second budget on a laptop. Re-measure on
Railway before relying on that there.

**Government warning, layout rules (October 2026).** Before the word-box rules in "Government warning check" the
warning came back 5 pass / 11 review / 4 fail on the 20 real labels, every one of which carries a correct warning;
after them 8 / 9 / 3, with no change to the field verdicts (the old and the new warning code were scored back to
back through the same scorer and images and gave the same 56/104, 27 flagged, 19 false alarms, 2 accepted; the
scorer's own inputs had moved since the table above), the sample set (15/15), the batch set (243/250, the same
seven misses, 0 planted defects passed), the stress table or the 40 planted real-label defects. The warning check
can only reach the fields through the pipeline's decision to run the extra OCR passes, and on these labels that
decision changes for one label whose field verdicts are the same either way. What changed, label by label:

- Three wine and spirits labels whose two OCR passes had read the same row differently (one with a stray mark from
  the artwork, one clean) went from REVIEW to PASS: the duplicate readings are no longer both assembled, so the
  wording is exact (a condensed wine warning, a rosé, a letter-spaced whiskey where the wording is now exact but the
  type is too small to measure the bold heading, so it stays REVIEW for that reason).
- A beer can whose last line started with a stray "4" outside the statement's left edge: REVIEW to PASS.
- A gin label hyphenating "SUR-GEON" and "BEVER-AGES" over line breaks: wording REVIEW to PASS (8 px type, so the
  bold check still asks for a look).
- The handwritten keg collar: FAIL to REVIEW. The neighbouring column ("ATTENTION-READ BEFORE TAPPING ... THIS KEG
  MAY RUPTURE ...") is now set aside by its boxes and quoted; what remains is a genuine reading difference on the
  first line ("(1)" lost, or "TO" read as "10", depending on which of two readings is kept), which the agent must
  look at.
- A tequila label went from 3 reading differences to 1 (same REVIEW). The rest are unchanged: two labels whose
  warning the OCR did not read at all (light text over a photo, a slanted panel), one whose 6 px body is unreadable,
  three REVIEWs for "to" read as "10"/"T0" or a one-word last line that the turned-view reader drops, one cropped at
  the image edge, and three honest bold measurements (1.05, 1.26 and 1.27 against the 1.30 threshold; one of them
  really is set in the same weight as its body).

**After the second code review (October 2026, local, Tesseract 5.5.3):** samples 15/15, median 0.87 s; batch
243/250 with the same seven misses, 0 planted defects reported as PASS; real labels as in the table above; stress
test identical to the table below. None of the round-two fixes changed a verdict on these sets: the cases they
close (a blend percentage taken for the ABV, a misread that agrees with the application, a clearly read wrong
heading word, an unreadable card hiding a mismatch, 753 mL against 750 mL) do not occur in them, which is why each
fix has its own tests.

**With the second read of the figures (October 2026, local, Tesseract 5.5.3).** Before is `main` at `c55fa57`
(with the warning layout rules), after is the same code plus the second read and the look-alike repair rule,
scored back to back on a quiet machine with the same scripts and images:

```bash
python scripts/bench.py --fail-under 1.0
python scripts/bench.py --set batch -j 4 --fail-under 0.95
python scripts/real_labels.py --defects
python scripts/real_labels.py --csv scripts/calibration_labels.csv --defects
python scripts/stress_test.py
```

| Set | Before | After |
|---|---|---|
| Samples (15) | 15/15, 0 planted defects passed, median 0.94 s | 15/15, 0, median 0.93 s |
| Batch (250) | 243/250, 0 planted defects passed | **248/250**, 0 |
| Real labels (20, 104 fields) | 56/104 as expected, 27 flagged, 19 false alarms; warning 8/9/3; 40/40 planted defects caught; median 2.65 s, max 5.0 s | **58/104**, 26 flagged, 18 false alarms; warning 8/9/3; 40/40 caught; median 2.2 s, max 4.3 s |
| Calibration labels (166, 913 fields) | 518/913, 200 flagged, 175 false alarms; **2 of 330 planted defects reported as MATCH** | **523/913**, 195 flagged, 175 false alarms; **0 of 330** |
| Stress test | table below, "before" | table below, "after"; no row worse |

Batch: five of the seven misses were a litre volume read without its decimal point ("LSL", "L5L", "L75L"); the
crop read "1.5L" / "1.75L" and the five labels (0033, 0042, 0107, 0168, 0188) now PASS, with "A closer read of
that line confirms it: the page was first read as '15L'" or "The statement was read on a closer look at the
line" in the note. The two left are a brand matched in title case (0047, not a figure) and label 0152, whose
"1.75 L" the page read as "L751" at confidence 0 and the crop as "17514": no figure either way, so the net
contents stays NOT FOUND (the label is a brand-capitalization case expected to come back REVIEW; it comes back
FAIL, a stricter verdict, never a PASS).

Real labels, every field that changed: 26162001000137 net contents NEAR MATCH to MATCH (the page read "730 ML",
the crop "750 ML"); 11145001000540 net contents MISMATCH to NOT FOUND, as the ground truth expects (the label
has no net contents; a "0L" read in a scrap of text is no longer a figure). The real set's other figure
problems are not misreads a crop can fix: a statement read only in a turned view, "ONE PINT" never read, a
handwritten keg collar, "PROOF 102" written proof-first.

Calibration labels, every field that changed. Better: an alcohol content MISMATCH to MATCH ("91.5%" re-read as
"51.5%"), and net contents NEAR MATCH or MISMATCH to MATCH on five labels ("90 mL", "900ml", "730ml", a stray
"900 ml" and a stray "50ML" re-read as what is printed). Seven net contents MISMATCHes on a non-figure ("811 L",
"88 L", "8 L", "0ML", "0 mL" and the two "1L" below) became NOT FOUND: still a false alarm, but no longer a wrong
number shown as if it were read. **Worse, two:** 26239001000345 prints "50 ML" and the page read "SOML", every
digit as a letter; 26260001000505 prints "3L" and the page read "SL" (which used to come back as "5 L", a NEAR
MATCH on the wrong figure). Both are now NOT FOUND, because a figure made of letters alone is no longer made
into a volume (next paragraph), and the crops of those lines did not read the digits either. That is the price
of the fix below, and it is paid as a false alarm, never as a false MATCH.

The two planted defects that used to pass: on 26248001000067 and 26252001000396 a filed "1 L" came back MATCH.
Both labels print their real volume (750ml, 700ML) in type the page pass did not read, and both had an "LL" read
somewhere else: an importer's "LLC" cut short ("IMPORTED BY: SANTI IMPORTS LL") on the first, a lone "LL" on the
second. The old repair turned "LL" into "1 L" and matched it. A volume made by the look-alike repair must now keep at least one digit read as a digit, is never a
two-character token, never sits between a capitalised word and a ZIP code ("CHICAGO IL 60607"), and a zero
volume is never a candidate. Both fields are now NOT FOUND, and 330 of 330 planted defects are caught.

Time: a crop costs about 0.1 s and runs only on labels whose figure was doubtful. A label whose figure used to
be a MISMATCH or NOT FOUND and is settled by its crop no longer goes through the three turned and contrast passes
(batch label 0042: 1.43 s to 0.76 s); the NEAR MATCH labels cost about the same as before (0.58-0.96 s). The
other medians moved within run-to-run noise.

**Stress test (`scripts/stress_test.py`, the 15 sample labels per condition, local).** "Before → after" is the
second read of the figures (same runs as the table above); no row got worse.

| Condition | Expected verdict (before → after) | Planted defect reported as PASS |
|---|---|---|
| DejaVu (baseline), Georgia, Times, Baskerville, Helvetica Neue, Optima, Gill Sans, Avenir Next Condensed | 15/15 each | 0 |
| Futura / Rockwell | 14/15 → 15/15 / 13/15 → 15/15 | 0 |
| Didot (hairline serifs) | 8/15 → 10/15 | 0 |
| Brand in Papyrus, Trattatello or Impact | 15/15 each | 0 |
| Brand in Chalkduster / Herculanum | 12/15 → 13/15 / 12/15 | 0 |
| Brand in Copperplate | 13/15 → 14/15 | 1, see below |
| JPEG quality 35 | 14/15 → 15/15 | 0 |
| Half or a third of the resolution | 14/15 each | 0 |
| Tilted 1.5° / 4° (10/15 and 8/15 before straightening was added) | 15/15 / 14/15 | 0 |
| Blur radius 1.2 | 15/15 | 0 |
| Sensor noise (failed by time-out before the optional passes got their own limit) | 14/15 → 15/15 | 0 |

The one defect counted as missed is the brand-capitalization sample ("Stone's Throw Cellars" against "STONE'S
THROW CELLARS"). Copperplate has no lowercase letters: it draws them as small capitals, so the rendered label really
does read in capitals and the planted difference disappears. Every other miss is either a clean label flagged for
a look or a defect caught with a different severity (FAIL where REVIEW was expected, or the reverse).

### RapidOCR: the second engine, measured four ways (October 2026, local)

Measured on the build that ships (after the layout-aware warning and the second read of the figures were
merged), on an 11-core Mac: Tesseract 5.5.3, rapidocr-onnxruntime 1.4.4 on onnxruntime 1.31.0. The arrangements:

- **(a)** Tesseract alone (`RAPID_ESCALATION=0`), the behaviour before this change;
- **(b)** Tesseract, then RapidOCR as the escalation, with the 5 s time budget (the shipped default; the same
  runs with `LABEL_TIME_BUDGET_S=0` gave identical verdicts);
- **(c)** RapidOCR as the primary reader, alone (`--reader rapid`, `RAPID_ESCALATION=0`);
- **(d)** RapidOCR primary with Tesseract's two upright passes as its escalation (`--reader rapid+tesseract`).

```bash
RAPID_ESCALATION=0 python scripts/real_labels.py --defects -j 1        # (a); drop the variable for (b)
python scripts/real_labels.py --defects -j 1 --reader rapid            # (c); rapid+tesseract for (d)
python scripts/real_labels.py --csv scripts/calibration_labels.csv --defects
python scripts/bench.py [--set batch -j 4] [--reader ...]
python scripts/stress_test.py [--reader ...]
```

| | (a) Tesseract | **(b) + RapidOCR escalation** | (c) RapidOCR alone | (d) RapidOCR + Tesseract |
|---|---|---|---|---|
| **Real labels (20, 104 fields, one at a time)** | | | | |
| Fields with the expected verdict | 58/104 | **62/104** | 44/104 | 59/104 |
| Flagged for review where MATCH was expected | 26 | 28 | 21 | 24 |
| False alarms (MISMATCH / NOT FOUND for text on the label) | 18 | **12** | 37 | 19 |
| Accepted without the expected look | 2 | 2 | 2 | 2 |
| Warning pass / review / fail | 8 / 9 / 3 | 8 / 9 / 3 | 0 / 2 / 18 | 4 / 6 / 10 |
| Planted wrong ABV or volume reported as MATCH | 0 of 40 | 0 of 40 | 0 of 40 | 0 of 40 |
| Time per label, median / max | 1.45 s / 2.74 s | 2.48 s / 4.21 s | 1.23 s / 2.17 s | 1.87 s / 2.77 s |
| **Calibration set (167 labels, 913 fields, four at a time)** | | | not run | not run |
| Fields with the expected verdict | 523/913 | **552/913** | | |
| False alarms | 175 | **130** | | |
| Accepted without the expected look | 20 | 18 | | |
| Warning pass / review / fail | 47 / 83 / 37 | 47 / 83 / 37 | | |
| Planted defects reported as MATCH | 0 of 330 | 0 of 330 | | |
| Time per label, median / max (four at a time) | 2.53 s / 7.46 s | 3.74 s / 7.60 s | | |
| **Samples (15, one at a time)** | 15/15 | 15/15 | 11/15 | 13/15 |
| median / max | 0.90 s / 1.12 s | 1.83 s / 3.10 s | 0.92 s / 1.78 s | 1.19 s / 1.86 s |
| **Batch (250, four at a time)** | 248/250 | **249/250** | 108/250 | 158/250 |
| median / p95 / max | 0.54 / 1.39 / 1.71 s | 0.62 / 3.24 / 5.39 s | 1.96 / 2.47 / 3.67 s | 2.07 / 2.83 / 4.00 s |
| Planted defects reported as PASS (samples and batch) | 0 | 0 | 0 | 0 |

**Stress test, (a) against (b):** no condition got worse, five got better: Didot 10/15 → 11/15, Chalkduster
brand 13/15 → 15/15, half resolution 14/15 → 15/15, tilted 4° 14/15 → 15/15 (Herculanum 12/15, third
resolution 14/15 and Copperplate 14/15 unchanged; every other condition 15/15 in both). The one missed defect
in both is the Copperplate brand-capitalization sample described above (the typeface has no lowercase).
Medians roughly double under (b) (1.2 s → 2.3-3.6 s, four labels at a time) because most stress labels carry a
planted defect, and a label that really is wrong always pays for every optional pass.

What the escalation buys, read honestly: six fewer false alarms on the 20 hand-checked labels and 45 fewer on
the 167 calibration labels, all of them brand names, class/types and bottler lines in display faces, white or
small text on photographs and coloured panels, which RapidOCR reads and Tesseract does not. The warning results
do not move: RapidOCR's reading of a statement is used only when Tesseract found none (next paragraph). It never
let a planted defect through on any set. The cost is time on the labels that need it, about one second of
RapidOCR plus the re-matching; clean labels that pass on the first read never pay it (the batch median barely
moves, its p95 does). The time budget never had to skip anything in these runs on this machine.

Why not RapidOCR first. As the primary reader (c) it is worse on every set: in small print it often runs words
together (`GLENMORAR`, `IndiaPaleAle`, `WOMENSHOULDNOTDRINKALCOHOLICBEVERAGESDURINGPREGNANCY`) and sometimes drops
a whole line (on a clean black-on-white tequila label it lost the `GOVERNMENT WARNING: (1) According to the Surgeon
General` line), which fails the warning's word-for-word and capitals checks on 18 of 20 real labels and turns
generated labels' clean fields into near misses (108/250). Tesseract as its escalation (d) repairs most fields on
the real labels but not the generated ones. These numbers are untuned: a comparison with the spaces removed would
likely close part of the gap, but (b) is already best on every count that matters, so Tesseract stays first.

Two rules keep RapidOCR's habits from hurting in (b), both covered by tests:

- **Warning: Tesseract's reading first.** The statement is searched without RapidOCR's lines, and with them only
  when that finds nothing. Letting RapidOCR's reading compete turned one real label's warning from REVIEW into
  FAIL (its `GOVERNMENTWARNING:` fails the capitals check); before that, sharing Tesseract's frame made it worse on
  six. RapidOCR's read is also a frame of its own (`View.engine`), so its lines are never taken for Tesseract's.
- **Disagreeing readings across engines** (`RAPID_TRUST_CONF`, in the thresholds table): Tesseract's word
  confidences and RapidOCR's line scores are not on one scale, so for the disagreeing-readings rule a RapidOCR line
  scored 0.90 or more counts as certain. A confident RapidOCR reading that says something else at the same place
  ("BARK" where Tesseract read "BARN") forces a look; a less confident one competes with its score as read.

Other measurements behind the defaults: RapidOCR reading the colour image (`RAPID_INPUT=color`) instead of
Tesseract's preprocessed grayscale came back 59/104 against 60/104 on the real labels (measured before the merges)
and costs a second resize, so grayscale stays. With four ONNX Runtime threads a read took 1.2 s per label (one
thread 2.6 s, the runtime's default 1.4 s); four concurrent reads of one shared engine took 2.7 s together and
returned exactly the sequential results.

### Meaning-changing warning differences (October 2026, local)

Before is `main` at `c514b36`, after is the same code with the rule in "Government warning check", both run back
to back on this machine (Tesseract 5.5.3, RapidOCR escalation on), one label at a time
(`scripts/real_labels.py -j 1`, with and without `--csv scripts/calibration_labels.csv`). Every label in both real
sets carries a correct warning, so any wording FAIL the rule added there would be a false alarm.

| Set | Warning before: pass / review / fail | After |
|---|---|---|
| Real labels (20) | 8 / 9 / 3 | 8 / 9 / 3 |
| Calibration labels (167) | 47 / 83 / 37 | 47 / 83 / 37 |
| Samples (15) | 15/15 expected verdicts | 15/15, with `warning_text_altered` now expected to FAIL |
| Batch (250) | 249/250 | 249/250 (the same miss, label 0047), with the three altered-warning labels now expected to FAIL |

No correct label is newly failed: the warning verdict is the same on every one of the 187 real labels, label by
label, and no wording note names a meaning change. Before the confidence and gap tests are applied, the
classifier finds candidates on only two calibration labels, both already failing on similarity (66 and 75:
"may" lost where half the statement was not read, and a stretch of lines OCR did not read); `meaning_min_score`
keeps them from being judged word by word, and the confidence and line tests would have rejected them too. Planted defects reported as PASS: 0 on the samples and the batch. The
sample `warning_text_altered` prints "can cause" for "may cause" and the batch's three altered labels print
"must not" for "should not": each replaces a modal, which changes what the warning says, so their expected
verdict is now FAIL (`scripts/generate_labels.py` records the same; its third variant, "impair" for "impairs",
stays a REVIEW). (The field counts on the calibration set moved on four labels between the two runs, a brand,
two net contents and a class/type, by which optional OCR passes fit the time budget on a loaded machine; the
warning rule cannot reach the fields.)

**Planted edits on real labels.** `prompts/06-evaluation-data/tools/edit_warning.py` erases words using
Tesseract's word boxes and paints the replacement in their place, closed up from the left. Nine labels with a
clear warning (wording PASS before), four edits each and an unedited control re-saved the same way; the
replacement was set in DejaVu Sans, as close as the tool's fonts come to these labels' sans-serif warnings, and
in the case of the words it replaces. Two changes to the tool for this run: ink is the colour farthest from the
local background (it took the darkest pixels, which painted black on the white-on-black 26265001000907), and
the words to replace must share one printed line (it erased the box around all of them, which on two labels
wiped two lines of the statement); where "should not drink" spans a line break, "not drink" -> "drink" was
planted instead. A sideways warning (26209001000730) was edited turned upright and turned back.

| Label | "should drink" | "can cause" | "men" | "improves" | Control |
|---|---|---|---|---|---|
| 26205001000418 (tequila) | FAIL | FAIL | FAIL | FAIL | PASS (as before) |
| 26265001000907 (wine, white on black) | FAIL (line break, justified) | FAIL | FAIL | FAIL | PASS |
| 11145001000540 | FAIL | FAIL | FAIL | FAIL | PASS |
| 12048001000331 | FAIL | FAIL | FAIL | FAIL | PASS |
| 26232001000435 | FAIL | FAIL | FAIL | FAIL | PASS |
| 26252001000585 (wine) | FAIL (line break) | FAIL | **REVIEW** | FAIL | PASS |
| 26258001000346 (keg collar) | FAIL | FAIL | FAIL | FAIL | PASS |
| 26209001000730 (sideways) | FAIL | FAIL | FAIL | FAIL | PASS |
| 26251001000627 | not planted | FAIL | FAIL | not planted | REVIEW (as the re-saved image reads) |

33 of 34 planted edits fail, each with a note naming it and a question naming the word ("Does the label print
"not" in "should not drink"?"). The miss: on 26252001000585 the painted "men" is legible but Tesseract read it
at confidence 13, so it asks for a look (one difference, "men" for "women"). The tool found no "should not drink"
or "impairs" on 26251001000627's Tesseract read; 17007001000031 was tried and dropped for the same reason. The
controls keep the verdict the label had before; 26251001000627's re-saved control reads REVIEW (another reading
of one line says "hea" for "health") with and without the rule, the JPEG re-save and not the rule. Replacement
words were read at confidence 77-96 on the other labels, which is what set `meaning_conf` at 70: at 80,
11145001000540's legible "MEN" (77) would have asked for a look. Painting a serif replacement into a sans warning (the tool's default font)
lowered the confidence of the painted word (52-78), which is why the run above matches the face.

## Assumptions

- Labels arrive as flat artwork files or straight-on scans, the way they are attached to applications. Scans
  tilted by up to 6° are straightened.
- The application data is trusted; the label is what is being verified.
- "Exact wording" of the warning means the words; punctuation and line breaks are not judged.
- The regulation (27 CFR 16.22(a)(2)) requires the heading in capitals and bold and the rest of the statement not
  bold. The bold check compares the two, so an all-bold statement asks for review; a heading heavier than an
  already-bold body is not detected. Type size (16.22(b)) is not checked: the image does not carry a physical scale.
- A capitalization-only difference matters for the brand name (flagged for review) but not for the other fields.
- English-language labels.

## Known limitations

- Photos taken at an angle, with glare, shadows or poor lighting are out of scope; expect NOT FOUND results and a
  warning-statement failure on such images. Tilt up to 6° is corrected; perspective correction would be next.
- On real approved labels the tool asks for a look on every label (see "Measured results"): brand names in display
  or curved typefaces, light text over photographs and handwriting produce false alarms; a single-digit misread
  survives when the second read of the line repeats it or the line was only read in a turned view.
- Tesseract struggles with decorative, script or outlined brand typography and with text on busy backgrounds. The
  guided matching tolerates a fair amount of noise, but a brand set in a script face may come back NOT FOUND.
  The RapidOCR escalation closes some of these (6 of 18 false alarms on the 20 real labels, 45 of 175 on the
  calibration set), not all: a
  handwritten keg collar, a brand on a tight curve and the smallest sideways print still fail, and RapidOCR
  itself sometimes runs words together (`GLENMORAR`), which the matcher reads as a near miss.
- The time budget trades accuracy for time on a slow machine: when the first read and the turned passes have
  used most of the 5 seconds, the escalation is skipped and the label keeps its first-pass verdict (named in
  the reader line). On a shared vCPU that is expected to be the common case for hard labels; the measured
  gains above assume the escalation ran.
- The budget is checked before each optional pass, against that pass's average cost; a pass that runs slower
  than its average still finishes. With four batch labels at a time on this machine the slowest label took
  5.4 s against the 5 s budget. The RapidOCR read itself is bounded separately by `RAPID_TIMEOUT_S` (10 s).
- A RapidOCR read cannot be interrupted: one that runs past `RAPID_TIMEOUT_S` is abandoned (the label goes on
  without it) but finishes on its thread, holding one of the `RAPID_WORKERS` slots until it does.
- An OpenCV 5.0.0 `resize` of an unpadded 1799-pixel-wide label crashed the process once on macOS/arm64 while
  this was measured; handing RapidOCR a fresh array padded to the detector's 32-pixel grid did not crash in
  140 reads, but it is a mitigation, not a root-cause fix.
- The bold heuristic is calibrated on synthetic labels and assumes the statement body is in regular weight. A
  real-world calibration set would be needed before trusting it unattended, which is why it only asks for review.
- Batch jobs are kept in memory and disappear on restart; for production they would go to a queue and a database.
- No authentication: the prototype assumes it runs on an internal network.
- Text at 90° is read only when the first, upright read leaves something missing; text at other angles (curved
  around a seal, set diagonally) is not read.
- A warning statement printed in two side-by-side panels (sentence (1) left, sentence (2) right) interleaves in
  top-to-bottom order and fails as wording; one above the other, or wrapped around a seal, assembles correctly.
- A one-word last line of a sideways warning ("PROBLEMS.") is dropped by the turned-view reader, which keeps only
  lines of two or more words, so that label asks for a look over a missing word that is on the label.
- The brand's small-print rule assumes the brand is printed larger than the bottler statement. A label whose
  brand is deliberately small next to a large fanciful name gets a NEAR MATCH for the agent to confirm.
- Uploads are size-checked from their Content-Length; a chunked upload without one is only limited per file after
  it arrives. A reverse proxy limit is the production answer.
- A letter-spaced brand is compared with its spaces closed up, so "B A R N O N E" matches both "BARN ONE" and
  "BAR NONE": OCR does not keep the wider gap between words.
- Two readings of the same place that disagree are only flagged when OCR was at least as confident of the one
  that disagrees; a label misprint that OCR also reads with less confidence than a misread of it would pass.
- A warning statement printed in two halves (the first sentence on the front, the second on the back) is
  assembled into one and can pass; each sentence must appear once and in order, and a sentence printed again
  elsewhere is reported, but the regulation's requirement that the statement be printed as one unit is not judged.
- A meaning-changing warning edit fails only when OCR read it surely. Measured on planted edits (see
  "Measured results"), the misses ask for a look instead: a replacement word read below confidence 70 (a
  legible "men" that Tesseract scored 13), a neighbour of a missing "not" read below it, and a "not" removed at
  a line break in ragged or centred type, where the gap cannot be judged. The word lists are finite: a
  replacement they do not name ("teenagers" for "women") asks for a look, never passes. With the cloud reader,
  which reports no confidences, or a statement only RapidOCR read, every wording difference only asks for a look.
- Words beyond the statement's column on two or more of its rows are taken for a neighbouring column and set
  aside (quoted, not judged). A label that deliberately added words to the statement on two lines, each sticking
  out past every other line, would be read the same way; the quote in the note is the agent's guard.
- Single-label checks and batch jobs share one pool of four Tesseract processes for the second pass and the
  optional passes; measured locally, a single check took 0.6-0.7 s during a 250-label batch against 0.4-0.7 s
  idle, but a smaller container will queue more.

## Security notes

Uploads are validated (image type, 20 MB and 40 megapixel limits, decompression-bomb and zip member/size limits)
and processed in memory; the app itself never writes them to disk, but the web framework spools any upload over
1 MB to a temporary file in `/tmp` for the duration of the request. Sample file names are whitelisted, the CSV
export neutralises spreadsheet formulas, and the container runs as an unprivileged user. The app makes no outbound
calls unless the cloud reader is enabled.
