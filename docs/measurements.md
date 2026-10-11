# Measurement history

How each figure in the README's "Measured results" was reached, in the order the work happened. Older
sections describe earlier builds and are kept as they were measured; the README's "current build at a
glance" is the summary of the build that ships.

## Before the second engine

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

## RapidOCR: the second engine, measured four ways (October 2026, local)

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

What the escalation buys, read honestly: six fewer false alarms on the 20 development labels and 45 fewer on
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

## Meaning-changing warning differences (October 2026, local)

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

**Planted edits on real labels.** A development script that is not in the repository erases words using
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

## Evidence strength: FAIL only on what was read clearly (October 2026, local)

Before is `main` at `eab19e9` (with the meaning-changing warning rule), after is the same code with the evidence-strength rule, each a git archive
run outside the synced folder, the two sides interleaved set by set on the same machine (Tesseract 5.5.3,
RapidOCR escalation on), real labels one at a time. Other sessions were loading this machine heavily (at one
point a load average above 100), and under load the 5-second budget skips the RapidOCR escalation, which changes
field verdicts from run to run; so both sides ran with `LABEL_TIME_BUDGET_S=0`, which on a quiet machine is what
the budget allows anyway (an earlier quiet run with the default budget gave the same verdicts on the 20 real
labels). Field and warning verdicts are identical before and after on every label of every set: only the overall
status, its summary and the new clarity fields changed.

```bash
LABEL_TIME_BUDGET_S=0 python scripts/bench.py --fail-under 1.0
LABEL_TIME_BUDGET_S=0 python scripts/bench.py --set batch --fail-under 0.95
LABEL_TIME_BUDGET_S=0 python scripts/real_labels.py -j 1 --defects --json real20.json
LABEL_TIME_BUDGET_S=0 python scripts/real_labels.py -j 1 --defects --csv scripts/calibration_labels.csv --json calib.json
LABEL_TIME_BUDGET_S=0 python scripts/stress_test.py --verbose
```

| Set | Before | After |
|---|---|---|
| Correct real labels (20): PASS / REVIEW / FAIL | 0 / 10 / 10 | **0 / 16 / 4** |
| Correct calibration labels (167): PASS / REVIEW / FAIL | 6 / 70 / 91 | **6 / 107 / 54** |
| All 187 correct real labels | 6 / 80 / 101 | **6 / 123 / 58** |
| Planted wrong figures on real labels (40 + 330): FAIL / REVIEW / PASS | 369 / 1 / 0 | 353 / 17 / **0** |
| Planted field reported as MATCH | 0 of 370 | 0 of 370 |
| Field verdicts (real 20 / calibration) | 62/104 and 576/913 as expected, 12 and 109 false alarms | unchanged |
| Samples (15) | 15/15, 0 planted defects passed, median 1.8 s | 15/15, 0, 1.7 s |
| Batch (250) | 249/250 (label 0047), 0 planted defects passed | 249/250, the same miss, 0 |
| Stress test (24 conditions × 15) | 0 planted defects passed | 0; Didot 11/15 → 8/15, third resolution 14/15 → 13/15, every other row the same |
| Time per real label, median / max (budget off) | 2.36 s / 4.23 s (20), 2.55 s / 7.33 s (167) | 2.38 s / 4.22 s, 2.53 s / 7.37 s |

An earlier evaluation of the live build (not in the repository) measured 1 / 21 / 23 PASS / REVIEW / FAIL on its 45 correct labels; on the 187 here the FAIL share
falls from 54% to 31%. What still fails a correct label is evidence the rule cannot see as unclear: a matcher
that took a partial or wrong span of clearly read text (50 text MISMATCHes read at 60+ / 80+), a misread figure
(always clear, by measurement), and 9 warning statements read at 90+ that still fail as wording.

**Separation.** Over the clean runs of both real sets (ground truth for every field):

| | Unclear (REVIEW) | Clear (FAIL) |
|---|---|---|
| Field false alarms (120): text MISMATCH | 29 | 50 |
| ... text NOT FOUND | 9 | 1 |
| ... figure NOT FOUND | 14 | 10 (5 of them an application value that cannot be read as a figure) |
| ... figure MISMATCH | 0 | 7 |
| Warning false alarms (40; every real label carries a correct warning) | 31 (25 failures on a statement read below 90, 6 "not found") | 9 |
| Real problems: planted wrong figures (368 that came back MISMATCH or NOT FOUND) | 21 (all NOT FOUND: the label's own figure was never read) | 347 |
| Real problems: net contents genuinely absent (3 labels) | 3 | 0 |

So 52 of 120 field false alarms and 31 of 40 warning false alarms no longer fail a label; 68 field false alarms
stay clear; 24 real problems became unclear, and 16 labels carrying a planted defect moved from FAIL to REVIEW
(on the other 5 another clear problem keeps the FAIL). Every one of the 16, with the planted field NOT FOUND
because the label's own figure was not read:

- Real labels (4): 11145001000540, 15117001000279, 15117001000284 (net contents; all three labels genuinely
  print none) and 26189001000502 (net contents; "16 fl oz" is printed but was not read).
- Calibration labels (12): alcohol content on 26210001000327, 26232001000450, 26251001000296, 26258001000801,
  26271001000532; net contents on 26210001000327, 26232001000450, 26251001000782, 26259001000159, 26261001000185,
  26264001000814, 26271001000514.

The same rule on the stress renderings moved 4 planted defects from FAIL to REVIEW, none to PASS: in Didot (hairline
serifs; the label as a whole read with 86-90% confident words) `missing_warning`, `missing_net_contents` and
`wrong_country` (its "Product of Ireland" read at 26 for the least sure word), and at a third of the resolution
`wrong_brand` (RapidOCR read "RIVEr BENd BREWiNG CO" at 72). 70 / 85 and a 6% unread-marks floor, the first
values tried, had moved 8; the looser 60 / 80 and 10% cost 2 more correct real labels that FAIL (58 against 56).

**Planted warning edits on real labels.** The 34 edits of the run above (9 labels; "should drink", "can cause",
"men", "improves") give the same verdict with this rule as without it, 33 FAIL and the one REVIEW: a
meaning-changing difference that `check_warning` fails is always clear, because it already requires the words
that make it to be read confidently one by one. Without that exception the statement-confidence floor sent all
four edits of 12048001000331 (statement read at 84-87) and one of 26209001000730 to REVIEW.

**Request a better image.** 11 of the 187 correct real labels (2 of the 20, 9 of the 167) get "Much of this
label could not be read clearly"; none of the synthetic labels do.
