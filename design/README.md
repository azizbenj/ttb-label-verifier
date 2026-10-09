# Label Check UI redesign: handoff

This folder is the design spec for the UI redesign. Read it before touching anything under `app/templates/` or `app/static/`.

- **Canvas (interactive, source of truth):** https://claude.ai/artifact/QRoVDnra5hD8R6jGh6JAk3
  A Claude Code session can read it directly with the Artifact tool: `action: "read"`, that URL, and `path: "project/<Board>.dc.html"` (list the files with `action: "list"`, `scope: "files"`). The same sources are copied here under `design/boards/` so nothing depends on the network.
- **Tokens:** `design/tokens.css`, paste at the top of `app/static/style.css`.
- **Boards:** `design/boards/*.html` (23 files, listed below). Each is a self-contained HTML page: the `<style>` block in `<helmet>` is the CSS to port, the markup is the DOM to produce, and the copy is final. Ignore the `<script src="./support.js">` line, the `<x-dc>` wrapper and the `<script type="text/x-dc">` block; `{{…}}`, `<sc-if>` and `<sc-for>` only appear in the interactive boards and mark state-driven parts (hover highlight, filters, decisions).
- **Images:** boards reference `/_blob/<id>`; `design/boards/ASSETS.md` maps each id to a file in `data/`.

## The idea in one line

Every verdict sits next to its proof: the label image is the primary surface, each field and warning check points at the region it was read from, and a batch is a triage queue with REVIEW first. Vocabulary is fixed: MATCH, NEAR MATCH, MISMATCH, NOT FOUND, SKIPPED, PASS, REVIEW, FAIL, "government warning". Per-field reasons come from the pipeline unchanged.

## Boards

| Board | What it specifies |
|---|---|
| `Main.html` | Direction, the five rules, canvas map |
| `Tokens.html` | Colour (with measured contrast), type, space, radius, elevation, motion; light and dark |
| `Components.html` | Verdict tags, status banner, evidence row, diff table, progress, filter chips, data table, drop zone, sample picker, toasts and alerts, empty states, buttons, inputs, focus |
| `Single-Empty.html` | Single-label form, drop zone, sample gallery, empty result slot |
| `Single-Loading.html` | Reading state: folded form bar, scanning image, skeleton rows, steps |
| `Single-Pass.html` | PASS result: split view, pins, hover highlight (interactive) |
| `Single-Review.html` | REVIEW: near-match evidence strip, decision buttons (interactive) |
| `Single-Fail.html` | FAIL: mismatch with number conversion, problem anchor (interactive) |
| `Evidence-Warning.html` | Government warning evidence: crop, four checks, context diff, bold meter, decision (interactive) |
| `Batch-Upload.html` | Batch form, template preview, "what you get back" |
| `Batch-Progress.html` | Honest progress: counts so far, estimate, rows streaming in |
| `Batch-Results.html` | Triage table: filters, search, sort, detail panel, bulk bar (interactive) |
| `Review-Queue.html` | Keyboard review queue with decisions (interactive) |
| `Batch-Export.html` | Export dialog: scope, include, format, quick exports |
| `Help.html` | Non-modal help drawer |
| `Errors.html` | Every error state with copy |
| `Dark.html` | Dark theme on the FAIL result |
| `Print.html` | Letter print report with reviewer decision block |
| `Tablet-Single.html` | 768 px result |
| `Narrow-Single.html` | 390 px result |
| `Narrow-Batch.html` | 390 px batch list |
| `Copy.html` | Every headline, label, helper and message |
| `Implementation.html` | Accessibility decisions, pipeline additions, template-by-template mapping, print CSS, a day plan |

## Order of work

1. `Tokens.html` → `tokens.css` into `style.css`; bundle Public Sans woff2 under `app/static/fonts/` (system fallback already in the stack). No CDN, no external requests.
2. `base.html`: identifier strip, neutral mark, tagline, help button, footer. See "Naming" below.
3. `index.html`: two tabs, form layout, drop zones, sample gallery (keeps the hidden `<select>`), empty states.
4. `partials/result.html`: banner (`role="status"`), split view with pins and `.hl` regions, evidence rows, warning section. Needs `box` percentages per field from the OCR word span (see `Implementation.html`, "What the pipeline must expose").
5. `partials/batch_status.html`: progress with tallies, triage header, filters, brand column, detail panel, keyboard map.
6. `partials/error.html` kinds, print CSS, help drawer, reduced-motion and dark media queries.
7. Day 2: decision endpoint and review queue, export dialog.

## Hard constraints

- Jinja + one CSS file + small vanilla JS. No build step, no framework, no external assets, inline SVG icons only.
- Semantic DOM: real `<table>`, `<button>`, `<label>` + `<input>`; the single-label form works without JS.
- 1280 px first; 768 px and 390 px with no horizontal scroll; WCAG 2.1 AA (contrast values are on the Tokens board); `prefers-reduced-motion` honoured.
- Colour is never the only signal: every tag has a glyph and a word, every pin a number.

## Naming

This is an unaffiliated prototype. Do not use "Department of the Treasury", "TTB", "Alcohol and Tobacco Tax and Trade Bureau", a seal, or agency-style attribution anywhere in the UI (31 U.S.C. § 333). The product is "Label Check"; the strip reads "Label Check · COLA compliance review · Prototype". Regulation citations (27 CFR parts 4, 5, 7, 16) and the term COLA are fine. `base.html` currently has a "TTB" seal, a "Label Check · TTB" title and a "TTB Compliance Division" footer: replace them as part of step 2.
