/* Label Check UI: tabs, help drawer, single-label flow (gallery, drop zone, result with evidence pins),
   batch flow (polling, filters). Dependency-free; every screen also works without this file. */
(function () {
  "use strict";
  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));
  const reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // --- tabs (WAI-ARIA tabs: roving tabindex, arrow keys) -------------------------------------------
  const tabs = $$(".modes .tab");
  function selectTab(tab, remember) {
    tabs.forEach((t) => {
      const on = t === tab;
      t.classList.toggle("on", on);
      t.setAttribute("aria-selected", on);
      t.tabIndex = on ? 0 : -1;
    });
    $$(".panel").forEach((p) => p.classList.toggle("on", p.id === "tab-" + tab.dataset.tab));
    if (remember) { try { localStorage.setItem("labelcheck.tab", tab.dataset.tab); } catch (e) { /* ignore */ } }
  }
  tabs.forEach((tab, i) => {
    tab.addEventListener("click", () => selectTab(tab, true));
    tab.addEventListener("keydown", (e) => {
      const step = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
      if (!step) return;
      const next = tabs[(i + step + tabs.length) % tabs.length];
      selectTab(next, true);
      next.focus();
    });
  });
  // The remembered tab, unless the server asked for one (a shared batch link opens on the batch tab).
  const asked = $(".modes[data-open-tab]");
  if (asked) { const tab = $(`.modes .tab[data-tab="${asked.dataset.openTab}"]`); if (tab) selectTab(tab, false); }
  else {
    try {
      const saved = localStorage.getItem("labelcheck.tab");
      const tab = saved && $(`.modes .tab[data-tab="${saved}"]`);
      if (tab) selectTab(tab, false);
    } catch (e) { /* ignore */ }
  }

  // --- modal layers: the page behind a drawer or overlay is inert, so focus and screen readers stay in it
  const PAGE = ".strip, header.masthead, main, footer.page-foot";
  function setPageInert(on) { $$(PAGE).forEach((el) => { el.inert = on; }); }

  // --- help drawer ----------------------------------------------------------------------------------
  const helpBtn = $("#help-btn"), drawer = $("#help-drawer"), scrim = $("#help-scrim");
  let lastFocus = null;
  function openHelp() {
    lastFocus = document.activeElement;
    drawer.hidden = false; scrim.hidden = false;
    setPageInert(true);
    helpBtn.setAttribute("aria-expanded", "true");
    $("#help-close").focus();
  }
  function closeHelp() {
    drawer.hidden = true; scrim.hidden = true;
    setPageInert(false);
    helpBtn.setAttribute("aria-expanded", "false");
    if (lastFocus) lastFocus.focus();
  }
  if (helpBtn && drawer) {
    helpBtn.addEventListener("click", () => (drawer.hidden ? openHelp() : closeHelp()));
    $("#help-close").addEventListener("click", closeHelp);
    scrim.addEventListener("click", closeHelp);
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !drawer.hidden) closeHelp(); });
  }

  // --- fetch helpers --------------------------------------------------------------------------------
  function escapeHtml(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
  const ALERT_ICON = (size) => `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 8v5"/><path d="M12 16h.01"/></svg>`;
  // detailHtml is trusted markup built here; everything that comes from a file name or the server is escaped.
  function alertCard(title, detailHtml, retry) {
    return `<div class="card alert err" role="alert" tabindex="-1" data-app-card data-kind="client">${ALERT_ICON(20)}<div class="grow"><strong>${escapeHtml(title)}</strong><p class="detail">${detailHtml}</p>${retry ? '<button class="btn small retry" type="button" data-act="retry">Try again</button>' : ""}</div></div>`;
  }
  function errorCard(title, detail, retry) { return alertCard(title, escapeHtml(detail), retry); }
  const OFFLINE = ["We couldn't reach the server.", "Your check was not sent. Check the network connection and try again; nothing you typed was lost."];
  // The app answers every request with one of its own cards (marked data-app-card), errors included.
  // Anything else (a proxy's error page during a redeploy) must not be pasted into the page.
  async function cardFrom(resp) {
    const text = await resp.text();
    const head = text.trim().slice(0, 400);
    if (head.startsWith("<") && head.includes("data-app-card")) return text;
    return errorCard("The server did not answer properly.", `Please try again in a moment (HTTP ${resp.status}).`);
  }
  function focusCard(slot) {
    const card = slot.querySelector("[role=status], [role=alert], [tabindex='-1']");
    if (card) card.focus({ preventScroll: false });
  }
  async function post(url, body, slot, button) {
    slot.classList.add("busy");
    const label = button ? button.textContent : "";
    if (button) { button.disabled = true; button.textContent = "Checking…"; }
    try {
      const resp = await fetch(url, { method: "POST", body, headers: { "X-Partial": "1" } });
      slot.innerHTML = await cardFrom(resp);
      return resp.ok;
    } catch (err) {
      slot.innerHTML = errorCard(OFFLINE[0], OFFLINE[1], true);
      return false;
    } finally {
      slot.classList.remove("busy");
      if (button) { button.disabled = false; button.textContent = label; }
    }
  }

  // --- single label: form, gallery, drop zone ------------------------------------------------------
  const singleForm = $("#single-form");
  const entry = $("#single-entry");
  const resultSlot = $("#result");
  const samples = (() => { try { return JSON.parse(($("#samples-data") || { textContent: "[]" }).textContent); } catch (e) { return []; } })();
  const picker = $("#sample-picker");
  const sampleField = $("#sample-field");
  const imageInput = $("#image-input");
  const drop = $("#drop");
  const preview = $("#preview");
  let objectUrl = null;

  function fmtBytes(n) { return n > 1024 * 1024 ? (n / 1048576).toFixed(1) + " MB" : Math.max(1, Math.round(n / 1024)) + " KB"; }
  function showPreview(src, name, meta) {
    if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
    if (!src) { preview.hidden = true; drop.hidden = false; $("#preview-img").removeAttribute("src"); return; }
    if (src.startsWith("blob:")) objectUrl = src;
    $("#preview-img").src = src;
    $("#preview-name").textContent = name;
    $("#preview-meta").textContent = meta;
    preview.hidden = false; drop.hidden = true;
  }
  const SAMPLE_KEYS = ["brand_name", "class_type", "alcohol_content", "net_contents", "bottler_name_address", "country_of_origin", "application_id"];
  let currentSample = null;
  function pressTile(name) { $$(".thumb[data-sample]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.sample === name))); }
  // Clear the fields a sample filled in, but keep anything the agent typed over since.
  function clearSampleFields() {
    if (!currentSample) return;
    SAMPLE_KEYS.forEach((k) => {
      const el = singleForm.querySelector(`[name="${k}"]`);
      if (el && el.value === (currentSample[k] || "")) el.value = "";
    });
    currentSample = null;
  }
  function applySample(name) {
    const s = samples.find((x) => x.name === name);
    if (!s) { clearSampleFields(); sampleField.value = ""; pressTile(""); showPreview(null); return; }
    clearSampleFields();
    clearDropError();
    currentSample = s;
    sampleField.value = s.name;
    pressTile(s.name);
    SAMPLE_KEYS.forEach((k) => {
      const el = singleForm.querySelector(`[name="${k}"]`);
      if (el) el.value = s[k] || "";
    });
    imageInput.value = "";
    showPreview(`/samples/${s.name}.png`, s.short + " (sample)", s.kind);
  }
  if (picker) picker.addEventListener("change", () => applySample(picker.value));
  $$(".thumb[data-sample]").forEach((b) => b.addEventListener("click", () => {
    const again = b.getAttribute("aria-pressed") === "true";   // clicking the chosen tile deselects it
    picker.value = again ? "" : b.dataset.sample;
    applySample(picker.value);
  }));
  const more = $("#more-samples");
  if (more) more.addEventListener("click", () => {
    const box = $("#gallery-defects");
    box.hidden = !box.hidden;
    more.setAttribute("aria-expanded", String(!box.hidden));
  });
  // --- errors: mark the inputs and the drop zone an error is about (Errors board) -------------------
  const dropB = drop ? $("#drop b") : null, dropS = drop ? $("#drop .small") : null;
  const maxBytes = imageInput ? +imageInput.dataset.maxBytes || 20 * 1048576 : 0;
  const maxMb = Math.round(maxBytes / 1048576);
  function setDropError(title, message, chooseAgain) {
    if (!drop) return;
    drop.classList.add("err");
    dropB.textContent = title;
    dropS.innerHTML = escapeHtml(message) + (chooseAgain ? '<span class="btn secondary small again">Choose another file</span>' : "");
  }
  function clearDropError() {
    if (!drop || !drop.classList.contains("err")) return;
    drop.classList.remove("err");
    dropB.textContent = dropB.dataset.t; dropS.textContent = dropS.dataset.t;
  }
  function markField(key, message) {
    const el = singleForm && singleForm.querySelector(`#f-${key}`);
    if (!el) return null;
    el.classList.add("err"); el.setAttribute("aria-invalid", "true");
    let m = document.getElementById(`f-${key}-m`);
    if (!m) { m = document.createElement("div"); m.className = "msg"; m.id = `f-${key}-m`; el.insertAdjacentElement("afterend", m); }
    m.innerHTML = ALERT_ICON(14) + escapeHtml(message);
    el.setAttribute("aria-describedby", m.id);
    return el;
  }
  function clearField(el) {
    el.classList.remove("err"); el.removeAttribute("aria-invalid"); el.removeAttribute("aria-describedby");
    const m = document.getElementById(`${el.id}-m`);
    if (m) m.remove();
  }
  function clearMarks() { if (singleForm) $$(".input.err, .input[aria-invalid]", singleForm).forEach(clearField); clearDropError(); }
  function wireErrorLinks(card) {
    $$("a[data-focus]", card).forEach((a) => a.addEventListener("click", (e) => {
      const el = document.getElementById(a.dataset.focus);
      if (el) { e.preventDefault(); el.focus(); }
    }));
  }
  // An error card from the server says which inputs it is about: mark them, return the first one.
  function applyErrorMarks(slot) {
    const card = slot.querySelector(".alert[data-app-card]");
    if (!card) return null;
    wireErrorLinks(card);
    let first = null;
    if (card.dataset.fields) {
      try { JSON.parse(card.dataset.fields).forEach((f) => { first = markField(f.key, f.message) || first; }); } catch (e) { /* ignore */ }
    }
    if (card.dataset.dropTitle) { setDropError(card.dataset.dropTitle, card.dataset.dropMessage, card.dataset.kind === "image_unreadable"); first = first || imageInput; }
    return first;
  }
  function labelText(el) {
    const l = singleForm.querySelector(`label[for="${el.id}"]`);
    return l && l.firstChild ? l.firstChild.textContent.trim() : el.name;
  }
  function showSingleError(html) {
    resultSlot.innerHTML = html; resultSlot.hidden = false;
    const card = resultSlot.querySelector(".alert");
    if (card) wireErrorLinks(card);
  }
  if (drop && dropB) { dropB.dataset.t = dropB.dataset.t || dropB.textContent; dropS.dataset.t = dropS.dataset.t || dropS.textContent; }
  if (singleForm) {
    singleForm.addEventListener("input", (e) => { if (e.target.matches(".input[aria-invalid]")) clearField(e.target); });
    // A plain form post that came back with an error (no JavaScript on that request): wire the links.
    if (resultSlot && !resultSlot.hidden) applyErrorMarks(resultSlot);
  }

  // One file or several (front, back, neck): the server stacks them into one label.
  function fileChosen(files) {
    picker.value = ""; sampleField.value = ""; pressTile(""); currentSample = null;
    clearDropError();
    const list = files ? Array.from(files.length !== undefined ? files : [files]) : [];
    if (!list.length) { showPreview(null); return; }
    const bad = list.find((f) => !/^image\//.test(f.type) && !/\.(png|jpe?g|tiff?|bmp|webp)$/i.test(f.name));
    if (bad) {
      setDropError("That isn't an image we can read", `'${bad.name}' could not be read as an image. Please upload a PNG or JPG of the label.`, true);
      imageInput.value = "";
      return;
    }
    const big = list.find((f) => f.size > maxBytes);
    if (big) {   // known before upload: say so at once (the server repeats the check)
      showSingleError(alertCard("This image is too big to check.", escapeHtml(`'${big.name}' is ${Math.round(big.size / 1048576)} MB; the limit is ${maxMb} MB. A 300 dpi scan is plenty: export it as a PNG or JPG and try again.`)));
      imageInput.value = "";
      showPreview(null);
      return;
    }
    if (resultSlot.querySelector('.alert[data-kind^="image"], .alert[data-kind="client"]') && entry && !entry.hidden) { resultSlot.innerHTML = ""; resultSlot.hidden = true; }
    const total = list.reduce((n, f) => n + f.size, 0);
    const name = list.length === 1 ? list[0].name : `${list.length} images: ${list.map((f) => f.name).join(", ")}`;
    const meta = list.length === 1 ? `${list[0].type || "image"} · ${fmtBytes(total)}` : `read together as one label · ${fmtBytes(total)}`;
    showPreview(URL.createObjectURL(list[0]), name, meta);
  }
  if (imageInput) imageInput.addEventListener("change", () => fileChosen(imageInput.files));
  if (drop) {
    ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); $("#drop b").textContent = "Release to add the label"; }));
    ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, () => { drop.classList.remove("over"); if (!drop.classList.contains("err")) $("#drop b").textContent = dropB.dataset.t; }));
    drop.addEventListener("drop", (e) => {
      e.preventDefault();
      const files = e.dataTransfer && e.dataTransfer.files;
      if (!files || !files.length) return;
      try { imageInput.files = files; } catch (err) { /* older browsers */ }
      fileChosen(files);
    });
  }
  const clearImage = $("#clear-image");
  // "Remove" drops the image; for a sample it also clears the fields the sample filled in.
  if (clearImage) clearImage.addEventListener("click", () => { imageInput.value = ""; picker.value = ""; applySample(""); });

  function showEntry(show) { entry.hidden = !show; resultSlot.hidden = show; }

  // --- result: pins, highlights, evidence strips, actions ----------------------------------------------
  function wireResult(slot) {
    const frame = $("#label-frame", slot);
    const img = frame && frame.querySelector("img");
    // Uploaded images arrive inline; samples by URL. Either way the crops reuse the same source.
    const setActive = (key) => {
      $$(".hl", slot).forEach((h) => h.classList.toggle("on", !!key && h.dataset.hl === key));
      $$("[data-row]", slot).forEach((r) => r.classList.toggle("on", !!key && r.dataset.row === key));
    };
    $$("[data-row]", slot).forEach((row) => {
      row.addEventListener("mouseenter", () => setActive(row.dataset.row));
      row.addEventListener("mouseleave", () => setActive(null));
      row.addEventListener("focusin", () => setActive(row.dataset.row));
      row.addEventListener("focusout", () => setActive(null));
    });
    // Evidence strips: open for problems and near matches, collapsed otherwise; the pin toggles them.
    $$("[data-strip][data-collapsed]", slot).forEach((s) => { s.hidden = true; });
    $$(".pin[data-pin]", slot).forEach((pin) => pin.addEventListener("click", () => {
      const strip = $(`[data-strip="${pin.dataset.pin}"]`, slot);
      if (!strip) { setActive(pin.dataset.pin); return; }
      strip.hidden = !strip.hidden;
      pin.setAttribute("aria-expanded", String(!strip.hidden));
    }));
    wireDecisions(slot);
    $$("[data-act]", slot).forEach((b) => b.addEventListener("click", (e) => {
      const act = b.dataset.act;
      if (act === "another" || act === "edit") e.preventDefault();
      if (act === "edit") { showEntry(true); singleForm.querySelector("input.input").focus(); window.scrollTo({ top: 0, behavior: reduced ? "auto" : "smooth" }); }
      if (act === "another") { showEntry(true); singleForm.reset(); applySample(""); showPreview(null); singleForm.querySelector("input.input").focus(); window.scrollTo({ top: 0, behavior: reduced ? "auto" : "smooth" }); }
      if (act === "print") window.print();
      if (act === "copy") {
        const rows = $$("tr.bad[data-row]", slot).map((r) => `${r.children[1].textContent.trim()}: application "${r.children[2].textContent.trim()}", label "${r.children[3].textContent.trim()}"`);
        const w = slot.querySelector("#warning-section .why.bad, #warning-section .chk.bad .t");
        if (w) rows.push("Government warning: " + w.textContent.trim());
        const text = rows.join("\n");
        if (navigator.clipboard) navigator.clipboard.writeText(text).then(() => { b.textContent = "Copied"; setTimeout(() => { b.innerHTML = b.dataset.html; }, 2000); });
        b.dataset.html = b.dataset.html || b.innerHTML;
      }
    }));
    if (!frame || !img || !objectUrl) return;
  }

  // --- decisions (Copy board): yes = the label is fine, no = it is wrong, skip; "Decided by you ... Undo" -----
  function showDecision(block, value) {
    const texts = {};
    $$(".answers [data-answer]", block).forEach((b) => { texts[b.dataset.answer] = b.firstChild.textContent.trim(); });
    block.dataset.decision = value;
    const answers = block.querySelector(".answers"), fine = block.querySelector(".fine"), done = block.querySelector(".decided");
    answers.hidden = !!value; if (fine) fine.hidden = !!value; done.hidden = !value;
    if (value) done.querySelector("[data-answer-text]").textContent = value === "skip" ? "skipped for now" : texts[value];
  }
  async function decide(block, value, wrap) {
    const job = wrap && wrap.dataset.job;
    if (job) {
      try {
        const body = new FormData(); body.append("index", wrap.dataset.index); body.append("value", value);
        const resp = await fetch(`/batch/${job}/decision`, { method: "POST", body, headers: { "X-Partial": "1" } });
        if (!resp.ok) { toast("That decision was not saved. Please try again."); return; }
        const row = document.querySelector(`tr.r[data-index="${wrap.dataset.index}"]`);
        if (row) row.dataset.decision = value === "clear" ? "" : value;
        const d = document.querySelector("[data-d=decision]");
        if (d && row && row.classList.contains("cur")) d.textContent = row.dataset.decision || "—";
      } catch (err) { toast("That decision was not saved. Please check the connection."); return; }
    }
    showDecision(block, value === "clear" ? "" : value);
  }
  function wireDecisions(scope) {
    const wrap = scope.closest("[data-job]") || scope.querySelector("[data-job]");
    $$(".decide [data-answer]", scope).forEach((b) => b.addEventListener("click", (e) => {
      e.preventDefault();
      const block = b.closest(".decide");
      const value = b.dataset.answer;
      const review = document.querySelector("[data-review]");
      decide(block, value, wrap || review).then(() => {
        // In the queue a decision moves you on (Review-Queue board).
        if (review && value !== "clear") { const next = $("[data-act=next]"); if (next) location.href = next.href; }
      });
    }));
  }
  // Y / N / S answer the first open question on the page; the review queue also moves with J / K and leaves with Esc.
  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input, select, textarea") || e.metaKey || e.ctrlKey || e.altKey) return;
    const dlg = document.querySelector("dialog[open]"); if (dlg) return;
    const review = document.querySelector("[data-review]");
    const k = e.key.toLowerCase();
    if (k === "y" || k === "n" || k === "s") {
      const open = $$(".decide").find((d) => !d.dataset.decision && !d.closest("[hidden]") && d.offsetParent !== null);
      if (!open) return;
      const btn = open.querySelector(`[data-answer="${{ y: "pass", n: "fail", s: "skip" }[k]}"]`);
      if (btn) { e.preventDefault(); btn.click(); }
      return;
    }
    if (!review) return;
    if (k === "j") { const a = $("[data-act=next]"); if (a) { e.preventDefault(); location.href = a.href; } }
    else if (k === "k") { const a = $("[data-act=prev]"); if (a) { e.preventDefault(); location.href = a.href; } }
    else if (e.key === "Escape") { const a = $("[data-act=back]"); if (a) location.href = a.href; }
  });
  let toastTimer = null;
  function toast(text) {
    let t = $(".toast");
    if (!t) { t = document.createElement("div"); t.className = "toast"; t.setAttribute("role", "status"); document.body.appendChild(t); }
    t.textContent = text; t.hidden = false;
    clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.hidden = true; }, 6000);
  }
  if (document.querySelector("[data-review]")) wireDecisions(document.body);

  if (singleForm) {
    singleForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      // Check before sending: list what is missing, mark each input, focus the first one. Nothing typed is lost.
      clearMarks();
      const missing = $$("input[data-missing]", singleForm).filter((el) => !el.value.trim());
      const noImage = !sampleField.value && !(imageInput.files && imageInput.files.length);
      if (missing.length || noImage) {
        missing.forEach((el) => markField(el.name, el.dataset.missing));
        const parts = [];
        if (missing.length) parts.push("Please fill in: " + missing.map((el) => `<a href="#${el.id}" data-focus="${el.id}">${escapeHtml(labelText(el))}</a>`).join(", ") + ".");
        if (noImage) {
          setDropError("The label image is missing", `Drop it here or choose a file · PNG, JPG, TIFF or WEBP · up to ${maxMb} MB`);
          parts.push("Please add a label image (PNG, JPG, TIFF or WEBP), or pick a sample.");
        }
        showSingleError(alertCard("We couldn't run this check.", parts.join(" ")));
        (missing[0] || imageInput).focus();
        return;
      }
      const ok = await post(singleForm.action, new FormData(singleForm), resultSlot, $("#single-submit"));
      resultSlot.hidden = false;
      if (ok && resultSlot.querySelector(".result")) {
        entry.hidden = true;
        wireResult(resultSlot);
        focusCard(resultSlot);
        return;
      }
      entry.hidden = false;   // keep the form in view next to the error
      const first = applyErrorMarks(resultSlot);
      const retry = resultSlot.querySelector("[data-act=retry]");
      if (retry) retry.addEventListener("click", () => singleForm.requestSubmit());
      if (first) first.focus(); else focusCard(resultSlot);
    });
  }

  // --- batch ----------------------------------------------------------------------------------------
  const batchForm = $("#batch-form");
  const batchEntry = $("#batch-entry");
  const batchSlot = $("#batch-result");
  let pollTimer = null;
  let reconnecting = false;

  // drop zones for the CSV and the images (the real inputs stay the accessible controls)
  $$("[data-drop]").forEach((zone) => {
    const input = zone.querySelector("input[type=file]");
    const label = zone.querySelector("b");
    const idle = label.textContent;
    const show = () => { if (input.files.length) { label.textContent = input.files.length === 1 ? input.files[0].name : `${input.files.length} files chosen`; zone.classList.add("chosen"); } else { label.textContent = idle; zone.classList.remove("chosen"); } };
    input.addEventListener("change", show);
    ["dragenter", "dragover"].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("over"); }));
    ["dragleave", "drop"].forEach((ev) => zone.addEventListener(ev, () => zone.classList.remove("over")));
    zone.addEventListener("drop", (e) => {
      e.preventDefault();
      if (!e.dataTransfer || !e.dataTransfer.files.length) return;
      try { input.files = e.dataTransfer.files; } catch (err) { /* older browsers */ }
      show();
    });
  });

  function showBatchEntry(show) { batchEntry.hidden = !show; batchSlot.hidden = show; }

  function wireBatch() {
    const box = batchSlot.querySelector("[data-job]");
    if (!box) return;
    if (box.dataset.status === "running") {
      pollTimer = setTimeout(async () => {
        const scroller = batchSlot.querySelector(".table-scroll");
        const keep = scroller ? scroller.scrollTop : 0;
        try {
          const resp = await fetch(`/batch/${box.dataset.job}`, { headers: { "X-Partial": "1" } });
          // 404: the job is gone (restart), show the app's message. Other errors (a proxy during a
          // redeploy): keep the last state and try again on the next tick.
          if (resp.ok || resp.status === 404) {
            batchSlot.innerHTML = await cardFrom(resp);
            if (!resp.ok) focusCard(batchSlot);
            const again = batchSlot.querySelector(".table-scroll");
            if (again) again.scrollTop = keep;   // your place is kept while rows stream in
            reconnecting = false;
          } else reconnecting = true;
        } catch (err) { reconnecting = true; }   // network blip: say so, keep polling
        const note = batchSlot.querySelector("[data-reconnect]");
        if (note) note.hidden = !reconnecting;
        wireBatch();
      }, reconnecting ? 2000 : 1000);
      return;
    }
    if (box.dataset.wired) return;
    box.dataset.wired = "1";
    $$("body > [data-overlay]").forEach((o) => { if (!batchSlot.contains(o)) o.remove(); });   // a previous batch's
    focusCard(batchSlot);

    const rows = $$("tr.r", batchSlot);
    const search = $(".search", batchSlot);
    const sort = $(".sort", batchSlot);
    const tbody = batchSlot.querySelector("table.batch tbody");
    const foot = batchSlot.querySelector("[data-foot] td");
    const detail = $(".detail", batchSlot);
    const bulkbar = $(".bulkbar", batchSlot);
    const overlay = $("[data-overlay]", batchSlot);
    const word = { PASS: "Pass", REVIEW: "Review", FAIL: "Fail" };
    let active = "", cur = -1;

    function visible() { return rows.filter((r) => !r.hidden); }
    function apply() {
      const q = (search && search.value || "").trim().toLowerCase();
      rows.forEach((r) => { r.hidden = !((!active || r.dataset.status === active) && (!q || r.dataset.text.includes(q))); });
      const n = visible().length;
      if (foot) foot.textContent = n === 0 ? (q ? `No labels match "${q}"${active ? " in " + word[active] : ""}. Clear the search or pick another filter.` : "No labels in this list.") : (n === rows.length ? "End of list" : `${n} of ${rows.length} labels shown`);
      if (cur < 0 || rows[cur].hidden) {          // keep the detail panel on something visible
        const first = visible()[0];
        if (first) setCursor(rows.indexOf(first), false); else { setCursor(-1); if (detail) detail.hidden = true; }
      }
    }
    function setFilter(f) {
      active = f;
      $$(".fchip", batchSlot).forEach((b) => { const on = b.dataset.filter === f; b.classList.toggle("on", on); b.setAttribute("aria-pressed", on); });
      apply();
    }
    $$(".fchip", batchSlot).forEach((b) => b.addEventListener("click", () => setFilter(b.dataset.filter)));
    if (search) search.addEventListener("input", apply);
    if (sort) sort.addEventListener("change", () => {
      const key = sort.value;
      const order = { REVIEW: 0, FAIL: 1, ERROR: 2, PASS: 3 };
      const sorted = rows.slice().sort((a, b) => {
        if (key === "id") return a.dataset.id.localeCompare(b.dataset.id, undefined, { numeric: true });
        if (key === "brand") return a.dataset.brand.localeCompare(b.dataset.brand);
        if (key === "time") return (+b.dataset.time) - (+a.dataset.time);
        return (order[a.dataset.status] - order[b.dataset.status]) || (+a.dataset.index - +b.dataset.index);
      });
      sorted.forEach((r) => tbody.insertBefore(r, foot ? foot.parentElement : null));
    });

    function fillDetail(r) {
      if (!detail) return;
      detail.hidden = false;
      detail.querySelector("[data-d=id]").textContent = r.dataset.id;
      detail.querySelector("[data-d=brand]").textContent = r.dataset.brand || "—";
      const st = detail.querySelector("[data-d=status]");
      st.className = "chip " + ({ PASS: "pass", REVIEW: "review", FAIL: "fail" }[r.dataset.status] || "none");
      st.textContent = r.dataset.status;
      detail.querySelector("[data-d=summary]").textContent = r.dataset.summary;
      detail.querySelector("[data-d=file]").textContent = r.dataset.file || "—";
      detail.querySelector("[data-d=time]").textContent = r.dataset.time ? (r.dataset.time / 1000).toFixed(1) + " s" : "—";
      detail.querySelector("[data-d=needs]").textContent = r.dataset.needs || "—";
      const dd = detail.querySelector("[data-d=decision]"); if (dd) dd.textContent = r.dataset.decision || "—";
      const frame = detail.querySelector("[data-d=frame]"), img = detail.querySelector("[data-d=img]"), hl = detail.querySelector("[data-d=hl]");
      if (r.dataset.img) {
        img.src = r.dataset.img; img.alt = (r.dataset.brand || "Label") + " label"; frame.hidden = false;
        if (r.dataset.aspect) frame.style.aspectRatio = `1 / ${r.dataset.aspect}`;
        const b = r.dataset.box ? r.dataset.box.split(",").map(Number) : null;
        if (b && b.length === 4) { hl.hidden = false; hl.style.left = b[0] + "%"; hl.style.top = b[1] + "%"; hl.style.width = b[2] + "%"; hl.style.height = b[3] + "%"; hl.classList.toggle("fail", r.dataset.status === "FAIL"); }
        else hl.hidden = true;
      } else { frame.hidden = true; }
    }
    function setCursor(i, scroll) {
      rows.forEach((r) => { r.classList.remove("cur"); r.setAttribute("aria-selected", "false"); });
      cur = i;
      if (i < 0) return;
      rows[i].classList.add("cur"); rows[i].setAttribute("aria-selected", "true");
      if (scroll !== false) rows[i].scrollIntoView({ block: "nearest" });
      fillDetail(rows[i]);
    }
    function move(step) {
      const vis = visible();
      if (!vis.length) return;
      const pos = Math.max(0, vis.indexOf(rows[cur]));
      const next = vis[Math.min(vis.length - 1, Math.max(0, (cur < 0 ? (step > 0 ? -1 : 0) : pos) + step))];
      setCursor(rows.indexOf(next));
      next.focus({ preventScroll: true });
    }
    async function openFull(r) {
      if (!overlay || !r) return;
      // Out of <main> so that the page behind can be made inert while the overlay is open.
      if (overlay.parentElement !== document.body) document.body.appendChild(overlay);
      overlay.hidden = false;
      setPageInert(true);
      overlay.querySelector("[data-overlay-title]").textContent = `${r.dataset.id} · ${r.dataset.brand}`;
      const body = overlay.querySelector("[data-overlay-body]");
      body.innerHTML = '<p class="muted">Loading…</p>';
      try {
        const resp = await fetch(`/batch/${box.dataset.job}/item/${r.dataset.index}`, { headers: { "X-Partial": "1" } });
        body.innerHTML = await cardFrom(resp);
        wireResult(body);
      } catch (err) { body.innerHTML = errorCard("Could not load this result.", "Please try again."); }
      overlay.querySelector("[data-act=close]").focus();
    }
    function closeFull() {
      if (overlay && !overlay.hidden) { overlay.hidden = true; setPageInert(false); if (cur >= 0) rows[cur].focus(); }
    }

    rows.forEach((r, i) => {
      r.addEventListener("click", (e) => {
        if (e.target.type === "checkbox") { syncBulk(); return; }
        if (e.target.closest("a.rowlink")) e.preventDefault();   // the link is for pages without JavaScript
        setCursor(i, false);
      });
      r.addEventListener("dblclick", () => openFull(r));
      r.addEventListener("focus", () => { if (cur !== i) setCursor(i, false); });
      r.querySelector("input[type=checkbox]").addEventListener("change", syncBulk);
    });
    if (detail) detail.querySelector("[data-act=open]").addEventListener("click", () => openFull(rows[cur]));
    if (overlay) { overlay.querySelector("[data-act=close]").addEventListener("click", closeFull); overlay.addEventListener("click", (e) => { if (e.target === overlay) closeFull(); }); }

    function syncBulk() {
      const checked = rows.filter((r) => r.querySelector("input[type=checkbox]").checked);
      rows.forEach((r) => r.classList.toggle("sel", r.querySelector("input[type=checkbox]").checked));
      if (!bulkbar) return;
      bulkbar.hidden = checked.length === 0;
      bulkbar.querySelector("[data-d=count]").textContent = checked.length;
      const link = bulkbar.querySelector("[data-act=export-selected]");
      link.href = `/batch/${box.dataset.job}/export.csv?ids=` + encodeURIComponent(checked.map((r) => r.dataset.id).join(","));
      const rep = $("[data-act=report-selected]", batchSlot);
      if (rep) rep.href = `/batch/${box.dataset.job}/report?ids=` + encodeURIComponent(checked.map((r) => r.dataset.id).join(","));
    }
    const all = $(".select-all", batchSlot);
    if (all) all.addEventListener("change", () => { visible().forEach((r) => { r.querySelector("input[type=checkbox]").checked = all.checked; }); syncBulk(); });
    if (bulkbar) bulkbar.querySelector("[data-act=clear]").addEventListener("click", () => { rows.forEach((r) => { r.querySelector("input[type=checkbox]").checked = false; }); if (all) all.checked = false; syncBulk(); });
    // Export dialog (Batch-Export board): scope, include, format; quick exports; a toast when it goes.
    const dlg = $("dialog[data-export]", batchSlot);
    const jobId = box.dataset.job;
    const filterName = () => { const on = $(".fchip.on", batchSlot); return on && on.dataset.filter ? on.dataset.filter[0] + on.dataset.filter.slice(1).toLowerCase() : "All"; };
    function exportUrl(scope, include, fmt) {
      const p = new URLSearchParams();
      const on = $(".fchip.on", batchSlot), q = (search && search.value.trim()) || "";
      if (scope === "selected") p.set("ids", rows.filter((r) => r.querySelector("input[type=checkbox]").checked).map((r) => r.dataset.id).join(","));
      else if (scope === "shown") { if (on && on.dataset.filter && !q) p.set("status", on.dataset.filter); else p.set("ids", visible().map((r) => r.dataset.id).join(",")); }
      if (fmt === "reports") return `/batch/${jobId}/report?${p}`;
      p.set("include", include.join(","));
      return `/batch/${jobId}/export.csv?${p}`;
    }
    function openExport(scope) {
      if (!dlg) { location.href = `/batch/${jobId}/export.csv`; return; }
      const selected = rows.filter((r) => r.querySelector("input[type=checkbox]").checked).length;
      $("[data-selected-count]", dlg).textContent = selected;
      $("[data-shown-label]", dlg).textContent = filterName();
      $("[data-shown-count]", dlg).textContent = visible().length;
      const sel = $('input[name=scope][value=selected]', dlg); sel.disabled = !selected;
      const want = $(`input[name=scope][value=${scope || (selected ? "selected" : "all")}]`, dlg);
      if (want && !want.disabled) want.checked = true; else $('input[name=scope][value=all]', dlg).checked = true;
      syncExport();
      dlg.showModal();
    }
    function syncExport() {
      if (!dlg) return;
      $$(".opt", dlg).forEach((o) => o.classList.toggle("on", o.querySelector("input").checked));
      const scope = $("input[name=scope]:checked", dlg).value, fmt = $("input[name=fmt]:checked", dlg).value;
      const count = scope === "all" ? rows.length : scope === "shown" ? visible().length : rows.filter((r) => r.querySelector("input[type=checkbox]").checked).length;
      const stamp = (box.dataset.date || new Date().toISOString().slice(0, 10));
      const scopeName = scope === "all" ? "all" : scope === "selected" ? "selected" : filterName().toLowerCase();
      $("[data-export-name]", dlg).textContent = `label-check_${box.dataset.slug || "batch"}_${scopeName}_${stamp}.${fmt === "reports" ? "html" : "csv"} · ${count} row${count === 1 ? "" : "s"}`;
      $("[data-act=download]", dlg).textContent = fmt === "reports" ? "Open the reports" : "Download CSV";
      $$(".ck input", dlg).forEach((c) => { c.disabled = fmt === "reports"; });
    }
    if (dlg) {
      dlg.addEventListener("change", syncExport);
      $$("[data-act=close]", dlg).forEach((b) => b.addEventListener("click", () => dlg.close()));
      $("[data-export-form]", dlg).addEventListener("submit", (e) => {
        e.preventDefault();
        const scope = $("input[name=scope]:checked", dlg).value, fmt = $("input[name=fmt]:checked", dlg).value;
        const include = $$(".ck input:checked", dlg).map((c) => c.value);
        const url = exportUrl(scope, include, fmt);
        const count = $("[data-export-name]", dlg).textContent.split("·")[1].trim();
        if (fmt === "reports") window.open(url, "_blank", "noopener"); else location.href = url;
        dlg.close();
        toast(fmt === "reports" ? `Opened ${count} as printable reports` : `Exported ${count} as ${$("[data-export-name]", dlg).textContent.split("·")[0].trim()}`);
      });
      $$("[data-quick]", dlg).forEach((a) => a.addEventListener("click", () => { dlg.close(); toast(`Exported the labels marked ${a.dataset.quick}`); }));
    }
    $$("[data-act=export]", batchSlot).forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); openExport(); }));
    const exportSel = $("[data-act=export-selected]", batchSlot);
    if (exportSel && dlg) exportSel.addEventListener("click", (e) => { e.preventDefault(); openExport("selected"); });
    const reviewLink = $("[data-act=review]", batchSlot);
    const newBatch = $("[data-act=newbatch]", batchSlot);
    if (newBatch) newBatch.addEventListener("click", (e) => { e.preventDefault(); showBatchEntry(true); batchForm.reset(); $$("[data-drop] b").forEach((b) => { b.textContent = b.closest("[data-drop]").dataset.drop === "csv" ? "Drop the CSV here, or choose a file" : "Drop the images or a zip here"; }); window.scrollTo({ top: 0, behavior: reduced ? "auto" : "smooth" }); });

    box.addEventListener("keydown", (e) => {
      if (e.target.matches("input, select, textarea")) return;
      if (overlay && !overlay.hidden) { if (e.key === "Escape") { e.preventDefault(); closeFull(); } return; }
      const k = e.key;
      if (k === "j" || k === "ArrowDown") { e.preventDefault(); move(1); }
      else if (k === "k" || k === "ArrowUp") { e.preventDefault(); move(-1); }
      else if (k === "x" && cur >= 0) { e.preventDefault(); const c = rows[cur].querySelector("input[type=checkbox]"); c.checked = !c.checked; syncBulk(); }
      else if (k === "Enter" && cur >= 0) { e.preventDefault(); openFull(rows[cur]); }
      else if (k === "r" && reviewLink) { e.preventDefault(); location.href = reviewLink.href; }
      else if (k === "1") setFilter("REVIEW"); else if (k === "2") setFilter("FAIL"); else if (k === "3") setFilter("PASS"); else if (k === "4" && rows.some((r) => r.dataset.status === "ERROR")) setFilter("ERROR"); else if (k === "0") setFilter("");
      else if (k === "Escape" && cur >= 0) { setCursor(-1); if (detail) detail.hidden = true; }
    });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeFull(); });

    // Open on what needs attention first.
    if ($$(".fchip.review", batchSlot).some((b) => !b.disabled) && rows.some((r) => r.dataset.status === "REVIEW")) setFilter("REVIEW");
    const first = visible()[0];
    if (first) setCursor(rows.indexOf(first), false);
  }

  async function startBatch(body, button) {
    clearTimeout(pollTimer);
    batchSlot.hidden = false;
    const ok = await post("/batch", body, batchSlot, button);
    if (ok) batchEntry.hidden = true;
    else {
      batchEntry.hidden = false;
      const retry = batchSlot.querySelector("[data-act=retry]");
      if (retry) retry.addEventListener("click", () => startBatch(body, button));
      focusCard(batchSlot);
    }
    wireBatch();
  }
  // A batch card the server rendered into the page (a shared /batch/<id> link): wire it like one just fetched.
  if (batchSlot && batchSlot.querySelector("[data-job]")) wireBatch();
  if (batchForm) {
    batchForm.addEventListener("submit", (e) => { e.preventDefault(); startBatch(new FormData(batchForm), batchForm.querySelector('button[type="submit"]')); });
    const sampleBtn = $("#sample-batch");
    if (sampleBtn) sampleBtn.addEventListener("click", (e) => {
      e.preventDefault();   // a submit button, so the sample batch also starts without JavaScript
      const fd = new FormData();
      fd.append("sample", "1");
      const reader = batchForm.querySelector('[name="reader"]:checked');
      if (reader) fd.append("reader", reader.value);
      startBatch(fd, sampleBtn);
    });
  }
})();
