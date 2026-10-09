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
  try {
    const saved = localStorage.getItem("labelcheck.tab");
    const tab = saved && $(`.modes .tab[data-tab="${saved}"]`);
    if (tab) selectTab(tab, false);
  } catch (e) { /* ignore */ }

  // --- help drawer ----------------------------------------------------------------------------------
  const helpBtn = $("#help-btn"), drawer = $("#help-drawer"), scrim = $("#help-scrim");
  let lastFocus = null;
  function openHelp() {
    lastFocus = document.activeElement;
    drawer.hidden = false; scrim.hidden = false;
    helpBtn.setAttribute("aria-expanded", "true");
    $("#help-close").focus();
  }
  function closeHelp() {
    drawer.hidden = true; scrim.hidden = true;
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
  function errorCard(title, detail) {
    return `<div class="card alert err" role="alert" tabindex="-1" data-app-card><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 8v5"/><path d="M12 16h.01"/></svg><div><strong>${escapeHtml(title)}</strong><div class="detail">${escapeHtml(detail)}</div></div></div>`;
  }
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
      slot.innerHTML = errorCard("We couldn't reach the server.", "Check your connection and try again.");
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
  const samples = (() => { try { return JSON.parse($("#samples-data").textContent); } catch (e) { return []; } })();
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
  function fileChosen(file) {
    picker.value = ""; sampleField.value = ""; pressTile(""); currentSample = null;
    drop.classList.remove("err");
    if (!file) { showPreview(null); return; }
    if (!/^image\//.test(file.type) && !/\.(png|jpe?g|tiff?|bmp|webp)$/i.test(file.name)) {
      drop.classList.add("err");
      $("#drop b").textContent = "That isn't an image";
      imageInput.value = "";
      return;
    }
    $("#drop b").textContent = "Drop the label here, or choose a file";
    showPreview(URL.createObjectURL(file), file.name, `${file.type || "image"} · ${fmtBytes(file.size)}`);
  }
  if (imageInput) imageInput.addEventListener("change", () => fileChosen(imageInput.files[0]));
  if (drop) {
    ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); $("#drop b").textContent = "Release to add the label"; }));
    ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, () => { drop.classList.remove("over"); $("#drop b").textContent = "Drop the label here, or choose a file"; }));
    drop.addEventListener("drop", (e) => {
      e.preventDefault();
      const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (!file) return;
      try { const dt = new DataTransfer(); dt.items.add(file); imageInput.files = dt.files; } catch (err) { /* older browsers */ }
      fileChosen(file);
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
    $$("[data-act]", slot).forEach((b) => b.addEventListener("click", () => {
      const act = b.dataset.act;
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

  if (singleForm) {
    singleForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const ok = await post(singleForm.action, new FormData(singleForm), resultSlot, $("#single-submit"));
      resultSlot.hidden = false;
      if (ok && resultSlot.querySelector(".result")) {
        entry.hidden = true;
        wireResult(resultSlot);
      } else {
        entry.hidden = false;   // keep the form in view next to the error
      }
      focusCard(resultSlot);
    });
  }

  // --- batch ----------------------------------------------------------------------------------------
  const batchForm = $("#batch-form");
  const batchEntry = $("#batch-entry");
  const batchSlot = $("#batch-result");
  let pollTimer = null;

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
          }
        } catch (err) { /* network blip: retry next tick */ }
        wireBatch();
      }, 1000);
      return;
    }
    if (box.dataset.wired) return;
    box.dataset.wired = "1";
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
      const order = { REVIEW: 0, FAIL: 1, PASS: 2 };
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
      st.className = "chip " + { PASS: "pass", REVIEW: "review", FAIL: "fail" }[r.dataset.status];
      st.textContent = r.dataset.status;
      detail.querySelector("[data-d=summary]").textContent = r.dataset.summary;
      detail.querySelector("[data-d=file]").textContent = r.dataset.file || "—";
      detail.querySelector("[data-d=time]").textContent = r.dataset.time ? (r.dataset.time / 1000).toFixed(1) + " s" : "—";
      detail.querySelector("[data-d=needs]").textContent = r.dataset.needs || "—";
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
      overlay.hidden = false;
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
    function closeFull() { if (overlay && !overlay.hidden) { overlay.hidden = true; if (cur >= 0) rows[cur].focus(); } }

    rows.forEach((r, i) => {
      r.addEventListener("click", (e) => { if (e.target.type === "checkbox") { syncBulk(); return; } setCursor(i, false); });
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
    }
    const all = $(".select-all", batchSlot);
    if (all) all.addEventListener("change", () => { visible().forEach((r) => { r.querySelector("input[type=checkbox]").checked = all.checked; }); syncBulk(); });
    if (bulkbar) bulkbar.querySelector("[data-act=clear]").addEventListener("click", () => { rows.forEach((r) => { r.querySelector("input[type=checkbox]").checked = false; }); if (all) all.checked = false; syncBulk(); });
    const newBatch = $("[data-act=newbatch]", batchSlot);
    if (newBatch) newBatch.addEventListener("click", () => { showBatchEntry(true); batchForm.reset(); $$("[data-drop] b").forEach((b) => { b.textContent = b.closest("[data-drop]").dataset.drop === "csv" ? "Drop the CSV here, or choose a file" : "Drop the images or a zip here"; }); window.scrollTo({ top: 0, behavior: reduced ? "auto" : "smooth" }); });

    box.addEventListener("keydown", (e) => {
      if (e.target.matches("input, select, textarea")) return;
      if (overlay && !overlay.hidden) { if (e.key === "Escape") { e.preventDefault(); closeFull(); } return; }
      const k = e.key;
      if (k === "j" || k === "ArrowDown") { e.preventDefault(); move(1); }
      else if (k === "k" || k === "ArrowUp") { e.preventDefault(); move(-1); }
      else if (k === "x" && cur >= 0) { e.preventDefault(); const c = rows[cur].querySelector("input[type=checkbox]"); c.checked = !c.checked; syncBulk(); }
      else if (k === "Enter" && cur >= 0) { e.preventDefault(); openFull(rows[cur]); }
      else if (k === "1") setFilter("REVIEW"); else if (k === "2") setFilter("FAIL"); else if (k === "3") setFilter("PASS"); else if (k === "0") setFilter("");
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
    if (ok) batchEntry.hidden = true; else { batchEntry.hidden = false; focusCard(batchSlot); }
    wireBatch();
  }
  if (batchForm) {
    batchForm.addEventListener("submit", (e) => { e.preventDefault(); startBatch(new FormData(batchForm), batchForm.querySelector('button[type="submit"]')); });
    const sampleBtn = $("#sample-batch");
    if (sampleBtn) sampleBtn.addEventListener("click", () => {
      const fd = new FormData();
      fd.append("sample", "1");
      const reader = batchForm.querySelector('[name="reader"]:checked');
      if (reader) fd.append("reader", reader.value);
      startBatch(fd, sampleBtn);
    });
  }
})();
