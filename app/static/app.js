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
  function pressTile(name) { $$(".thumb[data-sample]").forEach((b) => b.setAttribute("aria-pressed", b.dataset.sample === name)); }
  function applySample(name) {
    const s = samples.find((x) => x.name === name);
    sampleField.value = s ? s.name : "";
    pressTile(s ? s.name : "");
    if (!s) { showPreview(null); return; }
    ["brand_name", "class_type", "alcohol_content", "net_contents", "bottler_name_address", "country_of_origin", "application_id"].forEach((k) => {
      const el = singleForm.querySelector(`[name="${k}"]`);
      if (el) el.value = s[k] || "";
    });
    imageInput.value = "";
    showPreview(`/samples/${s.name}.png`, s.short + " (sample)", s.kind);
  }
  if (picker) picker.addEventListener("change", () => applySample(picker.value));
  $$(".thumb[data-sample]").forEach((b) => b.addEventListener("click", () => { picker.value = b.dataset.sample; applySample(b.dataset.sample); }));
  const more = $("#more-samples");
  if (more) more.addEventListener("click", () => {
    const box = $("#gallery-defects");
    box.hidden = !box.hidden;
    more.setAttribute("aria-expanded", String(!box.hidden));
  });
  function fileChosen(file) {
    picker.value = ""; sampleField.value = ""; pressTile("");
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
  if (clearImage) clearImage.addEventListener("click", () => { imageInput.value = ""; picker.value = ""; sampleField.value = ""; pressTile(""); showPreview(null); });

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

  // --- batch ------------------------------------------------------------------------------------------
  const batchForm = $("#batch-form");
  const batchSlot = $("#batch-result");
  let pollTimer = null;

  function wireBatch() {
    const box = batchSlot.querySelector("[data-job]");
    if (!box) return;
    if (box.dataset.status === "running") {
      pollTimer = setTimeout(async () => {
        try {
          const resp = await fetch(`/batch/${box.dataset.job}`, { headers: { "X-Partial": "1" } });
          // 404: the job is gone (restart), show the app's message. Other errors (a proxy during a
          // redeploy): keep the last state and try again on the next tick.
          if (resp.ok || resp.status === 404) {
            const html = await cardFrom(resp);
            batchSlot.innerHTML = html;
            if (!resp.ok) focusCard(batchSlot);
          }
        } catch (err) { /* network blip: retry next tick */ }
        wireBatch();
      }, 1000);
      return;
    }
    if (!box.dataset.wired) { box.dataset.wired = "1"; focusCard(batchSlot); }
    const rows = $$("tr.row", batchSlot);
    const search = $(".search", batchSlot);
    let active = "";
    function apply() {
      const q = (search && search.value || "").trim().toLowerCase();
      rows.forEach((r) => {
        const ok = (!active || r.dataset.status === active) && (!q || r.dataset.text.includes(q));
        r.hidden = !ok;
        const d = r.nextElementSibling;
        if (d && d.classList.contains("detail")) d.hidden = !ok || d.dataset.closed === "1";
      });
    }
    $$(".filter", batchSlot).forEach((btn) => btn.addEventListener("click", () => {
      active = btn.dataset.filter;
      $$(".filter", batchSlot).forEach((b) => { b.classList.toggle("active", b === btn); b.setAttribute("aria-pressed", b === btn); });
      apply();
    }));
    if (search) search.addEventListener("input", apply);
    rows.forEach((r) => {
      const open = async () => {
        let d = r.nextElementSibling;
        if (d && d.classList.contains("detail")) { d.hidden = !d.hidden; d.dataset.closed = d.hidden ? "1" : "0"; return; }
        d = document.createElement("tr");
        d.className = "detail";
        d.innerHTML = `<td colspan="${r.children.length}">Loading…</td>`;
        r.after(d);
        try {
          const resp = await fetch(`/batch/${box.dataset.job}/item/${r.dataset.index}`, { headers: { "X-Partial": "1" } });
          d.firstElementChild.innerHTML = await cardFrom(resp);
          wireResult(d.firstElementChild);
        } catch (err) { d.firstElementChild.textContent = "Could not load this result."; }
      };
      r.addEventListener("click", open);
      r.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
    });
  }

  if (batchForm) {
    batchForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      clearTimeout(pollTimer);
      await post(batchForm.action, new FormData(batchForm), batchSlot, batchForm.querySelector('button[type="submit"]'));
      wireBatch();
    });
    const sampleBtn = $("#sample-batch");
    if (sampleBtn) sampleBtn.addEventListener("click", async () => {
      clearTimeout(pollTimer);
      const fd = new FormData();
      fd.append("sample", "1");
      const reader = batchForm.querySelector('[name="reader"]:checked');
      if (reader) fd.append("reader", reader.value);
      await post("/batch", fd, batchSlot, sampleBtn);
      wireBatch();
    });
  }
})();
