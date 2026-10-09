/* Small, dependency-free UI glue: tabs, form submission, sample prefill, batch polling and filters. */
(function () {
  "use strict";
  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

  // --- tabs -----------------------------------------------------------------------------------
  const tabs = $$(".tab");
  tabs.forEach((tab, i) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => {
        t.classList.toggle("active", t === tab);
        t.setAttribute("aria-selected", t === tab);
        t.tabIndex = t === tab ? 0 : -1;
      });
      $$(".panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + tab.dataset.tab));
      try { localStorage.setItem("labelcheck.tab", tab.dataset.tab); } catch (e) { /* ignore */ }
    });
    tab.addEventListener("keydown", (e) => {  // arrow keys move between tabs (WAI-ARIA tabs pattern)
      const step = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
      if (!step) return;
      const next = tabs[(i + step + tabs.length) % tabs.length];
      next.click();
      next.focus();
    });
  });
  try {
    const saved = localStorage.getItem("labelcheck.tab");
    const tab = saved && $(`.tab[data-tab="${saved}"]`);
    if (tab) tab.click();
  } catch (e) { /* ignore */ }

  // --- helpers --------------------------------------------------------------------------------
  function errorCard(title, detail) {
    return `<div class="card error" role="alert" tabindex="-1"><strong>${title}</strong><p>${detail}</p></div>`;
  }

  // The app answers every request with an HTML card, errors included. Anything else (a proxy's
  // error page during a redeploy, say) must not be pasted into the page.
  async function cardFrom(resp) {
    const text = await resp.text();
    if (text.trim().startsWith("<div class=\"card")) return text;
    return errorCard("The server did not answer properly.", `Please try again in a moment (HTTP ${resp.status}).`);
  }

  function show(slot, html) {
    slot.innerHTML = html;
    const card = slot.querySelector(".card");
    if (card && card.hasAttribute("tabindex")) card.focus();  // keyboard and screen-reader users land on the result
  }

  async function post(url, body, slot, button) {
    slot.classList.add("busy");
    if (button) button.disabled = true;
    try {
      const resp = await fetch(url, { method: "POST", body });
      show(slot, await cardFrom(resp));
    } catch (err) {
      show(slot, errorCard("We couldn't reach the server.", "Check your connection and try again."));
    } finally {
      slot.classList.remove("busy");
      if (button) button.disabled = false;
    }
  }

  function postForm(form, slot) {
    return post(form.action, new FormData(form), slot, form.querySelector('button[type="submit"]'));
  }

  // --- single label ---------------------------------------------------------------------------
  const singleForm = $("#single-form");
  const samples = (() => { try { return JSON.parse($("#samples-data").textContent); } catch (e) { return []; } })();
  const picker = $("#sample-picker");
  const imageInput = $("#image-input");
  const preview = $("#preview");

  function showPreview(src) {
    const img = preview.querySelector("img");
    if (img.src.startsWith("blob:")) URL.revokeObjectURL(img.src);
    if (!src) { preview.hidden = true; img.removeAttribute("src"); return; }
    img.src = src;
    preview.hidden = false;
  }

  if (picker) {
    picker.addEventListener("change", () => {
      const s = samples.find((x) => x.name === picker.value);
      if (!s) { showPreview(null); return; }
      ["brand_name", "class_type", "alcohol_content", "net_contents", "bottler_name_address", "country_of_origin", "application_id"].forEach((k) => {
        const el = singleForm.querySelector(`[name="${k}"]`);
        if (el) el.value = s[k] || "";
      });
      imageInput.value = "";
      showPreview(`/samples/${s.name}.png`);
    });
  }
  if (imageInput) {
    imageInput.addEventListener("change", () => {
      if (!imageInput.files.length) { showPreview(null); return; }
      picker.value = "";
      showPreview(URL.createObjectURL(imageInput.files[0]));
    });
  }
  const clearImage = $("#clear-image");
  if (clearImage) clearImage.addEventListener("click", () => { imageInput.value = ""; picker.value = ""; showPreview(null); });

  if (singleForm) {
    singleForm.addEventListener("submit", (e) => {
      e.preventDefault();
      postForm(singleForm, $("#result"));
    });
  }

  // --- batch ----------------------------------------------------------------------------------
  const batchForm = $("#batch-form");
  const batchSlot = $("#batch-result");
  let pollTimer = null;

  function wireBatch() {
    const box = batchSlot.querySelector("[data-job]");
    if (!box) return;
    if (box.dataset.status === "running") {
      pollTimer = setTimeout(async () => {
        try {
          const resp = await fetch(`/batch/${box.dataset.job}`);
          // 404: the job is gone (restart), show the app's message. Other errors (a proxy during a
          // redeploy): keep the last state and try again on the next tick.
          if (resp.ok || resp.status === 404) {
            const html = await cardFrom(resp);
            if (resp.ok) batchSlot.innerHTML = html; else show(batchSlot, html);
          }
        } catch (err) { /* network blip: retry next tick */ }
        wireBatch();
      }, 1000);
      return;
    }
    if (!box.dataset.wired) { box.dataset.wired = "1"; box.focus(); }
    // filters
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
      $$(".filter", batchSlot).forEach((b) => {
        b.classList.toggle("active", b === btn);
        b.setAttribute("aria-pressed", b === btn);
      });
      apply();
    }));
    if (search) search.addEventListener("input", apply);
    // row expansion
    rows.forEach((r) => {
      const open = async () => {
        let d = r.nextElementSibling;
        if (d && d.classList.contains("detail")) {
          d.hidden = !d.hidden; d.dataset.closed = d.hidden ? "1" : "0"; return;
        }
        d = document.createElement("tr");
        d.className = "detail";
        d.innerHTML = `<td colspan="${r.children.length}">Loading…</td>`;
        r.after(d);
        try {
          const resp = await fetch(`/batch/${box.dataset.job}/item/${r.dataset.index}`);
          d.firstElementChild.innerHTML = await cardFrom(resp);
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
      await postForm(batchForm, batchSlot);
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
