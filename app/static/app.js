/* Small, dependency-free UI glue: tabs, form submission, sample prefill, batch polling and filters. */
(function () {
  "use strict";
  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

  // --- tabs -----------------------------------------------------------------------------------
  $$(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      $$(".tab").forEach((t) => { t.classList.toggle("active", t === tab); t.setAttribute("aria-selected", t === tab); });
      $$(".panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + tab.dataset.tab));
      try { localStorage.setItem("labelcheck.tab", tab.dataset.tab); } catch (e) { /* ignore */ }
    });
  });
  try {
    const saved = localStorage.getItem("labelcheck.tab");
    const tab = saved && $(`.tab[data-tab="${saved}"]`);
    if (tab) tab.click();
  } catch (e) { /* ignore */ }

  // --- helpers --------------------------------------------------------------------------------
  async function postForm(form, slot) {
    slot.classList.add("busy");
    const submit = form.querySelector('button[type="submit"]');
    if (submit) submit.disabled = true;
    try {
      const resp = await fetch(form.action, { method: "POST", body: new FormData(form) });
      slot.innerHTML = await resp.text();
    } catch (err) {
      slot.innerHTML = '<div class="card error" role="alert"><strong>We couldn\'t reach the server.</strong><p>Check your connection and try again.</p></div>';
    } finally {
      slot.classList.remove("busy");
      if (submit) submit.disabled = false;
    }
  }

  // --- single label ---------------------------------------------------------------------------
  const singleForm = $("#single-form");
  const samples = (() => { try { return JSON.parse($("#samples-data").textContent); } catch (e) { return []; } })();
  const picker = $("#sample-picker");
  const imageInput = $("#image-input");
  const preview = $("#preview");

  function showPreview(src) {
    if (!src) { preview.hidden = true; preview.querySelector("img").src = ""; return; }
    preview.querySelector("img").src = src;
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
          batchSlot.innerHTML = await resp.text();
        } catch (err) { /* keep the last state, retry next tick */ }
        wireBatch();
      }, 1000);
      return;
    }
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
      $$(".filter", batchSlot).forEach((b) => b.classList.toggle("active", b === btn));
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
          d.firstElementChild.innerHTML = await resp.text();
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
      batchSlot.classList.add("busy");
      sampleBtn.disabled = true;
      try {
        const resp = await fetch("/batch", { method: "POST", body: fd });
        batchSlot.innerHTML = await resp.text();
      } finally {
        batchSlot.classList.remove("busy");
        sampleBtn.disabled = false;
      }
      wireBatch();
    });
  }
})();
