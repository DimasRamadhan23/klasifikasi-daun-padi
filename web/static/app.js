(() => {
  const $ = (id) => document.getElementById(id);
  const MAX_BYTES = 10 * 1024 * 1024;
  const ALLOWED = ["image/jpeg", "image/png"];

  const input = $("file-input");
  const dropzone = $("dropzone");
  const preview = $("preview");
  const dzEmpty = $("dz-empty");
  const btnAnalyze = $("btn-analyze");
  const btnReset = $("btn-reset");
  const errorEl = $("error");
  const resultSec = $("hasil");

  let file = null;
  let previewUrl = null;
  let modelReady = false;

  const pct = (p) => (p * 100).toLocaleString("id-ID", { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + "%";
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function showError(msg) {
    errorEl.textContent = msg;
    errorEl.hidden = !msg;
  }

  function updateButtons() {
    btnAnalyze.disabled = !file || !modelReady;
    btnReset.disabled = !file;
  }

  function setFile(f) {
    showError("");
    if (!f) return;
    if (!ALLOWED.includes(f.type)) return showError("Format tidak didukung. Gunakan JPG, JPEG, atau PNG.");
    if (f.size > MAX_BYTES) return showError("Ukuran file melebihi 10 MB.");
    file = f;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(f);
    preview.src = previewUrl;
    preview.hidden = false;
    dzEmpty.hidden = true;
    updateButtons();
  }

  function reset() {
    file = null;
    input.value = "";
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    preview.hidden = true;
    preview.removeAttribute("src");
    dzEmpty.hidden = false;
    resultSec.hidden = true;
    showError("");
    updateButtons();
  }

  input.addEventListener("change", () => setFile(input.files[0]));
  btnReset.addEventListener("click", reset);

  ["dragenter", "dragover"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); }));
  dropzone.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

  // Tempel gambar dari clipboard (Cmd/Ctrl + V)
  window.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) setFile(item.getAsFile());
  });

  btnAnalyze.addEventListener("click", async () => {
    if (!file) return;
    showError("");
    btnAnalyze.classList.add("loading");
    btnAnalyze.disabled = true;
    btnAnalyze.querySelector(".btn-label").textContent = "Menganalisis…";

    try {
      const fd = new FormData();
      fd.append("image", file, file.name || "foto.png");
      const res = await fetch("/api/predict", { method: "POST", body: fd });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || `Terjadi kesalahan (HTTP ${res.status}).`);
      render(data);
    } catch (err) {
      showError(err.message || "Gagal menghubungi server.");
    } finally {
      btnAnalyze.classList.remove("loading");
      btnAnalyze.querySelector(".btn-label").textContent = "Analisis";
      updateButtons();
    }
  });

  function render(d) {
    $("res-label").textContent = d.label_id;
    $("res-latin").textContent = d.info.latin;
    $("res-conf").textContent = pct(d.confidence);
    $("low-conf").hidden = !d.low_confidence;
    $("res-input").src = d.input_image;
    $("res-cam").src = d.gradcam;
    $("res-time").textContent = `Waktu proses: ${d.elapsed_ms} ms`;

    $("res-probs").innerHTML = d.probabilities.map((p, i) => `
      <div class="prob${i === 0 ? " top" : ""}">
        <div class="prob-row"><span>${esc(p.name)} <code>${esc(p.class)}</code></span><span>${pct(p.probability)}</span></div>
        <div class="bar"><i data-w="${(p.probability * 100).toFixed(2)}"></i></div>
      </div>`).join("");

    const info = d.info;
    const list = (items) => items.map((s) => `<li>${esc(s)}</li>`).join("");
    $("res-info").innerHTML = `
      <h3>Tentang ${esc(info.name)} <span class="tag">${esc(info.type)}</span></h3>
      <p class="muted">${esc(info.summary)}</p>
      <div class="info-cols">
        <div><h4>Gejala yang perlu dicocokkan</h4><ul>${list(info.symptoms)}</ul></div>
        <div><h4>Penanganan umum</h4><ul>${list(info.management)}</ul></div>
      </div>
      <p class="muted" style="margin-top:14px">Cocokkan gejala di atas dengan kondisi tanaman di lapangan sebelum mengambil tindakan.</p>`;

    resultSec.hidden = false;
    requestAnimationFrame(() => {
      document.querySelectorAll("#res-probs .bar > i").forEach((el) => { el.style.width = el.dataset.w + "%"; });
    });
    resultSec.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // Cek status model saat halaman dibuka
  fetch("/api/health")
    .then((r) => r.json())
    .then((h) => {
      modelReady = !!h.model_loaded;
      if (!modelReady) {
        const b = $("model-banner");
        b.textContent = h.error || "Model belum dimuat di server.";
        b.hidden = false;
      }
      updateButtons();
    })
    .catch(() => showError("Server tidak dapat dihubungi."));
})();
