// Halaman Deteksi: memuat model TensorFlow.js dan menjalankan prediksi di browser.
// Model dibuat oleh tools/konversi_model.py dan diletakkan di folder model/.
// Model menerima piksel 0-255 ukuran 224x224 (preprocess_input sudah di dalam model)
// dan mengeluarkan "probabilities" serta "gradcam".
(() => {
  const MODEL_URL = "model/model.json";
  const META_URL = "model/metadata.json";
  const IMG_SIZE = 224;
  const LOW_CONFIDENCE = 0.7;
  const MAX_BYTES = 10 * 1024 * 1024;
  const ALLOWED = ["image/jpeg", "image/png"];

  const $ = (id) => document.getElementById(id);
  const input = $("file-input");
  const dropzone = $("dropzone");
  const preview = $("preview");
  const dzEmpty = $("dz-empty");
  const btnAnalyze = $("btn-analyze");
  const btnReset = $("btn-reset");
  const errorEl = $("error");
  const resultSec = $("hasil");
  const statusDot = $("status-dot");
  const statusText = $("status-text");
  const progress = $("progress");
  const banner = $("model-banner");

  let model = null;
  let classes = [];
  let file = null;
  let previewUrl = null;

  const pct = (p) =>
    (p * 100).toLocaleString("id-ID", { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + "%";
  const esc = (s) =>
    String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

  function setStatus(state, text) {
    statusDot.className = "dot " + state;
    statusText.textContent = text;
  }
  function showError(msg) {
    errorEl.textContent = msg;
    errorEl.hidden = !msg;
  }
  function showBanner(html) {
    banner.innerHTML = html;
    banner.hidden = false;
  }
  function updateButtons() {
    btnAnalyze.disabled = !file || !model;
    btnReset.disabled = !file;
  }

  // ---------------------------------------------------------------- model
  async function loadModel() {
    if (location.protocol === "file:") {
      setStatus("err", "Model tidak bisa dimuat");
      showBanner(
        "<p><b>Halaman dibuka langsung sebagai file.</b> Browser memblokir pemuatan model dari <code>file://</code>.</p>" +
          "<p>Jalankan lewat server lokal, misalnya klik kanan <code>index.html</code> → <b>Open with Live Server</b>, " +
          "atau jalankan <code>python3 -m http.server</code> di folder <code>website/</code> lalu buka " +
          "<code>http://localhost:8000</code>.</p>"
      );
      return;
    }
    if (!window.tf) {
      setStatus("err", "TensorFlow.js gagal dimuat");
      showBanner("<p>Pustaka TensorFlow.js tidak dapat diunduh. Periksa koneksi internet lalu muat ulang halaman.</p>");
      return;
    }
    try {
      setStatus("loading", "Memuat model…");
      const meta = await fetch(META_URL).then((r) => {
        if (!r.ok) throw new Error("metadata");
        return r.json();
      });
      classes = meta.classes;
      progress.hidden = false;
      model = await tf.loadGraphModel(MODEL_URL, {
        onProgress: (f) => {
          progress.firstElementChild.style.width = Math.round(f * 100) + "%";
          setStatus("loading", `Memuat model… ${Math.round(f * 100)}%`);
        },
      });
      progress.hidden = true;
      // Pemanasan agar prediksi pertama tidak lambat.
      tf.tidy(() => model.execute({ citra: tf.zeros([1, IMG_SIZE, IMG_SIZE, 3]) }, ["probabilities"]));
      setStatus("ok", `Model siap · ${classes.length} kelas · backend ${tf.getBackend()}`);
    } catch (err) {
      console.error(err);
      model = null;
      progress.hidden = true;
      setStatus("err", "Model belum tersedia");
      showBanner(
        "<p><b>File model tidak ditemukan di folder <code>model/</code>.</b></p>" +
          "<p>Konversi model hasil notebook dengan <code>tools/konversi_model.py</code> " +
          "(lihat WEBSITE.md bagian “Cara menyiapkan model”).</p>"
      );
    }
    updateButtons();
  }

  // ---------------------------------------------------------------- input
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
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.add("drag");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.remove("drag");
    })
  );
  dropzone.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
  window.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) setFile(item.getAsFile());
  });

  // ---------------------------------------------------------------- prediksi
  // Colormap "jet" seperti matplotlib, x di [0, 1].
  function jet(x) {
    const ch = (o) => tf.clipByValue(tf.sub(1.5, tf.abs(tf.sub(tf.mul(x, 4), o))), 0, 1);
    return tf.stack([ch(3), ch(2), ch(1)], -1);
  }

  async function analyze() {
    if (!file || !model) return;
    showError("");
    btnAnalyze.classList.add("loading");
    btnAnalyze.disabled = true;
    btnAnalyze.querySelector(".btn-label").textContent = "Menganalisis…";
    const t0 = performance.now();

    let bitmap;
    try {
      bitmap = await createImageBitmap(file);
    } catch {
      showError("File tidak dapat dibaca sebagai gambar.");
      finish();
      return;
    }

    try {
      // Sama dengan notebook: decode 3 kanal → resize bilinear 224x224 (tanpa menjaga rasio).
      const resized = tf.tidy(() =>
        tf.image.resizeBilinear(tf.browser.fromPixels(bitmap).toFloat(), [IMG_SIZE, IMG_SIZE], false, true)
      );
      bitmap.close();

      const [probT, camT] = model.execute({ citra: resized.expandDims(0) }, ["probabilities", "gradcam"]);
      const probs = Array.from(await probT.data());

      const imgU8 = tf.tidy(() => resized.clipByValue(0, 255).round().div(255));
      const overlay = tf.tidy(() => {
        const cam = camT.squeeze([0, 3]);
        return tf.add(tf.mul(imgU8, 0.55), tf.mul(jet(cam), 0.45)).clipByValue(0, 1);
      });
      await tf.browser.toPixels(imgU8, $("cv-input"));
      await tf.browser.toPixels(overlay, $("cv-cam"));
      tf.dispose([resized, probT, camT, imgU8, overlay]);

      render(probs, Math.round(performance.now() - t0));
    } catch (err) {
      console.error(err);
      showError("Terjadi kesalahan saat memproses gambar: " + err.message);
    }
    finish();
  }

  function finish() {
    btnAnalyze.classList.remove("loading");
    btnAnalyze.querySelector(".btn-label").textContent = "Analisis";
    updateButtons();
  }

  btnAnalyze.addEventListener("click", analyze);

  // ---------------------------------------------------------------- tampilan hasil
  function render(probs, ms) {
    const ranked = classes
      .map((c, i) => ({ key: c, p: probs[i], info: window.DATA_PENYAKIT[c] || { nama: c } }))
      .sort((a, b) => b.p - a.p);
    const top = ranked[0];
    const info = top.info;

    $("res-label").textContent = info.nama;
    $("res-latin").textContent = info.latin || "";
    $("res-conf").textContent = pct(top.p);
    $("low-conf").hidden = top.p >= LOW_CONFIDENCE;
    $("res-time").textContent = `Diproses di perangkat Anda dalam ${ms} ms`;

    $("res-probs").innerHTML = ranked
      .map(
        (r, i) => `
      <div class="prob${i === 0 ? " top" : ""}">
        <div class="prob-row"><span>${esc(r.info.nama)} <code>${esc(r.key)}</code></span><span>${pct(r.p)}</span></div>
        <div class="bar"><i data-w="${(r.p * 100).toFixed(2)}"></i></div>
      </div>`
      )
      .join("");

    const list = (items = []) => items.map((s) => `<li>${esc(s)}</li>`).join("");
    $("res-info").innerHTML = `
      <h3>Tentang ${esc(info.nama)} ${info.jenis ? `<span class="tag">${esc(info.jenis)}</span>` : ""}</h3>
      <p class="muted">${esc(info.ringkasan || "")}</p>
      <div class="info-cols">
        <div><h4>Gejala yang perlu dicocokkan</h4><ul>${list(info.gejala)}</ul></div>
        <div><h4>Penanganan umum</h4><ul>${list(info.penanganan)}</ul></div>
      </div>
      <p class="muted small" style="margin:14px 0 0">Cocokkan gejala dengan kondisi tanaman di lapangan.
        <a href="penyakit.html#${encodeURIComponent(top.key)}">Baca selengkapnya →</a></p>`;

    resultSec.hidden = false;
    requestAnimationFrame(() => {
      document.querySelectorAll("#res-probs .bar > i").forEach((el) => (el.style.width = el.dataset.w + "%"));
    });
    resultSec.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  loadModel();
})();
