# Struktur Website Klasifikasi Penyakit Daun Padi

Website statis (HTML, CSS, JavaScript) untuk mendeteksi penyakit daun padi dari foto.
Model ResNet-50 dari notebook [Rice_Leaf_Disease-2.ipynb](Rice_Leaf_Disease-2.ipynb) dijalankan
**langsung di browser** dengan TensorFlow.js, jadi tidak perlu server Python dan foto tidak dikirim ke mana pun.

---

## 1. Struktur folder

```
Klasifikasi Penyakit Daun Padi/
├── Rice_Leaf_Disease-2.ipynb        # notebook penelitian (pelatihan model)
├── WEBSITE.md                       # dokumen ini
│
├── website/                         # ← seluruh website ada di sini
│   ├── index.html                   # halaman Beranda
│   ├── deteksi.html                 # halaman Deteksi (unggah foto + hasil)
│   ├── penyakit.html                # halaman info 3 penyakit
│   ├── tentang.html                 # halaman Tentang Model
│   │
│   ├── css/
│   │   └── style.css                # semua gaya tampilan (terang & gelap, responsif)
│   │
│   ├── js/
│   │   ├── main.js                  # dipakai semua halaman: menu ponsel, tanda halaman aktif
│   │   ├── data-penyakit.js         # data ringkas tiap penyakit untuk hasil deteksi
│   │   └── deteksi.js               # memuat model TF.js, prediksi, Grad-CAM, tampilkan hasil
│   │
│   ├── assets/
│   │   └── img/
│   │       └── logo.svg             # logo & favicon
│   │
│   └── model/                       # model TensorFlow.js (hasil konversi)
│       ├── README.md
│       ├── model.json               # ← dibuat oleh tools/konversi_model.py
│       ├── group1-shard*of*.bin     # ← bobot model (±48 MB total)
│       └── metadata.json            # ← daftar kelas
│
└── tools/
    ├── konversi_model.py            # konversi .keras → TensorFlow.js
    └── requirements.txt             # dependensi skrip konversi
```

## 2. Halaman

| File | Halaman | Isi |
|---|---|---|
| `index.html` | Beranda | Judul dan penjelasan singkat, tombol "Mulai deteksi", 3 langkah cara kerja, kartu 3 penyakit (tautan ke `penyakit.html`). |
| `deteksi.html` | Deteksi | Status pemuatan model, area unggah (klik / seret dan lepas / tempel Cmd+V), tips foto, lalu hasil: nama penyakit, % keyakinan, citra 224×224, heatmap Grad-CAM, probabilitas 3 kelas, gejala dan penanganan. |
| `penyakit.html` | Penyakit | Satu bagian per penyakit (`#blast`, `#blight`, `#tungro`): penyebab, cara penyebaran, gejala, penanganan umum. |
| `tentang.html` | Tentang | Metrik performa, arsitektur, dataset, hasil per kelas, penjelasan Grad-CAM, batasan model. |

Semua halaman memakai **header** (logo + menu: Beranda, Deteksi, Penyakit, Tentang) dan **footer**
(catatan bahwa website ini alat bantu, bukan pengganti diagnosis ahli) yang sama.

## 3. Fungsi tiap file CSS & JS

**`css/style.css`** dibagi menjadi beberapa bagian bernomor:
1. Variabel warna (`:root`), dengan versi gelap otomatis lewat `prefers-color-scheme`
2. Dasar (font, tautan, `.wrap`)
3. Header & navigasi (menu berubah jadi tombol ☰ di layar ≤ 640px)
4. Komponen umum (`.card`, `.btn`, `.banner`, `.tag`)
5. Beranda, 6. Deteksi, 7. Penyakit, 8. Tentang
9. Footer
10. Responsif (titik henti 900px dan 640px)

**`js/main.js`** buka/tutup menu di ponsel, menandai menu halaman aktif, dan mengisi tahun di footer.

**`js/data-penyakit.js`** berisi objek `window.DATA_PENYAKIT` dengan kunci `blast`, `blight`, `tungro`
(harus sama dengan `classes` di `model/metadata.json`). Setiap entri berisi nama, nama latin, jenis, ringkasan, gejala, dan penanganan.

**`js/deteksi.js`** menjalankan alur deteksi:
1. Memuat `model/metadata.json` dan `model/model.json` (dengan indikator progres).
2. Menerima foto JPG/PNG maksimum 10 MB.
3. Preprocessing **sama dengan notebook**: decode RGB → `resizeBilinear` 224×224 (tanpa menjaga rasio).
4. `model.execute({citra}, ["probabilities", "gradcam"])`.
5. Menggambar citra dan overlay Grad-CAM (colormap *jet*, alpha 0,45) ke `<canvas>`.
6. Menampilkan probabilitas, peringatan bila keyakinan < 70%, serta info penyakit.

Pustaka luar hanya satu: **TensorFlow.js 4.22.0** dari CDN jsDelivr (dimuat di `deteksi.html`).

## 4. Model TensorFlow.js

`tools/konversi_model.py` mengubah `resnet50_daun_padi.keras` menjadi model TF.js dengan:

- **Input** `citra`: `[1, 224, 224, 3]`, piksel 0–255. `preprocess_input` ResNet-50 sudah ada di dalam model.
- **Output** `probabilities`: `[1, 3]`, probabilitas softmax.
- **Output** `gradcam`: `[1, 224, 224, 1]`, heatmap 0–1 di layer `conv5_block3_out` untuk kelas teratas.

Grad-CAM dihitung di dalam model memakai turunan analitik kepala klasifikasi
(GAP → Dense ReLU → Dense Softmax). Skrip memeriksa otomatis bahwa hasilnya sama dengan
Grad-CAM `GradientTape` di notebook sebelum menyimpan. Bobot disimpan dalam float16 (±48 MB);
pakai `--float32` untuk bobot penuh (±94 MB).

## 5. Cara menyiapkan model

1. Unduh `resnet50_daun_padi.keras` dan `class_names.json` dari Google Drive
   (`MyDrive/PROJECT/hasil_skripsi/`), lalu simpan misalnya di folder `model_asli/`.
2. Pasang dependensi konversi (Python 3.10–3.12):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r tools/requirements.txt
   pip install --no-deps tensorflowjs==4.22.0
   ```
3. Jalankan konversi:
   ```bash
   python tools/konversi_model.py \
     --model model_asli/resnet50_daun_padi.keras \
     --classes model_asli/class_names.json \
     --out website/model
   ```

## 6. Cara menjalankan website

Website **harus dibuka lewat server lokal**, bukan klik dua kali file HTML, karena browser memblokir
pemuatan model dari `file://`. Pilih salah satu:

- **VS Code:** pasang ekstensi *Live Server*, klik kanan `website/index.html` → **Open with Live Server**.
- **Terminal:**
  ```bash
  cd website
  python3 -m http.server 8000
  ```
  lalu buka http://localhost:8000.

Untuk dipublikasikan, folder `website/` bisa langsung diunggah ke hosting statis seperti
GitHub Pages, Netlify, atau Vercel.

## 7. Catatan

- Model hanya mengenal 3 kelas (blast, blight, tungro). Foto daun sehat tetap akan diberi salah satu label,
  karena itu ada peringatan keyakinan rendah.
- Versi web memakai bobot float16, jadi probabilitasnya bisa berbeda sangat sedikit dari notebook (sekitar 0,1%).
- Pemuatan pertama mengunduh ±48 MB. Setelah itu browser menyimpannya di cache.
