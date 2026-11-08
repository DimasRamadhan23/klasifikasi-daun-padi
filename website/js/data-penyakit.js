// Ringkasan tiap kelas untuk ditampilkan di halaman Deteksi.
// Kunci harus sama dengan isi model/metadata.json → "classes" (nama folder dataset).
// Versi lengkap ada di penyakit.html.
window.DATA_PENYAKIT = {
  blast: {
    nama: "Blas",
    latin: "Pyricularia oryzae (Magnaporthe oryzae)",
    jenis: "Jamur",
    ringkasan:
      "Penyakit jamur yang menyerang daun, leher malai, dan buku batang. Berkembang cepat pada kelembapan tinggi dan pemupukan nitrogen berlebih.",
    gejala: [
      "Bercak berbentuk belah ketupat (oval meruncing di kedua ujung).",
      "Bagian tengah bercak abu-abu keputihan dengan tepi cokelat kemerahan.",
      "Bercak dapat menyatu sehingga daun mengering.",
    ],
    penanganan: [
      "Gunakan varietas tahan blas yang dianjurkan di daerah setempat.",
      "Kurangi pupuk nitrogen berlebih; pupuk secara berimbang (N, P, K).",
      "Atur jarak tanam agar kanopi tidak terlalu lembap.",
      "Bila serangan meluas, gunakan fungisida terdaftar sesuai anjuran petugas.",
    ],
  },
  blight: {
    nama: "Hawar Daun Bakteri",
    latin: "Xanthomonas oryzae pv. oryzae",
    jenis: "Bakteri",
    ringkasan:
      "Dikenal juga sebagai penyakit kresek. Bakteri masuk lewat luka atau pori air di tepi daun dan menyebar melalui air irigasi, hujan, dan angin.",
    gejala: [
      "Lesi bergaris kuning hingga oranye, dimulai dari ujung atau tepi daun.",
      "Lesi memanjang ke pangkal daun dengan batas bergelombang.",
      "Daun menjadi keabu-abuan, mengering, dan tampak seperti terbakar.",
    ],
    penanganan: [
      "Tanam varietas tahan hawar daun bakteri.",
      "Hindari nitrogen berlebih; tambahkan kalium.",
      "Atur pengairan berselang dan jaga drainase sawah.",
      "Musnahkan sisa tanaman terinfeksi dan bersihkan pematang.",
    ],
  },
  tungro: {
    nama: "Tungro",
    latin: "RTBV & RTSV (virus tungro padi)",
    jenis: "Virus",
    ringkasan:
      "Penyakit virus yang ditularkan wereng hijau (Nephotettix virescens). Tanaman yang sudah terinfeksi tidak dapat disembuhkan.",
    gejala: [
      "Daun kuning hingga kuning-oranye, dimulai dari ujung daun.",
      "Tanaman kerdil dan jumlah anakan berkurang.",
      "Daun muda menggulung atau terpuntir.",
    ],
    penanganan: [
      "Kendalikan wereng hijau sebagai vektor sejak persemaian.",
      "Cabut dan benamkan tanaman bergejala (eradikasi).",
      "Tanam serempak dan gunakan varietas tahan tungro.",
      "Laporkan serangan ke petugas POPT/PPL setempat.",
    ],
  },
};
