"""Informasi singkat tiap kelas penyakit yang dikenali model.

Kunci dictionary harus sama dengan isi class_names.json (label folder dataset).
"""

DISEASE_INFO = {
    "blast": {
        "name": "Blas",
        "latin": "Pyricularia oryzae (Magnaporthe oryzae)",
        "type": "Jamur",
        "color": "#c2410c",
        "summary": (
            "Penyakit jamur yang menyerang daun, leher malai, dan buku batang. "
            "Berkembang cepat pada kelembapan tinggi dan pemupukan nitrogen berlebih."
        ),
        "symptoms": [
            "Bercak berbentuk belah ketupat (oval meruncing di kedua ujung).",
            "Bagian tengah bercak berwarna abu-abu keputihan dengan tepi cokelat kemerahan.",
            "Bercak dapat menyatu sehingga daun mengering.",
            "Pada serangan leher malai (blas leher), malai patah dan bulir hampa.",
        ],
        "management": [
            "Gunakan varietas tahan blas yang dianjurkan di daerah setempat.",
            "Kurangi pupuk nitrogen berlebih; pupuk secara berimbang (N, P, K).",
            "Atur jarak tanam (misalnya jajar legowo) agar kanopi tidak terlalu lembap.",
            "Bersihkan sisa jerami dan gulma inang yang terinfeksi.",
            "Bila serangan meluas, gunakan fungisida terdaftar sesuai anjuran petugas.",
        ],
    },
    "blight": {
        "name": "Hawar Daun Bakteri",
        "latin": "Xanthomonas oryzae pv. oryzae",
        "type": "Bakteri",
        "color": "#b45309",
        "summary": (
            "Dikenal juga sebagai penyakit kresek. Bakteri masuk lewat luka atau pori air "
            "di tepi daun dan menyebar melalui air irigasi, hujan, dan angin."
        ),
        "symptoms": [
            "Lesi bergaris berwarna kuning hingga oranye, dimulai dari ujung atau tepi daun.",
            "Lesi memanjang ke pangkal daun dengan batas bergelombang.",
            "Daun berubah menjadi keabu-abuan, mengering, lalu tampak seperti terbakar.",
            "Pada tanaman muda, seluruh tanaman dapat layu (gejala kresek).",
        ],
        "management": [
            "Tanam varietas tahan hawar daun bakteri.",
            "Hindari pemupukan nitrogen berlebih; tambahkan kalium.",
            "Atur pengairan berselang (intermittent) dan jaga drainase sawah.",
            "Hindari melukai daun saat memindah bibit; jangan memotong ujung bibit.",
            "Musnahkan sisa tanaman terinfeksi dan bersihkan pematang.",
        ],
    },
    "tungro": {
        "name": "Tungro",
        "latin": "Rice tungro bacilliform virus (RTBV) & Rice tungro spherical virus (RTSV)",
        "type": "Virus",
        "color": "#a16207",
        "summary": (
            "Penyakit virus yang ditularkan oleh wereng hijau (Nephotettix virescens). "
            "Tanaman yang sudah terinfeksi tidak dapat disembuhkan."
        ),
        "symptoms": [
            "Daun berwarna kuning hingga kuning-oranye, dimulai dari ujung daun.",
            "Tanaman kerdil dan jumlah anakan berkurang.",
            "Daun muda menggulung atau terpuntir.",
            "Malai pendek, pembungaan terlambat, dan banyak bulir hampa.",
        ],
        "management": [
            "Kendalikan wereng hijau sebagai vektor sejak persemaian.",
            "Cabut dan benamkan tanaman bergejala (eradikasi) agar tidak menjadi sumber virus.",
            "Tanam serempak dalam satu hamparan dan lakukan rotasi varietas.",
            "Gunakan varietas tahan tungro atau tahan wereng hijau.",
            "Laporkan serangan ke petugas POPT/PPL setempat untuk pengamatan wilayah.",
        ],
    },
}


def get_info(class_name: str) -> dict:
    """Kembalikan info penyakit; aman untuk kelas yang tidak terdaftar."""
    info = DISEASE_INFO.get(class_name)
    if info is None:
        return {
            "name": class_name,
            "latin": "",
            "type": "",
            "color": "#4d7c0f",
            "summary": "Informasi untuk kelas ini belum tersedia.",
            "symptoms": [],
            "management": [],
        }
    return info
