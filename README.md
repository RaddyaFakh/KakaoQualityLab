# 🍫 Sistem Klasifikasi Mutu Biji Kakao Otomatis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

Sistem berbasis **Computer Vision** yang dirancang untuk mengotomatisasi proses uji belah (*cut-test*) pada biji kakao. Aplikasi ini mengimplementasikan arsitektur *Deep Learning* **DenseNet-121** (melalui *Transfer Learning*) untuk mengklasifikasikan mutu biji kakao secara cepat, objektif, dan konsisten berdasarkan standar mutu **SNI 2323**.

---

## ✨ Fitur Utama

- **Deteksi Cerdas (AI)**: Mendukung dua metode input citra—unggah file (JPG/PNG) dan tangkapan langsung dari kamera (*webcam*).
- **Klasifikasi 4 Kelas Mutu**:
  - `Fermented` (Mutu Baik / Cokelat Merata)
  - `Unfermented` (Mentah / Slaty)
  - `Moldy` (Biji Berjamur)
  - `Broken Beans` (Cacat Fisik / Pecah)
- **Tingkat Kepercayaan Model**: Menampilkan skor probabilitas (*Confidence Level*) beserta distribusi grafik batang.
- **Pencatatan Riwayat (Log)**: Merekam hasil analisis selama sesi berjalan, yang dapat diekspor langsung ke dalam format `.CSV`.
- **UI Profesional & Responsif**: Dibangun dengan komponen antarmuka yang bersih (*clean design*) lengkap dengan dukungan *Dark Mode*.

## 🚀 Instalasi & Persiapan Lingkungan

### 1. Kloning Repositori
Buka terminal dan kloning repositori ini ke komputer lokal Anda:
```bash
git clone https://github.com/username-anda/nama-repo-anda.git
cd nama-repo-anda
```

### 2. Instalasi Dependensi
Pastikan Python sudah terpasang. Sangat disarankan untuk menggunakan *virtual environment*.
```bash
# Membuat virtual environment
python -m venv venv
# Aktivasi virtual environment (Windows)
venv\Scripts\activate
# Aktivasi virtual environment (Mac/Linux)
source venv/bin/activate

# Menginstal paket yang dibutuhkan
pip install streamlit tensorflow numpy pandas pillow
```

### 3. Persiapan Model
Aplikasi ini memerlukan model *Deep Learning* yang sudah dilatih.
- Pastikan file `densenet121_kakao1.h5` diletakkan tepat di *root directory* (satu folder dengan `app.py`).

### 4. Menjalankan Aplikasi
Eksekusi perintah berikut untuk menjalankan server Streamlit:
```bash
streamlit run app.py
```
Aplikasi akan secara otomatis terbuka di *browser* pada alamat `http://localhost:8501`.

## 📂 Struktur Direktori Proyek

```text
├── .streamlit/
│   └── config.toml           # Konfigurasi tema Streamlit (Dark Mode by default)
├── assets/
│   └── contoh_kelas/         # Aset gambar referensi untuk tiap kelas mutu (opsional)
├── app.py                    # Script utama antarmuka Streamlit
├── densenet121_kakao1.h5     # File bobot model DenseNet-121 (Harus ditambahkan)
├── requirements.txt          # Daftar dependensi modul Python
└── README.md                 # Dokumentasi proyek
```

## 👨‍💻 Penulis

Dikembangkan oleh **Muhammad Raddya Fakhreza**  
Universitas Gunadarma  

---
*Disclaimer: Hasil klasifikasi dari sistem ini ditujukan sebagai alat bantu augmentasi pengujian pasca-panen, bukan sebagai pengganti mutlak verifikasi kualitas secara manual.*
