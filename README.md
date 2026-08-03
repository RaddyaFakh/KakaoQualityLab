# Dashboard Klasifikasi Mutu Biji Kakao — Versi Perbarui

## Yang diperbaiki dari kode sebelumnya
Di kode lama, label arsitektur pada hasil deteksi tertukar dengan file model
yang dimuat:

- `model_a` dimuat dari `densenet121_kakao1.h5`, tapi ditampilkan sebagai
  **"MobileNetV2"**.
- `model_b` dimuat dari `mobilenetv2_kakao1.h5`, tapi ditampilkan sebagai
  **"ResNet50"** — arsitektur ini bahkan tidak pernah dimuat sama sekali.

Pada versi ini, label sudah disesuaikan dengan file yang benar-benar dimuat:
`model_a` -> **DenseNet-121**, `model_b` -> **MobileNetV2**. Ini juga
memperbaiki nama kolom di Riwayat Analisis dan pesan error saat model gagal
dimuat.

## Yang ditambahkan
- Tema visual baru (palet warna terinspirasi warna penampang biji kakao saat
  uji belah/cut-test, tipografi Space Grotesk + Inter + IBM Plex Mono).
- Beranda: bagian "Tentang Biji Kakao & Proses Pasca-Panen", fakta singkat,
  galeri 4 kelas mutu dengan slot foto contoh, dan langkah penggunaan.
- Deteksi: badge kelas berwarna, banner kesepakatan/ketidaksepakatan kedua
  model, progress bar kepercayaan, dan grafik distribusi probabilitas untuk
  seluruh kelas (bukan hanya kelas dengan skor tertinggi).
- Riwayat: ringkasan (total analisis, tingkat kesepakatan model, kelas
  terbanyak), highlight warna per kelas pada tabel, dan tombol hapus riwayat.

## Cara menjalankan
```bash
pip install streamlit tensorflow pillow pandas numpy
streamlit run app.py
```
Pastikan `densenet121_kakao1.h5` dan `mobilenetv2_kakao1.h5` berada di folder
yang sama dengan `app.py`.

## Menambahkan foto contoh di Beranda
Simpan foto ke `assets/contoh_kelas/` — lihat `BACA_SAYA.txt` di dalam folder
tersebut untuk nama file yang dikenali sistem. Sebelum foto ditambahkan,
sistem menampilkan ilustrasi warna sebagai pengganti sementara agar dashboard
tetap tampil rapi.
