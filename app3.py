from __future__ import annotations

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# 1. KONSTANTA & KONFIGURASI


APP_TITLE = "Sistem Klasifikasi Mutu Biji Kakao"
MODEL_PATH = "densenet121_kakao1.h5"
ASSET_DIR = "assets/contoh_kelas"
IMAGE_SIZE = (224, 224)      # dimensi input yang diharapkan model DenseNet-121
CONFIDENCE_TARGET = 85       # ambang kepercayaan acuan penelitian (%)

# Design tokens yang dipakai pada HTML inline (selaras dengan CSS di bawah).
COLOR_ROAST = "#2B1810"
COLOR_CACAO = "#6B4226"
COLOR_GOLD = "#C69749"

# Urutan kelas HARUS identik dengan urutan output layer saat model dilatih.
CLASS_NAMES = [
    "Broken Beans Cocoa (Cacat Fisik/Pecah)",
    "Fermented Cocoa (Mutu Baik/Cokelat Merata)",
    "Moldy Cocoa (Berjamur)",
    "Unfermented Cocoa (Mentah/Slaty)",
]
CLASS_NAMES_BASE = [name.split(" (")[0] for name in CLASS_NAMES]

# Metadata tampilan untuk tiap kelas mutu.
CLASS_INFO = {
    "Fermented Cocoa": {
        "short": "Fermented",
        "status": "Mutu Baik",
        "desc": "Fermentasi sempurna, penampang cokelat merata. "
                "Mutu utama dan layak untuk grading ekspor.",
        "color": "#5C7A52",
        "asset": "fermented",
    },
    "Unfermented Cocoa": {
        "short": "Unfermented",
        "status": "Proses Belum Optimal",
        "desc": "Fermentasi belum sempurna, penampang keunguan (slaty) "
                "dan rasa masih sepat.",
        "color": "#6B5B73",
        "asset": "unfermented",
    },
    "Moldy Cocoa": {
        "short": "Moldy",
        "status": "Ditolak",
        "desc": "Terkontaminasi jamur akibat pengeringan atau penyimpanan "
                "yang kurang tepat.",
        "color": "#74806B",
        "asset": "moldy",
    },
    "Broken Beans Cocoa": {
        "short": "Broken Beans",
        "status": "Cacat Fisik",
        "desc": "Biji retak atau pecah akibat penanganan mekanis sehingga "
                "menurunkan mutu fisik biji.",
        "color": "#C9762E",
        "asset": "broken",
    },
}



# 2. GAYA VISUAL (CSS)


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');

:root {
    --roast-900: #2B1810;
    --cacao-700: #6B4226;
    --pulp-050: #FBF6EE;
    --pulp-100: #F3E9DA;
    --gold-500: #C69749;
    --line: rgba(43, 24, 16, 0.10);
    --ink-muted: rgba(43, 24, 16, 0.62);

    /* Token yang nilainya berbeda antara mode terang & gelap */
    --app-bg: var(--pulp-050);
    --text-primary: var(--roast-900);
    --text-muted: var(--ink-muted);
    --card-bg: #FFFFFF;
    --card-metric: var(--cacao-700);
}

/* Override token di atas saat browser/OS memakai preferensi dark mode */
@media (prefers-color-scheme: dark) {
    :root {
        --app-bg: #1C130D;
        --text-primary: #F3E9DA;
        --text-muted: rgba(243, 233, 218, 0.62);
        --line: rgba(255, 255, 255, 0.10);
        --card-bg: #2A1D15;
        --card-metric: #D9A85C;
    }
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: var(--app-bg); }

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.01em;
}

div[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--card-metric);
}
div[data-testid="stMetricLabel"] {
    font-family: 'Inter', sans-serif;
    opacity: 0.72;
}

/* ---- Eyebrow / label bagian ---- */
.section-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--cacao-700);
    opacity: 0.85;
    margin-bottom: 0.35rem;
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] { background-color: var(--roast-900); }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {
    color: var(--pulp-100) !important;
}
section[data-testid="stSidebar"] hr { border-color: rgba(255, 255, 255, 0.15); }

.brand-row { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 1.4rem; }
.brand-mark {
    width: 38px; height: 38px; border-radius: 10px; flex-shrink: 0;
    background: linear-gradient(135deg, var(--gold-500), var(--cacao-700));
    display: flex; align-items: center; justify-content: center;
    font-family: 'Space Grotesk', sans-serif; font-weight: 700;
    color: white; font-size: 1.05rem;
}
.brand-title {
    font-family: 'Space Grotesk', sans-serif; font-weight: 700;
    font-size: 1.02rem; color: var(--pulp-050); line-height: 1.15;
}
.brand-sub { font-size: 0.72rem; color: var(--pulp-100); opacity: 0.65; margin-top: 0.1rem; }

/* ---- Navigasi sidebar bergaya pill ---- */
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 0.4rem; }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 9px;
    padding: 0.7rem 1rem !important;
    margin: 0 !important;
    transition: background 0.15s ease, border-color 0.15s ease;
    cursor: pointer;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(198, 151, 73, 0.14);
    border-color: rgba(198, 151, 73, 0.40);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(90deg, var(--gold-500), var(--cacao-700) 130%);
    border-color: var(--gold-500);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
    color: var(--roast-900) !important;
    font-weight: 700 !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { display: none; }

/* ---- Tombol ---- */
.stButton > button, .stDownloadButton > button {
    background-color: var(--cacao-700);
    color: var(--pulp-050);
    border: none;
    border-radius: 7px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    padding: 0.55rem 1.4rem;
    transition: background-color 0.15s ease, color 0.15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background-color: var(--gold-500);
    color: var(--roast-900);
}

/* ---- Hero ---- */
.hero-box {
    background: linear-gradient(120deg, var(--roast-900) 0%, var(--cacao-700) 100%);
    padding: 2.4rem 2.6rem;
    border-radius: 14px;
    margin-bottom: 1.6rem;
    box-shadow: 0 12px 30px rgba(43, 24, 16, 0.18);
}
.hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem; letter-spacing: 0.22em; text-transform: uppercase;
    color: var(--gold-500); margin-bottom: 0.7rem;
}
.hero-box h1 { color: var(--pulp-050) !important; font-size: 2.05rem; margin: 0 0 0.5rem 0; }
.hero-box p {
    color: var(--pulp-100); font-size: 1rem; max-width: 680px;
    margin: 0; line-height: 1.55;
}

.swatch-strip { display: flex; gap: 0.6rem; margin-top: 1.4rem; flex-wrap: wrap; }
.swatch-chip {
    display: flex; align-items: center; gap: 0.45rem;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 999px;
    padding: 0.32rem 0.8rem;
    font-size: 0.78rem; color: var(--pulp-050);
    font-family: 'IBM Plex Mono', monospace;
}
.swatch-dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }

/* ---- Badge kelas ---- */
.class-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.35rem 0.85rem; border-radius: 999px;
    font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 0.92rem;
    color: white; margin-bottom: 0.6rem;
}

/* ---- Kartu informasi ---- */
.info-card {
    background: var(--card-bg);
    border: 1px solid var(--line);
    border-top: 3px solid var(--gold-500);
    border-radius: 10px;
    padding: 1rem 1.1rem;
    height: 100%;
    box-shadow: 0 2px 8px rgba(43, 24, 16, 0.05);
}
.info-card .angka { font-family: 'IBM Plex Mono', monospace; font-size: 1.3rem; color: var(--card-metric); font-weight: 600; }
.info-card .label { font-size: 0.82rem; color: var(--text-primary); opacity: 0.75; margin-top: 0.2rem; }

/* ---- Penanda langkah ---- */
.step-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 28px; height: 28px; border-radius: 50%;
    background: var(--cacao-700); color: white;
    font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 0.85rem;
    margin-right: 0.6rem; flex-shrink: 0;
}

/* ---- Footer khusus ---- */
.app-footer {
    margin-top: 2.5rem; padding-top: 1.1rem;
    border-top: 1px solid var(--line);
    font-size: 0.8rem; color: var(--text-muted); line-height: 1.6;
}

/* Sembunyikan footer bawaan Streamlit */
footer { visibility: hidden; }
</style>
"""



# 3. MODEL & PRA-PEMROSESAN


@st.cache_resource
def load_model():

    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as error:  # noqa: BLE001 - tampilkan pesan ramah ke pengguna
        st.error(f"Gagal memuat model: {error}")
        return None


def preprocess_image(image: Image.Image) -> np.ndarray:
    
    image = image.resize(IMAGE_SIZE)
    image_array = np.array(image.convert("RGB")).astype(np.float32)
    return np.expand_dims(image_array, axis=0)



# 4. FUNGSI BANTU TAMPILAN


def class_badge_html(kelas: str) -> str:
    """Menghasilkan potongan HTML badge berwarna untuk sebuah kelas mutu."""
    info = CLASS_INFO.get(kelas, {"color": "#999999"})
    return f'<span class="class-badge" style="background:{info["color"]};">{kelas}</span>'


def find_asset(base_name: str) -> str | None:
    """Mencari berkas contoh citra kelas di folder aset (jpg/jpeg/png)."""
    for ext in ("jpg", "jpeg", "png"):
        path = os.path.join(ASSET_DIR, f"{base_name}.{ext}")
        if os.path.exists(path):
            return path
    return None


def render_class_visual(base_name: str, color: str, short_label: str) -> None:
    
    path = find_asset(base_name)
    if path:
        st.image(path, use_container_width=True)
    else:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, {color}22, {color}55);
                border: 1.5px dashed {color};
                border-radius: 10px; height: 150px;
                display: flex; align-items: center; justify-content: center;
                font-family: 'Space Grotesk', sans-serif; font-weight: 600;
                color: {color}; font-size: 0.95rem;">{short_label}</div>
            """,
            unsafe_allow_html=True,
        )


def render_footer() -> None:
    st.markdown(
        f"""
        <div class="app-footer">
            <b>{APP_TITLE}</b> &nbsp;·&nbsp; Model: DenseNet-121 (Transfer Learning) &nbsp;·&nbsp;
            Framework: TensorFlow &amp; Streamlit<br>
            Mengacu pada standar mutu biji kakao SNI 2323. Hasil klasifikasi
            ditujukan sebagai alat bantu pengujian, bukan pengganti verifikasi manual.
        </div>
        """,
        unsafe_allow_html=True,
    )



# 5. HALAMAN: BERANDA


def render_home_page() -> None:
    """Menyusun konten halaman Beranda."""
    swatch_html = "".join(
        f'<span class="swatch-chip">'
        f'<span class="swatch-dot" style="background:{info["color"]};"></span>'
        f'{info["short"]}</span>'
        for info in CLASS_INFO.values()
    )
    st.markdown(
        f"""
        <div class="hero-box">
            <div class="hero-eyebrow">Computer Vision · Pasca-Panen</div>
            <h1>Klasifikasi Mutu Biji Kakao Otomatis</h1>
            <p>Penerapan <i>computer vision</i> berbasis arsitektur
            <b>DenseNet-121</b> untuk membantu proses uji belah (cut-test)
            biji kakao secara otomatis, konsisten, dan objektif.</p>
            <div class="swatch-strip">{swatch_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Tentang biji kakao & fakta singkat ---
    col_teks, col_fakta = st.columns([1.6, 1])
    with col_teks:
        st.markdown("### Tentang Biji Kakao & Proses Pasca-Panen")
        st.markdown(
            "Setelah buah kakao (pod) dipanen dan dibelah, biji basah yang masih "
            "terbungkus pulp menjalani tahap **fermentasi** selama beberapa hari di "
            "dalam kotak kayu atau tumpukan berlapis daun pisang. Selama proses ini "
            "suhu tumpukan meningkat akibat aktivitas mikroba, dan warna keping biji "
            "berubah dari keunguan menjadi cokelat — perubahan inilah yang menjadi "
            "cikal bakal aroma khas cokelat.\n\n"
            "Setelah fermentasi, biji dijemur hingga kadar airnya turun ke tingkat "
            "aman untuk disimpan, lalu disortasi. Pada tahap sortasi inilah mutu biji "
            "ditentukan, secara tradisional dengan **uji belah (cut-test)**: sejumlah "
            "sampel biji dibelah dua dan penampangnya diamati secara visual oleh "
            "petugas.\n\n"
            "Cara manual ini efektif namun bergantung pada ketelitian dan pengalaman "
            "petugas, serta memakan waktu jika dilakukan pada volume besar. Sistem ini "
            "dirancang untuk mendukung proses tersebut secara lebih cepat dan konsisten "
            "menggunakan citra digital."
        )
    with col_fakta:
        st.markdown("### Fakta Singkat")
        fakta = [
            ("5–6 hari", "Durasi fermentasi umum"),
            ("± 7%", "Target kadar air setelah penjemuran"),
            ("Uji Belah", "Metode grading manual konvensional"),
            ("SNI 2323", "Acuan standar mutu biji kakao nasional"),
        ]
        fc1, fc2 = st.columns(2)
        for idx, (angka, label) in enumerate(fakta):
            target = fc1 if idx % 2 == 0 else fc2
            with target:
                st.markdown(
                    f"""
                    <div class="info-card" style="margin-bottom:0.8rem;">
                        <div class="angka">{angka}</div>
                        <div class="label">{label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # --- Galeri 4 kelas mutu ---
    st.markdown("### Kelas Mutu yang Dideteksi Sistem")
    gallery_cols = st.columns(4)
    for gcol, (kelas, info) in zip(gallery_cols, CLASS_INFO.items()):
        with gcol:
            render_class_visual(info["asset"], info["color"], info["short"])
            st.markdown(
                f'<div style="margin-top:0.7rem;">{class_badge_html(kelas)}</div>',
                unsafe_allow_html=True,
            )
            st.caption(info["desc"])

    # --- Cara menggunakan ---
    st.markdown("### Cara Menggunakan Sistem")
    langkah = [
        (
            "Pilih Sumber Citra",
            "Buka menu Deteksi Cerdas (AI), lalu pilih unggah berkas foto "
            "atau ambil langsung lewat kamera.",
        ),
        (
            "Jalankan Analisis",
            "Klik tombol Mulai Analisis Citra — sistem menjalankan model "
            "DenseNet-121 untuk mengklasifikasikan mutu biji.",
        ),
        (
            "Tinjau & Simpan Hasil",
            "Tinjau hasil klasifikasi beserta tingkat kepercayaannya; setiap "
            "analisis otomatis tercatat di menu Riwayat Analisis dan siap diunduh.",
        ),
    ]
    for i, (judul, isi) in enumerate(langkah, start=1):
        st.markdown(
            f"""
            <div style="display:flex; align-items:flex-start; margin-bottom:0.9rem;">
                <span class="step-badge">{i}</span>
                <div><b>{judul}</b><br><span style="opacity:0.85;">{isi}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# 6. HALAMAN: DETEKSI CERDAS (AI)

def render_detection_page(model) -> None:
    """Menyusun halaman input citra dan menjalankan inferensi model."""
    st.title("Pusat Pengujian & Deteksi Citra")
    st.write(
        "Pilih sumber citra biji kakao di bawah ini untuk memulai klasifikasi "
        "otomatis menggunakan model DenseNet-121."
    )

    metode_input = st.radio(
        "Metode input citra:",
        ["Unggah File Foto", "Ambil Foto via Kamera"],
    )

    image_to_analyze: Image.Image | None = None
    sumber_label = ""

    if metode_input == "Unggah File Foto":
        uploaded_file = st.file_uploader(
            "Pilih file citra (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"]
        )
        if uploaded_file:
            image_to_analyze = Image.open(uploaded_file)
            sumber_label = "Data Sekunder"
    else:
        camera_file = st.camera_input("Posisikan biji kakao di tengah kamera")
        if camera_file:
            image_to_analyze = Image.open(camera_file)
            sumber_label = "Data Primer"

    if image_to_analyze is None:
        return

    st.markdown("### Pratinjau Citra Uji")
    st.image(image_to_analyze, width=350)

    if not st.button("Mulai Analisis Citra"):
        return

    if model is None:
        st.error(
            f"Model belum termuat. Pastikan berkas `{MODEL_PATH}` berada di "
            "folder yang sama dengan aplikasi ini."
        )
        return

    # --- Inferensi ---
    with st.spinner("Model sedang menganalisis citra..."):
        processed_img = preprocess_image(image_to_analyze)
        start_time = time.time()
        prediksi = model.predict(processed_img)
        waktu_inferensi = time.time() - start_time

    idx = int(np.argmax(prediksi))
    kelas = CLASS_NAMES_BASE[idx]
    skor = float(np.max(prediksi) * 100)

    st.success("Analisis selesai.")
    st.markdown("### Hasil Klasifikasi")

    warna = CLASS_INFO[kelas]["color"]
    st.markdown(
        f"""
        <div style="background:{warna}18; border-left:4px solid {warna};
        padding:0.8rem 1rem; border-radius:6px; margin-bottom:1rem;">
        Citra ini diklasifikasikan sebagai <b>{kelas}</b>
        ({CLASS_INFO[kelas]['status']}).
        </div>
        """,
        unsafe_allow_html=True,
    )

    res_col, chart_col = st.columns([1, 1.3])
    with res_col:
        with st.container(border=True):
            st.markdown("##### DenseNet-121")
            st.markdown(class_badge_html(kelas), unsafe_allow_html=True)
            st.metric(label="Tingkat Kepercayaan", value=f"{skor:.2f}%")
            st.progress(min(100, max(0, int(round(skor)))))
            st.metric(label="Waktu Inferensi", value=f"{waktu_inferensi:.4f} detik")

    with chart_col:
        st.markdown("##### Distribusi Probabilitas per Kelas")
        label_pendek = [CLASS_INFO[c]["short"] for c in CLASS_NAMES_BASE]
        df_prob = pd.DataFrame(
            {"Probabilitas (%)": (prediksi[0] * 100).round(2)},
            index=label_pendek,
        )
        st.bar_chart(df_prob)

    # --- Catat ke riwayat sesi ---
    waktu_sekarang = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.riwayat_data.append(
        {
            "Waktu": waktu_sekarang,
            "Sumber": sumber_label,
            "Hasil DenseNet-121": f"{kelas} ({skor:.1f}%)",
            "_raw": kelas,
            "_skor": skor,
        }
    )


# 7. HALAMAN: RIWAYAT ANALISIS

def _highlight_kelas(val: str) -> str:
    """Memberi warna latar sel tabel sesuai kelas mutu yang terdeteksi."""
    for kelas, info in CLASS_INFO.items():
        if isinstance(val, str) and val.startswith(kelas):
            return (
                f'background-color: {info["color"]}22; '
                f'color: {info["color"]}; font-weight:600;'
            )
    return ""


def render_history_page() -> None:
    """Menyusun halaman rekap riwayat pengujian dalam satu sesi."""
    st.title("Log & Riwayat Hasil Analisis")
    st.write(
        "Daftar berikut memuat seluruh aktivitas klasifikasi biji kakao "
        "menggunakan model DenseNet-121 selama sesi aplikasi berjalan."
    )

    if not st.session_state.riwayat_data:
        st.info(
            "Belum ada riwayat pengujian. Silakan lakukan deteksi melalui menu "
            "'Deteksi Cerdas (AI)' terlebih dahulu."
        )
        return

    df_full = pd.DataFrame(st.session_state.riwayat_data)
    total = len(df_full)
    rata_skor = df_full["_skor"].mean()
    kelas_dominan = df_full["_raw"].mode()[0]

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Analisis", total)
    m2.metric("Rata-rata Tingkat Kepercayaan", f"{rata_skor:.1f}%")
    m3.metric("Kelas Paling Sering Terdeteksi", CLASS_INFO[kelas_dominan]["short"])

    st.markdown("---")

    df_display = df_full.drop(columns=["_raw", "_skor"])
    try:
        styled = df_display.style.map(_highlight_kelas, subset=["Hasil DenseNet-121"])
    except AttributeError:
        # Fallback untuk versi pandas lama (< 2.1.0)
        styled = df_display.style.applymap(_highlight_kelas, subset=["Hasil DenseNet-121"])
    st.dataframe(styled, use_container_width=True)

    st.markdown("---")
    dl_col, clear_col = st.columns([3, 1])
    with dl_col:
        st.caption(
            "Unduh data untuk menyimpan hasil klasifikasi mutu biji kakao"
        )
        csv_data = df_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Unduh Tabel Riwayat (.CSV)",
            data=csv_data,
            file_name="log_analisis_kakao.csv",
            mime="text/csv",
        )
    with clear_col:
        st.write("")
        if st.button("Hapus Riwayat"):
            st.session_state.riwayat_data = []
            st.rerun()


# 8. SIDEBAR & NAVIGASI

def build_sidebar() -> str:
    st.sidebar.markdown(
        """
        <div class="brand-row">
            <div class="brand-mark">K</div>
            <div>
                <div class="brand-title">Kakao Quality Lab</div>
                <div class="brand-sub">Klasifikasi Mutu Biji Kakao</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    halaman = st.sidebar.radio(
        "Pilih halaman:",
        ["Beranda", "Deteksi Cerdas (AI)", "Riwayat Analisis"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        <div style="background: rgba(198,151,73,0.14); border-left:3px solid var(--gold-500);
        padding:0.8rem 1rem; border-radius:6px; font-size:0.85rem; line-height:1.55;">
        <b>Tips Pengambilan Gambar</b><br>Gunakan pencahayaan konstan saat mengambil data
        menggunakan kamera agar hasilnya sesuai dengan target
        {CONFIDENCE_TARGET}%.
        </div>
        """,
        unsafe_allow_html=True,
    )
    return halaman

# 9. ENTRY POINT

def main() -> None:
    """Titik masuk aplikasi: konfigurasi, gaya, routing halaman."""
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Inisialisasi state riwayat sesi.
    if "riwayat_data" not in st.session_state:
        st.session_state.riwayat_data = []

    model = load_model()
    halaman = build_sidebar()

    if halaman == "Beranda":
        render_home_page()
    elif halaman == "Deteksi Cerdas (AI)":
        render_detection_page(model)
    elif halaman == "Riwayat Analisis":
        render_history_page()

    render_footer()


if __name__ == "__main__":
    main()
