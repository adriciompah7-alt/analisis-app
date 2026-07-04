# -*- coding: utf-8 -*-
"""
K-Means Clustering — Pengelompokan Data Pemilih (versi Streamlit)
==================================================================
Alur kerja:
  1. Upload Data
  2. Preprocessing (cek missing value, bersihkan data, statistik deskriptif)
  3. Pilih Jumlah Cluster (Manual)
  4. Elbow Method (validasi / opsi ubah K)
  5. Centroid Awal
  6. Proses Iterasi
  7. Hasil Akhir

Konvergensi berbasis perubahan cluster assignment (sama seperti perhitungan
manual Excel) — bukan berdasarkan pergeseran centroid.

Cara menjalankan:
    pip install -r requirements.txt
    streamlit run app_streamlit.py

CATATAN PERBAIKAN (fix bug):
  - Sebelumnya terjadi TypeError:
      "update_layout() got multiple values for keyword argument 'xaxis'"
    Ini terjadi karena dict PLOTLY_LAYOUT sudah memiliki key 'xaxis'/'yaxis'/'legend',
    lalu di beberapa tempat key yang sama juga dikirim ulang sebagai keyword argument
    terpisah saat memanggil fig.update_layout(**PLOTLY_LAYOUT, xaxis=..., ...).
    Python tidak mengizinkan keyword yang sama muncul dua kali.
  - Perbaikan: gabungkan (merge) dict override ke PLOTLY_LAYOUT terlebih dahulu
    menjadi satu dict baru, baru kemudian di-unpack sekali dengan **.
    Diterapkan pada: grafik Elbow (Step 3) dan grafik Donut (Step 6).
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ══════════════════════════════════════════════════════════════════════════
# KONFIGURASI
# ══════════════════════════════════════════════════════════════════════════
COLS = ['Status Memilih', 'Umur (N)', 'Status Kawin', 'Jenis Kelamin', 'RT (N)', 'Disabilitas']
DEFAULT_K = 6
STEPS = [
    ("📂", "Upload Data"), ("🧹", "Preprocessing"), ("🎯", "Pilih Cluster (Manual)"),
    ("📈", "Elbow Method"), ("🧭", "Centroid Awal"), ("🔄", "Proses Iterasi"), ("🏁", "Hasil Akhir"),
]
# Palet cluster "Aurora" — violet, cyan, pink, kuning madu, teal, biru.
# Setiap warna sengaja dipilih cukup terang/vivid agar teks putih di atasnya tetap terbaca.
CLUSTER_COLORS = ['#A855F7', '#22D3EE', '#F472B6', '#FBBF24', '#2DD4BF', '#818CF8']


def cluster_text_color(hex_color: str) -> str:
    """Pilih warna teks (putih/hitam) otomatis berdasarkan kecerahan warna latar,
    supaya kontras selalu terjaga di badge & tabel apa pun warnanya."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return '#171123' if luminance > 0.62 else '#FFFFFF'


st.set_page_config(page_title="K-Means · Data Pemilih", page_icon="🗳️", layout="wide",
                    initial_sidebar_state="expanded")


def ccolor(c):
    return CLUSTER_COLORS[(int(c) - 1) % len(CLUSTER_COLORS)]


def badge(c, label=None):
    color = ccolor(c)
    txt = cluster_text_color(color)
    label = label or f"C{c}"
    return (f'<span style="display:inline-flex;align-items:center;justify-content:center;'
            f'padding:3px 14px;border-radius:20px;font-size:11.5px;font-weight:800;'
            f'background:{color};color:{txt};box-shadow:0 2px 12px {color}66;'
            f'letter-spacing:.02em;">{label}</span>')


def style_cluster_col(df, col='CLUSTER'):
    def _c(val):
        color = ccolor(val)
        txt = cluster_text_color(color)
        return f'background-color:{color};color:{txt};font-weight:800;border-radius:6px'
    styler = df.style.map(_c, subset=[col]) if col in df.columns else df.style
    return styler.set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#1B1533'), ('color', '#C9C3E8'),
                                     ('font-size', '10.5px'), ('text-transform', 'uppercase'),
                                     ('letter-spacing', '.04em'), ('font-weight', '700')]},
    ])


PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Plus Jakarta Sans, sans-serif', color='#C9C3E8', size=12),
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(gridcolor='#2A2350', zeroline=False),
    yaxis=dict(gridcolor='#2A2350', zeroline=False),
    legend=dict(bgcolor='rgba(0,0,0,0)'),
)


def merged_layout(base: dict, **overrides) -> dict:
    """Gabungkan dict layout dasar (PLOTLY_LAYOUT) dengan override secara aman.

    Ini adalah util untuk menghindari TypeError "got multiple values for keyword
    argument ..." yang muncul kalau kita menulis:
        fig.update_layout(**PLOTLY_LAYOUT, xaxis=..., legend=...)
    padahal PLOTLY_LAYOUT sudah punya key 'xaxis'/'legend' di dalamnya.

    Untuk key yang berupa dict (seperti 'xaxis', 'yaxis', 'legend'), override
    di-merge (bukan menimpa total) supaya style dasar (gridcolor, dsb) tetap
    terbawa kecuali memang di-override secara eksplisit.
    """
    result = dict(base)
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(result.get(key), dict):
            result[key] = {**result[key], **val}
        else:
            result[key] = val
    return result


def render_stepper(current, steps):
    """Render indikator langkah horizontal bergaya (mirip progress bar bertahap)."""
    n = len(steps)
    html = '<div style="display:flex;align-items:center;overflow-x:auto;padding:4px 2px 18px;">'
    for i, (icon, label) in enumerate(steps):
        if i < current:
            circle_bg, circle_border, txt_color = 'var(--teal)', 'var(--teal)', '#062A26'
            content = '✓'
        elif i == current:
            circle_bg = 'linear-gradient(135deg,var(--accent),var(--accent3))'
            circle_border, txt_color = 'var(--accent)', '#FFFFFF'
            content = str(i + 1)
        else:
            circle_bg, circle_border, txt_color = 'transparent', '#332B5C', 'var(--text3)'
            content = str(i + 1)
        label_color = 'var(--teal)' if i < current else ('var(--accent2)' if i == current else 'var(--text3)')
        html += (
            f'<div style="display:flex;align-items:center;flex-shrink:0;">'
            f'<div style="width:26px;height:26px;border-radius:50%;background:{circle_bg};'
            f'border:1.5px solid {circle_border};display:flex;align-items:center;justify-content:center;'
            f'font-size:11px;font-weight:700;color:{txt_color};flex-shrink:0;">{content}</div>'
            f'<span style="font-size:11px;font-weight:700;color:{label_color};margin:0 10px;white-space:nowrap;">{label}</span>'
        )
        if i < n - 1:
            line_color = 'var(--teal)' if i < current else '#2A2350'
            html += f'<div style="height:1.5px;width:22px;background:{line_color};margin-right:4px;flex-shrink:0;"></div>'
        html += '</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_nav_buttons(step, total_steps, next_label="Lanjut →", next_disabled=False, on_next=None):
    """Render tombol navigasi Kembali / Lanjut secara simetris & konsisten di setiap step."""
    st.write("")
    st.markdown("<hr style='margin:6px 0 16px;opacity:.35'>", unsafe_allow_html=True)
    col_back, col_spacer, col_next = st.columns([1, 2, 1])
    with col_back:
        if step > 0:
            if st.button("← Kembali", use_container_width=True, key=f"back_{step}"):
                goto(step - 1)
    with col_next:
        if step < total_steps - 1 and on_next is not None:
            if st.button(next_label, type="primary", use_container_width=True,
                         disabled=next_disabled, key=f"next_{step}"):
                on_next()


# ══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════
defaults = {
    'page': 'home', 'step': 0, 'raw_df': None, 'clean_df': None,
    'manual_k': DEFAULT_K, 'K': DEFAULT_K, 'wcss': None,
    'centroids_input': None, 'iterations': None, 'hasil': None,
    'centroid_akhir': None, 'silhouette': None, 'current_iter': 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def goto(i):
    st.session_state.page = 'wizard'
    st.session_state.step = i
    st.rerun()


def goto_home():
    st.session_state.page = 'home'
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# FUNGSI INTI K-MEANS MANUAL (identik dengan perhitungan manual Excel)
# ══════════════════════════════════════════════════════════════════════════
def run_kmeans_manual(X, centroid_awal, K, max_iter=100):
    X = np.array(X, dtype=float)
    centroid = np.array(centroid_awal, dtype=float)
    all_iterations = []
    cluster_lama = None
    final_cluster = None

    for iterasi in range(1, max_iter + 1):
        centroid_lama = centroid.copy()
        jarak = cdist(X, centroid, metric='euclidean')
        cluster = np.argmin(jarak, axis=1) + 1
        anggota = {f'C{c}': int((cluster == c).sum()) for c in range(1, K + 1)}

        centroid_baru = []
        for c in range(1, K + 1):
            pts = X[cluster == c]
            centroid_baru.append(pts.mean(axis=0).tolist() if len(pts) > 0 else centroid[c - 1].tolist())
        centroid_baru = np.array(centroid_baru)

        konvergen = (cluster_lama is not None) and np.array_equal(cluster, cluster_lama)

        all_iterations.append({
            'iterasi': iterasi,
            'centroid_lama': centroid_lama.tolist(),
            'centroid_baru': centroid_baru.tolist(),
            'anggota': anggota,
            'konvergen': konvergen,
            'jarak': jarak,
            'cluster': cluster.copy(),
        })

        cluster_lama = cluster.copy()
        centroid = centroid_baru
        final_cluster = cluster

        if konvergen:
            break

    return all_iterations, centroid, final_cluster


# ══════════════════════════════════════════════════════════════════════════
# CSS — TEMA "AURORA" · VIOLET · PINK · CYAN (kontras teks dicek per elemen)
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root{
  --accent:#A855F7;        /* violet utama */
  --accent2:#F472B6;        /* pink */
  --accent3:#22D3EE;        /* cyan */
  --accent-glow:rgba(168,85,247,.30);
  --teal:#2DD4BF; --gold:#FBBF24; --coral:#FB7185; --blue:#818CF8;
  --bg:#0B0817; --surface:#150F2B; --surface2:#1D1640; --border:#2E2657;
  --text1:#F6F3FF; --text2:#B7AFDE; --text3:#726A9C;
}

html, body, [class*="css"] { font-family:'Plus Jakarta Sans', system-ui, sans-serif !important; }
.stApp{
  background:
    radial-gradient(ellipse 900px 500px at 10% -10%, rgba(168,85,247,.16), transparent 60%),
    radial-gradient(ellipse 800px 500px at 100% 0%, rgba(34,211,238,.10), transparent 55%),
    var(--bg) !important;
}
.main .block-container{ padding-top:1.6rem; padding-bottom:3rem; max-width:1180px; }

/* ── Sembunyikan chrome bawaan yang mengganggu ── */
#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header[data-testid="stHeader"]{background:transparent;}

/* ── Sidebar ── */
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#120C26 0%,#0B0817 100%);
  border-right:1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container{ padding-top:1.2rem; }

/* ── Judul & teks umum ── */
h1{ font-weight:800 !important; letter-spacing:-.03em; color:#fff !important; font-size:1.9rem !important; }
h2,h3{ font-weight:700 !important; letter-spacing:-.02em; color:#fff !important; }
p, li, span, label, .stMarkdown{ color:var(--text2); }
.stCaption, [data-testid="stCaptionContainer"]{ color:var(--text3) !important; }

/* ── Progress bar ── */
div[data-testid="stProgress"] > div > div{
  background:linear-gradient(90deg,var(--accent),var(--accent2),var(--accent3)) !important;
  border-radius:99px;
}
div[data-testid="stProgress"] > div{ background:var(--surface2); border-radius:99px; }

/* ── Tombol umum ── */
.stButton > button{
  border-radius:11px !important; font-weight:700 !important; font-size:13.5px !important;
  border:1px solid var(--border) !important; background:var(--surface2) !important;
  color:var(--text1) !important; transition:all .18s ease !important; padding:.55rem 1rem !important;
}
.stButton > button:hover{
  border-color:var(--accent) !important; color:#fff !important; transform:translateY(-1px);
  box-shadow:0 6px 20px rgba(168,85,247,.22) !important;
}
.stButton > button[kind="primary"]{
  background:linear-gradient(135deg,var(--accent),var(--accent2)) !important;
  border:none !important; color:#FFFFFF !important; text-shadow:0 1px 3px rgba(0,0,0,.25);
  box-shadow:0 6px 22px var(--accent-glow) !important;
}
.stButton > button[kind="primary"]:hover{ filter:brightness(1.1); color:#FFFFFF !important; transform:translateY(-2px); }
.stButton > button[kind="primary"]:disabled{
  background:var(--surface2) !important; color:var(--text3) !important; box-shadow:none !important; opacity:.6;
}
.stDownloadButton > button{
  background:linear-gradient(135deg,var(--teal),#0E9C87) !important; border:none !important;
  color:#04231C !important; font-weight:800 !important; border-radius:11px !important;
  box-shadow:0 6px 22px rgba(45,212,191,.28) !important;
}
.stDownloadButton > button:hover{ filter:brightness(1.08); transform:translateY(-2px); }

/* Tombol navigasi sidebar */
section[data-testid="stSidebar"] .stButton > button{
  text-align:left !important; justify-content:flex-start !important;
  background:transparent !important; border:1px solid transparent !important;
  color:var(--text2) !important; font-weight:600 !important; margin-bottom:2px !important;
}
section[data-testid="stSidebar"] .stButton > button:hover{
  background:var(--surface) !important; color:#fff !important; border-color:var(--border) !important;
  box-shadow:none !important; transform:none;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"]{
  background:linear-gradient(90deg, rgba(168,85,247,.22), rgba(244,114,182,.12)) !important;
  border:1px solid rgba(168,85,247,.45) !important; color:#fff !important;
  box-shadow:inset 3px 0 0 var(--accent) !important;
}

/* Tombol angka cepat (K quick-select) — kecil & center */
div[data-testid="column"] .stButton > button{ padding:.45rem .3rem !important; }

/* ── Metric cards ── */
div[data-testid="stMetric"]{
  background:linear-gradient(160deg, var(--surface), var(--surface2));
  border:1px solid var(--border); border-radius:16px;
  padding:14px 16px 10px; box-shadow:0 4px 16px rgba(0,0,0,.28);
}
div[data-testid="stMetricLabel"]{ color:var(--text3) !important; font-size:10.5px !important; font-weight:700 !important; text-transform:uppercase; letter-spacing:.06em; }
div[data-testid="stMetricValue"]{
  background:linear-gradient(90deg,var(--accent2),var(--accent3));
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
  font-weight:800 !important; font-size:1.75rem !important;
}
div[data-testid="stMetricDelta"]{ font-weight:700 !important; }

/* ── Alerts (success / info / warning / error) ── */
div[data-testid="stAlertContainer"]{
  border-radius:13px !important; border:1px solid var(--border) !important; font-size:13.5px !important;
  background:var(--surface) !important;
}

/* ── Expander (blok centroid) ── */
div[data-testid="stExpander"]{
  border:1px solid var(--border) !important; border-radius:15px !important; overflow:hidden;
  background:linear-gradient(160deg, var(--surface), #150F2B) !important; margin-bottom:10px;
}
div[data-testid="stExpander"] summary{ font-weight:700 !important; color:var(--text1) !important; }

/* ── DataFrame / Table ── */
div[data-testid="stDataFrame"]{
  border:1px solid var(--border) !important; border-radius:13px !important; overflow:hidden;
}

/* ── Inputs ── */
div[data-testid="stNumberInput"] input, div[data-testid="stTextInput"] input{
  background:var(--surface2) !important; border:1px solid var(--border) !important;
  color:var(--text1) !important; border-radius:9px !important; font-family:'JetBrains Mono',monospace !important;
}
div[data-testid="stNumberInput"] input:focus{ border-color:var(--accent) !important; box-shadow:0 0 0 3px var(--accent-glow) !important; }

/* ── Slider ── */
div[data-testid="stSlider"] div[role="slider"]{ background:var(--accent) !important; border-color:#fff !important; }
div[data-testid="stSlider"] > div > div > div{ background:linear-gradient(90deg,var(--accent),var(--accent3)) !important; }

/* ── File uploader ── */
div[data-testid="stFileUploaderDropzone"]{
  background:var(--surface) !important; border:2px dashed var(--border) !important; border-radius:18px !important;
}
div[data-testid="stFileUploaderDropzone"]:hover{ border-color:var(--accent) !important; }

/* ── Hero banner (halaman utama / step 0) ── */
.hero-banner{
  background:linear-gradient(135deg, rgba(168,85,247,.20), rgba(34,211,238,.10));
  border:1px solid var(--border); border-radius:22px; padding:28px 32px; margin-bottom:22px;
  position:relative; overflow:hidden;
}
.hero-banner::before{
  content:''; position:absolute; top:-70px; right:-70px; width:240px; height:240px; border-radius:50%;
  background:radial-gradient(circle, rgba(168,85,247,.35), transparent 70%);
}
.hero-banner::after{
  content:''; position:absolute; bottom:-90px; left:20%; width:260px; height:260px; border-radius:50%;
  background:radial-gradient(circle, rgba(34,211,238,.22), transparent 70%);
}
.hero-eyebrow{
  display:inline-flex; align-items:center; gap:7px; padding:4px 13px; border-radius:99px;
  background:rgba(168,85,247,.18); border:1px solid rgba(168,85,247,.4); font-size:10.5px;
  font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:#E9D5FF; margin-bottom:12px;
  position:relative; z-index:1;
}
.hero-title{
  font-size:1.85rem; font-weight:800; letter-spacing:-.03em; margin:0 0 8px 0; position:relative; z-index:1;
  background:linear-gradient(90deg,#FFFFFF, #E9D5FF 60%, #A5F3FC);
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
}
.hero-desc{ font-size:13.5px; color:var(--text2); line-height:1.65; max-width:760px; margin:0; position:relative; z-index:1; }

/* ── Sidebar brand ── */
.brand-box{ display:flex; align-items:center; gap:11px; padding:2px 4px 14px; border-bottom:1px solid var(--border); margin-bottom:12px; }
.brand-icon{
  width:40px; height:40px; border-radius:12px; background:linear-gradient(135deg,var(--accent),var(--accent2));
  display:flex; align-items:center; justify-content:center; font-size:19px;
  box-shadow:0 0 20px var(--accent-glow); flex-shrink:0;
}
.brand-title{ font-size:14.5px; font-weight:800; color:#fff; line-height:1.25; }
.brand-sub{
  font-size:9.5px; letter-spacing:.06em; font-weight:700;
  background:linear-gradient(90deg,var(--accent2),var(--accent3));
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
}

/* ── Section label kecil ── */
.sec-label{
  font-size:10.5px; font-weight:800; letter-spacing:.09em; text-transform:uppercase; color:var(--text2);
  margin:22px 0 10px; padding-left:11px; border-left:3px solid var(--accent2);
}
.sec-label:first-of-type{ margin-top:6px; }

/* ── Card generik ── */
.card{
  background:linear-gradient(160deg, var(--surface), var(--surface2));
  border:1px solid var(--border); border-radius:18px; padding:18px 20px; margin-bottom:14px;
  box-shadow:0 4px 18px rgba(0,0,0,.22);
}

/* ── Container bordered (st.container(border=True)) — jadikan senada tema ── */
div[data-testid="stVerticalBlockBorderWrapper"]{
  border-color: var(--border) !important; border-radius:16px !important;
  background:linear-gradient(160deg, rgba(255,255,255,.02), rgba(255,255,255,.005)) !important;
}

/* ── Divider tipis ── */
hr{ border-color:var(--border) !important; opacity:.6; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGASI — disembunyikan total di halaman Dashboard/Beranda
# ══════════════════════════════════════════════════════════════════════════
if st.session_state.page == 'home':
    # Sembunyikan sidebar (termasuk panah collapse-nya) khusus di halaman Beranda,
    # supaya benar-benar hanya dashboard yang tampil, tanpa "slide bar".
    st.markdown("""
    <style>
    section[data-testid="stSidebar"]{ display:none !important; }
    div[data-testid="stSidebarCollapsedControl"]{ display:none !important; }
    </style>
    """, unsafe_allow_html=True)
else:
    with st.sidebar:
        st.markdown("""
        <div class="brand-box">
          <div class="brand-icon">🗳️</div>
          <div>
            <div class="brand-title">K-Means<br>Data Pemilih</div>
            <div class="brand-sub">ANALISIS PENGELOMPOKAN</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🏠  Dashboard", key="nav_home", use_container_width=True,
                     type="primary" if st.session_state.page == 'home' else "secondary"):
            goto_home()

        st.markdown("<hr style='margin:8px 0 10px;opacity:.4'>", unsafe_allow_html=True)

        for i, (icon, label) in enumerate(STEPS):
            mark = "✅" if i < st.session_state.step else ("▶" if i == st.session_state.step else "")
            btn_label = f"{icon}  {i+1}. {label}   {mark}"
            is_active = (st.session_state.page == 'wizard' and i == st.session_state.step)
            if st.button(btn_label, key=f"nav_{i}", use_container_width=True,
                         type="primary" if is_active else "secondary"):
                goto(i)

        st.markdown("<hr style='margin:16px 0 10px;opacity:.4'>", unsafe_allow_html=True)
        st.caption("Dibangun dengan Streamlit + scikit-learn")

# ══════════════════════════════════════════════════════════════════════════
# HALAMAN DASHBOARD / BERANDA — hanya hero banner + 1 tombol, tanpa apa pun lagi
# ══════════════════════════════════════════════════════════════════════════
if st.session_state.page == 'home':
    st.markdown(f"""
    <div class="hero-banner" style="padding:38px 36px;">
      <div class="hero-eyebrow">🗳️ Sistem Analisis Clustering</div>
      <div class="hero-title" style="font-size:2.3rem;">Pengelompokan Data Pemilih dengan K-Means</div>
      <p class="hero-desc" style="max-width:640px;">
        Kelompokkan data pemilih berdasarkan usia, status kawin, jenis kelamin, wilayah RT, dan
        disabilitas secara otomatis — lengkap dengan validasi Elbow Method, visualisasi tiap iterasi,
        dan skor Silhouette untuk mengukur kualitas hasil.
      </p>
    </div>
    """, unsafe_allow_html=True)

    col_cta1, col_cta2 = st.columns([1, 3])
    with col_cta1:
        if st.button("🚀 Mulai Analisis Baru", type="primary", use_container_width=True, key="cta_new"):
            for k, v in defaults.items():
                if k not in ('page',):
                    st.session_state[k] = v
            goto(0)

# ══════════════════════════════════════════════════════════════════════════
# HALAMAN WIZARD (LANGKAH ANALISIS)
# ══════════════════════════════════════════════════════════════════════════
elif st.session_state.page == 'wizard':
    step = st.session_state.step
    render_stepper(step, STEPS)

    # ── Header halaman (hero banner) ──
    icon, label = STEPS[step]
    step_descriptions = {
        0: "Unggah data pemilih mentah (Excel/CSV) sebagai titik awal analisis clustering.",
        1: "Periksa nilai kosong dan bersihkan data sebelum masuk ke tahap clustering.",
        2: "Tentukan jumlah cluster (K) secara manual sesuai kebutuhan analisis Anda.",
        3: "Validasi pilihan K menggunakan grafik Within-Cluster Sum of Squares (WCSS).",
        4: "Tetapkan koordinat centroid awal untuk setiap cluster sebelum iterasi dimulai.",
        5: "Ikuti setiap iterasi algoritma K-Means hingga mencapai konvergensi.",
        6: "Ringkasan akhir hasil pengelompokan data pemilih beserta evaluasi kualitasnya.",
    }
    st.markdown(f"""
    <div class="hero-banner">
      <div class="hero-eyebrow"><span>{icon}</span> Tahap {step+1} dari {len(STEPS)}</div>
      <div class="hero-title">{label}</div>
      <p class="hero-desc">{step_descriptions.get(step, "")}</p>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # STEP 0 — UPLOAD DATA
    # ══════════════════════════════════════════════════════════════════
    if step == 0:
        st.markdown('<div class="sec-label">Format Data yang Dibutuhkan</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.write("Upload file **Excel (.xlsx)** atau **CSV** dengan kolom: " + ", ".join(f"`{c}`" for c in COLS))
            st.info("💡 Data mentah akan dibersihkan pada tahap **Preprocessing** berikutnya.")
            uploaded = st.file_uploader("Pilih file", type=['xlsx', 'xls', 'csv'], label_visibility="collapsed")

        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded) if uploaded.name.lower().endswith('.csv') else pd.read_excel(uploaded)
                df.columns = [c.strip() for c in df.columns]
                missing_cols = [c for c in COLS if c not in df.columns]
                if missing_cols:
                    st.error(f"Kolom tidak ditemukan: {', '.join(missing_cols)}")
                else:
                    df_cols = df[COLS].copy()
                    st.session_state.raw_df = df_cols
                    st.session_state.clean_df = None
                    st.session_state.wcss = None
                    st.session_state.iterations = None
                    st.session_state.hasil = None
                    st.success(f"✅ **{uploaded.name}** berhasil dimuat — **{len(df_cols)}** baris data mentah.")
            except Exception as e:
                st.error(f"Gagal baca file: {e}")

        if st.session_state.raw_df is not None:
            df_cols = st.session_state.raw_df
            missing_rows = int(df_cols.isna().any(axis=1).sum())
            st.markdown('<div class="sec-label">Ringkasan Data</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Data Mentah", len(df_cols))
            c2.metric("Baris Kosong", missing_rows)
            c3.metric("Avg Umur (N)", round(float(df_cols['Umur (N)'].mean(skipna=True)), 2))
            c4.metric("Avg RT (N)", round(float(df_cols['RT (N)'].mean(skipna=True)), 2))

            st.markdown('<div class="sec-label">Pratinjau Data Mentah — 10 Baris Pertama</div>', unsafe_allow_html=True)
            st.dataframe(df_cols.head(10), use_container_width=True, hide_index=True)

        render_nav_buttons(step, len(STEPS),
                            next_label="Lanjut ke Preprocessing →",
                            next_disabled=(st.session_state.raw_df is None),
                            on_next=lambda: goto(1))

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — PREPROCESSING
    # ══════════════════════════════════════════════════════════════════════════
    elif step == 1:
        if st.session_state.raw_df is None:
            st.warning("Silakan upload data terlebih dahulu.")
            render_nav_buttons(step, len(STEPS))
        else:
            st.caption(
                "Baris dengan nilai kosong pada salah satu kolom dibuang; sisanya dipakai apa adanya "
                "tanpa filtering tambahan (identik dengan perhitungan manual di Excel)."
            )
            df_raw = st.session_state.raw_df
            missing_counts = {c: int(df_raw[c].isna().sum()) for c in COLS}
            df_clean = df_raw.dropna(subset=COLS).reset_index(drop=True)
            st.session_state.clean_df = df_clean

            total_awal = len(df_raw)
            total_bersih = len(df_clean)
            total_dibuang = total_awal - total_bersih

            st.success(
                f"🧹 Preprocessing selesai — dari **{total_awal}** baris mentah, **{total_bersih}** baris "
                f"valid dipakai untuk clustering ({total_dibuang} baris dibuang karena nilai kosong)."
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Data Awal", total_awal)
            c2.metric("Data Bersih", total_bersih)
            c3.metric("Dibuang", total_dibuang)
            c4.metric("% Data Valid", f"{(total_bersih/total_awal*100) if total_awal else 0:.1f}%")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown('<div class="sec-label">Missing Value per Kolom</div>', unsafe_allow_html=True)
                st.dataframe(
                    pd.DataFrame(list(missing_counts.items()), columns=["Kolom", "Jumlah Missing"]),
                    use_container_width=True, hide_index=True,
                )
            with col_b:
                st.markdown('<div class="sec-label">Statistik Deskriptif (Data Bersih)</div>', unsafe_allow_html=True)
                if total_bersih:
                    st.dataframe(df_clean[COLS].astype(float).describe().T[['min', 'max', 'mean', 'std']],
                                 use_container_width=True)
                else:
                    st.info("Tidak ada data bersih untuk dihitung statistiknya.")

            st.markdown('<div class="sec-label">Pratinjau Data Bersih — 10 Baris Pertama</div>', unsafe_allow_html=True)
            st.dataframe(df_clean.head(10), use_container_width=True, hide_index=True)

            render_nav_buttons(step, len(STEPS),
                                next_label="Lanjut ke Pilih Cluster →",
                                next_disabled=(total_bersih == 0),
                                on_next=lambda: goto(2))

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — PILIH JUMLAH CLUSTER (MANUAL)
    # ══════════════════════════════════════════════════════════════════════════
    elif step == 2:
        if st.session_state.clean_df is None:
            st.warning("Lakukan preprocessing data terlebih dahulu.")
            render_nav_buttons(step, len(STEPS))
        else:
            st.caption(
                "Tentukan sendiri jumlah cluster (K) berdasarkan pertimbangan Anda (mis. jumlah kategori "
                "RT/wilayah, kebutuhan analisis, dsb). Elbow Method pada tahap berikutnya dapat dipakai "
                "untuk memvalidasi atau menyesuaikan pilihan ini."
            )
            with st.container(border=True):
                st.markdown('<div class="sec-label">Masukkan Nilai K</div>', unsafe_allow_html=True)
                k_val = st.number_input(
                    "Masukkan Nilai K (2 – 10)", min_value=2, max_value=10,
                    value=int(st.session_state.manual_k), step=1, label_visibility="collapsed",
                )
                st.session_state.manual_k = int(k_val)

                st.markdown('<div class="sec-label">Atau Pilih Cepat</div>', unsafe_allow_html=True)
                quick_cols = st.columns(9)
                for i, kk in enumerate(range(2, 11)):
                    if quick_cols[i].button(str(kk), key=f"quickk_{kk}",
                                             type="primary" if kk == st.session_state.manual_k else "secondary",
                                             use_container_width=True):
                        st.session_state.manual_k = kk
                        st.rerun()

                st.write("")
                st.markdown(f'K terpilih (manual): {badge(1, str(st.session_state.manual_k))}', unsafe_allow_html=True)

            render_nav_buttons(step, len(STEPS),
                                next_label="Tetapkan K & Lanjut ke Elbow Method →",
                                on_next=lambda: (st.session_state.update(K=st.session_state.manual_k), goto(3)))

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — ELBOW METHOD
    # ══════════════════════════════════════════════════════════════════════════
    elif step == 3:
        if st.session_state.clean_df is None:
            st.warning("Lakukan preprocessing data terlebih dahulu.")
            render_nav_buttons(step, len(STEPS))
        else:
            X = st.session_state.clean_df[COLS].values

            if st.session_state.wcss is None:
                with st.spinner("Menghitung WCSS untuk K=1 hingga K=10..."):
                    wcss = []
                    for k in range(1, 11):
                        model = KMeans(n_clusters=k, random_state=42, n_init=10)
                        model.fit(X)
                        wcss.append(round(float(model.inertia_), 4))
                    st.session_state.wcss = wcss

            wcss = st.session_state.wcss
            st.info(f"📌 K yang Anda pilih secara manual sebelumnya: **{st.session_state.manual_k}**. "
                    f"Gunakan grafik & tabel di bawah untuk memvalidasi, atau ubah pilihan K jika diperlukan.")

            st.markdown('<div class="sec-label">Grafik Elbow — WCSS vs K</div>', unsafe_allow_html=True)
            ks = list(range(1, 11))
            chosen_k = int(st.session_state.K)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ks, y=wcss, mode='lines+markers', line=dict(color='#A855F7', width=3, shape='spline'),
                marker=dict(size=8, color='#150F2B', line=dict(color='#A855F7', width=2)),
                fill='tozeroy', fillcolor='rgba(168,85,247,.14)', name='WCSS', hovertemplate='K=%{x}<br>WCSS=%{y:,.0f}<extra></extra>',
            ))
            if chosen_k in ks:
                fig.add_trace(go.Scatter(
                    x=[chosen_k], y=[wcss[chosen_k - 1]], mode='markers', marker=dict(size=15, color='#22D3EE', symbol='circle',
                                                                                        line=dict(color='#FFFFFF', width=1.5)),
                    name=f'K terpilih ({chosen_k})', hovertemplate=f'K terpilih = {chosen_k}<extra></extra>',
                ))
            # --- FIX: jangan unpack **PLOTLY_LAYOUT lalu kirim ulang key 'xaxis'/'yaxis'.
            # Gabungkan dulu overridenya via merged_layout(), baru unpack SEKALI.
            fig.update_layout(**merged_layout(
                PLOTLY_LAYOUT,
                height=280,
                showlegend=False,
                xaxis=dict(title='K', dtick=1),
                yaxis=dict(title='WCSS'),
            ))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            st.markdown('<div class="sec-label">Tabel Nilai WCSS</div>', unsafe_allow_html=True)
            chart_df = pd.DataFrame({'K': ks, 'WCSS': wcss}).set_index('K')
            st.dataframe(chart_df.T, use_container_width=True)

            with st.container(border=True):
                st.markdown('<div class="sec-label">Konfirmasi Jumlah Cluster (K)</div>', unsafe_allow_html=True)
                k_confirm = st.number_input(
                    "K", min_value=2, max_value=10, value=int(st.session_state.K), step=1, key="k_confirm",
                    label_visibility="collapsed",
                )
                st.session_state.K = int(k_confirm)
                st.markdown(f'K terpilih: {badge(1, str(st.session_state.K))}', unsafe_allow_html=True)

            render_nav_buttons(step, len(STEPS),
                                next_label="Atur Centroid Awal →",
                                on_next=lambda: goto(4))

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4 — CENTROID AWAL
    # ══════════════════════════════════════════════════════════════════════════
    elif step == 4:
        if st.session_state.clean_df is None:
            st.warning("Lakukan preprocessing data terlebih dahulu.")
            render_nav_buttons(step, len(STEPS))
        else:
            K = st.session_state.K
            df_clean = st.session_state.clean_df

            st.info("💡 Nilai centroid harus diisi dalam skala data: " + ", ".join(f"**{c}**" for c in COLS) + ".")

            if st.session_state.centroids_input is None or len(st.session_state.centroids_input) != K:
                st.session_state.centroids_input = [[0.0] * len(COLS) for _ in range(K)]

            col_fill, col_run = st.columns([1, 1])
            with col_fill:
                if st.button("🎲 Isi Acak dari Data", use_container_width=True):
                    n = len(df_clean)
                    idx = np.random.choice(n, size=min(K, n), replace=False) if n > 0 else []
                    for c, di in enumerate(idx):
                        st.session_state.centroids_input[c] = [float(df_clean.iloc[di][col]) for col in COLS]
                    st.rerun()

            st.markdown('<div class="sec-label">Koordinat Centroid per Cluster</div>', unsafe_allow_html=True)
            for c in range(K):
                with st.expander(f"Centroid C{c+1}", expanded=(c == 0)):
                    st.markdown(
                        f'<div style="height:4px;border-radius:4px;background:{ccolor(c+1)};margin-bottom:12px;width:60px;'
                        f'box-shadow:0 0 10px {ccolor(c+1)}77;"></div>',
                        unsafe_allow_html=True,
                    )
                    row_cols = st.columns(len(COLS))
                    for ci, col in enumerate(COLS):
                        val = row_cols[ci].number_input(
                            col, value=float(st.session_state.centroids_input[c][ci]),
                            key=f"cen_{c}_{ci}", format="%.4f",
                        )
                        st.session_state.centroids_input[c][ci] = val

            st.write("")
            st.markdown("<hr style='margin:6px 0 16px;opacity:.35'>", unsafe_allow_html=True)
            col_back, col_run_btn = st.columns([1, 2])
            with col_back:
                if st.button("← Kembali", use_container_width=True, key="back_4"):
                    goto(3)
            with col_run_btn:
                if st.button("▶ Jalankan K-Means", type="primary", use_container_width=True):
                    X = df_clean[COLS].values
                    with st.spinner("Menjalankan K-Means..."):
                        all_iterations, centroid_akhir, final_cluster = run_kmeans_manual(
                            X, st.session_state.centroids_input, K
                        )
                        score = 0.0
                        if final_cluster is not None and len(set(final_cluster)) > 1:
                            score = round(float(silhouette_score(X, final_cluster)), 4)

                        hasil_rows = []
                        for i in range(len(X)):
                            row = {c: round(float(X[i, ci]), 6) for ci, c in enumerate(COLS)}
                            row['CLUSTER'] = int(final_cluster[i])
                            hasil_rows.append(row)

                        st.session_state.iterations = all_iterations
                        st.session_state.centroid_akhir = centroid_akhir.tolist()
                        st.session_state.silhouette = score
                        st.session_state.hasil = hasil_rows
                        st.session_state.current_iter = 0

                    goto(5)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5 — PROSES ITERASI
    # ══════════════════════════════════════════════════════════════════════════
    elif step == 5:
        if not st.session_state.iterations:
            st.warning("Jalankan K-Means terlebih dahulu di tahap Centroid Awal.")
            render_nav_buttons(step, len(STEPS))
        else:
            iterations = st.session_state.iterations
            K = st.session_state.K
            total = len(iterations)
            grand_total = len(st.session_state.hasil)

            st.success(f"Selesai dalam **{total} iterasi** · Konvergen pada iterasi {total} ✅")

            idx = st.slider("Pilih Iterasi", 1, total, st.session_state.current_iter + 1) - 1
            st.session_state.current_iter = idx
            it = iterations[idx]

            st.markdown(f'<div class="sec-label">📊 Anggota Cluster — Iterasi {idx+1}</div>', unsafe_allow_html=True)
            cols_m = st.columns(K + 1)
            for c in range(K):
                cnt = it['anggota'][f'C{c+1}']
                pct = (cnt / grand_total * 100) if grand_total else 0
                cols_m[c].metric(f"Cluster {c+1}", cnt, f"{pct:.1f}%")
            cols_m[K].metric("Total", grand_total)

            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown(f'<div class="sec-label">Centroid Lama (Iterasi {idx+1})</div>', unsafe_allow_html=True)
                st.dataframe(
                    pd.DataFrame(it['centroid_lama'], columns=COLS, index=[f"C{c+1}" for c in range(K)]),
                    use_container_width=True,
                )
            with col_r:
                st.markdown(f'<div class="sec-label">Centroid Baru (Rata-Rata Iterasi {idx+1})</div>', unsafe_allow_html=True)
                cen_baru_df = pd.DataFrame(it['centroid_baru'], columns=COLS, index=[f"C{c+1}" for c in range(K)])
                cen_baru_df.insert(0, 'n', [it['anggota'][f'C{c+1}'] for c in range(K)])
                st.dataframe(cen_baru_df, use_container_width=True)

            st.markdown('<div class="sec-label">Tabel Jarak Euclidean & Klaster (10 baris pertama, selengkapnya di ekspor Excel)</div>', unsafe_allow_html=True)
            jarak = it['jarak']
            cluster = it['cluster']
            tbl_rows = []
            hasil = st.session_state.hasil
            for i in range(min(10, len(jarak))):
                row = {col: hasil[i][col] for col in COLS}
                for c in range(K):
                    row[f"D ke C{c+1}"] = round(float(jarak[i, c]), 6)
                row['Min'] = round(float(jarak[i].min()), 6)
                row['Klaster'] = int(cluster[i])
                tbl_rows.append(row)
            tbl_df = pd.DataFrame(tbl_rows)
            st.dataframe(style_cluster_col(tbl_df, 'Klaster'), use_container_width=True)

            if it['konvergen'] and idx > 0:
                st.success(
                    f"✅ **Bukti Konvergensi** — Cluster assignment iterasi **{idx}** = iterasi **{idx+1}** "
                    f"(identik, tidak ada data yang berpindah cluster). Algoritma dinyatakan **KONVERGEN**."
                )

            render_nav_buttons(step, len(STEPS),
                                next_label="Lihat Hasil Akhir 🏁",
                                on_next=lambda: goto(6))

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 6 — HASIL AKHIR
    # ══════════════════════════════════════════════════════════════════════════
    elif step == 6:
        if not st.session_state.hasil:
            st.warning("Jalankan K-Means terlebih dahulu.")
            render_nav_buttons(step, len(STEPS))
        else:
            hasil = st.session_state.hasil
            cen = st.session_state.centroid_akhir
            sil = st.session_state.silhouette
            K = st.session_state.K
            nIter = len(st.session_state.iterations)

            df_hasil = pd.DataFrame(hasil)
            counts = [int((df_hasil['CLUSTER'] == c + 1).sum()) for c in range(K)]
            grand_total = sum(counts)

            st.success(
                f"🎉 K-Means selesai dalam **{nIter} iterasi** dan konvergen. "
                f"K = **{K}** · Total data: **{grand_total}**"
            )

            cols_m = st.columns(K + 1)
            for c in range(K):
                pct = counts[c] / grand_total * 100 if grand_total else 0
                cols_m[c].metric(f"Cluster {c+1}", counts[c], f"{pct:.1f}%")
            cols_m[K].metric("Total Data", grand_total)

            st.markdown('<div class="sec-label">📈 Rekap Anggota per Iterasi</div>', unsafe_allow_html=True)
            rekap_rows = []
            for i, it in enumerate(st.session_state.iterations):
                row = {"Iterasi": i + 1}
                for c in range(K):
                    row[f"C{c+1}"] = it['anggota'][f'C{c+1}']
                row["Total"] = grand_total
                row["Status"] = "✓ Konvergen" if it['konvergen'] else ""
                rekap_rows.append(row)
            rekap_rows.append({
                "Iterasi": "Hasil Akhir",
                **{f"C{c+1}": counts[c] for c in range(K)},
                "Total": grand_total, "Status": "",
            })
            st.dataframe(pd.DataFrame(rekap_rows), use_container_width=True, hide_index=True)

            col_sil, col_don = st.columns([1, 1])
            with col_sil:
                st.markdown('<div class="sec-label">Silhouette Score</div>', unsafe_allow_html=True)
                qual = "Sangat Baik" if sil > 0.7 else "Baik" if sil > 0.5 else "Cukup" if sil > 0.25 else "Lemah"
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=sil, number={'suffix': f"  ·  {qual}", 'font': {'size': 22, 'color': '#F472B6'}},
                    gauge={
                        'axis': {'range': [-1, 1], 'tickcolor': '#726A9C', 'tickfont': {'size': 9}},
                        'bar': {'color': '#A855F7', 'thickness': .35},
                        'bgcolor': 'rgba(0,0,0,0)', 'borderwidth': 0,
                        'steps': [
                            {'range': [-1, .25], 'color': 'rgba(251,113,133,.18)'},
                            {'range': [.25, .5], 'color': 'rgba(251,191,36,.18)'},
                            {'range': [.5, 1], 'color': 'rgba(45,212,191,.22)'},
                        ],
                    },
                ))
                # --- FIX: 'margin' di-merge dulu (bukan konflik), lalu 'height' aman
                # karena bukan key dari PLOTLY_LAYOUT. Tetap dirapikan lewat merged_layout()
                # agar konsisten dan tahan bila PLOTLY_LAYOUT berubah nanti.
                gauge.update_layout(**merged_layout(
                    PLOTLY_LAYOUT,
                    margin=dict(l=20, r=20, t=10, b=10),
                    height=200,
                ))
                st.plotly_chart(gauge, use_container_width=True, config={'displayModeBar': False})
                st.caption("Rentang: −1 (buruk) hingga +1 (sempurna). Nilai > 0.5 = baik.")
            with col_don:
                st.markdown('<div class="sec-label">Distribusi Data per Cluster</div>', unsafe_allow_html=True)
                donut = go.Figure(go.Pie(
                    labels=[f"Cluster {c+1}" for c in range(K)], values=counts, hole=.62,
                    marker=dict(colors=[ccolor(c + 1) for c in range(K)], line=dict(color='#0B0817', width=2)),
                    textinfo='percent',
                    textfont=dict(size=11, color=[cluster_text_color(ccolor(c + 1)) for c in range(K)]),
                    hovertemplate='%{label}: %{value}<extra></extra>',
                ))
                # --- FIX: sebelumnya 'legend' dikirim baik lewat **PLOTLY_LAYOUT (yang sudah
                # punya key 'legend') MAUPUN sebagai keyword eksplisit -> TypeError.
                # Sekarang di-merge sekali via merged_layout().
                donut.update_layout(**merged_layout(
                    PLOTLY_LAYOUT,
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=200,
                    showlegend=True,
                    legend=dict(orientation='h', y=-.15, font=dict(size=10)),
                ))
                st.plotly_chart(donut, use_container_width=True, config={'displayModeBar': False})

            st.markdown('<div class="sec-label">Centroid Akhir</div>', unsafe_allow_html=True)
            st.dataframe(
                pd.DataFrame(cen, columns=COLS, index=[f"C{c+1}" for c in range(K)]),
                use_container_width=True,
            )

            st.markdown('<div class="sec-label">Tabel Hasil Akhir — 50 Baris Pertama (lengkap di file ekspor)</div>', unsafe_allow_html=True)
            st.dataframe(style_cluster_col(df_hasil.head(50), 'CLUSTER'), use_container_width=True, hide_index=True)

            # ── Ekspor ke Excel ──
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_hasil[COLS + ['CLUSTER']].to_excel(writer, sheet_name='Hasil_Akhir', index=False)
                pd.DataFrame(cen, columns=COLS, index=[f"C{c+1}" for c in range(K)]).to_excel(
                    writer, sheet_name='Centroid_Akhir'
                )
                for it in st.session_state.iterations:
                    jarak = it['jarak']
                    cluster = it['cluster']
                    rows = []
                    for i in range(len(jarak)):
                        row = {col: hasil[i][col] for col in COLS}
                        for c in range(K):
                            row[f"C{c+1}"] = round(float(jarak[i, c]), 8)
                        row['jarak_min'] = round(float(jarak[i].min()), 8)
                        row['Cluster'] = int(cluster[i])
                        rows.append(row)
                    pd.DataFrame(rows).to_excel(writer, sheet_name=f"Iterasi_{it['iterasi']}", index=False)
            output.seek(0)

            st.write("")
            st.markdown("<hr style='margin:6px 0 16px;opacity:.35'>", unsafe_allow_html=True)
            col_back, col_dl = st.columns([1, 2])
            with col_back:
                if st.button("← Kembali", use_container_width=True, key="back_6"):
                    goto(5)
            with col_dl:
                st.download_button(
                    "💾 Ekspor ke Excel (.xlsx)", data=output, file_name="KMeans_Pemilih.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )