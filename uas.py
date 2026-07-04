# -*- coding: utf-8 -*-
"""
K-Means Clustering — Pengelompokan Data Pemilih
Disesuaikan dengan dataset: Status Memilih, Umur (N), Status Kawin,
Jenis Kelamin, RT (N), Disabilitas — menggunakan K = 6 cluster (default).

Alur kerja:
  1. Upload Data
  2. Preprocessing (cek missing value, bersihkan data, statistik deskriptif)
  3. Pilih Jumlah Cluster (Manual)
  4. Elbow Method (validasi / opsi ubah K)
  5. Centroid Awal
  6. Proses Iterasi
  7. Hasil Akhir

Konvergensi berbasis perubahan cluster assignment (sama seperti perhitungan manual Excel).
"""

from flask import Flask, request, jsonify, send_file, render_template_string
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import io
import traceback

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

SESSION = {}

# ── Kolom yang dipakai untuk clustering (sesuai dataset yang diunggah) ──
COLS = ['Status Memilih', 'Umur (N)', 'Status Kawin', 'Jenis Kelamin', 'RT (N)', 'Disabilitas']
COL_LABELS = ['Status Memilih', 'Umur (N)', 'Status Kawin', 'Jenis Kelamin', 'RT (N)', 'Disabilitas']
DEFAULT_K = 6

# ══════════════════════════════════════════════════════════════════════════════
LANDING_HTML = r"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>K-Means · Data Pemilih</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}html{scroll-behavior:smooth}body{font-family:'Plus Jakarta Sans',system-ui,sans-serif;background:#0A091A;color:#E2E0F0;overflow-x:hidden;min-height:100vh}:root{--accent:#F5B942;--accent-dim:#C88A1A;--accent-glow:rgba(245,185,66,.22);--teal:#2BC9A2;--coral:#F0694A;--amber:#F5B942;--blue:#5BA4F5;--purple:#9B8EFF;--bg:#0A091A;--bg2:#0F0E1A;--surface:#17152E;--surface2:#1E1C38;--border:#252340;--text1:#E2E0F0;--text2:#8F8DB0;--text3:#504E72}::-webkit-scrollbar{width:5px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:#252340;border-radius:99px}.navbar{position:fixed;top:0;left:0;right:0;z-index:100;padding:0 48px;height:64px;display:flex;align-items:center;justify-content:space-between;background:rgba(10,9,26,.8);backdrop-filter:blur(20px);border-bottom:1px solid rgba(37,35,64,.5)}.nav-logo{display:flex;align-items:center;gap:10px}.nav-logo-icon{width:36px;height:36px;border-radius:9px;background:linear-gradient(135deg,var(--amber),var(--accent-dim));display:flex;align-items:center;justify-content:center;font-size:18px;box-shadow:0 0 20px var(--accent-glow)}.nav-logo-text{font-size:15px;font-weight:700;color:#fff;letter-spacing:-.3px}.nav-logo-sub{font-size:10px;color:var(--text3);font-weight:500;letter-spacing:.05em}.nav-cta{padding:8px 18px;border-radius:8px;font-size:13px;font-weight:700;background:linear-gradient(135deg,var(--amber),var(--accent-dim));color:#fff;text-decoration:none;cursor:pointer;box-shadow:0 0 20px var(--accent-glow);transition:all .2s}.nav-cta:hover{filter:brightness(1.12);transform:translateY(-1px)}.hero{min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:100px 24px 60px;position:relative;overflow:hidden}.hero-grid{position:absolute;inset:0;background-image:linear-gradient(rgba(245,185,66,.04) 1px,transparent 1px),linear-gradient(90deg,rgba(245,185,66,.04) 1px,transparent 1px);background-size:50px 50px;mask-image:radial-gradient(ellipse 80% 60% at 50% 50%,#000 30%,transparent 100%)}.glow-orb{position:absolute;border-radius:50%;filter:blur(80px);pointer-events:none}.glow-1{width:500px;height:500px;background:rgba(245,185,66,.1);top:-100px;left:50%;transform:translateX(-50%)}.hero-content{position:relative;z-index:1;max-width:800px}.hero-eyebrow{display:inline-flex;align-items:center;gap:8px;padding:6px 14px;border-radius:99px;background:rgba(245,185,66,.1);border:1px solid rgba(245,185,66,.2);font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--amber);margin-bottom:28px}.eyebrow-dot{width:6px;height:6px;border-radius:50%;background:var(--teal);animation:pulse 2s ease infinite}@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}.hero-title{font-size:clamp(28px,5vw,62px);font-weight:800;line-height:1.08;letter-spacing:-.04em;color:#fff;margin-bottom:20px}.hero-title .grad{background:linear-gradient(135deg,var(--amber) 20%,var(--coral) 80%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}.hero-desc{font-size:16px;line-height:1.7;color:var(--text2);max-width:580px;margin:0 auto 40px}.hero-actions{display:flex;align-items:center;justify-content:center;gap:12px;flex-wrap:wrap}.btn-hero-primary{display:inline-flex;align-items:center;gap:8px;padding:14px 28px;border-radius:12px;font-size:15px;font-weight:700;text-decoration:none;background:linear-gradient(135deg,var(--amber),var(--accent-dim));color:#fff;box-shadow:0 4px 30px var(--accent-glow);transition:all .2s}.btn-hero-primary:hover{transform:translateY(-2px);filter:brightness(1.1)}.footer{padding:24px 48px;border-top:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;font-size:12px;color:var(--text3)}.footer a{color:var(--amber);text-decoration:none}
</style>
</head>
<body>
<nav class="navbar">
  <div class="nav-logo">
    <div class="nav-logo-icon">🗳️</div>
    <div><div class="nav-logo-text">K-Means Clustering</div><div class="nav-logo-sub">DATA PEMILIH</div></div>
  </div>
  <a href="/app" class="nav-cta">Mulai Analisis →</a>
</nav>
<section class="hero">
  <div class="hero-grid"></div>
  <div class="glow-orb glow-1"></div>
  <div class="hero-content">
    <div class="hero-eyebrow"><span class="eyebrow-dot"></span>Sistem Analisis Data Pemilih</div>
    <h1 class="hero-title">Pengelompokan<br><span class="grad">Data Pemilih</span><br>dengan K-Means</h1>
    <p class="hero-desc">Klasterisasi berdasarkan Status Memilih, Umur, Status Kawin, Jenis Kelamin, RT, dan Disabilitas — lengkap dengan preprocessing, pemilihan cluster manual, dan elbow method.</p>
    <div class="hero-actions"><a href="/app" class="btn-hero-primary">🗳️ Mulai Analisis Sekarang</a></div>
  </div>
</section>
<footer class="footer"><div>K-Means Clustering · Data Pemilih</div><div>Flask + Chart.js + scikit-learn</div></footer>
</body>
</html>"""

HTML = r"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>K-Means · Data Pemilih</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}html,body{height:100%;overflow:hidden}body{font-family:'Plus Jakarta Sans',system-ui,sans-serif;font-size:14px;display:flex;background:#0F0E1A;color:#E2E0F0}::-webkit-scrollbar{width:4px;height:4px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:#2E2B4A;border-radius:99px}:root{--accent:#F5B942;--accent-dim:#C88A1A;--accent-glow:rgba(245,185,66,.18);--teal:#2BC9A2;--coral:#F0694A;--amber:#F5B942;--blue:#5BA4F5;--purple:#9B8EFF;--bg:#0F0E1A;--bg2:#15132A;--bg3:#1C1A35;--surface:#201E3A;--surface2:#272548;--border:#2E2B50;--border2:#3A3760;--text1:#E2E0F0;--text2:#8F8DB0;--text3:#504E72;--sidebar:#12102A;--r4:4px;--r8:8px;--r12:12px;--r16:16px;--r24:24px}
.sidebar{width:230px;flex-shrink:0;background:var(--sidebar);display:flex;flex-direction:column;height:100vh;border-right:1px solid var(--border)}.sidebar-brand{padding:28px 22px 20px;border-bottom:1px solid var(--border)}.sidebar-brand .logo{display:flex;align-items:center;gap:10px;margin-bottom:6px}.logo-icon{width:34px;height:34px;border-radius:var(--r8);background:linear-gradient(135deg,var(--amber),var(--accent-dim));display:flex;align-items:center;justify-content:center;font-size:18px;box-shadow:0 0 16px var(--accent-glow)}.logo-text{font-size:15px;font-weight:700;color:#fff;line-height:1.2}.logo-sub{font-size:10px;color:var(--text3);margin-top:2px;line-height:1.4}.sidebar-nav{flex:1;padding:12px 10px;overflow-y:auto}.nav-group{font-size:9px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--text3);padding:6px 12px 4px;margin-top:8px}.nav-item{display:flex;align-items:center;gap:10px;padding:9px 12px;border-radius:var(--r8);color:var(--text2);font-size:13px;font-weight:500;cursor:pointer;transition:all .15s;margin-bottom:2px;user-select:none;position:relative}.nav-icon{font-size:15px;width:20px;text-align:center;flex-shrink:0}.nav-item:hover{background:var(--surface);color:var(--text1)}.nav-item.active{background:var(--surface2);color:var(--amber)}.nav-item.active::before{content:'';position:absolute;left:0;top:20%;bottom:20%;width:3px;border-radius:0 3px 3px 0;background:var(--amber)}.nav-item.done .nav-label::after{content:' ✓';color:var(--teal);font-size:11px}.sidebar-foot{padding:14px 22px;font-size:10px;color:var(--text3);border-top:1px solid var(--border)}.sidebar-foot strong{color:var(--amber)}.back-home{display:flex;align-items:center;gap:8px;padding:9px 12px;border-radius:var(--r8);color:var(--text3);font-size:12px;font-weight:500;cursor:pointer;transition:all .15s;margin-bottom:6px;text-decoration:none}.back-home:hover{background:var(--surface);color:var(--text2)}.main{flex:1;display:flex;flex-direction:column;height:100vh;overflow:hidden}.topbar{background:var(--bg2);border-bottom:1px solid var(--border);padding:0 20px;height:52px;flex-shrink:0;display:flex;align-items:center;gap:0;overflow-x:auto}.pb-step{display:flex;align-items:center;flex-shrink:0}.pb-dot{width:24px;height:24px;border-radius:50%;border:1.5px solid var(--border2);display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:700;color:var(--text3);transition:all .3s;flex-shrink:0}.pb-dot.active{background:var(--amber);border-color:var(--amber);color:#fff;box-shadow:0 0 12px var(--accent-glow)}.pb-dot.done{background:var(--teal);border-color:var(--teal);color:#fff}.pb-label{font-size:10px;color:var(--text3);margin-left:6px;white-space:nowrap;transition:color .3s;font-weight:500}.pb-label.active{color:var(--amber);font-weight:600}.pb-label.done{color:var(--teal)}.pb-line{height:1px;width:18px;background:var(--border);margin:0 6px;transition:background .3s;flex-shrink:0}.pb-line.done{background:var(--teal);opacity:.5}.content{flex:1;overflow-y:auto;padding:28px 36px 60px}.pane{display:none}.pane.active{display:block}.page-title{font-size:21px;font-weight:700;color:#fff;margin-bottom:3px;letter-spacing:-.3px}.page-sub{font-size:13px;color:var(--text2);margin-bottom:24px;line-height:1.6}.sec-label{font-size:9px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--text3);margin-bottom:10px}.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r16);padding:20px 22px;margin-bottom:14px}.card-header{padding:14px 20px;font-size:12px;font-weight:600;color:var(--text2);border-bottom:1px solid var(--border)}.metric-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin-bottom:14px}.metric{background:var(--bg3);border:1px solid var(--border);border-radius:var(--r12);padding:14px 16px;position:relative;overflow:hidden}.metric::before{content:'';position:absolute;inset:0;background:linear-gradient(135deg,var(--m-color,var(--amber)),transparent);opacity:.05}.metric-label{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--text3);margin-bottom:6px}.metric-value{font-size:22px;font-weight:700;color:var(--m-color,var(--amber))}.btn{display:inline-flex;align-items:center;gap:6px;padding:9px 18px;border-radius:var(--r8);font-size:13px;font-weight:600;font-family:inherit;cursor:pointer;border:1px solid var(--border2);background:var(--surface2);color:var(--text1);transition:all .15s;white-space:nowrap}.btn:hover{background:var(--surface);border-color:var(--text3)}.btn:active{transform:scale(.98)}.btn-primary{background:linear-gradient(135deg,var(--amber),var(--accent-dim));color:#fff;border:none;box-shadow:0 2px 12px var(--accent-glow)}.btn-primary:hover{filter:brightness(1.1)}.btn-teal{background:var(--teal);color:#0a1a15;border:none;font-weight:700}.btn-teal:hover{filter:brightness(1.08)}.btn-ghost{background:transparent;border:1px solid var(--amber);color:var(--amber)}.btn-ghost:hover{background:var(--accent-glow)}.btn-row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.btn-row.right{justify-content:flex-end}.upload-zone{border:2px dashed var(--border2);border-radius:var(--r24);padding:56px 24px;text-align:center;cursor:pointer;transition:all .2s;background:var(--bg3);position:relative;overflow:hidden}.upload-zone:hover,.upload-zone.drag{border-color:var(--amber);background:rgba(245,185,66,.06)}.upload-icon{font-size:52px;margin-bottom:14px;display:block}.upload-zone h3{font-size:16px;font-weight:700;color:#fff;margin-bottom:6px}.upload-zone p{font-size:12px;color:var(--text2)}.upload-hint{margin-top:12px;display:inline-flex;gap:8px;font-size:11px;color:var(--text3)}.upload-hint span{background:var(--surface2);border:1px solid var(--border2);padding:3px 10px;border-radius:20px}.table-wrap{overflow-x:auto;border-radius:var(--r8);border:1px solid var(--border)}.table-scroll{max-height:300px;overflow-y:auto}table{width:100%;border-collapse:collapse;font-size:12px}thead th{background:var(--bg3);padding:9px 12px;text-align:left;font-weight:600;font-size:10px;color:var(--text3);border-bottom:1px solid var(--border);white-space:nowrap;position:sticky;top:0;z-index:1}tbody td{padding:8px 12px;border-bottom:1px solid var(--border);color:var(--text1);white-space:nowrap}tbody tr:last-child td{border-bottom:none}tbody tr:hover td{background:var(--surface)}.mono{font-family:'JetBrains Mono',monospace;font-size:11px}.badge{display:inline-flex;align-items:center;justify-content:center;min-width:26px;height:22px;padding:0 7px;border-radius:20px;font-size:10px;font-weight:700}.b1{background:rgba(245,185,66,.15);color:#F5B942;border:1px solid rgba(245,185,66,.25)}.b2{background:rgba(43,201,162,.12);color:#2BC9A2;border:1px solid rgba(43,201,162,.25)}.b3{background:rgba(240,105,74,.12);color:#F0694A;border:1px solid rgba(240,105,74,.25)}.b4{background:rgba(91,164,245,.12);color:#5BA4F5;border:1px solid rgba(91,164,245,.25)}.b5{background:rgba(155,142,255,.12);color:#9B8EFF;border:1px solid rgba(155,142,255,.25)}.b6{background:rgba(240,160,220,.12);color:#F0A0DC;border:1px solid rgba(240,160,220,.25)}.k-selector{display:flex;gap:6px;flex-wrap:wrap;margin:12px 0}.k-btn{width:38px;height:38px;border-radius:var(--r8);border:1px solid var(--border2);background:var(--surface);font-size:13px;font-weight:700;color:var(--text2);cursor:pointer;transition:all .15s;font-family:inherit}.k-btn:hover{border-color:var(--amber);color:var(--amber)}.k-btn.selected{background:var(--amber);color:#0f0e1a;border-color:var(--amber);box-shadow:0 2px 10px var(--accent-glow)}.centroid-block{border:1px solid var(--border);border-radius:var(--r12);overflow:hidden;margin-bottom:10px;transition:border-color .2s}.centroid-block:focus-within{border-color:var(--amber)}.centroid-header{padding:10px 16px;font-weight:700;font-size:12px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:8px}.centroid-body{padding:14px 16px;display:grid;grid-template-columns:repeat(3,1fr);gap:12px;background:var(--bg3)}.form-group label{display:block;font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--text3);margin-bottom:5px}.form-group input{width:100%;padding:8px 11px;border:1px solid var(--border2);border-radius:var(--r8);font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--text1);background:var(--surface);transition:border-color .15s;outline:none}.form-group input:focus{border-color:var(--amber);box-shadow:0 0 0 3px var(--accent-glow)}.iter-nav{display:flex;align-items:center;gap:8px;flex-wrap:wrap;background:var(--surface);border:1px solid var(--border);border-radius:var(--r12);padding:10px 14px;margin-bottom:14px}.iter-nav-info{flex:1;font-size:13px;font-weight:600;color:var(--text1)}.iter-badge{display:inline-block;font-size:10px;font-weight:700;background:rgba(43,201,162,.12);color:var(--teal);border:1px solid rgba(43,201,162,.25);padding:2px 10px;border-radius:20px;margin-left:8px}.alert{display:flex;align-items:flex-start;gap:10px;padding:12px 16px;border-radius:var(--r12);font-size:13px;margin-bottom:14px;line-height:1.5}.alert-info{background:rgba(245,185,66,.08);color:#F5C76A;border:1px solid rgba(245,185,66,.2)}.alert-success{background:rgba(43,201,162,.1);color:#2BC9A2;border:1px solid rgba(43,201,162,.2)}.alert-warn{background:rgba(240,105,74,.08);color:#F0947F;border:1px solid rgba(240,105,74,.2)}.score-row{display:flex;align-items:center;gap:14px;margin:8px 0 4px}.score-num{font-size:32px;font-weight:700;color:var(--teal)}.score-track{flex:1;height:6px;background:var(--bg3);border-radius:99px;overflow:hidden}.score-fill{height:100%;background:linear-gradient(90deg,var(--teal),var(--amber));border-radius:99px;transition:width .9s ease}.score-qual{font-size:12px;font-weight:700;color:var(--text2)}.cen-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:8px;margin-bottom:16px}.cen-card{padding:14px 16px;border-radius:var(--r12);border:1px solid var(--border);background:var(--bg3);border-top:3px solid var(--c-color,var(--amber))}.cen-card-title{font-size:12px;font-weight:700;margin-bottom:10px;color:var(--c-color,var(--amber))}.cen-card-row{font-size:11px;color:var(--text2);margin-bottom:4px;line-height:1.4}.cen-card-row span{font-family:'JetBrains Mono',monospace;color:var(--text1);font-size:11px}.spinner{display:inline-block;width:18px;height:18px;border:2px solid var(--border2);border-top-color:var(--amber);border-radius:50%;animation:spin .7s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}.loading-overlay{display:none;align-items:center;justify-content:center;gap:10px;padding:40px;color:var(--text2);font-size:13px}.perp-zero{color:var(--teal);font-weight:600}.chart-wrap{position:relative;width:100%}.big-k-input{display:flex;align-items:center;gap:12px;margin:14px 0}.big-k-input input{width:90px;padding:12px 14px;font-size:22px;font-weight:800;text-align:center;font-family:'JetBrains Mono',monospace;border:1.5px solid var(--border2);border-radius:var(--r12);background:var(--bg3);color:var(--amber);outline:none}.big-k-input input:focus{border-color:var(--amber);box-shadow:0 0 0 3px var(--accent-glow)}
</style>
</head>
<body>
<aside class="sidebar">
  <div class="sidebar-brand">
    <div class="logo"><div class="logo-icon">🗳️</div><div class="logo-text">K-Means<br>Data Pemilih</div></div>
    <div class="logo-sub">Pengelompokan Data Pemilih<br>Preprocessing → Manual K → Elbow</div>
  </div>
  <nav class="sidebar-nav" id="sideNav">
    <div class="nav-group">Alur Kerja</div>
    <div class="nav-item active" data-tab="0"><span class="nav-icon">📂</span><span class="nav-label">Upload Data</span></div>
    <div class="nav-item" data-tab="1"><span class="nav-icon">🧹</span><span class="nav-label">Preprocessing</span></div>
    <div class="nav-item" data-tab="2"><span class="nav-icon">🎯</span><span class="nav-label">Pilih Cluster (Manual)</span></div>
    <div class="nav-item" data-tab="3"><span class="nav-icon">📈</span><span class="nav-label">Elbow Method</span></div>
    <div class="nav-item" data-tab="4"><span class="nav-icon">🧭</span><span class="nav-label">Centroid Awal</span></div>
    <div class="nav-item" data-tab="5"><span class="nav-icon">🔄</span><span class="nav-label">Proses Iterasi</span></div>
    <div class="nav-item" data-tab="6"><span class="nav-icon">🏁</span><span class="nav-label">Hasil Akhir</span></div>
    <div class="nav-group" style="margin-top:16px">Navigasi</div>
    <a href="/" class="back-home">← Kembali ke Beranda</a>
  </nav>
  <div class="sidebar-foot">Dibangun dengan <strong>Flask</strong> + Chart.js</div>
</aside>

<div class="main">
  <div class="topbar" id="topbar">
    <div class="pb-step"><div class="pb-dot active" id="pd0">1</div><div class="pb-label active" id="pl0">Upload</div></div>
    <div class="pb-line" id="pline0"></div>
    <div class="pb-step"><div class="pb-dot" id="pd1">2</div><div class="pb-label" id="pl1">Preprocessing</div></div>
    <div class="pb-line" id="pline1"></div>
    <div class="pb-step"><div class="pb-dot" id="pd2">3</div><div class="pb-label" id="pl2">Pilih K</div></div>
    <div class="pb-line" id="pline2"></div>
    <div class="pb-step"><div class="pb-dot" id="pd3">4</div><div class="pb-label" id="pl3">Elbow</div></div>
    <div class="pb-line" id="pline3"></div>
    <div class="pb-step"><div class="pb-dot" id="pd4">5</div><div class="pb-label" id="pl4">Centroid</div></div>
    <div class="pb-line" id="pline4"></div>
    <div class="pb-step"><div class="pb-dot" id="pd5">6</div><div class="pb-label" id="pl5">Iterasi</div></div>
    <div class="pb-line" id="pline5"></div>
    <div class="pb-step"><div class="pb-dot" id="pd6">7</div><div class="pb-label" id="pl6">Hasil</div></div>
  </div>

  <div class="content">
    <!-- TAB 0: UPLOAD -->
    <div class="pane active" id="pane0">
      <div class="page-title">Upload Dataset Pemilih</div>
      <div class="page-sub">File Excel (.xlsx) atau CSV dengan kolom: <code>Status Memilih</code>, <code>Umur (N)</code>, <code>Status Kawin</code>, <code>Jenis Kelamin</code>, <code>RT (N)</code>, <code>Disabilitas</code><br>
      <strong style="color:var(--amber)">Data mentah akan dibersihkan pada tahap Preprocessing berikutnya.</strong></div>
      <input type="file" id="fileInput" accept=".xlsx,.xls,.csv" style="display:none">
      <div class="upload-zone" id="uploadZone" onclick="document.getElementById('fileInput').click()">
        <span class="upload-icon">🗳️</span>
        <h3>Seret file ke sini atau klik untuk memilih</h3>
        <p>Dataset pemilih mentah — akan diproses pada tahap Preprocessing</p>
        <div class="upload-hint"><span>.xlsx</span><span>.xls</span><span>.csv</span></div>
      </div>
      <div id="uploadResult" style="display:none;margin-top:20px">
        <div id="uploadAlert" class="alert alert-success"></div>
        <div class="metric-row" id="uploadMetrics"></div>
        <div class="card" id="previewTableCard" style="padding:0">
          <div class="card-header">Pratinjau Data Mentah — 10 Baris Pertama</div>
          <div class="table-wrap" style="border:none;border-radius:0"><div class="table-scroll"><table id="previewTable"></table></div></div>
        </div>
        <div class="btn-row right" style="margin-top:8px">
          <button class="btn btn-primary" onclick="goTo(1)">Lanjut ke Preprocessing →</button>
        </div>
      </div>
    </div>

    <!-- TAB 1: PREPROCESSING -->
    <div class="pane" id="pane1">
      <div class="page-title">Preprocessing Data</div>
      <div class="page-sub">Pemeriksaan nilai kosong (missing value) dan pembersihan data sebelum clustering. Baris dengan nilai kosong pada salah satu kolom dibuang; sisanya dipakai apa adanya tanpa filtering tambahan.</div>
      <div class="loading-overlay" id="preLoading" style="display:flex"><div class="spinner"></div> Memproses data...</div>
      <div id="preResult" style="display:none">
        <div class="alert alert-success" id="preAlert"></div>
        <div class="metric-row" id="preMetrics"></div>

        <div class="card">
          <div class="sec-label">Missing Value per Kolom (Data Mentah)</div>
          <div class="table-wrap"><table id="missingTable"></table></div>
        </div>

        <div class="card">
          <div class="sec-label">Statistik Deskriptif (Data Setelah Dibersihkan)</div>
          <div class="table-wrap"><table id="descTable"></table></div>
        </div>

        <div class="card" style="padding:0">
          <div class="card-header">Pratinjau Data Bersih — 10 Baris Pertama</div>
          <div class="table-wrap" style="border:none;border-radius:0"><div class="table-scroll"><table id="cleanPreviewTable"></table></div></div>
        </div>

        <div class="btn-row right">
          <button class="btn btn-primary" onclick="goTo(2)">Lanjut ke Pilih Cluster →</button>
        </div>
      </div>
    </div>

    <!-- TAB 2: PILIH CLUSTER MANUAL -->
    <div class="pane" id="pane2">
      <div class="page-title">Pilih Jumlah Cluster (Manual)</div>
      <div class="page-sub">Tentukan sendiri jumlah cluster (K) yang ingin digunakan berdasarkan pertimbangan Anda (mis. jumlah kategori RT/wilayah, kebutuhan analisis, dsb). Elbow Method pada tahap berikutnya dapat dipakai untuk memvalidasi atau menyesuaikan pilihan ini.</div>
      <div class="card">
        <div class="sec-label">Masukkan Nilai K</div>
        <div class="big-k-input">
          <input type="number" id="manualKInput" min="2" max="10" value="6">
          <div style="font-size:12px;color:var(--text2);line-height:1.5">Rentang yang didukung: <strong style="color:#fff">2 – 10</strong> cluster.</div>
        </div>
        <div class="sec-label" style="margin-top:16px">Atau Pilih Cepat</div>
        <div class="k-selector" id="manualKSelector"></div>
        <div style="margin-top:8px;font-size:12px;color:var(--text2)">K terpilih (manual): <strong id="manualKLabel" style="color:var(--amber)">6</strong></div>
      </div>
      <div class="btn-row right">
        <button class="btn btn-primary" onclick="confirmManualK()">Tetapkan K &amp; Lanjut ke Elbow Method →</button>
      </div>
    </div>

    <!-- TAB 3: ELBOW -->
    <div class="pane" id="pane3">
      <div class="page-title">Elbow Method</div>
      <div class="page-sub">WCSS dihitung untuk K=1 hingga K=10 sebagai validasi terhadap K yang Anda pilih secara manual. Anda tetap dapat menyesuaikan K di sini bila diperlukan.</div>
      <div class="loading-overlay" id="elbowLoading"><div class="spinner"></div> Menghitung WCSS...</div>
      <div id="elbowResult" style="display:none">
        <div class="alert alert-info" id="elbowManualNote"></div>
        <div class="card">
          <div class="sec-label">Grafik Elbow — WCSS vs K</div>
          <div class="chart-wrap" style="height:250px"><canvas id="elbowChart"></canvas></div>
        </div>
        <div class="card">
          <div class="sec-label">Tabel Nilai WCSS</div>
          <div class="table-wrap"><table id="wcssTable"></table></div>
        </div>
        <div class="card">
          <div class="sec-label">Konfirmasi Jumlah Cluster (K)</div>
          <div class="k-selector" id="kSelector"></div>
          <div style="margin-top:8px;font-size:12px;color:var(--text2)">K terpilih: <strong id="kSelectedLabel" style="color:var(--amber)">6</strong></div>
        </div>
        <div class="btn-row right">
          <button class="btn btn-primary" onclick="goToCentroid()">Atur Centroid Awal →</button>
        </div>
      </div>
    </div>

    <!-- TAB 4: CENTROID -->
    <div class="pane" id="pane4">
      <div class="page-title">Centroid Awal</div>
      <div class="page-sub">Masukkan koordinat centroid awal per cluster dalam skala data asli (tanpa normalisasi tambahan — data sudah dalam skala 0–1 sesuai dataset).</div>
      <div class="alert alert-info">💡 Nilai centroid harus diisi dalam skala data: <strong>Status Memilih</strong>, <strong>Umur (N)</strong>, <strong>Status Kawin</strong>, <strong>Jenis Kelamin</strong>, <strong>RT (N)</strong>, <strong>Disabilitas</strong>.</div>
      <div id="centroidForms"></div>
      <div class="btn-row" style="margin-top:16px">
        <button class="btn btn-ghost" onclick="autoFillCentroids()">🎲 Isi Acak dari Data</button>
        <button class="btn btn-primary" onclick="runKMeans()" style="margin-left:auto">▶ Jalankan K-Means</button>
      </div>
    </div>

    <!-- TAB 5: ITERASI -->
    <div class="pane" id="pane5">
      <div class="page-title">Proses Iterasi</div>
      <div id="iterSubtitle" class="page-sub">Menjalankan algoritma...</div>
      <div class="loading-overlay" id="iterLoading" style="display:flex"><div class="spinner"></div> Menjalankan K-Means...</div>
      <div id="iterResult" style="display:none">
        <div class="iter-nav">
          <button class="btn" onclick="changeIter(-1)">‹ Sebelumnya</button>
          <div class="iter-nav-info" id="iterNavLabel">—</div>
          <button class="btn" onclick="changeIter(1)">Selanjutnya ›</button>
          <button class="btn btn-primary" onclick="goTo(6)">Lihat Hasil Akhir 🏁</button>
        </div>
        <div id="iterBody"></div>
      </div>
    </div>

    <!-- TAB 6: HASIL -->
    <div class="pane" id="pane6">
      <div class="page-title">Hasil Akhir Pengelompokan</div>
      <div class="page-sub">Ringkasan hasil clustering data pemilih.</div>
      <div id="hasilContent"></div>
    </div>
  </div>
</div>

<script>
const S={wcss:[],K:6,manualK:6,previewData:[],iterations:[],hasil:[],centroidAkhir:[],silhouette:0,currentIter:0,elbowChart:null,preprocessed:false};
const C_COLOR=['#F5B942','#2BC9A2','#F0694A','#5BA4F5','#9B8EFF','#F0A0DC'];
const COLS=['Status Memilih','Umur (N)','Status Kawin','Jenis Kelamin','RT (N)','Disabilitas'];
const COL_LABELS=['Status Memilih','Umur (N)','Status Kawin','Jenis Kelamin','RT (N)','Disabilitas'];

function fmt(v,maxDec=6){const n=parseFloat(v);if(isNaN(n))return v;return n.toFixed(maxDec).replace(/(\.\d*?)0+$/,'$1').replace(/\.$/,'');}
function fmtNum(v){return parseFloat(v).toLocaleString('id-ID');}
function hexToRgb(hex){const r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5,7),16);return `${r},${g},${b}`;}

function goTo(idx){
  document.querySelectorAll('.pane').forEach((p,i)=>p.classList.toggle('active',i===idx));
  document.querySelectorAll('.nav-item').forEach((n,i)=>n.classList.toggle('active',i===idx));
  updateProgress(idx);
}
function updateProgress(active){
  for(let i=0;i<7;i++){
    const dot=document.getElementById('pd'+i),lbl=document.getElementById('pl'+i);
    dot.className='pb-dot'+(i<active?' done':i===active?' active':'');
    lbl.className='pb-label'+(i<active?' done':i===active?' active':'');
    dot.textContent=i<active?'✓':i+1;
    if(i<6)document.getElementById('pline'+i).className='pb-line'+(i<active?' done':'');
    const nav=document.querySelectorAll('.nav-item')[i];
    if(i<active)nav.classList.add('done');else nav.classList.remove('done');
  }
}
document.getElementById('sideNav').addEventListener('click',e=>{const item=e.target.closest('.nav-item');if(item)goTo(parseInt(item.dataset.tab));});

// Upload
const uz=document.getElementById('uploadZone');
uz.addEventListener('dragover',e=>{e.preventDefault();uz.classList.add('drag')});
uz.addEventListener('dragleave',()=>uz.classList.remove('drag'));
uz.addEventListener('drop',e=>{e.preventDefault();uz.classList.remove('drag');handleFile(e.dataTransfer.files[0])});
document.getElementById('fileInput').addEventListener('change',e=>handleFile(e.target.files[0]));

function handleFile(file){
  if(!file)return;
  const fd=new FormData();fd.append('file',file);
  fetch('/upload',{method:'POST',body:fd})
    .then(r=>{if(!r.ok)throw new Error('Server error '+r.status);return r.json()})
    .then(data=>{
      if(data.error){alert('Error: '+data.error);return;}
      S.previewData=data.preview;
      S.preprocessed=false;
      document.getElementById('preResult').style.display='none';
      document.getElementById('preLoading').style.display='flex';
      document.getElementById('uploadAlert').innerHTML=`✅ <strong>${data.filename}</strong> berhasil dimuat — <strong>${data.total_awal}</strong> baris data mentah${data.stats.missing_rows>0?` (<span style="color:var(--coral)">${data.stats.missing_rows} baris mengandung nilai kosong</span>)`:''}.`;
      const s=data.stats;
      document.getElementById('uploadMetrics').innerHTML=`
        <div class="metric" style="--m-color:#F5B942"><div class="metric-label">Total Data Mentah</div><div class="metric-value">${s.jumlah_data}</div></div>
        <div class="metric" style="--m-color:#F0694A"><div class="metric-label">Baris Kosong</div><div class="metric-value">${s.missing_rows}</div></div>
        <div class="metric" style="--m-color:#2BC9A2"><div class="metric-label">Avg Umur (N)</div><div class="metric-value">${s.avg_umur}</div></div>
        <div class="metric" style="--m-color:#5BA4F5"><div class="metric-label">Avg RT (N)</div><div class="metric-value">${s.avg_rt}</div></div>`;
      const cols=data.columns;
      let th='<thead><tr><th>#</th>'+cols.map(c=>`<th>${c}</th>`).join('')+'</tr></thead>';
      let tb='<tbody>'+data.preview.map((row,i)=>'<tr><td style="color:var(--text3)">'+(i+1)+'</td>'+cols.map(c=>`<td class="mono">${row[c]??''}</td>`).join('')+'</tr>').join('')+'</tbody>';
      document.getElementById('previewTable').innerHTML=th+tb;
      document.getElementById('uploadResult').style.display='block';
      uz.style.display='none';
      updateProgress(0);
    }).catch(err=>alert('Gagal upload: '+err));
}

// Preprocessing
function goToPreprocess(){
  document.getElementById('preLoading').style.display='flex';
  document.getElementById('preResult').style.display='none';
  fetch('/preprocess',{method:'POST'}).then(r=>r.json()).then(data=>{
    if(data.error){alert('Error: '+data.error);return;}
    S.preprocessed=true;
    S.previewData=data.preview;
    document.getElementById('preLoading').style.display='none';
    document.getElementById('preResult').style.display='block';
    document.getElementById('preAlert').innerHTML=`🧹 Preprocessing selesai — dari <strong>${data.total_awal}</strong> baris mentah, <strong>${data.total_bersih}</strong> baris valid dipakai untuk clustering (${data.total_dibuang} baris dibuang karena nilai kosong).`;
    document.getElementById('preMetrics').innerHTML=`
      <div class="metric" style="--m-color:#F5B942"><div class="metric-label">Data Awal</div><div class="metric-value">${data.total_awal}</div></div>
      <div class="metric" style="--m-color:#2BC9A2"><div class="metric-label">Data Bersih</div><div class="metric-value">${data.total_bersih}</div></div>
      <div class="metric" style="--m-color:#F0694A"><div class="metric-label">Dibuang</div><div class="metric-value">${data.total_dibuang}</div></div>
      <div class="metric" style="--m-color:#5BA4F5"><div class="metric-label">% Data Valid</div><div class="metric-value">${data.total_awal>0?((data.total_bersih/data.total_awal)*100).toFixed(1):'0.0'}%</div></div>`;

    let mth='<thead><tr><th>Kolom</th><th>Jumlah Missing</th></tr></thead>';
    let mtb='<tbody>'+COLS.map(c=>`<tr><td>${c}</td><td class="mono" style="color:${(data.missing_counts[c]||0)>0?'#F0694A':'#2BC9A2'}">${data.missing_counts[c]||0}</td></tr>`).join('')+'</tbody>';
    document.getElementById('missingTable').innerHTML=mth+mtb;

    let dth='<thead><tr><th>Kolom</th><th>Min</th><th>Max</th><th>Mean</th><th>Std Dev</th></tr></thead>';
    let dtb='<tbody>'+COLS.map(c=>{const d=data.desc[c]||{};return `<tr><td>${c}</td><td class="mono">${fmt(d.min)}</td><td class="mono">${fmt(d.max)}</td><td class="mono">${fmt(d.mean)}</td><td class="mono">${fmt(d.std)}</td></tr>`;}).join('')+'</tbody>';
    document.getElementById('descTable').innerHTML=dth+dtb;

    let cth='<thead><tr><th>#</th>'+COLS.map(c=>`<th>${c}</th>`).join('')+'</tr></thead>';
    let ctb='<tbody>'+data.preview.map((row,i)=>'<tr><td style="color:var(--text3)">'+(i+1)+'</td>'+COLS.map(c=>`<td class="mono">${row[c]??''}</td>`).join('')+'</tr>').join('')+'</tbody>';
    document.getElementById('cleanPreviewTable').innerHTML=cth+ctb;
  }).catch(err=>alert('Gagal preprocessing: '+err));
}

// Pilih Cluster Manual
function renderManualKSelector(){
  const ks=document.getElementById('manualKSelector');ks.innerHTML='';
  for(let k=2;k<=10;k++){
    const b=document.createElement('button');b.className='k-btn'+(k===S.manualK?' selected':'');b.textContent=k;
    b.onclick=()=>{setManualK(k);};
    ks.appendChild(b);
  }
}
function setManualK(k){
  S.manualK=k;S.K=k;
  document.getElementById('manualKInput').value=k;
  document.getElementById('manualKLabel').textContent=k;
  document.querySelectorAll('#manualKSelector .k-btn').forEach(b=>b.classList.toggle('selected',parseInt(b.textContent)===k));
}
document.getElementById('manualKInput').addEventListener('input',e=>{
  let v=parseInt(e.target.value);
  if(isNaN(v))return;
  if(v<2)v=2;if(v>10)v=10;
  setManualK(v);
});
function confirmManualK(){
  if(!S.previewData){/* no-op guard */}
  S.K=S.manualK;
  goTo(3);
}

// Elbow
function goToElbow(){
  document.getElementById('elbowLoading').style.display='flex';
  document.getElementById('elbowResult').style.display='none';
  fetch('/elbow',{method:'POST'}).then(r=>r.json()).then(data=>{
    if(data.error){alert('Error: '+data.error);return;}
    S.wcss=data.wcss;
    document.getElementById('elbowLoading').style.display='none';
    document.getElementById('elbowResult').style.display='block';
    document.getElementById('elbowManualNote').innerHTML=`📌 K yang Anda pilih secara manual sebelumnya: <strong>${S.manualK}</strong>. Gunakan grafik &amp; tabel di bawah untuk memvalidasi, atau ubah pilihan K jika diperlukan.`;
    renderElbow(data.wcss,data.table);renderKSelector();
  }).catch(err=>alert('Gagal: '+err));
}

function renderElbow(wcss,table){
  if(S.elbowChart)S.elbowChart.destroy();
  const ctx=document.getElementById('elbowChart').getContext('2d');
  S.elbowChart=new Chart(ctx,{type:'line',data:{labels:Array.from({length:10},(_,i)=>i+1),datasets:[{label:'WCSS',data:wcss,borderColor:'#F5B942',backgroundColor:ctx=>{const g=ctx.chart.ctx.createLinearGradient(0,0,0,250);g.addColorStop(0,'rgba(245,185,66,.25)');g.addColorStop(1,'rgba(245,185,66,0)');return g;},pointBackgroundColor:'#201E3A',pointBorderColor:'#F5B942',pointBorderWidth:2.5,pointRadius:6,fill:true,tension:0.35,borderWidth:2.5}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{title:{display:true,text:'K',color:'#504E72'},grid:{color:'rgba(46,43,80,.5)'},ticks:{color:'#504E72'}},y:{grid:{color:'rgba(46,43,80,.5)'},ticks:{color:'#504E72',callback:v=>v.toLocaleString('id-ID')}}}}});
  let html='<thead><tr><th>K</th>'+table.map(r=>`<th>${r.K}</th>`).join('')+'</tr></thead>';
  html+='<tbody><tr><td style="font-weight:700">WCSS</td>'+table.map(r=>`<td class="mono">${r.WCSS.toLocaleString('id-ID')}</td>`).join('')+'</tr></tbody>';
  document.getElementById('wcssTable').innerHTML=html;
}

function renderKSelector(){
  const ks=document.getElementById('kSelector');ks.innerHTML='';
  for(let k=2;k<=8;k++){
    const b=document.createElement('button');b.className='k-btn'+(k===S.K?' selected':'');b.textContent=k;
    b.onclick=()=>{S.K=k;document.querySelectorAll('#kSelector .k-btn').forEach(x=>x.classList.remove('selected'));b.classList.add('selected');document.getElementById('kSelectedLabel').textContent=k;};
    ks.appendChild(b);
  }
  document.getElementById('kSelectedLabel').textContent=S.K;
}

function goToCentroid(){if(!S.K){alert('Pilih K terlebih dahulu.');return;}goTo(4);renderCentroidForms();}

function renderCentroidForms(){
  const K=S.K,cf=document.getElementById('centroidForms');cf.innerHTML='';
  for(let c=0;c<K;c++){
    cf.innerHTML+=`<div class="centroid-block"><div class="centroid-header" style="color:${C_COLOR[c%6]};background:rgba(${hexToRgb(C_COLOR[c%6])},.07)"><span style="width:8px;height:8px;border-radius:50%;background:${C_COLOR[c%6]};display:inline-block"></span>Centroid C${c+1}</div><div class="centroid-body">${COLS.map((col,ci)=>`<div class="form-group"><label>${COL_LABELS[ci]}</label><input type="number" id="c${c}_${ci}" value="0" step="any" placeholder="0"></div>`).join('')}</div></div>`;
  }
}

function autoFillCentroids(){
  const data=S.previewData;if(!data.length)return;
  const idx=[];while(idx.length<S.K){const i=Math.floor(Math.random()*data.length);if(!idx.includes(i))idx.push(i);}
  idx.forEach((di,c)=>{COLS.forEach((col,ci)=>{const el=document.getElementById(`c${c}_${ci}`);if(el)el.value=parseFloat(data[di][col]||0).toFixed(4);});});
}

function runKMeans(){
  const K=S.K,centroids=[];
  for(let c=0;c<K;c++){centroids.push(COLS.map((_,ci)=>{const el=document.getElementById(`c${c}_${ci}`);return parseFloat(el?el.value:0)||0;}));}
  goTo(5);
  document.getElementById('iterLoading').style.display='flex';
  document.getElementById('iterResult').style.display='none';
  fetch('/kmeans',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({k:K,centroids})})
    .then(r=>{if(!r.ok)throw new Error('Server error '+r.status);return r.json();})
    .then(data=>{
      if(data.error){alert('Error: '+data.error);return;}
      S.iterations=data.iterations;S.hasil=data.hasil;S.centroidAkhir=data.centroid_akhir;S.silhouette=data.silhouette;S.currentIter=0;
      document.getElementById('iterLoading').style.display='none';
      document.getElementById('iterResult').style.display='block';
      document.getElementById('iterSubtitle').textContent=`Selesai dalam ${data.total_iterasi} iterasi · Konvergen pada iterasi ${data.total_iterasi} ✅`;
      showIter(0);renderHasil();
    }).catch(err=>alert('Error: '+err));
}

function changeIter(d){const ni=S.currentIter+d;if(ni>=0&&ni<S.iterations.length){S.currentIter=ni;showIter(ni);}}

function showIter(idx){
  const it=S.iterations[idx],K=S.K,total=S.iterations.length,grandTotal=S.hasil.length;
  document.getElementById('iterNavLabel').innerHTML=`Iterasi ke-<strong>${idx+1}</strong> dari <strong>${total}</strong>${it.konvergen?'<span class="iter-badge">✓ Konvergen</span>':''}`;

  let html='';
  html+=`<div class="card" style="border-color:rgba(245,185,66,.25);background:rgba(245,185,66,.03)"><div class="sec-label" style="color:var(--amber)">📊 Anggota Cluster — Iterasi ${idx+1}</div><div class="metric-row">`;
  for(let c=0;c<K;c++){const cnt=it.anggota['C'+(c+1)]??0,pct=grandTotal>0?((cnt/grandTotal)*100).toFixed(1):'0.0';html+=`<div class="metric" style="--m-color:${C_COLOR[c%6]}"><div class="metric-label">Cluster ${c+1}</div><div class="metric-value">${cnt}</div><div style="font-size:10px;color:var(--text3);margin-top:3px">${pct}%</div></div>`;}
  html+=`<div class="metric" style="--m-color:#fff;border:2px solid var(--border2)"><div class="metric-label">Total</div><div class="metric-value" style="color:#fff">${grandTotal}</div></div></div></div>`;

  html+=`<div class="card"><div class="sec-label">Centroid Lama (Input Iterasi ${idx+1})</div><div class="table-wrap"><table><thead><tr><th>Cluster</th>${COL_LABELS.map(c=>`<th>${c}</th>`).join('')}</tr></thead><tbody>`;
  for(let c=0;c<K;c++)html+=`<tr><td><span class="badge b${(c%6)+1}">C${c+1}</span></td>${COLS.map((_,ci)=>`<td class="mono">${fmt(it.centroid_lama[c][ci])}</td>`).join('')}</tr>`;
  html+='</tbody></table></div></div>';

  html+=`<div class="card"><div class="sec-label">Tabel Jarak Euclidean & Klaster (10 baris pertama ditampilkan, sisanya di ekspor Excel)</div><div class="table-wrap"><div class="table-scroll"><table><thead><tr><th>#</th>${COL_LABELS.map(c=>`<th>${c}</th>`).join('')}${Array.from({length:K},(_,c)=>`<th>D ke C${c+1}</th>`).join('')}<th>Min</th><th>Klaster</th></tr></thead><tbody>`;
  it.tabel.slice(0,200).forEach((row,i)=>{
    const cl=row.Cluster,h=S.hasil[i]||{};
    html+=`<tr><td style="color:var(--text3)">${i+1}</td>${COLS.map(c=>`<td class="mono">${fmt(h[c])}</td>`).join('')}${Array.from({length:K},(_,c)=>`<td class="mono">${fmt(row['C'+(c+1)])}</td>`).join('')}<td class="mono perp-zero">${fmt(row.jarak_min)}</td><td><span class="badge b${(cl-1)%6+1}">C${cl}</span></td></tr>`;
  });
  html+='</tbody></table></div></div></div>';

  html+=`<div class="card"><div class="sec-label">Centroid Baru (Hasil Rata-Rata Iterasi ${idx+1})</div><div class="table-wrap"><table><thead><tr><th>Cluster</th><th>n</th>${COL_LABELS.map(c=>`<th>${c}</th>`).join('')}</tr></thead><tbody>`;
  for(let c=0;c<K;c++)html+=`<tr><td><span class="badge b${(c%6)+1}">C${c+1}</span></td><td class="mono">${it.anggota['C'+(c+1)]}</td>${COLS.map((_,ci)=>`<td class="mono" style="color:${C_COLOR[c%6]};font-weight:600">${fmt(it.centroid_baru[c][ci])}</td>`).join('')}</tr>`;
  html+='</tbody></table></div></div>';

  if(it.konvergen && idx>0){
    html+=`<div class="card" style="border-color:rgba(43,201,162,.35);background:rgba(43,201,162,.04)"><div class="sec-label" style="color:var(--teal)">✅ Bukti Konvergensi — Cluster Assignment Iterasi ${idx} = Iterasi ${idx+1}</div><div style="font-size:12px;color:var(--text2);margin-bottom:12px;line-height:1.6">Assignment cluster pada iterasi <strong style="color:#fff">${idx}</strong> dan iterasi <strong style="color:#fff">${idx+1}</strong> <strong style="color:var(--teal)">identik</strong> — tidak ada data yang berpindah cluster. Algoritma dinyatakan KONVERGEN.</div><div style="padding:10px 14px;background:rgba(43,201,162,.1);border-radius:8px;border:1px solid rgba(43,201,162,.2);font-size:12px;color:var(--teal);font-weight:600;text-align:center">🔒 K-Means KONVERGEN pada iterasi ${idx+1}</div></div>`;
  }
  document.getElementById('iterBody').innerHTML=html;
}

function renderHasil(){
  const K=S.K,cen=S.centroidAkhir,hasil=S.hasil,sil=S.silhouette,nIter=S.iterations.length;
  const counts=Array.from({length:K},(_,c)=>hasil.filter(r=>r.CLUSTER===c+1).length);
  const qual=sil>0.7?'Sangat Baik':sil>0.5?'Baik':sil>0.25?'Cukup':'Lemah';
  const fillPct=Math.max(0,Math.min(100,((sil+1)/2)*100)).toFixed(1);
  const grandTotal=counts.reduce((a,b)=>a+b,0);

  let html=`<div class="alert alert-success">🎉 K-Means selesai dalam <strong>${nIter} iterasi</strong> dan konvergen. K = <strong>${K}</strong> · Total data: <strong>${grandTotal}</strong></div>`;
  html+='<div class="metric-row">';
  for(let c=0;c<K;c++){const pct=((counts[c]/grandTotal)*100).toFixed(1);html+=`<div class="metric" style="--m-color:${C_COLOR[c%6]}"><div class="metric-label">Cluster ${c+1}</div><div class="metric-value">${counts[c]}</div><div style="font-size:10px;color:var(--text3);margin-top:3px">${pct}%</div></div>`;}
  html+=`<div class="metric" style="--m-color:#fff;border:2px solid var(--border2)"><div class="metric-label">Total Data</div><div class="metric-value" style="color:#fff">${grandTotal}</div></div></div>`;

  html+=`<div class="card"><div class="sec-label">📈 Rekap Anggota per Iterasi</div><div class="table-wrap"><table><thead><tr><th>Iterasi</th>${Array.from({length:K},(_,c)=>`<th>C${c+1}</th>`).join('')}<th>Total</th><th>Status</th></tr></thead><tbody>`;
  S.iterations.forEach((it,i)=>{html+=`<tr><td style="font-weight:700;color:var(--text2)">Iterasi ${i+1}</td>${Array.from({length:K},(_,c)=>`<td class="mono">${it.anggota['C'+(c+1)]??0}</td>`).join('')}<td class="mono" style="color:var(--text3)">${grandTotal}</td><td>${it.konvergen?'<span style="color:var(--teal);font-size:11px;font-weight:700">✓ Konvergen</span>':''}</td></tr>`;});
  html+=`<tr style="background:var(--bg3);border-top:2px solid var(--border2)"><td style="font-weight:800;color:#fff">Hasil Akhir</td>${counts.map(c=>`<td class="mono" style="font-weight:800;color:#fff">${c}</td>`).join('')}<td class="mono" style="font-weight:800;color:var(--amber)">${grandTotal}</td><td></td></tr></tbody></table></div></div>`;

  html+=`<div class="card"><div class="sec-label">Silhouette Score</div><div class="score-row"><div class="score-num">${sil.toFixed(4)}</div><div class="score-track"><div class="score-fill" style="width:${fillPct}%"></div></div><div class="score-qual">${qual}</div></div><div style="font-size:11px;color:var(--text3);margin-top:4px">Rentang: −1 (buruk) hingga +1 (sempurna). Nilai > 0.5 = baik.</div></div>`;

  html+=`<div class="sec-label" style="margin-top:4px">Centroid Akhir</div><div class="cen-grid">`;
  for(let c=0;c<K;c++)html+=`<div class="cen-card" style="--c-color:${C_COLOR[c%6]}"><div class="cen-card-title">C${c+1}</div>${COLS.map((col,ci)=>`<div class="cen-card-row">${COL_LABELS[ci]}<br><span>${parseFloat(cen[c][ci]).toLocaleString('id-ID',{maximumFractionDigits:4})}</span></div>`).join('')}</div>`;
  html+='</div>';

  html+=`<div class="card"><div class="sec-label">Distribusi Data per Cluster</div><div class="chart-wrap" style="height:220px"><canvas id="hasilChart"></canvas></div></div>`;

  html+=`<div class="card" style="padding:0"><div class="card-header">Tabel Hasil Akhir — 50 Baris Pertama (lengkap di file ekspor)</div><div class="table-wrap" style="border:none;border-radius:0"><div class="table-scroll"><table><thead><tr><th>#</th>${COL_LABELS.map(c=>`<th>${c}</th>`).join('')}<th>CLUSTER</th></tr></thead><tbody>`;
  hasil.slice(0,50).forEach((row,i)=>{html+=`<tr><td style="color:var(--text3)">${i+1}</td>${COLS.map(c=>`<td class="mono">${parseFloat(row[c]).toLocaleString('id-ID',{maximumFractionDigits:4})}</td>`).join('')}<td><span class="badge b${(row.CLUSTER-1)%6+1}">C${row.CLUSTER}</span></td></tr>`;});
  html+=`</tbody></table></div></div><div class="btn-row right" style="padding:14px 20px"><a href="/export" class="btn btn-teal">💾 Ekspor ke Excel (.xlsx)</a></div></div>`;

  document.getElementById('hasilContent').innerHTML=html;
  setTimeout(()=>{
    const ctx=document.getElementById('hasilChart');if(!ctx)return;
    new Chart(ctx.getContext('2d'),{type:'bar',data:{labels:Array.from({length:K},(_,c)=>`C${c+1} (${counts[c]})`),datasets:[{data:counts,backgroundColor:C_COLOR.slice(0,K),borderRadius:8,borderSkipped:false}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{grid:{display:false},ticks:{color:'#504E72',font:{size:10}}},y:{beginAtZero:true,grid:{color:'rgba(46,43,80,.5)'},ticks:{color:'#504E72'}}}}});
  },100);
}

renderManualKSelector();

const _goTo=goTo;
window.goTo=function(idx){
  _goTo(idx);
  if(idx===1 && !S.preprocessed)goToPreprocess();
  if(idx===2){document.getElementById('manualKInput').value=S.manualK;renderManualKSelector();}
  if(idx===3 && S.wcss.length===0)goToElbow();
  if(idx===6 && S.hasil.length>0)renderHasil();
};
</script>
</body>
</html>"""

# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def landing():
    return render_template_string(LANDING_HTML)

@app.route('/app')
def index():
    return render_template_string(HTML)


@app.route('/upload', methods=['POST'])
def upload():
    """
    Tahap Upload HANYA membaca & memvalidasi file — TIDAK melakukan
    cleaning di sini. Pembersihan data (drop baris dengan nilai kosong)
    dipindahkan ke tahap Preprocessing (/preprocess) agar sesuai dengan
    alur kerja: Upload -> Preprocessing -> Pilih Cluster -> Elbow -> ...
    Kolom di luar COLS (mis. 'Unnamed: 6' sisa Excel) otomatis diabaikan.
    """
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'Tidak ada file'}), 400

        fname = file.filename.lower()
        try:
            df = pd.read_csv(file) if fname.endswith('.csv') else pd.read_excel(file)
        except Exception as e:
            return jsonify({'error': f'Gagal baca file: {str(e)}'}), 400

        df.columns = [c.strip() for c in df.columns]

        missing = [c for c in COLS if c not in df.columns]
        if missing:
            return jsonify({'error': f'Kolom tidak ditemukan: {", ".join(missing)}'}), 400

        df_cols = df[COLS].copy()
        total_awal = len(df_cols)
        missing_counts = {c: int(df_cols[c].isna().sum()) for c in COLS}
        missing_rows = int(df_cols.isna().any(axis=1).sum())

        # Simpan data MENTAH (belum dibersihkan) untuk diproses di /preprocess
        SESSION['raw_df'] = df_cols.to_dict('records')
        SESSION.pop('df', None)
        SESSION.pop('X', None)

        stats = {
            'jumlah_data':  int(total_awal),
            'missing_rows': missing_rows,
            'avg_umur':     round(float(df_cols['Umur (N)'].mean(skipna=True)), 4) if total_awal else 0,
            'avg_rt':       round(float(df_cols['RT (N)'].mean(skipna=True)), 4) if total_awal else 0,
        }

        return jsonify({
            'success':  True,
            'stats':    stats,
            'preview':  df_cols.head(10).fillna('').to_dict('records'),
            'columns':  COLS,
            'filename': file.filename,
            'total_awal': total_awal,
            'missing_counts': missing_counts,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/preprocess', methods=['POST'])
def preprocess():
    """
    Tahap Preprocessing: membuang baris dengan nilai kosong di salah satu
    kolom COLS (tanpa filtering/outlier-removal tambahan — sama seperti
    perhitungan manual Excel), lalu menyimpan data bersih untuk dipakai
    pada tahap-tahap berikutnya (Pilih Cluster, Elbow, K-Means).
    """
    try:
        raw_records = SESSION.get('raw_df', [])
        if not raw_records:
            return jsonify({'error': 'Upload data terlebih dahulu'}), 400

        df_raw = pd.DataFrame(raw_records)
        total_awal = len(df_raw)
        missing_counts = {c: int(df_raw[c].isna().sum()) for c in COLS}

        df_clean = df_raw.dropna(subset=COLS).reset_index(drop=True)
        total_bersih = len(df_clean)
        total_dibuang = total_awal - total_bersih

        X = df_clean[COLS]
        SESSION['df'] = df_clean.to_dict('records')
        SESSION['X'] = X.values.tolist()

        desc = {}
        for c in COLS:
            col = X[c].astype(float) if total_bersih else pd.Series([], dtype=float)
            desc[c] = {
                'min':  round(float(col.min()), 4) if total_bersih else 0,
                'max':  round(float(col.max()), 4) if total_bersih else 0,
                'mean': round(float(col.mean()), 4) if total_bersih else 0,
                'std':  round(float(col.std()), 4) if total_bersih > 1 else 0,
            }

        return jsonify({
            'success': True,
            'total_awal': total_awal,
            'total_bersih': total_bersih,
            'total_dibuang': total_dibuang,
            'missing_counts': missing_counts,
            'desc': desc,
            'preview': df_clean[COLS].head(10).fillna('').to_dict('records'),
            'columns': COLS,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/elbow', methods=['POST'])
def elbow():
    try:
        X = np.array(SESSION.get('X', []))
        if len(X) == 0:
            return jsonify({'error': 'Lakukan preprocessing data terlebih dahulu'}), 400

        wcss = []
        for k in range(1, 11):
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            model.fit(X)
            wcss.append(round(float(model.inertia_), 4))

        return jsonify({'success': True, 'wcss': wcss, 'table': [{'K': k, 'WCSS': w} for k, w in enumerate(wcss, 1)]})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/kmeans', methods=['POST'])
def run_kmeans():
    """
    Konvergensi berbasis PERUBAHAN CLUSTER ASSIGNMENT (bukan gerakan
    centroid) — identik dengan cara Excel menghitung manual. Tanpa
    min_iters, tanpa normalisasi tambahan (data sudah dalam skala 0-1).
    K berasal dari pilihan manual pengguna (divalidasi/disesuaikan pada
    tahap Elbow Method).
    """
    try:
        data = request.get_json()
        X_original = np.array(SESSION.get('X', []))
        df_records = SESSION.get('df', [])

        if len(X_original) == 0:
            return jsonify({'error': 'Lakukan preprocessing data terlebih dahulu'}), 400

        K = int(data.get('k', DEFAULT_K))
        centroid_input = data.get('centroids', [])
        if len(centroid_input) != K:
            return jsonify({'error': f'Jumlah centroid harus {K}'}), 400

        X = X_original.copy()
        centroid = np.array(centroid_input, dtype=float)

        max_iter = 100
        all_iterations = []
        final_cluster = None
        cluster_lama = None

        for iterasi in range(1, max_iter + 1):
            centroid_lama = centroid.copy()

            jarak = cdist(X, centroid, metric='euclidean')
            cluster = np.argmin(jarak, axis=1) + 1
            anggota = {f'C{c}': int((cluster == c).sum()) for c in range(1, K + 1)}

            centroid_baru = []
            for c in range(1, K + 1):
                pts = X[cluster == c]
                centroid_baru.append(
                    pts.mean(axis=0).tolist() if len(pts) > 0
                    else centroid[c - 1].tolist()
                )
            centroid_baru = np.array(centroid_baru)

            perpindahan = np.sqrt(np.sum((centroid_baru - centroid_lama) ** 2, axis=1))

            konvergen = (cluster_lama is not None) and np.array_equal(cluster, cluster_lama)

            iter_data = {
                'iterasi': iterasi,
                'centroid_lama': centroid_lama.tolist(),
                'centroid_baru': centroid_baru.tolist(),
                'perpindahan': [round(float(p), 6) for p in perpindahan],
                'anggota': anggota,
                'konvergen': konvergen,
                'tabel': [
                    {
                        **{f'C{c}': round(float(jarak[i, c - 1]), 8) for c in range(1, K + 1)},
                        'jarak_min': round(float(jarak[i].min()), 8),
                        'Cluster': int(cluster[i]),
                    }
                    for i in range(len(X))
                ]
            }
            all_iterations.append(iter_data)

            cluster_lama = cluster.copy()
            centroid = centroid_baru
            final_cluster = cluster

            if konvergen:
                break

        score = 0.0
        if final_cluster is not None and len(set(final_cluster)) > 1:
            score = round(float(silhouette_score(X_original, final_cluster)), 4)

        centroid_akhir = centroid.tolist()

        hasil_rows = []
        for i in range(len(X_original)):
            row = {c: round(float(X_original[i, ci]), 6) for ci, c in enumerate(COLS)}
            row['CLUSTER'] = int(final_cluster[i])
            hasil_rows.append(row)

        SESSION.update({
            'hasil': hasil_rows,
            'centroid_akhir': centroid_akhir,
            'iterations': all_iterations,
            'K': K,
            'silhouette': score,
        })

        return jsonify({
            'success': True,
            'iterations': all_iterations,
            'centroid_akhir': centroid_akhir,
            'silhouette': score,
            'hasil': hasil_rows,
            'total_iterasi': len(all_iterations),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/export')
def export_excel():
    try:
        hasil          = SESSION.get('hasil', [])
        centroid_akhir = SESSION.get('centroid_akhir', [])
        iterations     = SESSION.get('iterations', [])
        K              = SESSION.get('K', DEFAULT_K)

        if not hasil:
            return jsonify({'error': 'Jalankan K-Means terlebih dahulu'}), 400

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_hasil = pd.DataFrame(hasil)
            cols_order = [c for c in COLS + ['CLUSTER'] if c in df_hasil.columns]
            df_hasil[cols_order].to_excel(writer, sheet_name='Hasil_Akhir', index=False)

            pd.DataFrame(centroid_akhir, columns=COLS,
                         index=[f'C{i+1}' for i in range(K)]).to_excel(writer, sheet_name='Centroid_Akhir')

            for it in iterations:
                df_it = pd.DataFrame(it['tabel'])
                for ci, col in enumerate(COLS):
                    df_it.insert(ci, col, [hasil[i][col] for i in range(len(df_it))])
                df_it.to_excel(writer, sheet_name=f"Iterasi_{it['iterasi']}", index=False)

        output.seek(0)
        return send_file(output, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                         as_attachment=True, download_name='KMeans_Pemilih.xlsx')
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)