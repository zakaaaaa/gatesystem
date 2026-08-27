"""
evaluasi_sistem.py
==================
Modul evaluasi end-to-end Sistem GATE untuk subbab 4.3.6.4
(Pengujian Akurasi Klasifikasi).

Mengisi kebutuhan metodologis subbab 3.10: Confusion Matrix +
Accuracy, Precision, Recall, F1-Score. Reduksi biner mengikuti 3.10.1:
    - Prediksi sistem : HIGH / MEDIUM  -> POSITIF (terindikasi judol)
                        LOW            -> NEGATIF (tidak terindikasi)
    - Label manual    : 1 (judol)      -> POSITIF
                        0 (non-judol)  -> NEGATIF

Sumber prediksi sistem  : scan_log.json (dihasilkan screening_gate.py)
Sumber label sebenarnya : ground_truth.csv (diisi manual oleh anotator)

CARA PAKAI
----------
1) Buat lembar sampel untuk dilabel manual (stratified random sampling):
       python evaluasi_sistem.py template
   -> menghasilkan evaluasi/ground_truth_template.csv

   PENTING: agar Precision/Akurasi/F1 dapat dihitung, set sampel WAJIB
   mengandung situs NEGATIF (non-judol). Karena daftar input utama
   (CNS Gambling) seluruhnya positif, jalankan dahulu screening_gate.py
   pada sebuah daftar kontrol (situs berita/edukasi/e-commerce/.go.id)
   sehingga prediksinya ikut tersimpan di scan_log.json, lalu sertakan
   URL kontrol tsb saat pelabelan.

2) Salin template menjadi ground_truth.csv, lalu isi kolom 'label_manual'
   dengan 1 (judol) atau 0 (non-judol) untuk setiap baris.

3) Hitung metrik evaluasi:
       python evaluasi_sistem.py eval
   -> evaluasi/hasil_evaluasi.json, grafik_confusion_matrix.png,
      grafik_metrik_evaluasi.png
"""

import os
import sys
import json
import csv
import random
from datetime import datetime
from collections import Counter

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("[WARN] matplotlib/numpy tidak tersedia. Grafik tidak akan dibuat.")

# ================= LINGKUNGAN & BASE DIR =================
# Sama seperti screening_gate.py: di Colab semua path diarahkan ke folder
# GATESystem di Drive, sehingga file ini bisa langsung di-COPAS ke satu cell.
IS_COLAB = 'google.colab' in sys.modules
DRIVE_BASE = '/content/drive/MyDrive/GATESystem'   # samakan dgn screening_gate.py

if IS_COLAB:
    try:
        if not os.path.exists('/content/drive/MyDrive'):
            from google.colab import drive
            drive.mount('/content/drive')
    except Exception as e:
        print(f"[WARN] Gagal mount Drive otomatis: {e}")
    BASE_DIR = os.environ.get('BASE_DIR', DRIVE_BASE)
else:
    BASE_DIR = os.environ.get('BASE_DIR', '.')


def _p(name):
    return name if os.path.isabs(name) else os.path.join(BASE_DIR, name)


# ================= KONFIGURASI =================
SCAN_LOG_FILE   = _p('scan_log.json')
EVAL_DIR        = _p('evaluasi')
TEMPLATE_FILE   = os.path.join(EVAL_DIR, 'ground_truth_template.csv')
GROUND_TRUTH    = os.path.join(EVAL_DIR, 'ground_truth.csv')
RESULT_JSON     = os.path.join(EVAL_DIR, 'hasil_evaluasi.json')

# Jumlah sampel per kategori prediksi untuk template pelabelan.
# (Stratified sampling agar tiap tingkat risiko terwakili proporsional.)
SAMPLE_PER_CATEGORY = {'HIGH': 150, 'MEDIUM': 120, 'LOW': 100}

RANDOM_SEED = 42  # reproducibility lembar sampel & figur

os.makedirs(EVAL_DIR, exist_ok=True)


# ================= UTIL =================
def normalize_url(url):
    """Normalisasi URL untuk penjodohan prediksi <-> label."""
    if url is None:
        return ''
    u = url.strip().lower()
    for pre in ('https://', 'http://'):
        if u.startswith(pre):
            u = u[len(pre):]
    if u.startswith('www.'):
        u = u[4:]
    return u.rstrip('/')


# Kategori akses yang DIKELUARKAN dari perhitungan metrik (tidak bisa dibaca
# kontennya secara wajar): blokir pemerintah, situs mati, error, tantangan bot.
EXCLUDED_ACCESS = {'BLOCKED_KOMDIGI', 'DEAD', 'ERROR', 'BOT_CHALLENGE'}
# Jika '0': situs tak terakses TIDAK dikeluarkan, melainkan dianggap prediksi
# NEGATIF (sistem tidak menandainya sbg judol). Untuk framing dataset 2000+300.
EXCLUDE_INACCESSIBLE = os.environ.get('EXCLUDE_INACCESSIBLE', '1') == '1'


def load_predictions():
    """Baca scan_log.json -> dict {url: {risk_level, final_score, status, access_category}}.

    Memuat SEMUA entri (termasuk FAILED) agar kategori akses (mati/terblokir)
    bisa diketahui dan dikeluarkan dari ground-truth.
    """
    if not os.path.exists(SCAN_LOG_FILE):
        print(f"[ERROR] {SCAN_LOG_FILE} tidak ditemukan. Jalankan screening_gate.py dulu.")
        sys.exit(1)

    with open(SCAN_LOG_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    preds = {}
    for entry in data:
        key = normalize_url(entry['url'])
        status = entry.get('status')
        # access_category default 'OK' untuk SUCCESS (kompat scan_log lama tanpa field),
        # 'DEAD' untuk FAILED tanpa kategori.
        acc = entry.get('access_category')
        if not acc:
            acc = 'OK' if status == 'SUCCESS' else 'DEAD'
        sd = entry.get('score_data') or {}
        preds[key] = {
            'risk_level':      sd.get('risk_level'),
            'final_score':     sd.get('final_score'),
            'status':          status,
            'access_category': acc,
            'url_asli':        entry['url'],
        }
    return preds


def to_binary_prediction(risk_level):
    """HIGH/MEDIUM -> 1 (positif), LOW -> 0 (negatif). Sesuai subbab 3.10.1."""
    return 1 if risk_level in ('HIGH', 'MEDIUM') else 0


# ================= MODE 1: TEMPLATE =================
def build_template():
    """Buat lembar sampel berstrata untuk pelabelan manual."""
    random.seed(RANDOM_SEED)
    preds = load_predictions()

    # Kelompokkan URL per kategori prediksi (hanya yang bisa diakses/OK)
    by_cat = {'HIGH': [], 'MEDIUM': [], 'LOW': []}
    for url, p in preds.items():
        if p['status'] != 'SUCCESS' or p['access_category'] in EXCLUDED_ACCESS:
            continue
        by_cat.setdefault(p['risk_level'], []).append(url)

    rows = []
    for cat, n_target in SAMPLE_PER_CATEGORY.items():
        pool = by_cat.get(cat, [])
        chosen = random.sample(pool, min(n_target, len(pool)))
        for url in chosen:
            rows.append([
                preds[url]['url_asli'],
                cat,
                preds[url]['final_score'],
                ''  # label_manual: diisi 1 (judol) / 0 (non-judol)
            ])
        print(f"[INFO] Kategori {cat:<6}: tersedia {len(pool):>5}, "
              f"diambil {len(chosen)} sampel")

    random.shuffle(rows)  # acak urutan agar pelabelan tidak bias kategori
    with open(TEMPLATE_FILE, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['url', 'prediksi_sistem', 'skor_final', 'label_manual'])
        w.writerows(rows)

    print(f"\n[OK] Template pelabelan disimpan: {TEMPLATE_FILE} ({len(rows)} baris)")
    print("     Langkah berikut:")
    print("       1. Salin menjadi ground_truth.csv")
    print("       2. Isi kolom 'label_manual' (1=judol, 0=non-judol)")
    print("       3. Tambahkan baris URL KONTROL non-judol bila perlu")
    print("          (URL kontrol harus sudah pernah discan agar ada prediksinya)")
    print("       4. Jalankan: python evaluasi_sistem.py eval")


# ================= MODE 2: EVALUASI =================
def evaluate():
    """Hitung confusion matrix + Accuracy/Precision/Recall/F1."""
    if not os.path.exists(GROUND_TRUTH):
        print(f"[ERROR] {GROUND_TRUTH} tidak ditemukan.")
        print("        Buat dulu via 'template', isi label_manual, "
              "lalu simpan sebagai ground_truth.csv")
        sys.exit(1)

    preds = load_predictions()

    # Baca ground truth
    gt_rows = []
    with open(GROUND_TRUTH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            gt_rows.append(r)

    # Penjodohan prediksi <-> label
    TP = TN = FP = FN = 0
    matched, unmatched, unlabeled, invalid = 0, 0, 0, 0
    excluded = Counter()          # per kategori akses yang dikeluarkan
    detail_rows = []

    for r in gt_rows:
        raw_label = (r.get('label_manual') or '').strip()
        if raw_label == '':
            unlabeled += 1
            continue
        if raw_label not in ('0', '1'):
            invalid += 1
            print(f"[WARN] label_manual tidak valid ('{raw_label}') untuk {r.get('url')}")
            continue

        key = normalize_url(r.get('url'))
        if key not in preds:
            unmatched += 1
            continue

        p = preds[key]
        acc = p['access_category']
        inaccessible = (p['status'] != 'SUCCESS' or acc in EXCLUDED_ACCESS)
        y_true = int(raw_label)

        if inaccessible and EXCLUDE_INACCESSIBLE:
            # Mode rigor: keluarkan situs tak terbaca dari perhitungan.
            excluded[acc] += 1
            detail_rows.append([r.get('url'), p['risk_level'] or '-',
                                p['final_score'] if p['final_score'] is not None else '-',
                                '-', raw_label, f'EXCLUDED:{acc}'])
            continue

        if inaccessible:
            y_pred = 0      # tak terakses -> sistem tidak menandai judol -> negatif
        else:
            y_pred = to_binary_prediction(p['risk_level'])
        matched += 1

        if   y_true == 1 and y_pred == 1: TP += 1; cell = 'TP'
        elif y_true == 0 and y_pred == 0: TN += 1; cell = 'TN'
        elif y_true == 0 and y_pred == 1: FP += 1; cell = 'FP'
        else:                             FN += 1; cell = 'FN'

        detail_rows.append([
            r.get('url'), p['risk_level'] or '-',
            p['final_score'] if p['final_score'] is not None else '-',
            y_pred, y_true, cell
        ])

    total = TP + TN + FP + FN
    if total == 0:
        print("[ERROR] Tidak ada data terlabel yang berhasil dijodohkan. "
              "Periksa kolom label_manual dan apakah URL sudah discan.")
        sys.exit(1)

    # Metrik (subbab 3.10.2). Penjaga pembagian nol.
    def safe_div(a, b):
        return a / b if b else 0.0

    accuracy  = safe_div(TP + TN, total)
    precision = safe_div(TP, TP + FP)
    recall    = safe_div(TP, TP + FN)
    f1        = safe_div(2 * precision * recall, precision + recall)

    result = {
        'tanggal_evaluasi': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'jumlah_sampel_dievaluasi': total,
        'confusion_matrix': {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN},
        'metrik': {
            'accuracy':  round(accuracy, 4),
            'precision': round(precision, 4),
            'recall':    round(recall, 4),
            'f1_score':  round(f1, 4),
        },
        'metrik_persen': {
            'accuracy':  round(accuracy * 100, 2),
            'precision': round(precision * 100, 2),
            'recall':    round(recall * 100, 2),
            'f1_score':  round(f1 * 100, 2),
        },
        'catatan_data': {
            'baris_terlabel_terjodoh': matched,
            'tanpa_prediksi (belum discan)': unmatched,
            'belum_dilabel': unlabeled,
            'label_tidak_valid': invalid,
        },
        'dikeluarkan_dari_metrik': {
            'total': sum(excluded.values()),
            'rincian': dict(excluded),
            'keterangan': 'Situs tak terbaca wajar (blokir Komdigi / mati / '
                          'tantangan bot) dikeluarkan agar metrik mencerminkan '
                          'kinerja pada konten judol yang benar-benar dapat diakses.',
        },
        'komposisi_kelas_sebenarnya': {
            'positif (judol)':     TP + FN,
            'negatif (non-judol)': TN + FP,
        },
    }

    # Simpan JSON + CSV rincian
    with open(RESULT_JSON, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    detail_path = os.path.join(EVAL_DIR, 'rincian_evaluasi.csv')
    with open(detail_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['url', 'prediksi_sistem', 'skor_final',
                    'y_pred(biner)', 'y_true(biner)', 'sel'])
        w.writerows(detail_rows)

    # Cetak ringkasan
    print("\n" + "=" * 60)
    print("  HASIL EVALUASI SISTEM GATE (Pengujian Akurasi Klasifikasi)")
    print("=" * 60)
    print(f"  Sampel dievaluasi : {total}")
    print(f"  Kelas positif     : {TP + FN}  | Kelas negatif: {TN + FP}")
    print("-" * 60)
    print("  Confusion Matrix (positif = judol [HIGH/MEDIUM]):")
    print(f"                    Pred POSITIF   Pred NEGATIF")
    print(f"    Aktual POSITIF      TP={TP:<6}     FN={FN:<6}")
    print(f"    Aktual NEGATIF      FP={FP:<6}     TN={TN:<6}")
    print("-" * 60)
    print(f"  Accuracy  : {accuracy*100:6.2f}%   (TP+TN)/(total)")
    print(f"  Precision : {precision*100:6.2f}%   TP/(TP+FP)")
    print(f"  Recall    : {recall*100:6.2f}%   TP/(TP+FN)")
    print(f"  F1-Score  : {f1*100:6.2f}%   2PR/(P+R)")
    print("-" * 60)
    if excluded:
        print(f"  Dikeluarkan dari metrik (tak terbaca wajar): {sum(excluded.values())}")
        for k, v in excluded.most_common():
            print(f"      {k:<16}: {v}")
    print("-" * 60)
    if unmatched:
        print(f"  [!] {unmatched} URL belum punya prediksi (belum discan).")
    if unlabeled:
        print(f"  [!] {unlabeled} baris belum dilabel (label_manual kosong).")
    if (TN + FP) == 0:
        print("  [!] PERINGATAN: tidak ada sampel NEGATIF. Precision/Akurasi/F1")
        print("      tidak bermakna tanpa situs kontrol non-judol. Lihat docstring.")
    print("=" * 60)
    print(f"  Tersimpan: {RESULT_JSON}")
    print(f"            {detail_path}")

    if PLOT_AVAILABLE:
        _plot_confusion_matrix(TP, TN, FP, FN)
        _plot_metrics(accuracy, precision, recall, f1)

    return result


def _plot_confusion_matrix(TP, TN, FP, FN):
    cm = np.array([[TP, FN], [FP, TN]])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    labels = ['POSITIF\n(judol)', 'NEGATIF\n(non-judol)']
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Prediksi\nPOSITIF', 'Prediksi\nNEGATIF'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Aktual\nPOSITIF', 'Aktual\nNEGATIF'])
    names = [['TP', 'FN'], ['FP', 'TN']]
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{names[i][j]}\n{cm[i, j]}",
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    color='white' if cm[i, j] > thresh else 'black')
    ax.set_title('Confusion Matrix Sistem GATE\n(positif = HIGH/MEDIUM)',
                 fontsize=13, fontweight='bold', pad=15)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out = os.path.join(EVAL_DIR, 'grafik_confusion_matrix.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Grafik   : {out}")


def _plot_metrics(acc, prec, rec, f1):
    fig, ax = plt.subplots(figsize=(7, 5))
    names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    vals = [acc * 100, prec * 100, rec * 100, f1 * 100]
    colors = ['#1565c0', '#2e7d32', '#ef6c00', '#6a1b9a']
    bars = ax.bar(names, vals, color=colors, edgecolor='white', linewidth=0.8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"{v:.2f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.set_ylabel('Nilai (%)')
    ax.set_title('Metrik Evaluasi Klasifikasi Sistem GATE',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(EVAL_DIR, 'grafik_metrik_evaluasi.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Grafik   : {out}")


# ================= ENTRY =================
def main():
    # Saat di-copas ke cell Colab (tanpa argumen), default langsung 'eval'.
    mode = sys.argv[1] if len(sys.argv) > 1 else 'eval'
    if mode == 'template':
        build_template()
    elif mode == 'eval':
        evaluate()
    else:
        print("Penggunaan:")
        print("  python evaluasi_sistem.py template   # buat lembar sampel pelabelan")
        print("  python evaluasi_sistem.py eval       # hitung metrik dari ground_truth.csv")


if __name__ == '__main__':
    main()
