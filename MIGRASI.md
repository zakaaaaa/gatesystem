# Panduan Migrasi Sistem GATE ke PC Lain

Total folder saat ini ± **5,8 GB**. Sebagian besar adalah data/output yang bisa
di-generate ulang. Bagian ini menjelaskan apa yang wajib dipindah, apa yang
opsional, dan langkah setup di PC baru.

---

## 1. Yang WAJIB dipindah (inti sistem, ± 150 MB)

| Item | Ukuran | Keterangan |
|---|---|---|
| `screening.py`, `screening_demo.py` | kecil | pipeline demo |
| `_arsip/skrip/` | 172 KB | seluruh skrip yang dibahas di laporan |
| `best-4.pt` | 50 MB | **model YOLO — tidak bisa di-generate ulang tanpa training** |
| `data/` | 556 KB | `CNS Gambling 09022026.txt`, bobot, jsonl referensi |
| `dataset_eval_combined.txt`, `link_300.txt` | kecil | daftar URL input |
| `teks_dataset/` | 52 MB | dataset teks untuk cascade IndoBERT |
| `cm_test_set/` | 40 MB | test set independen (subbab 4.3.3.4 / 4.3.6.4) |
| `evaluasi/` | 228 KB | `ground_truth.csv`, hasil evaluasi, grafik |
| `dokumen/` | 3,5 MB | catatan matematis, draf, instruksi |
| `requirements.txt`, `setup.sh`, `.gitignore`, `MIGRASI.md` | kecil | file setup ini |
| `script/` | 8 KB | `jalankan_semua.sh`, `popup_kode.html` |

## 2. Opsional — bukti/hasil untuk laporan (pindah kalau masih dipakai di dokumen KP)

| Item | Ukuran | Bisa di-generate ulang? |
|---|---|---|
| `runs/` | 3,9 MB | ya (`yolo val`) — tapi berisi confusion matrix jadi |
| `laporan_validasi/` | 1,3 MB | ya, dari skrip cascade |
| `gambar/` | 2,1 MB | gambar UML/rumus/hasil untuk laporan — sebaiknya ikut |
| `hasil_analisa_judol/` | **3,2 GB** | ya — output screening (screenshot high/medium/low) |
| `dataset_demo_refresh/` | 171 MB | ya — materi demo seminar |
| `evidence/`, `evidence_demo/` | 65 MB | ya — screenshot hasil pipeline |
| `dataset_baru/`, `dataset_demo/`, `dataset_cns_baru/` | 187 MB | ya — output dataset builder |
| `new_dataset_candidate/` | 1,3 MB | ya |

## 3. Bisa di-SKIP (regenerasi penuh, paling berat)

| Item | Ukuran | Cara regenerasi |
|---|---|---|
| `train/` | **2,0 GB** | dataset training YOLO — hanya perlu kalau mau **melatih ulang** model. Kalau `best-4.pt` sudah dibawa, ini tidak wajib. |
| `__pycache__/` | 40 KB | otomatis |
| `*.log` (`dataset_demo_refresh.log`, dll) | — | otomatis |
| `.DS_Store` | — | artefak macOS |

> **Ringkas:** bawa bagian 1 + 2 saja → ± 640 MB. Tambah `train/` hanya kalau
> berencana retrain YOLO.

## 4. Model IndoBERT (`indobert_gate/`)

Folder ini **tidak ada di working directory** — model di-fine-tune di Google
Colab dan disimpan di `MyDrive/GATESystem/indobert_gate/`. Kalau mau menjalankan
`evaluasi_cascade.py` di PC baru, salin folder itu dari Drive ke root project.
Kalau tidak, cascade cukup dijalankan ulang di Colab lewat
`_arsip/skrip/finetune_indobert_colab.py`.

---

## 5. Cara memindahkan

```bash
# Dari PC lama, buat arsip TANPA data berat + venv + cache:
cd ~/Downloads/laporan-kp
tar --exclude='kode/train' \
    --exclude='kode/__pycache__' \
    --exclude='kode/.venv' \
    --exclude='kode/.DS_Store' \
    --exclude='*.log' \
    -czf gate-migrasi.tar.gz kode/

# Sertakan train/ terpisah kalau perlu retrain:
tar -czf gate-train.tar.gz kode/train/
```

Pindahkan `gate-migrasi.tar.gz` (dan `best-4.pt` sudah termasuk di dalamnya).

---

## 6. Setup di PC baru

**Prasyarat:** Python 3.10.x (dikembangkan di 3.10.5). Git opsional.

```bash
tar -xzf gate-migrasi.tar.gz
cd kode

# otomatis: buat .venv, install deps, download Chromium
bash setup.sh
```

Atau manual:

```bash
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python -m playwright install chromium
# Linux tambahan: python -m playwright install-deps chromium
```

### Verifikasi

```bash
source .venv/bin/activate
python -c "import cv2, ultralytics, playwright, torch; print('import OK')"
python -c "from ultralytics import YOLO; YOLO('best-4.pt'); print('model OK')"
python screening_demo.py --help        # atau lihat argparse di file
```

---

## 7. Alternatif: push ke GitHub (repo privat)

Bisa. Yang penting **jangan** ikutkan data super-berat (`train/`,
`hasil_analisa_judol/`, `dataset_demo_refresh/`) — sudah diblok di `.gitignore`.
Sisa yang ikut ± **400 MB** (termasuk `best-4.pt` 50 MB), aman untuk repo privat.

Batas GitHub yang relevan:
- 100 MB per file (hard). `best-4.pt` = 50 MB → aman tanpa Git LFS.
- 2 GB per push. Repo dianjurkan < 1 GB.
- Push pertama ± 400 MB agak lama, wajar.

```bash
cd ~/Downloads/laporan-kp/kode
git init
git add .
git status                      # cek train/ & hasil_analisa_judol/ TIDAK muncul
git commit -m "Sistem GATE - kode, model, dataset inti"

# Opsi A: pakai GitHub CLI (kalau sudah install 'gh')
gh repo create gate-kp --private --source=. --push

# Opsi B: bikin repo privat kosong dulu di github.com, lalu:
git branch -M main
git remote add origin https://github.com/<user>/gate-kp.git
git push -u origin main
```

Di PC baru:

```bash
git clone https://github.com/<user>/gate-kp.git
cd gate-kp
bash setup.sh
```

Lalu salin manual `train/` (kalau mau retrain) dan `indobert_gate/` (kalau mau
cascade) — keduanya tidak ada di repo.

> Kalau `best-4.pt` mau dikeluarkan dari git juga: tambahkan `best-4.pt` ke
> `.gitignore`, atau pakai Git LFS (`git lfs track "*.pt"`).

---

## 8. Catatan versi & platform

- `requirements.txt` = versi yang dipin dari environment kerja (paling aman).
- `requirements-full.txt` = `pip freeze` lengkap; sebagian pin mungkin
  spesifik macOS/arm64. Pakai `requirements.txt` sebagai acuan utama.
- **GPU NVIDIA:** install `torch==2.2.2 torchvision==0.17.2` dari
  `--index-url https://download.pytorch.org/whl/cu121` **sebelum**
  `pip install -r requirements.txt`.
- `nest-asyncio` dan `google-colab` hanya dipakai di jalur Google Colab —
  tidak perlu di PC lokal.
- `.claude/settings.local.json` berisi izin lokal Claude Code, aman untuk
  tidak ikut dipindah.
