# Arsip Skrip Sistem GATE

Skrip di `skrip/` **tidak dipakai untuk demonstrasi seminar**, tetapi seluruhnya
diuraikan di dalam Laporan Kerja Praktik. Disimpan di sini agar folder kerja tetap
ringkas tanpa menghilangkan jejak kode yang dibahas di laporan.

| Skrip | Peran di laporan |
|---|---|
| `screening_gate.py` | Pipeline skrining utama — subbab 4.3.4 dan 4.3.5 |
| `evaluasi_sistem.py` | Penghitungan confusion matrix dan metrik — subbab 4.3.6.4 |
| `ambil_teks.py` | Pengambilan ulang teks halaman untuk analisis tekstual |
| `ambil_screenshot_cm.py` | Penyiapan test set independen — subbab 4.3.3.4 |
| `scan_nonjudol_visual.py` | Pencacahan frekuensi visual non-judol untuk Frequency Ratio |
| `kumpul_nonjudol.py`, `kumpul_negatif_game.py` | Pengumpulan sampel pembanding (hard negative) |
| `kumpul_link_cns.py`, `screening_cns.py` | Pengumpulan dan penyaringan daftar CNS |
| `siapkan_pool_judol.py`, `bangun_dataset_judol.py` | Penyusunan dataset uji |
| `recek_kategori.py` | Pemeriksaan ulang kategori hasil |
| `kumpul_demo_refresh.py` | Penyegaran daftar tautan untuk materi demo |
| `finetune_indobert_colab.py`, `evaluasi_cascade.py` | Eksplorasi cascade IndoBERT — **tidak masuk laporan** |

Yang tetap berada di `kode/`: `screening_demo.py` dan `screening.py` (dipakai demo),
`best-4.pt` (model), `CNS Gambling 09022026.txt`, dan `dataset_eval_combined.txt`.
