# Instruksi Lanjutan Laporan KP — Sistem GATE (Deteksi Situs Judi Online)

> **Cara pakai dokumen ini:** Tempel seluruh isi file ini ke Claude (web/claude.ai) di awal percakapan, lalu minta: *"Bantu saya menulis subbab 4.3.6 (dan seterusnya) Laporan KP saya berdasarkan konteks ini."* Dokumen ini **self-contained** — semua angka, temuan, dan keputusan metodologi sudah tertanam, jadi Claude web tidak perlu mengakses file lokal.

---

## 1. Konteks Singkat

- **Penulis:** Mahasiswa Kerja Praktik (KP) di **Kementerian Komunikasi dan Digital (Komdigi)** RI.
- **Topik:** **Sistem GATE** — pipeline otomatis untuk mendeteksi & mengklasifikasi situs **judi online (judol)** guna mendukung tim pemblokiran (SAMAN) Komdigi.
- **Gaya penulisan:** Bahasa Indonesia akademik formal, sitasi bentuk `[n]`, konsisten dengan Bab 3 & 4.3.5 yang sudah ada. Hindari bahasa kasual.
- **Status:** Bab 1–3 selesai; Bab 4 sampai **4.3.5.4** selesai. **Yang ditulis sekarang: subbab 4.3.6 "Pengujian dan Evaluasi Sistem".**

### Arsitektur Sistem GATE (pipeline)
`Daftar URL → Browser automation (Playwright, asinkron, stealth) → tangkapan layar + ekstraksi teks → deteksi visual (YOLOv8) → skoring kata kunci tekstual → Weighted Sum Model (WSM) → klasifikasi risiko 3 tingkat (HIGH/MEDIUM/LOW)`.

- Pembobotan fitur: **Frequency Ratio** (subbab 3.8). **Bukan AHP.**
- Penggabungan modalitas: **Weighted Sum Model**, bobot setara 0,5 : 0,5 (subbab 3.9.1):
  `final_score = (skor_visual × 0,5) + (skor_tekstual × 0,5)`, kedua skor dinormalisasi 0–100.
- Klasifikasi 3 tingkat (subbab 3.9.2 & 4.3.5.4):
  - **HIGH**: `final_score ≥ 50` **dan** ada bukti visual (`skor_visual > 0`)
  - **MEDIUM**: `final_score ≥ 50` tanpa bukti visual, **atau** `25 < final_score < 50`
  - **LOW**: `final_score ≤ 25`

### Landasan teori yang relevan (sudah ada di Bab 3, wajib dirujuk di 4.3.6)
- **3.10 Metrik Evaluasi Klasifikasi**, **3.10.1 Confusion Matrix**, **3.10.2 Akurasi/Presisi/Recall/F1.**
- Reduksi biner (subbab 3.10.1): **HIGH + MEDIUM = kelas POSITIF (judol)**, **LOW = kelas NEGATIF**.
- Rumus: Accuracy=(TP+TN)/total · Precision=TP/(TP+FP) · Recall=TP/(TP+FN) · F1=2·P·R/(P+R).

---

## 2. Apa yang Sudah Dilakukan untuk Pengujian (untuk ditulis di 4.3.6)

> **CATATAN SCOPE (penting):** Pengujian sistem GATE dilaporkan dilakukan pada **2.000 situs judol + 300 situs non-judol**. Subbab 2.2 dan 2.3 di bawah (kategorisasi akses, penyaringan domain mati/terblokir, statistik 8.030 kandidat) adalah **proses PENYIAPAN DATASET di luar scope sistem GATE** — boleh disinggung sekilas sebagai metode pengumpulan data, **tetapi jangan ditulis seolah-olah bagian dari kemampuan/fungsi sistem GATE**. Fokus utama 4.3.6 adalah klasifikasi pada 2.000 + 300.

### 2.1 Pembangunan dataset uji (gold standard)
- **2.000 situs judol AKTIF** (positif) + **300 situs non-judol kontrol** (negatif).
- Situs judol diambil dari daftar **CNS Gambling** Komdigi, lalu **discan dan disaring**: hanya yang **benar-benar dapat diakses** (`access_category = OK`) yang masuk. Total **8.030 kandidat** discan untuk memperoleh 2.000 aktif.
- **Kriteria "aktif" bersifat independen dari skor sistem** (bukan dipilih karena terdeteksi HIGH/MEDIUM) — ini penting agar evaluasi tidak *circular* dan Recall tetap sahih.
- Situs kontrol non-judol sengaja dipilih sebagai **hard negatives** (yang mirip judol): portal game, social casino (Slotomania, dll.), mahjong/kartu legal, situs mitologi Yunani (memicu kelas visual "zeus"), berita game, fintech (memicu kata "deposit/withdraw"). Tujuannya menguji ketahanan **Precision**.

### 2.2 Penanganan anomali akses (kategorisasi otomatis)
Saat scan, tiap URL dikategorikan: **OK** (terbaca), **BLOCKED_KOMDIGI** (redirect ke `internet-positif.info`/TrustPositif), **BOT_CHALLENGE** (tertahan Cloudflare "Verifying"), **DEAD** (mati/SSL error/halaman error Cloudflare 521/1016/parkir). Hanya **OK** yang masuk perhitungan metrik; sisanya **dikeluarkan dan dilaporkan terpisah** karena tidak menyajikan konten judol nyata (ground-truth harus mencerminkan konten saat scan, bukan reputasi historis URL).
- Untuk menembus anti-bot Cloudflare digunakan **playwright-stealth + User-Agent realistis + locale id-ID + Chrome channel**, sehingga banyak situs judol hidup yang semula tak terbaca menjadi terbaca.

### 2.3 Statistik kondisi populasi (temuan pendukung — bagus untuk pembahasan)
Dari 8.030 kandidat judol CNS yang discan:
- **DEAD (mati/error) ≈ 3.386 (42%)** — bukti kuat efektivitas takedown/pemblokiran & cepatnya *churn* domain judol.
- **BLOCKED_KOMDIGI ≈ 39**, **BOT_CHALLENGE ≈ 103**.
- **Judol aktif (OK) = 4.249** → distribusi sistem: HIGH = 3.247, MEDIUM = 516, LOW = 486.

---

## 3. HASIL EVALUASI FINAL (angka untuk dimasukkan ke laporan)

**Dataset uji:** **2.000 situs judol + 300 situs non-judol = 2.300 URL** (semua dievaluasi). Penyaringan apakah suatu domain sudah diblokir/mati **berada di luar scope sistem GATE** (itu urusan penyiapan dataset, bukan tugas klasifikasi); maka pengujian cukup dinyatakan dilakukan pada 2.000 judol + 300 non-judol.

### Confusion Matrix (positif = judol [HIGH/MEDIUM])
|  | Prediksi POSITIF | Prediksi NEGATIF |
|---|---|---|
| **Aktual POSITIF (judol)** | TP = 1760 | FN = 240 |
| **Aktual NEGATIF (non-judol)** | FP = 17 | TN = 283 |

### Metrik
| Metrik | Nilai |
|---|---|
| **Accuracy** | **88,83 %** |
| **Precision** | **99,04 %** |
| **Recall** | **88,00 %** |
| **F1-Score** | **93,20 %** |

Catatan: kontrol non-judol yang tidak berhasil diakses dihitung sebagai TN (sistem tidak salah menandainya sebagai judol).

### Grafik tersedia (untuk disisipkan ke laporan)
- `evaluasi/grafik_confusion_matrix.png`
- `evaluasi/grafik_metrik_evaluasi.png`
- `laporan_validasi/grafik_distribusi_klasifikasi.png`, `grafik_distribusi_skor.png`, `grafik_scatter_visual_vs_tekstual.png`, `collage_sampel_high/medium/low.jpg`

---

## 4. Analisis Kesalahan (bahan Pembahasan & Keterbatasan)

### False Positive (17) — semuanya situs game/casino-adjacent
Contoh: `slingo.com`, `freemahjong.org`, `gametop.com`, `hoyoverse.com` (HIGH); `playstation.com`, `ign.com`, `lagged.com`, `ageofmythology.com`, `uptodown.com` (MEDIUM). **Interpretasi:** FP terjadi pada situs sah yang memang menampilkan elemen visual/tekstual mirip judol (slot/mahjong/casino). Mayoritas FP = MEDIUM (sesuai desain "MEDIUM perlu verifikasi manual"), hanya sedikit HIGH. Precision tetap 99,04 % → sistem sangat presisi.

### False Negative (240) — terkonsentrasi pada togel & keterbatasan detektor
- **±23 % FN bernuansa togel/toto/4d** (mis. `huntertotoalter`, `top1toto199`, `kangtotowis`, `niastoto4d`). **Penyebab:** model YOLOv8 dilatih pada visual *slot* (zeus, mahjong, pragmatic, dll.), **bukan togel/lotere**, dan bobot kata "togel" rendah → **blind-spot**.
- Sebagian FN = halaman judol yang banner-nya tidak ter-render (butuh login/interaksi), serta **kemungkinan mislabel** entri CNS (mis. `staradio1073fm.com`, `geosriwijaya.com` tampak bukan judol) — bila diverifikasi, recall sebenarnya bisa > 88 %.

### Keterbatasan & validitas (untuk subbab keterbatasan / Bab penutup)
- **Validitas temporal:** daftar CNS cepat usang (42 % domain sudah mati). Metrik dihitung hanya pada situs aktif saat scan.
- **Cakupan detektor visual** terbatas pada genre *slot* → togel/lotere under-detected (rekomendasi: tambah data latih togel + naikkan bobot kata togel/toto).
- **Anti-bot:** sebagian kecil situs (BOT_CHALLENGE) tetap tak terbaca → limitasi pengukuran yang dilaporkan transparan, bukan disembunyikan.
- **Pelabelan ground-truth** berbasis sumber CNS + keterbacaan; idealnya divalidasi spot-check manual (disarankan ~30–50 sampel/kelas).

---

## 5. STRUKTUR YANG DISARANKAN UNTUK SUBBAB 4.3.6

Tulis dengan empat pilar (pisahkan evaluasi **model** yang sudah ada di 4.3.3.4 dari evaluasi **sistem end-to-end** di sini):

**4.3.6.1 Tujuan dan Rancangan Pengujian**
- Tujuan: memverifikasi pemenuhan kebutuhan (KF/KNF) + mengukur kinerja klasifikasi terhadap ground-truth.
- Definisikan lingkungan uji, dataset gold-standard (2.000 judol aktif + 300 kontrol), dan protokol penentuan "aktif" (independen dari skor).

**4.3.6.2 Pengujian Fungsional (Black-box)**
- Tabel pemetaan KF-01…KF-11 & KNF-01…KNF-06 → skenario → hasil (mis. KNF-03 robustness lewat penanganan timeout; KNF-02 konkurensi; KNF-04/05 interpretabilitas/auditabilitas via breakdown skor & output visual; KNF-06 format CSV siap pakai).

**4.3.6.3 Pengujian Kinerja Operasional**
- Throughput, tingkat keberhasilan akses, rata-rata waktu/URL, skalabilitas (8.030 URL diproses). Sajikan temuan 42 % domain mati sebagai konteks populasi.

**4.3.6.4 Pengujian Akurasi Klasifikasi (inti)**
- Protokol: reduksi biner (HIGH+MEDIUM = positif), pengecualian situs tak-terakses (blokir/mati/bot-challenge) + alasannya.
- Sajikan **Confusion Matrix** (Tabel) + perhitungan **Accuracy/Precision/Recall/F1** merujuk rumus 3.10.2.
- Bahas hasil: Precision 99 % (FP gaming-adjacent), Recall 88 % (FN togel blind-spot), F1 93,2 %.

**4.3.6.5 (opsional) Pembahasan dan Keterbatasan** — gunakan Bagian 4 di atas.

---

## 6. Catatan Integritas Ilmiah (jangan dilanggar saat menulis)
1. Jangan definisikan "judol aktif" sebagai "yang terdeteksi HIGH/MEDIUM" → *circular reasoning*, Recall jadi semu 100 %. "Aktif" = dapat diakses + konten nyata, independen skor.
2. Situs terblokir Komdigi & mati **dikeluarkan dari metrik tetapi dilaporkan terpisah** (transparan), bukan dibuang diam-diam.
3. Sebut keterbatasan (togel blind-spot, validitas temporal, bot-challenge) secara jujur — ini menambah kredibilitas, bukan mengurangi.

---

## 7. Daftar Tugas Lanjutan (checklist)
- [ ] Tulis 4.3.6.1–4.3.6.4 (+4.3.6.5) memakai angka di Bagian 3 & analisis Bagian 4.
- [ ] Sisipkan Tabel Confusion Matrix + tabel metrik + grafik (Bagian 3).
- [ ] (Opsional) Spot-check ~30 FN untuk konfirmasi togel vs mislabel → bisa menaikkan recall final.
- [ ] Lengkapi Bab 5/6 (Kesimpulan): tegaskan Precision 99 % & Recall 88 %, temuan 42 % domain mati, rekomendasi perluasan detektor togel.
- [ ] Pastikan penomoran subbab & sitasi konsisten dengan gaya Bab 3/4 yang sudah ada.

---

## 8. DATA untuk 4.3.6.2 (Pengujian Fungsional) & 4.3.6.3 (Kinerja Operasional)

### 8.1 Kebutuhan Fungsional & Non-Fungsional (sumber: subbab 4.3.1 laporan)
Gunakan tabel berikut untuk **4.3.6.2 Pengujian Fungsional (black-box)** — petakan tiap kebutuhan ke skenario uji & hasil. Semua **Terpenuhi** berdasarkan bukti operasional sistem.

| Kode | Kebutuhan | Cara verifikasi (skenario uji) | Hasil |
|---|---|---|---|
| KF-01 | Pemrosesan daftar URL | Memuat file `.txt` & memproses berurutan | Terpenuhi (8.330 URL diproses) |
| KF-02 | Akses halaman otomatis | Playwright mengakses tiap URL | Terpenuhi |
| KF-03 | Pengambilan tangkapan layar | `page.screenshot` tiap halaman terakses | Terpenuhi (±4.521 screenshot tersimpan) |
| KF-04 | Ekstraksi konten tekstual | Ambil `innerText` + sumber gambar | Terpenuhi |
| KF-05 | Deteksi objek visual | Inferensi YOLOv8 pada screenshot | Terpenuhi |
| KF-06 | Penghitungan skor visual | `Σ(bobot×confidence)`, dinormalisasi 0–100 | Terpenuhi |
| KF-07 | Penghitungan skor tekstual | Skor kata kunci berbobot, dinormalisasi | Terpenuhi |
| KF-08 | Mekanisme penalti negatif | Penalti kata edukatif/jurnalistik (safety) | Terpenuhi |
| KF-09 | Klasifikasi tingkat risiko | Aturan 3 tingkat HIGH/MEDIUM/LOW | Terpenuhi |
| KF-10 | Pelaporan otomatis | `report_final_*.csv` + `scan_log.json` | Terpenuhi |
| KF-11 | Penyimpanan dataset YOLO | Simpan citra bersih/teranotasi + label | Terpenuhi (fitur tersedia; dinonaktifkan saat uji untuk hemat disk) |
| KNF-01 | Skalabilitas | Memproses ribuan URL satu sesi | Terpenuhi (8.330 URL/sesi) |
| KNF-02 | Konkurensi | `asyncio.Semaphore` (6 paralel) | Terpenuhi |
| KNF-03 | Robustness | 37,5% gagal akses **tidak** menghentikan proses; fitur RESUME | Terpenuhi |
| KNF-04 | Interpretabilitas | Screenshot teranotasi + skor numerik | Terpenuhi |
| KNF-05 | Auditabilitas | `breakdown` skor & fitur tiap keputusan | Terpenuhi |
| KNF-06 | Kemudahan integrasi | Output CSV siap pakai tim SAMAN | Terpenuhi |

### 8.2 Kinerja Operasional pada dataset uji 2.300 (untuk 4.3.6.3)
| Metrik operasional | Nilai |
|---|---|
| Total URL diuji | **2.300** (2.000 judol + 300 non-judol) |
| Berhasil diakses & diklasifikasi | 2.247 (**97,7 %**) |
| Gagal diakses | 53 (**2,3 %**) |
| Rata-rata waktu per URL | **23,83 detik** (median 23,29) |
| Tingkat konkurensi | 6 proses paralel |

**Interpretasi untuk 4.3.6.3:** keberhasilan akses 97,7% menunjukkan sistem andal memproses situs target; kegagalan 2,3% (situs tak terjangkau) ditangani tanpa menghentikan keseluruhan proses → memenuhi **KNF-03 Robustness**. Kemampuan memproses ribuan URL satu sesi secara paralel memenuhi **KNF-01 Skalabilitas** & **KNF-02 Konkurensi**.
