# Aturan Keputusan Cascade IndoBERT — GATE

**Status: PRE-REGISTERED.** Ditetapkan 10 Juli 2026, SEBELUM model dilatih dan
sebelum melihat satu pun hasil pada zona ambigu. Tidak boleh diubah setelah
evaluasi berjalan; perubahan apa pun harus dicatat sebagai deviasi.

## Definisi zona ambigu
URL diteruskan ke tingkat 2 (IndoBERT) jika dan hanya jika klasifikasi tingkat 1
(keyword berbobot + YOLO, konfigurasi Tabel 4.7) menghasilkan kategori **MEDIUM**.
Kategori HIGH dan LOW tingkat 1 bersifat final dan tidak disentuh cascade.

Dasar: MEDIUM adalah kelas ketidakpastian yang diakui sistem sendiri (skor
25–50, atau skor >= 50 tanpa konfirmasi visual). Pada dataset evaluasi 4.3.6,
zona ini berisi 254 URL (241 judol, 13 non-judol) = 11% dataset.

## Aturan keputusan tingkat 2
Untuk URL zona ambigu dengan probabilitas IndoBERT p = P(promosi judol | teks):

- p >= 0,5  ->  kategori akhir **HIGH** (terdeteksi)
- p <  0,5  ->  kategori akhir **LOW** (tidak terdeteksi)

Ambang 0,5 dipilih apriori sebagai titik keputusan Bayes standar untuk
klasifikasi biner, tanpa penyetelan terhadap data evaluasi.

## Analisis sensitivitas (dideklarasikan di muka)
Sebagai uji ketahanan (bukan pemilihan ambang), hasil juga dilaporkan pada
ambang 0,3 dan 0,7. Ambang utama tetap 0,5 apa pun hasilnya.

## Kasus tepi
- URL zona ambigu yang mati/tak terakses saat re-scrape (27 dari 254):
  dikeluarkan dari evaluasi cascade dan dilaporkan terpisah (konsisten dengan
  perlakuan situs tak terakses pada 4.3.6).
- Teks kosong/terlalu pendek saat inferensi: kategori tingkat 1 dipertahankan
  (MEDIUM), dicatat sebagai "tidak dapat diverifikasi".

## Metrik yang dilaporkan
1. Dalam zona: dari 227 URL medium yang terakses — berapa keputusan berubah,
   benar/salahnya perubahan (vs ground truth CNS/kontrol).
2. Keseluruhan: confusion matrix + Accuracy/Precision/Recall/F1 sebelum vs
   sesudah cascade pada dataset evaluasi yang sama persis (reduksi biner sama:
   HIGH = positif; LOW = negatif; MEDIUM sisa pasca-cascade tidak ada karena
   semua diputuskan tingkat 2, kecuali kasus tepi "tidak dapat diverifikasi"
   yang tetap dihitung positif sebagaimana perlakuan MEDIUM pada 4.3.6).
3. Kinerja klasifikasi IndoBERT pada test set internal (train/val/test
   80/10/10, group-aware split, seed 42).
