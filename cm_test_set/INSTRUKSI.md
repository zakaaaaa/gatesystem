# Paket Koreksi Anotasi — Test Set Independen untuk Confusion Matrix best-4.pt

## Apa ini?
100 screenshot BERSIH yang diambil segar pada 20 Juli 2026 dari domain-domain yang
**tidak beririsan** dengan 988 gambar training (dicek per domain). Model best-4.pt
sudah membuat **draf label otomatis** (folder `labels_draft/`, 338 kotak). Tugasmu
BUKAN menganotasi dari nol — hanya **mengoreksi draf**.

Komposisi sampling: 50 domain eks-kategori HIGH, 25 MEDIUM, 25 LOW — supaya ada
gambar dengan banyak objek, sedikit objek, dan tanpa objek. CATATAN: kategori lama
hanya untuk pengambilan sampel; beberapa domain kini mungkin sudah berganti konten
(jadi situs biasa/parkir). Anotasi APA YANG TERLIHAT SEKARANG di gambar — situs
non-judol tanpa objek justru berguna sebagai sampel negatif.

## Langkah di Make Sense (makesense.ai)

1. Buka https://www.makesense.ai → **Get Started** → drop SEMUA gambar dari folder
   `images/` → pilih **Object Detection**.
2. Saat diminta membuat label list: pilih **Load labels from file** → gunakan
   `labels_names.txt` (URUTAN 10 nama ini JANGAN diubah — id-nya harus cocok).
3. **Actions → Import Annotations** → pilih format **YOLO** → drop semua file .txt
   dari folder `labels_draft/`.
4. Periksa gambar SATU PER SATU dan koreksi:
   - **HAPUS** kotak yang salah (bukan objek kelas itu / deteksi ngaco).
   - **PERBAIKI** kelas yang keliru (mis. koin terdeteksi sebagai casino).
   - **TAMBAHKAN kotak untuk objek yang TERLEWAT model** ← INI YANG PALING PENTING.
     Kalau langkah ini dilewati, recall di confusion matrix jadi bohong.
   - Kelas `jp` (banner jackpot) TETAP dianotasi bila ada — perlu untuk CM 10 kelas.
5. Selesai semua → **Actions → Export Annotations** → format **YOLO (zip)**.
6. Simpan zip hasil ekspor ke folder ini dengan nama `labels_final.zip`, lalu bilang
   ke Claude: "labels final sudah ada".

## Estimasi waktu
± 1 jam (kebanyakan gambar hanya perlu dicek sekilas; gambar LOW umumnya kosong).

## Catatan metodologi (untuk laporan / jawaban dosen)
- Metode ini = *model-assisted annotation*: pra-label otomatis + verifikasi dan
  koreksi manual. Praktik standar industri; ditulis apa adanya di laporan.
- Ground truth final adalah hasil koreksianmu, bukan keluaran model.
- Setelah `labels_final.zip` ada, Claude akan menghitung confusion matrix +
  metrik per kelas best-4.pt pada test set ini dan menambahkannya ke 4.3.3.4
  sebagai pengujian pada data yang belum pernah dilihat model.
