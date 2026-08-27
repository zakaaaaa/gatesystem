# Draf Naskah Laporan — Penambahan Komponen Cascade IndoBERT

Berkas ini berisi teks siap-tempel untuk laporan, disusun mengikuti gaya, istilah,
dan konvensi penomoran `Laporan_KP_Final.docx`. Tiga bagian:

- **Bagian A** — Revisi paragraf kajian pustaka (subbab 3.7 Analisis Tekstual).
- **Bagian B** — Subbab landasan teori baru (usulan **3.11**).
- **Bagian C** — Subbab hasil & pembahasan baru (usulan **4.3.7**).

Angka yang menunggu hasil training ditandai `【ISI: …】`. Setelah training selesai
dan skrip evaluasi cascade dijalankan, seluruh placeholder akan saya isikan.

> **Catatan penempatan penomoran.** Landasan teori cascade ditempatkan sebagai
> subbab **3.11** (setelah 3.10) semata untuk menghindari penomoran ulang 3.8–3.10
> beserta seluruh referensi silangnya di Bab IV. Bila secara alur kamu lebih suka
> teori ini menyusul langsung 3.7 Analisis Tekstual, subbab dapat dipindah dan
> diberi nomor 3.8 (konsekuensinya 3.8–3.10 lama bergeser). Hasil implementasi
> ditempatkan sebagai **4.3.7**, menyusul 4.3.6, tanpa penomoran ulang apa pun.
> Dua sitasi baru diberi nomor **[26]** dan **[27]** (lanjutan dari [25]); daftar
> pustakanya ada di akhir berkas.

---

## BAGIAN A — Revisi Paragraf Kajian Pustaka (Subbab 3.7)

Pada subbab **3.7 Analisis Tekstual**, ganti paragraf kedua (yang saat ini diakhiri
kalimat *"Temuan ini mendukung pilihan desain sistem GATE yang menggunakan keyword
matching berbobot sebagai metode analisis tekstual primer."*) dengan paragraf
berikut:

> Dalam konteks moderasi konten berbahasa Indonesia, perbandingan antara berbagai
> model untuk klasifikasi komentar spam judi online menemukan bahwa meskipun
> IndoBERT mencapai performa tertinggi, model ringan berbasis CNN dan SVM dengan
> ekstraksi fitur kata kunci tetap relevan karena efisiensi komputasi yang
> signifikan dalam skenario deployment skala besar [2]. Temuan ini melandasi
> rancangan analisis tekstual sistem GATE yang bersifat bertingkat (hibrida):
> pencarian kata kunci berbobot difungsikan sebagai penilai primer yang ringan dan
> mudah diinterpretasikan untuk menilai seluruh halaman, sementara model bahasa
> kontekstual IndoBERT difungsikan secara selektif hanya sebagai verifikator pada
> subset kasus yang penilaian primernya tergolong ambigu. Dengan strategi ini,
> sistem memperoleh manfaat pemahaman kontekstual IndoBERT untuk membedakan promosi
> judol dari pembahasan yang sah mengenai perjudian, tanpa menanggung beban
> komputasinya pada keseluruhan volume URL. Landasan teori pendekatan bertingkat
> ini diuraikan pada subbab 3.11, sedangkan penerapan dan evaluasinya dibahas pada
> subbab 4.3.7.

*(Paragraf berikutnya — tentang RNN/CNN 93,07%/92,80% [16] — dan paragraf
implementasi `calculate_risk_score()` tetap seperti semula, tidak perlu diubah.)*

---

## BAGIAN B — Landasan Teori Baru (Subbab 3.11)

### 3.11  Model Bahasa Kontekstual dan Klasifikasi Bertingkat

Pendekatan analisis tekstual berbasis kata kunci berbobot yang diuraikan pada
subbab 3.7 sampai 3.8 unggul dalam hal kecepatan dan keterbukaan interpretasi,
namun memiliki keterbatasan mendasar: penilaian dilakukan berdasarkan kehadiran
kata secara harfiah tanpa mempertimbangkan konteks kalimat yang melingkupinya.
Akibatnya, sebuah halaman berita yang memuat frasa "situs slot ilegal diblokir"
dan sebuah halaman promosi yang memuat frasa "slot gacor maxwin" sama-sama memicu
kata kunci perjudian, meskipun maksud keduanya berlawanan. Untuk menangani kasus
semacam ini secara lebih cermat, sistem GATE melengkapi penilai primer dengan
model bahasa kontekstual, yang diterapkan melalui skema klasifikasi bertingkat.

#### 3.11.1  IndoBERT sebagai Model Bahasa Kontekstual

BERT (Bidirectional Encoder Representations from Transformers) adalah arsitektur
model bahasa berbasis Transformer yang mempelajari representasi kata secara
dua-arah, sehingga makna sebuah kata ditentukan oleh keseluruhan konteks kalimat
di kiri maupun kanannya. Berbeda dari pencarian kata kunci yang memperlakukan
setiap istilah secara independen, representasi kontekstual memungkinkan model
membedakan makna kata yang sama pada konteks yang berbeda. IndoBERT merupakan
varian BERT yang dilatih khusus pada korpus besar teks berbahasa Indonesia
(Indo4B), sehingga lebih peka terhadap struktur dan kosakata bahasa Indonesia
dibanding model multibahasa umum [26].

Dalam praktik klasifikasi, IndoBERT digunakan melalui mekanisme fine-tuning, yaitu
melanjutkan pelatihan model yang telah memiliki pengetahuan bahasa umum dengan
sejumlah data berlabel spesifik terhadap tugas yang dituju. Sebuah lapisan
klasifikasi ditambahkan di atas keluaran model, lalu seluruh parameter disesuaikan
agar model dapat memetakan teks masukan ke kelas target. Pada sistem GATE, tugas
yang dituju adalah klasifikasi biner: membedakan teks halaman yang merupakan
**promosi perjudian** dari teks yang **bukan promosi** (misalnya berita, edukasi,
atau situs sah yang kebetulan bersinggungan kosakata dengan perjudian).

#### 3.11.2  Klasifikasi Bertingkat (Cascade Classification)

Klasifikasi bertingkat adalah strategi yang menyusun beberapa penilai secara
berurutan, dari penilai yang cepat dan berbiaya rendah menuju penilai yang lebih
teliti namun berbiaya tinggi. Penilai pertama menyelesaikan mayoritas kasus yang
mudah diputuskan, dan hanya meneruskan kasus yang sulit atau meragukan ke penilai
berikutnya. Strategi ini telah lama digunakan dalam sistem klasifikasi untuk
menyeimbangkan akurasi dan efisiensi komputasi, sebagaimana pada detektor objek
klasik yang menolak sebagian besar kandidat pada tahap-tahap awal yang ringan [27].

Keunggulan utama pendekatan bertingkat adalah biaya komputasi tetap terkendali:
penilai berbiaya tinggi hanya dijalankan pada sebagian kecil masukan, bukan pada
keseluruhan volume data. Dalam konteks sistem GATE, penilai primer berupa kombinasi
deteksi visual dan pencarian kata kunci berbobot menyelesaikan sebagian besar URL
secara definitif, sedangkan verifikasi kontekstual berbasis IndoBERT hanya
dijalankan pada URL yang penilaian primernya berada pada zona ketidakpastian.
Dengan demikian, keputusan yang sudah tegas tidak diproses ulang, dan sumber daya
difokuskan pada kasus yang paling memerlukan pertimbangan tambahan.

---

## BAGIAN C — Hasil dan Pembahasan Baru (Subbab 4.3.7)

### 4.3.7  Peningkatan Presisi Zona Ambigu melalui Verifikasi Kontekstual (Cascade IndoBERT)

Hasil evaluasi pada subbab 4.3.6 menunjukkan bahwa sistem GATE mencapai kinerja
klasifikasi yang tinggi, namun juga memperlihatkan bahwa kesalahan klasifikasi
cenderung terkonsentrasi pada kategori risiko MEDIUM. Kategori ini pada dasarnya
merupakan zona ketidakpastian sistem, yaitu situasi ketika skor komposit tidak cukup
tinggi untuk dinyatakan HIGH namun tidak cukup rendah untuk dinyatakan LOW. Pada
dataset evaluasi yang terdiri atas 2.000 situs judol aktif dan 300 situs non-judol,
kategori MEDIUM memuat 254 URL, atau sekitar 11% dari total data yang berhasil
diakses. Dari 17 kesalahan klasifikasi positif (false positive) yang tercatat,
sebanyak 13 di antaranya berada tepat pada kategori MEDIUM. Temuan ini menjadi dasar
penerapan lapisan verifikasi kontekstual: alih-alih mengubah keseluruhan mekanisme
penilaian, sistem cukup meninjau ulang subset kasus ambigu tersebut menggunakan
model bahasa yang lebih peka konteks, sesuai kerangka klasifikasi bertingkat yang
dijelaskan pada subbab 3.11.2.

#### 4.3.7.1  Rancangan Cascade dan Definisi Zona Ambigu

Lapisan verifikasi dirancang sebagai tingkat kedua yang tidak mengubah keputusan
tingkat pertama untuk kategori HIGH dan LOW. Sebuah URL diteruskan ke verifikasi
IndoBERT jika dan hanya jika penilai primer mengklasifikasikannya sebagai MEDIUM.
Pemilihan MEDIUM sebagai satu-satunya zona ambigu didasarkan pada makna kategori
tersebut yang secara inheren merupakan pernyataan ketidakpastian sistem, sehingga
kategori inilah yang paling memerlukan pertimbangan tambahan.

Untuk menjaga objektivitas evaluasi, aturan keputusan cascade ditetapkan terlebih
dahulu sebelum model dilatih dan sebelum satu pun hasil pada zona ambigu diamati.
Aturan tersebut adalah sebagai berikut. Untuk setiap URL pada zona MEDIUM, IndoBERT
menghasilkan probabilitas *p* bahwa teks halaman merupakan promosi judol. Apabila
*p* ≥ 0,5, kategori akhir URL dinaikkan menjadi HIGH (terdeteksi sebagai judol);
apabila *p* < 0,5, kategori akhir diturunkan menjadi LOW (tidak terdeteksi). Ambang
0,5 dipilih sebagai titik keputusan biner standar tanpa penyetelan terhadap data
evaluasi. Sebagai uji ketahanan, hasil juga dilaporkan pada ambang 0,3 dan 0,7,
namun ambang utama tetap 0,5.

Terdapat dua kasus tepi yang perlakuannya juga ditetapkan di muka. Pertama, URL zona
MEDIUM yang tidak dapat diakses ulang pada saat pengambilan teks (misalnya domain
telah mati) dikeluarkan dari evaluasi cascade dan dilaporkan terpisah, konsisten
dengan perlakuan situs tak terakses pada subbab 4.3.6. Kedua, URL yang teksnya
kosong atau terlalu pendek untuk dinilai mempertahankan kategori MEDIUM-nya dan
dicatat sebagai "tidak dapat diverifikasi"; sesuai reduksi biner pada subbab 4.3.6,
kategori MEDIUM tetap dihitung sebagai terdeteksi.

#### 4.3.7.2  Pembangunan Dataset Pelatihan IndoBERT

Fine-tuning IndoBERT memerlukan data teks berlabel. Kelas positif (promosi judol)
dibangun dari teks halaman situs-situs judol aktif yang **bukan** merupakan bagian
dari dataset evaluasi, dengan hanya mengambil halaman yang penilai primernya menilai
HIGH atau MEDIUM guna menekan risiko salah label dari sumber daftar. Kelas negatif
(bukan promosi) dibangun dari dua jenis sumber: artikel berita dan edukasi berbahasa
Indonesia yang membahas perjudian daring dari sejumlah portal berita nasional, serta
situs sah yang beririsan kosakata dengan perjudian seperti situs permainan daring,
kasino sosial, teknologi finansial, dan lembaga pemerintah. Jenis kedua ini penting
karena merepresentasikan justru kasus-kasus yang paling rentan disalahklasifikasikan
oleh penilai primer.

Untuk mencegah kebocoran data (data leakage) yang dapat menggelembungkan hasil
evaluasi secara semu, diterapkan pemisahan tegas antara data pelatihan dan data uji
cascade. Seluruh domain yang termasuk dalam 254 URL zona MEDIUM dataset evaluasi
dikeluarkan dari data pelatihan, baik atas dasar kesamaan teks maupun kesamaan
domain. Setelah kurasi, penghapusan duplikat, dan penyaringan halaman yang terlalu
pendek, diperoleh **2.243** contoh berlabel (1.640 promosi judol dan 603 bukan
promosi). Data dibagi menjadi himpunan latih, validasi, dan uji dengan proporsi
80:10:10 (1.795 : 228 : 220) secara group-aware, yaitu pembagian dilakukan pada
tingkat domain agar tidak ada domain yang muncul pada lebih dari satu himpunan.

#### 4.3.7.3  Proses Fine-tuning

Model dasar yang digunakan adalah `indobert-base-p1` [26]. Teks masukan dipotong
hingga panjang maksimum 512 token, sesuai kapasitas maksimum arsitektur. Pelatihan
dilakukan selama 3 epoch dengan laju pembelajaran (learning rate) 2 × 10⁻⁵ dan
fungsi kerugian cross-entropy berbobot kelas untuk mengimbangi ketidakseimbangan
proporsi antara kelas positif dan negatif. Model terbaik dipilih berdasarkan skor
F1 tertinggi pada himpunan validasi. Pada himpunan uji internal, model mencapai
akurasi 【ISI: akurasi test%】, presisi 【ISI】, recall 【ISI】, dan F1-score
【ISI】 dalam membedakan promosi judol dari bukan promosi.

#### 4.3.7.4  Hasil Verifikasi Zona Ambigu dan Dampaknya terhadap Kinerja Sistem

Dari 254 URL kategori MEDIUM pada dataset evaluasi, sebanyak 227 URL berhasil
diakses ulang dan teksnya dapat diverifikasi, sedangkan 27 URL sisanya tidak dapat
diakses dan dilaporkan terpisah. Terhadap 227 URL tersebut, lapisan IndoBERT
mengubah keputusan pada 【ISI: jumlah】 kasus, yang terdiri atas 【ISI】 URL yang
dinaikkan dari MEDIUM menjadi HIGH dan 【ISI】 URL yang diturunkan menjadi LOW. Dari
seluruh perubahan tersebut, 【ISI】 perubahan terbukti benar bila diperiksa terhadap
data acuan.

Dampak lapisan verifikasi terhadap kinerja klasifikasi sistem secara keseluruhan
disajikan pada Tabel 4.【ISI】, yang membandingkan metrik sebelum dan sesudah
penerapan cascade pada dataset evaluasi yang sama.

| Metrik    | Sebelum Cascade | Sesudah Cascade |
|-----------|-----------------|-----------------|
| TP        | 1760            | 【ISI】          |
| FN        | 240             | 【ISI】          |
| FP        | 17              | 【ISI】          |
| TN        | 283             | 【ISI】          |
| Akurasi   | 88,83%          | 【ISI】          |
| Presisi   | 99,04%          | 【ISI】          |
| Recall    | 88,00%          | 【ISI】          |
| F1-Score  | 93,20%          | 【ISI】          |

Uji ketahanan terhadap pemilihan ambang menunjukkan bahwa pada ambang 0,3 dan 0,7,
hasil sistem berturut-turut menjadi 【ISI: ringkasan singkat】, yang
mengindikasikan bahwa keputusan cascade 【ISI: relatif stabil / sensitif】 terhadap
pergeseran ambang di sekitar nilai standar.

#### 4.3.7.5  Keterbatasan

Lapisan verifikasi cascade dirancang untuk memperbaiki ketepatan keputusan pada zona
ambigu, sehingga cakupan perbaikannya terbatas pada kategori MEDIUM. Kesalahan
klasifikasi yang terjadi di luar zona tersebut — khususnya situs judol yang telah
dinilai LOW oleh penilai primer dengan skor rendah yang meyakinkan, seperti sebagian
situs bertema togel/toto yang menjadi titik buta model deteksi visual — tidak
tersentuh oleh cascade dan tetap menjadi keterbatasan sistem sebagaimana diuraikan
pada subbab 4.3.6. Selain itu, karena teks halaman dipotong hingga 512 token,
informasi pada bagian bawah halaman yang panjang tidak ikut dipertimbangkan; dan
karena tokenizer IndoBERT dilatih pada teks berbahasa Indonesia baku, istilah yang
sengaja diobfuskasi (misalnya penulisan "g4c0r") berpotensi tidak dikenali secara
optimal.

---

## Sitasi Baru (untuk Daftar Pustaka)

Tambahkan pada daftar pustaka, menyesuaikan format yang dipakai laporan:

- **[26]** B. Wilie, K. Vincentio, G. I. Winata, S. Cahyawijaya, et al., "IndoNLU:
  Benchmark and Resources for Evaluating Indonesian Natural Language Understanding,"
  dalam *Proceedings of the 1st Conference of the Asia-Pacific Chapter of the ACL
  and the 10th International Joint Conference on Natural Language Processing*, 2020,
  hlm. 843–857.
- **[27]** P. Viola and M. Jones, "Rapid Object Detection using a Boosted Cascade of
  Simple Features," dalam *Proceedings of the 2001 IEEE Computer Society Conference
  on Computer Vision and Pattern Recognition (CVPR)*, 2001, vol. 1, hlm. I-511–I-518.

*(Catatan: [26] adalah rujukan resmi IndoBERT/IndoNLU; [27] adalah rujukan klasik
konsep cascade classification. Jika kamu punya rujukan cascade yang lebih dekat ke
domain NLP/moderasi konten, [27] dapat diganti.)*
