"""
bangun_dataset_judol.py
=======================
Saring hasil scan (scan_log.json) untuk situs judol yang BENAR-BENAR AKTIF
(access_category == OK: bisa diakses + konten nyata, BUKAN blokir Komdigi /
Cloudflare / mati), independen dari skor sistem.

- Jika judol aktif >= TARGET (default 2000): bangun dataset final +
  dataset_eval_combined.txt + evaluasi/ground_truth.csv.
- Jika < TARGET: laporkan kekurangan & instruksi menambah pool.

Pakai:
    python bangun_dataset_judol.py            # target 2000
    python bangun_dataset_judol.py 1500       # target lain
"""
import os, sys, json, csv, random
from collections import Counter

SCAN_LOG    = 'scan_log.json'
POOL_FILE   = 'pool_judol.txt'
CNS_FILE    = 'CNS Gambling 09022026.txt'
NONJUDOL    = 'dataset_nonjudol_300.txt'
OUT_JUDOL   = 'dataset_judol_aktif.txt'
OUT_COMBINED= 'dataset_eval_combined.txt'
GROUND_TRUTH= 'evaluasi/ground_truth.csv'
TARGET      = int(sys.argv[1]) if len(sys.argv) > 1 else 2000


def norm(u):
    u = u.strip().lower()
    for p in ('https://', 'http://'):
        if u.startswith(p):
            u = u[len(p):]
    return u[4:].rstrip('/') if u.startswith('www.') else u.rstrip('/')


def load_lines(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8', errors='ignore') as f:
        return [ln.strip().split()[0] for ln in f if ln.strip()]


def main():
    if not os.path.exists(SCAN_LOG):
        print(f"[ERROR] {SCAN_LOG} tidak ditemukan. Scan dulu pool-nya."); sys.exit(1)

    data = json.load(open(SCAN_LOG, encoding='utf-8'))

    # Definisi pool judol: dari pool_judol.txt bila ada, selain itu dari CNS.
    judol_src = POOL_FILE if os.path.exists(POOL_FILE) else CNS_FILE
    judol_pool = {norm(d) for d in load_lines(judol_src)}
    nonjudol_set = {norm(d) for d in load_lines(NONJUDOL)}
    print(f"[INFO] Pool judol dari: {judol_src} ({len(judol_pool)} URL)")

    # Telusuri scan_log untuk entri judol
    cat_stats = Counter()          # kategori akses semua judol yang discan
    risk_stats = Counter()         # risk level judol AKTIF
    active = {}                    # norm -> url_asli (judol aktif/OK)
    judol_scanned = 0

    for e in data:
        k = norm(e['url'])
        if k not in judol_pool or k in nonjudol_set:
            continue
        judol_scanned += 1
        status = e.get('status')
        acc = e.get('access_category') or ('OK' if status == 'SUCCESS' else 'DEAD')
        cat_stats[acc] += 1
        if status == 'SUCCESS' and acc == 'OK':
            active[k] = e['url']
            rl = (e.get('score_data') or {}).get('risk_level', '?')
            risk_stats[rl] += 1

    n_active = len(active)
    print("\n" + "=" * 56)
    print("  REKAP PENYARINGAN JUDOL AKTIF")
    print("=" * 56)
    print(f"  Judol discan        : {judol_scanned}")
    print(f"  Kategori akses      :")
    for c, v in cat_stats.most_common():
        print(f"      {c:<16}: {v}")
    print(f"  >> JUDOL AKTIF (OK) : {n_active}")
    print(f"     Distribusi skor  : " +
          ", ".join(f"{k}={v}" for k, v in risk_stats.most_common()))
    print(f"  Target              : {TARGET}")
    print("=" * 56)

    if n_active < TARGET:
        kurang = TARGET - n_active
        print(f"\n[BELUM CUKUP] Kurang {kurang} situs judol aktif.")
        print(f"  Tambah pool lalu scan lagi:")
        print(f"    python siapkan_pool_judol.py {max(kurang*3, 3000)}")
        print(f"    INPUT_FILE=pool_judol.txt python screening_gate.py")
        print(f"    python bangun_dataset_judol.py {TARGET}")
        sys.exit(1)          # belum cukup -> orchestrator lanjut nambah pool

    # Cukup -> pilih TARGET (acak terseed agar representatif & reproducible)
    keys = sorted(active.keys())
    random.seed(42)
    random.shuffle(keys)
    chosen = [active[k] for k in keys[:TARGET]]

    with open(OUT_JUDOL, 'w', encoding='utf-8') as f:
        f.write('\n'.join(chosen) + '\n')

    nonjudol = load_lines(NONJUDOL)
    with open(OUT_COMBINED, 'w', encoding='utf-8') as f:
        f.write('\n'.join(chosen + nonjudol) + '\n')

    os.makedirs('evaluasi', exist_ok=True)
    with open(GROUND_TRUTH, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['url', 'prediksi_sistem', 'skor_final', 'label_manual'])
        for u in chosen:
            w.writerow([u, '', '', 1])
        for u in nonjudol:
            w.writerow([u, '', '', 0])

    # Cek apakah non-judol sudah discan (untuk eval butuh prediksinya)
    scanned = {norm(e['url']) for e in data}
    nj_belum = [u for u in nonjudol if norm(u) not in scanned]

    print(f"\n[OK] Dataset final dibuat:")
    print(f"  - {OUT_JUDOL} ({len(chosen)} judol aktif)")
    print(f"  - {OUT_COMBINED} ({len(chosen)+len(nonjudol)} total)")
    print(f"  - {GROUND_TRUTH} (label terisi: judol=1, non-judol=0)")
    if nj_belum:
        print(f"\n[!] {len(nj_belum)} situs non-judol belum discan. Jalankan dulu:")
        print(f"    INPUT_FILE={NONJUDOL} python screening_gate.py")
    print(f"\nLalu hitung metrik:  python evaluasi_sistem.py eval")
    sys.exit(0)              # cukup -> orchestrator lanjut ke tahap evaluasi


if __name__ == '__main__':
    main()
