"""
siapkan_pool_judol.py
=====================
Ambil batch URL judol BARU dari CNS Gambling untuk pool scanning.
Anti-duplikat: tidak mengambil URL yang sudah ada di pool, di scan_log,
di dataset non-judol, maupun di dataset judol lama.

Bisa dipanggil berulang untuk MENAMBAH batch (mis. kalau situs aktif
belum cukup 2000):
    python siapkan_pool_judol.py 5000     # tambah 5000 URL baru ke pool
    python siapkan_pool_judol.py 3000     # tambah 3000 lagi
"""
import os, sys, json, random

CNS_FILE     = 'CNS Gambling 09022026.txt'
POOL_FILE    = 'pool_judol.txt'           # master pool judol yang akan discan
SCAN_LOG     = 'scan_log.json'
NONJUDOL     = 'dataset_nonjudol_300.txt'
JUDOL_LAMA   = 'dataset_judol_1000.txt'   # sampel lama (pre-stealth) -> dihindari


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
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000

    if not os.path.exists(CNS_FILE):
        print(f"[ERROR] {CNS_FILE} tidak ditemukan."); sys.exit(1)

    cns = load_lines(CNS_FILE)
    cns_uniq, seen = [], set()
    for d in cns:
        k = norm(d)
        if k not in seen:
            seen.add(k); cns_uniq.append(d)

    # Kumpulkan semua yang HARUS dihindari
    avoid = set()
    for path in (POOL_FILE, NONJUDOL, JUDOL_LAMA):
        for d in load_lines(path):
            avoid.add(norm(d))
    if os.path.exists(SCAN_LOG):
        try:
            for e in json.load(open(SCAN_LOG, encoding='utf-8')):
                avoid.add(norm(e['url']))
        except Exception:
            pass

    kandidat = [d for d in cns_uniq if norm(d) not in avoid]
    print(f"[INFO] CNS unik: {len(cns_uniq)} | sudah dipakai/hindari: {len(avoid)} "
          f"| kandidat baru: {len(kandidat)}")

    if not kandidat:
        print("[WARN] Tidak ada kandidat baru tersisa di CNS."); return

    existing_pool = load_lines(POOL_FILE)
    random.seed(1000 + len(existing_pool))          # batch berbeda tiap pemanggilan
    batch = random.sample(kandidat, min(n, len(kandidat)))

    with open(POOL_FILE, 'a', encoding='utf-8') as f:
        f.write('\n'.join(batch) + '\n')

    total = len(existing_pool) + len(batch)
    print(f"[OK] +{len(batch)} URL ditambahkan ke {POOL_FILE} (total pool: {total})")
    print(f"     Scan dgn:  INPUT_FILE={POOL_FILE} python screening_gate.py")
    print(f"     (RESUME aktif: hanya URL baru yang akan discan)")


if __name__ == '__main__':
    main()
