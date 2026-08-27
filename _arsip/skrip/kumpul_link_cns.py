import os
import csv
import socket
import requests
import urllib3
from concurrent.futures import ThreadPoolExecutor

urllib3.disable_warnings()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CNS_FILE = os.path.join(BASE_DIR, 'CNS Gambling 09022026.txt')

# Direktori output BARU, terpisah dari evidence/ & dataset_baru/ yang lama
OUT_DIR = os.path.join(BASE_DIR, 'dataset_cns_baru')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_AKTIF = 200      # jumlah link aktif yang dikejar
TARGET_TOTAL = 2000     # total link di dataset akhir
MAX_SCAN = 5000         # batas maksimal domain yang dicek liveness (biar tidak lari terlalu lama)

# Halaman blokir Internet Positif / TrustPositif
BLOCK_IPS = {'36.86.63.185', '195.35.23.222'}
BLOCK_HOSTS = ('internetpositif', 'internet-positif', 'internetsehat',
               'trustpositif', 'aduankonten', 'kominfo')

def load_descending():
    domains = []
    with open(CNS_FILE, encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith('total'):
                continue
            dom = line.split()[0]
            dom = dom.replace('https://', '').replace('http://', '').split('/')[0]
            if dom:
                domains.append(dom)
    domains.reverse()  # descending: baris paling bawah CNS -> pertama
    # buang duplikat sambil jaga urutan
    seen, out = set(), []
    for d in domains:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out

def cek_liveness(dom):
    """Return (dom, status, final_url). status: AKTIF / BLOKIR / MATI"""
    try:
        ips = {info[4][0] for info in socket.getaddrinfo(dom, 443)}
    except Exception:
        return dom, 'MATI', ''
    if ips & BLOCK_IPS:
        return dom, 'BLOKIR', ''
    url = f'https://{dom}'
    try:
        r = requests.get(url, timeout=12, verify=False, allow_redirects=True,
                         headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        final = r.url.lower()
        if any(h in final for h in BLOCK_HOSTS):
            return dom, 'BLOKIR', r.url
        if r.status_code < 400 and len(r.content) > 200:
            return dom, 'AKTIF', r.url
        return dom, 'MATI', r.url
    except Exception:
        return dom, 'MATI', ''

def main():
    descending = load_descending()
    print(f"[INFO] Total domain unik di CNS (descending): {len(descending)}")

    aktif, status_log = [], []
    scanned = 0

    with ThreadPoolExecutor(max_workers=40) as ex:
        # proses per batch agar bisa berhenti begitu 200 aktif tercapai
        batch = 120
        for i in range(0, min(len(descending), MAX_SCAN), batch):
            chunk = descending[i:i + batch]
            for dom, st, final in ex.map(cek_liveness, chunk):
                scanned += 1
                status_log.append((dom, st, final))
                if st == 'AKTIF' and len(aktif) < TARGET_AKTIF:
                    aktif.append(dom)
            n_aktif = len(aktif)
            print(f"[PROGRES] dicek {scanned:>4} | aktif {n_aktif:>3}/{TARGET_AKTIF}")
            if n_aktif >= TARGET_AKTIF:
                break

    # susun dataset 2000: link aktif dulu, lalu isi dari descending sampai 2000
    dataset, seen = [], set()
    for d in aktif:
        dataset.append(d); seen.add(d)
    for d in descending:
        if len(dataset) >= TARGET_TOTAL:
            break
        if d not in seen:
            dataset.append(d); seen.add(d)

    with open(os.path.join(OUT_DIR, 'link_aktif.txt'), 'w') as f:
        f.write('\n'.join(aktif) + '\n')
    with open(os.path.join(OUT_DIR, 'dataset_2000.txt'), 'w') as f:
        f.write('\n'.join(f"{d} 1" for d in dataset) + '\n')
    with open(os.path.join(OUT_DIR, 'status_cek.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['domain', 'status', 'final_url'])
        w.writerows(status_log)

    n_block = sum(1 for _, s, _ in status_log if s == 'BLOKIR')
    n_mati = sum(1 for _, s, _ in status_log if s == 'MATI')
    print("\n===== RINGKASAN =====")
    print(f"Domain dicek        : {scanned}")
    print(f"  AKTIF             : {len(aktif)}")
    print(f"  BLOKIR            : {n_block}")
    print(f"  MATI/timeout      : {n_mati}")
    print(f"link_aktif.txt      : {len(aktif)} link")
    print(f"dataset_2000.txt    : {len(dataset)} link")
    print(f"Output di           : {OUT_DIR}")

if __name__ == '__main__':
    main()
