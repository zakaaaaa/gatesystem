import os
import csv
import json
import socket
import asyncio
import requests
import urllib3
import cv2
from concurrent.futures import ThreadPoolExecutor
from playwright.async_api import async_playwright
from ultralytics import YOLO

urllib3.disable_warnings()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CNS_FILE = os.path.join(BASE_DIR, 'CNS Gambling 09022026.txt')
MODEL_PATH = os.path.join(BASE_DIR, 'best-4.pt')

OUT_DIR = os.path.join(BASE_DIR, 'dataset_demo_refresh')
DIR_EVIDENCE = os.path.join(OUT_DIR, 'evidence')
for cat in ['high', 'medium', 'low']:
    os.makedirs(os.path.join(DIR_EVIDENCE, cat), exist_ok=True)

STATE_FILE = os.path.join(OUT_DIR, 'state.json')
LOG_CSV = os.path.join(OUT_DIR, 'log_screening.csv')

TARGET_HIGH = int(os.environ.get('TARGET_HIGH', 150))
TARGET_TOTAL = int(os.environ.get('TARGET_TOTAL', 300))
PRECHECK_BATCH = int(os.environ.get('PRECHECK_BATCH', 300))     # domain dicek liveness per batch
PRECHECK_WORKERS = 40
PLAYWRIGHT_CONCURRENCY = 8

BLOCK_IPS = {'36.86.63.185', '195.35.23.222'}
BLOCK_HOSTS = ('internetpositif', 'internet-positif', 'internetsehat',
               'trustpositif', 'aduankonten', 'kominfo')

# Sumber domain yang SUDAH PERNAH dipakai/dicek di proyek ini -> dikecualikan
EXCLUDE_FILES = [
    'dataset_eval_combined.txt',
    'dataset_judol_aktif.txt',
    'judol.txt',
    os.path.join('dataset_cns_baru', 'status_cek.csv'),
    os.path.join('dataset_cns_baru', 'dataset_2000.txt'),
    os.path.join('dataset_cns_baru', 'link_aktif.txt'),
]

def norm_domain(raw):
    d = raw.strip()
    if not d:
        return ''
    d = d.split(',')[0].strip()  # untuk baris csv "domain,status,..."
    d = d.replace('https://', '').replace('http://', '').split('/')[0]
    d = d.split()[0] if d.split() else d
    return d.lower()

def load_exclude_set():
    excl = set()
    for rel in EXCLUDE_FILES:
        path = os.path.join(BASE_DIR, rel)
        if not os.path.exists(path):
            continue
        with open(path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                dom = norm_domain(line)
                if dom and dom != 'domain':
                    excl.add(dom)
    return excl

def load_candidates():
    excl = load_exclude_set()
    domains = []
    with open(CNS_FILE, encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith('total'):
                continue
            dom = norm_domain(line)
            if dom:
                domains.append(dom)
    domains.reverse()  # descending: paling segar duluan (konsisten dgn kumpul_link_cns.py)
    seen, out = set(), []
    for d in domains:
        if d in seen or d in excl:
            continue
        seen.add(d)
        out.append(d)
    print(f"[INFO] Total domain CNS unik: {len(domains) if False else len(set(domains))} | dikecualikan (sudah pernah dipakai): {len(excl)} | kandidat baru: {len(out)}")
    return out

def cek_liveness(dom):
    try:
        ips = {info[4][0] for info in socket.getaddrinfo(dom, 443)}
    except Exception:
        return dom, 'MATI'
    if ips & BLOCK_IPS:
        return dom, 'BLOKIR'
    url = f'https://{dom}'
    try:
        r = requests.get(url, timeout=12, verify=False, allow_redirects=True,
                          headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        final = r.url.lower()
        if any(h in final for h in BLOCK_HOSTS):
            return dom, 'BLOKIR'
        if r.status_code < 400 and len(r.content) > 200:
            return dom, 'AKTIF'
        return dom, 'MATI'
    except Exception:
        return dom, 'MATI'

# ---------- Skoring IDENTIK dengan screening_cns.py / screening.py ----------
def calculate_frequency_weights():
    N = 966
    stats = {
        'zeus':         {'judol': 390, 'non': 6},
        'mahjong':      {'judol': 317, 'non': 10},
        'pragmatic':    {'judol': 516, 'non': 6},
        'casino':       {'judol': 59,  'non': 13},
        'koin':         {'judol': 203, 'non': 14},
        'vulgar':       {'judol': 167, 'non': 11},
        'pool':         {'judol': 218, 'non': 13},
        'pg':           {'judol': 397, 'non': 3},
        'mahjong_card': {'judol': 99,  'non': 14}
    }
    SCALING_FACTOR = 4.5
    final_weights = {}
    for label, count in stats.items():
        f_judol = count['judol'] / N * 100
        f_non = count['non'] / N * 100
        ratio = f_judol / (f_non + 1)
        final_weights[label] = round(ratio * SCALING_FACTOR, 2)
    return final_weights

VISUAL_WEIGHTS = calculate_frequency_weights()

def calculate_text_weights():
    N_JUDOL, N_NON = 2249, 1116
    stats = {
        'pragmatic':    {'judol': 1041, 'non': 1},
        'togel':        {'judol': 1654, 'non': 10},
        'rtp':          {'judol': 1539, 'non': 12},
        'olympus':      {'judol': 729,  'non': 0},
        'gacor':        {'judol': 1125, 'non': 8},
        'pg soft':      {'judol': 559,  'non': 0},
        'mahjong ways': {'judol': 498,  'non': 0},
        'maxwin':       {'judol': 465,  'non': 2},
        'zeus':         {'judol': 455,  'non': 2},
        'sicbo':        {'judol': 347,  'non': 0},
        'scatter':      {'judol': 616,  'non': 9},
        'baccarat':     {'judol': 393,  'non': 8},
        'withdraw':     {'judol': 1273, 'non': 56},
        'casino':       {'judol': 1755, 'non': 105},
        'deposit':      {'judol': 1554, 'non': 104},
        'slot':         {'judol': 2033, 'non': 205},
        'bet':          {'judol': 1459, 'non': 225},
        'live':         {'judol': 1961, 'non': 334},
        'pola':         {'judol': 316,  'non': 50},
        'rungkad':      {'judol': 47,   'non': 1},
        'dealer':       {'judol': 113,  'non': 28},
        'judi':         {'judol': 549,  'non': 479}
    }
    ratios = {}
    for kw, count in stats.items():
        f_judol = count['judol'] / N_JUDOL * 100
        f_non = count['non'] / N_NON * 100
        ratios[kw] = f_judol / (f_non + 1)
    SCALING = 80 / max(ratios.values())
    return {kw: round(r * SCALING, 2) for kw, r in ratios.items()}

TEXT_WEIGHTS = calculate_text_weights()

SAFETY_WORDS = ['polisi', 'ditangkap', 'hukum', 'berita', 'edukasi',
                'pemerintah', 'learning', 'education', 'repository']
SAFETY_PENALTY = 80

print(f"[INFO] Memuat model dari {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

def calculate_score(detections, text_content):
    text_lower = text_content.lower()
    raw_visual = sum(VISUAL_WEIGHTS.get(det['label'], 0) * det['conf'] for det in detections)
    norm_visual = min(raw_visual, 100)
    raw_text = sum(score for word, score in TEXT_WEIGHTS.items() if word in text_lower)
    for safe in SAFETY_WORDS:
        if safe in text_lower:
            raw_text -= SAFETY_PENALTY
    norm_text = max(min(raw_text, 100), 0)
    final_score = round((norm_visual * 0.5) + (norm_text * 0.5), 2)
    if final_score >= 50:
        risk = 'high' if norm_visual > 0 else 'medium'
    elif final_score > 25:
        risk = 'medium'
    else:
        risk = 'low'
    return risk, final_score

def matched_keywords(text_content):
    tl = text_content.lower()
    return [kw for kw in TEXT_WEIGHTS if kw in tl]

def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return [], None
    results = model.predict(img, conf=0.25, verbose=False)
    detections = []
    annotated = img.copy()
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            label = results[0].names[int(box.cls[0])]
            conf = float(box.conf[0])
            if label in VISUAL_WEIGHTS:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append({'label': label, 'conf': conf, 'box': (x1, y1, x2, y2)})
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return detections, annotated

# ---------- State ----------
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {'idx': 0, 'high': [], 'medium': [], 'low': [],
            'n_precheck': 0, 'n_aktif_precheck': 0,
            'n_blocked_browse': 0, 'n_error': 0}

def save_state(state):
    tmp = STATE_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)

if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, 'w', newline='') as f:
        csv.writer(f).writerow(['domain', 'risk', 'score', 'visual', 'keywords'])

def log_row(domain, risk, score, visual, keywords):
    with open(LOG_CSV, 'a', newline='') as f:
        csv.writer(f).writerow([domain, risk, score, visual, keywords])

# ---------- Playwright screening (concurrent) ----------
sem = asyncio.Semaphore(PLAYWRIGHT_CONCURRENCY)

async def scan_one(context, domain, state):
    full_url = f'https://{domain}'
    safe_name = domain.replace('/', '_')[:60]
    temp_path = os.path.join(OUT_DIR, f"{safe_name}_temp.jpg")
    async with sem:
        page = await context.new_page()
        try:
            await page.goto(full_url, timeout=30000, wait_until='domcontentloaded')
            await asyncio.sleep(5)

            if any(h in page.url.lower() for h in BLOCK_HOSTS):
                state['n_blocked_browse'] += 1
                print(f"[BLOKIR-BROWSE] {domain}")
                return

            page_text = await page.evaluate(
                "document.body.innerText + ' ' + Array.from(document.images).map(i=>i.src).join(' ')")
            await page.screenshot(path=temp_path, full_page=False)

            detections, annotated = process_image(temp_path)
            risk, score = calculate_score(detections, page_text)

            visual_tags = ','.join(f"{d['label']}:{d['conf']:.2f}" for d in detections)
            kws = ','.join(matched_keywords(page_text))

            ev = os.path.join(DIR_EVIDENCE, risk, f"{safe_name}.jpg")
            cv2.imwrite(ev, annotated)

            entry = {'domain': domain, 'risk': risk, 'score': score,
                     'visual': visual_tags, 'keywords': kws}
            state[risk].append(entry)
            log_row(domain, risk, score, visual_tags, kws)
            print(f"[{risk.upper():6}] {domain}  skor={score}  (H={len(state['high'])} M={len(state['medium'])} L={len(state['low'])})")
        except Exception as e:
            state['n_error'] += 1
            print(f"[ERROR] {domain} => {type(e).__name__}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            await page.close()

async def run():
    candidates = load_candidates()
    state = load_state()
    print(f"[RESUME] idx={state['idx']}  HIGH={len(state['high'])} MEDIUM={len(state['medium'])} LOW={len(state['low'])}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1366, 'height': 2500}, ignore_https_errors=True)

        while True:
            n_high = len(state['high'])
            n_total = n_high + len(state['medium']) + len(state['low'])
            if n_high >= TARGET_HIGH and n_total >= TARGET_TOTAL:
                print("[SELESAI] Kuota tercapai.")
                break
            if state['idx'] >= len(candidates):
                print("[HABIS] Kandidat CNS baru habis sebelum kuota tercapai.")
                break

            chunk = candidates[state['idx']: state['idx'] + PRECHECK_BATCH]
            state['idx'] += len(chunk)

            with ThreadPoolExecutor(max_workers=PRECHECK_WORKERS) as ex:
                aktif_chunk = []
                for dom, st in ex.map(cek_liveness, chunk):
                    state['n_precheck'] += 1
                    if st == 'AKTIF':
                        state['n_aktif_precheck'] += 1
                        aktif_chunk.append(dom)

            print(f"[PRECHECK] batch {len(chunk)} dicek -> {len(aktif_chunk)} AKTIF | total dicek {state['n_precheck']}")

            tasks = [scan_one(context, dom, state) for dom in aktif_chunk]
            if tasks:
                await asyncio.gather(*tasks)

            save_state(state)
            n_high = len(state['high'])
            n_total = n_high + len(state['medium']) + len(state['low'])
            print(f"[PROGRES] idx={state['idx']}/{len(candidates)} | HIGH={n_high}/{TARGET_HIGH} | TOTAL={n_total}/{TARGET_TOTAL} | blokir-browse={state['n_blocked_browse']} error={state['n_error']}")

        await browser.close()

    # ---- Susun hasil akhir 300 ----
    high_sorted = sorted(state['high'], key=lambda x: -x['score'])
    other_sorted = sorted(state['medium'], key=lambda x: -x['score']) + \
                   sorted(state['low'], key=lambda x: -x['score'])

    if len(high_sorted) >= TARGET_TOTAL:
        final = high_sorted[:TARGET_TOTAL]
    else:
        final = high_sorted + other_sorted[: TARGET_TOTAL - len(high_sorted)]

    with open(os.path.join(OUT_DIR, 'link_300.txt'), 'w') as f:
        f.write('\n'.join(e['domain'] for e in final) + '\n')
    with open(os.path.join(OUT_DIR, 'link_300.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['domain', 'risk', 'score', 'visual', 'keywords'])
        w.writeheader()
        w.writerows(final)

    n_final_high = sum(1 for e in final if e['risk'] == 'high')
    print("\n===== RINGKASAN AKHIR =====")
    print(f"Kandidat CNS diperiksa (precheck)  : {state['n_precheck']}")
    print(f"  AKTIF (precheck)                 : {state['n_aktif_precheck']}")
    print(f"Berhasil discreening (H+M+L)        : {len(state['high']) + len(state['medium']) + len(state['low'])}")
    print(f"  HIGH                              : {len(state['high'])}")
    print(f"  MEDIUM                            : {len(state['medium'])}")
    print(f"  LOW                               : {len(state['low'])}")
    print(f"  BLOKIR saat browse (SNI-block)     : {state['n_blocked_browse']}")
    print(f"  ERROR                             : {state['n_error']}")
    print(f"\nHASIL AKHIR link_300.txt            : {len(final)} link ({n_final_high} HIGH)")
    print(f"Output di                           : {OUT_DIR}")

if __name__ == '__main__':
    asyncio.run(run())
