import os
import cv2
import csv
import asyncio
from playwright.async_api import async_playwright
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'

# ---- INPUT & OUTPUT terpisah dari yang lama ----
INPUT_FILE = os.path.join(BASE_DIR, 'dataset_cns_baru', 'link_aktif.txt')
MODEL_PATH = os.path.join(BASE_DIR, 'best-4.pt')

OUT_ROOT = os.path.join(BASE_DIR, 'dataset_cns_baru', 'hasil')
DIR_EVIDENCE = os.path.join(OUT_ROOT, 'evidence')
for cat in ['high', 'medium', 'low']:
    os.makedirs(os.path.join(DIR_EVIDENCE, cat), exist_ok=True)

# Halaman blokir Internet Positif / TrustPositif -> jangan dihitung sebagai low, tandai BLOKIR
BLOCK_HOSTS = ('internetpositif', 'internet-positif', 'internetsehat',
               'trustpositif', 'aduankonten', 'kominfo')

# ---------- Skoring IDENTIK dengan screening.py ----------
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

RESULTS = []      # semua hasil (untuk log lengkap)
HARVEST = []      # hanya medium & high

async def scan_url(context, url):
    full_url = url if url.startswith('http') else f'https://{url}'
    safe_name = url.replace('https://', '').replace('http://', '').replace('/', '_')[:40]
    temp_path = os.path.join(OUT_ROOT, f"{safe_name}_temp.jpg")
    page = await context.new_page()
    try:
        await page.goto(full_url, timeout=30000, wait_until='domcontentloaded')
        await asyncio.sleep(5)

        # Deteksi kalau ternyata dibelokkan ke halaman blokir
        if any(h in page.url.lower() for h in BLOCK_HOSTS):
            print(f"[BLOKIR] {url} => dibelokkan ke {page.url}")
            RESULTS.append({'url': url, 'risk': 'blocked', 'score': 0, 'visual': '', 'keywords': ''})
            return

        page_text = await page.evaluate(
            "document.body.innerText + ' ' + Array.from(document.images).map(i=>i.src).join(' ')")
        await page.screenshot(path=temp_path, full_page=False)

        detections, annotated = process_image(temp_path)
        risk, score = calculate_score(detections, page_text)

        visual_tags = ','.join(f"{d['label']}:{d['conf']:.2f}" for d in detections)
        kws = ','.join(matched_keywords(page_text))

        RESULTS.append({'url': url, 'risk': risk, 'score': score,
                        'visual': visual_tags, 'keywords': kws})

        # simpan evidence
        ev = os.path.join(DIR_EVIDENCE, risk, f"{safe_name}.jpg")
        cv2.imwrite(ev, annotated)

        # hanya medium & high yang masuk daftar harvest
        if risk in ('medium', 'high'):
            HARVEST.append({'url': url, 'risk': risk, 'score': score,
                            'visual': visual_tags, 'keywords': kws})

        print(f"[SELESAI] {url} => {risk.upper()} (Skor: {score})")
    except Exception as e:
        print(f"[GAGAL] {url} => {type(e).__name__}")
        RESULTS.append({'url': url, 'risk': 'error', 'score': 0, 'visual': '', 'keywords': ''})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        await page.close()

async def main():
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Input tidak ada: {INPUT_FILE}")
        return
    with open(INPUT_FILE) as f:
        urls = [line.strip().split()[0] for line in f if line.strip()]
    print(f"[INFO] Target URL: {len(urls)}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1366, 'height': 2500}, ignore_https_errors=True)
        for i, url in enumerate(urls, 1):
            print(f"[{i}/{len(urls)}] {url}")
            await scan_url(context, url)
        await browser.close()

    # ---- tulis output ----
    with open(os.path.join(OUT_ROOT, 'log_lengkap.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['url', 'risk', 'score', 'visual', 'keywords'])
        w.writeheader(); w.writerows(RESULTS)

    HARVEST.sort(key=lambda x: (-{'high': 2, 'medium': 1}[x['risk']], -x['score']))
    with open(os.path.join(OUT_ROOT, 'daftar_medium_high.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['url', 'risk', 'score', 'visual', 'keywords'])
        w.writeheader(); w.writerows(HARVEST)
    with open(os.path.join(OUT_ROOT, 'daftar_medium_high.txt'), 'w') as f:
        f.write('\n'.join(h['url'] for h in HARVEST) + '\n')

    n_high = sum(1 for h in HARVEST if h['risk'] == 'high')
    n_med = sum(1 for h in HARVEST if h['risk'] == 'medium')
    n_low = sum(1 for r in RESULTS if r['risk'] == 'low')
    n_block = sum(1 for r in RESULTS if r['risk'] == 'blocked')
    n_err = sum(1 for r in RESULTS if r['risk'] == 'error')
    print("\n===== RINGKASAN =====")
    print(f"Total dipindai : {len(RESULTS)}")
    print(f"  HIGH         : {n_high}")
    print(f"  MEDIUM       : {n_med}")
    print(f"  LOW          : {n_low}")
    print(f"  BLOKIR       : {n_block}")
    print(f"  ERROR        : {n_err}")
    print(f"DAFTAR medium+high : {len(HARVEST)} link  -> daftar_medium_high.txt / .csv")
    print(f"Output di      : {OUT_ROOT}")

if __name__ == '__main__':
    asyncio.run(main())
