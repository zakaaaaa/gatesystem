import os
import cv2
import asyncio
from playwright.async_api import async_playwright
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
INPUT_FILE = os.path.join(BASE_DIR, 'dataset_eval_c.txt')
MODEL_PATH = os.path.join(BASE_DIR, 'best-4.pt')

DIR_EVIDENCE = os.path.join(BASE_DIR, 'evidence')
DIR_DATASET = os.path.join(BASE_DIR, 'dataset_baru')

for cat in ['high', 'medium', 'low']:
    os.makedirs(os.path.join(DIR_EVIDENCE, cat), exist_ok=True)

os.makedirs(os.path.join(DIR_DATASET, 'images_clean'), exist_ok=True)
os.makedirs(os.path.join(DIR_DATASET, 'images_verification'), exist_ok=True)
os.makedirs(os.path.join(DIR_DATASET, 'labels'), exist_ok=True)

# Konfigurasi skoring WAJIB identik dengan screening_gate.py (Tabel 4.7 laporan);
# file ini hanya versi ringan pipeline-nya, bukan varian skoring
def calculate_frequency_weights():
    # Frekuensi kehadiran label pada 966 sampel judol dan 966 non-judol (deteksi YOLO)
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
    # Frekuensi kemunculan tiap kata kunci pada sampel situs judol (n=2249) dan
    # non-judol (n=1116), hasil empirical frequency analysis dari konten teks.
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
        ratios[kw] = f_judol / (f_non + 1)     # Frequency Ratio + Laplace smoothing
    SCALING = 80 / max(ratios.values())         # normalisasi: bobot maksimum = 80
    return {kw: round(r * SCALING, 2) for kw, r in ratios.items()}


TEXT_WEIGHTS = calculate_text_weights()

SAFETY_WORDS = ['polisi', 'ditangkap', 'hukum', 'berita', 'edukasi',
                'pemerintah', 'learning', 'education', 'repository']
SAFETY_PENALTY = 80

print(f"[INFO] Memuat model dari {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
NAME_TO_ID = {v: k for k, v in model.names.items()}

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

def process_image_and_dataset(img_path, filename_base):
    img = cv2.imread(img_path)
    if img is None: return [], None

    img_h, img_w = img.shape[:2]
    results = model.predict(img, conf=0.25, verbose=False)
    
    detections = []
    annotated_img = img.copy()
    
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            label = results[0].names[int(box.cls[0])]
            conf = float(box.conf[0])
            xyxy = map(int, box.xyxy[0].tolist())
            
            if label in VISUAL_WEIGHTS:
                x1, y1, x2, y2 = xyxy
                detections.append({'label': label, 'conf': conf, 'box': (x1,y1,x2,y2)})
                
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated_img, f"{label} {conf:.2f}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imwrite(os.path.join(DIR_DATASET, 'images_clean', f"{filename_base}.jpg"), img)
    cv2.imwrite(os.path.join(DIR_DATASET, 'images_verification', f"{filename_base}.jpg"), annotated_img)
    
    with open(os.path.join(DIR_DATASET, 'labels', f"{filename_base}.txt"), 'w') as f:
        for det in detections:
            x1, y1, x2, y2 = det['box']
            cls_id = NAME_TO_ID.get(det['label'], 0)
            x_center, y_center = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            w, h = (x2 - x1), (y2 - y1)
            f.write(f"{cls_id} {x_center/img_w:.6f} {y_center/img_h:.6f} {w/img_w:.6f} {h/img_h:.6f}\n")

    return detections, annotated_img

async def scan_url(context, url):
    full_url = url if url.startswith('http') else f'https://{url}'
    safe_name = url.replace('https://', '').replace('http://', '').replace('/', '_')[:40]
    
    print(f"[PROSES] {full_url}")
    page = await context.new_page()
    
    temp_path = os.path.join(BASE_DIR, f"{safe_name}_temp.jpg")
    
    try:
        await page.goto(full_url, timeout=30000, wait_until='domcontentloaded')
        await asyncio.sleep(5)
        
        page_text = await page.evaluate("document.body.innerText + ' ' + Array.from(document.images).map(i=>i.src).join(' ')")
        
        await page.screenshot(path=temp_path, full_page=False)
        
        detections, annotated_img = process_image_and_dataset(temp_path, safe_name)
        
        risk, score = calculate_score(detections, page_text)
        
        evidence_path = os.path.join(DIR_EVIDENCE, risk, f"{safe_name}.jpg")
        cv2.imwrite(evidence_path, annotated_img)
        
        print(f"[SELESAI] {url} => {risk.upper()} (Skor: {score})")
        
    except Exception:
        print(f"[GAGAL] {url} => Error/Timeout")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        await page.close()

async def main():
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Buat file {INPUT_FILE} terlebih dahulu berisi daftar URL.")
        return

    with open(INPUT_FILE, 'r') as f:
        urls = [line.strip().split()[0] for line in f if line.strip()]

    print(f"[INFO] Ditemukan {len(urls)} target URL.")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1366, 'height': 2500}, ignore_https_errors=True)
        
        for url in urls:
            await scan_url(context, url)
            
        await browser.close()
    print("[INFO] Proses Selesai! Cek folder 'evidence' dan 'dataset_baru'.")

if __name__ == '__main__':
    asyncio.run(main())