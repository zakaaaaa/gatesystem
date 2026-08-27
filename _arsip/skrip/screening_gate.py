import os
import json
import asyncio
import aiofiles
import cv2
import logging
import csv
import threading
import sys
import time
import random
import numpy as np
from datetime import datetime
from collections import Counter
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from ultralytics import YOLO

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

IS_COLAB = 'google.colab' in sys.modules
DRIVE_BASE = '/content/drive/MyDrive/GATESystem'

if IS_COLAB:
    try:
        if not os.path.exists('/content/drive/MyDrive'):
            from google.colab import drive
            drive.mount('/content/drive')
    except Exception:
        pass
    BASE_DIR = os.environ.get('BASE_DIR', DRIVE_BASE)
else:
    try:
        _here = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        _here = '.'
    BASE_DIR = os.environ.get('BASE_DIR', _here)


def _p(name):
    return name if os.path.isabs(name) else os.path.join(BASE_DIR, name)


INPUT_FILE      = _p(os.environ.get('INPUT_FILE', 'dataset_eval_combined.txt'))
MODEL_PATH      = _p(os.environ.get('MODEL_PATH', 'best-4.pt'))
OUTPUT_DIR      = _p('hasil_analisa_judol')
DATASET_GEN_DIR = _p('new_dataset_candidate')
JSON_LOG_FILE   = _p('scan_log.json')
REPORT_DIR      = _p('laporan_validasi')
CONCURRENCY     = int(os.environ.get('CONCURRENCY', 6))
MAX_TARGETS     = int(os.environ.get('MAX_TARGETS', 0))
CONFIDENCE_THRESHOLD = 0.25

TIMEOUT_MS     = int(os.environ.get('TIMEOUT_MS', 30000))
NETWORKIDLE_MS = int(os.environ.get('NETWORKIDLE_MS', 8000))
SLEEP_LOAD     = float(os.environ.get('SLEEP_LOAD', 1.5))
SLEEP_SETTLE   = float(os.environ.get('SLEEP_SETTLE', 1.5))
BLOCK_HEAVY    = os.environ.get('BLOCK_HEAVY', '1') == '1'

RESUME          = os.environ.get('RESUME', '1') == '1'
GEN_DATASET     = os.environ.get('GEN_DATASET', '1') == '1'
USE_STEALTH     = os.environ.get('USE_STEALTH', '1') == '1'
BROWSER_CHANNEL = os.environ.get('BROWSER_CHANNEL', 'chrome')
CF_WAIT_MAX     = int(os.environ.get('CF_WAIT_MAX', 20000))
PER_URL_TIMEOUT = int(os.environ.get('PER_URL_TIMEOUT', 90000))
USER_AGENT      = os.environ.get('USER_AGENT',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36')

BLOCK_SIGNATURES = ('internet-positif', 'internetpositif', 'trustpositif',
                    'trust positif', 'aduankonten', 'kominfo.go.id',
                    'laman ini diblokir', 'situs ini diblokir', 'internet positif')

CHALLENGE_SIGNATURES = ('just a moment', 'performing security verification',
                        'verifying you are human', 'verify you are human',
                        'checking your browser', 'enable javascript and cookies',
                        'cf-browser-verification', 'attention required',
                        'needs to review the security',
                        'melakukan verifikasi keamanan', 'melakukan verifikasi',
                        'memverifikasi bahwa anda bukan bot', 'anda bukan bot',
                        'layanan keamanan untuk melindungi')

DEAD_SIGNATURES = ('web server is down', 'origin dns error', 'origin is unreachable',
                   'error code 521', 'error code 522', 'error code 523', 'error code 525',
                   'error code 520', 'error code 526', 'error 1016', 'error 1000',
                   'error 1001', 'error 1002', 'error 1003', "this site can",
                   'server not found', 'this domain is for sale', 'domain for sale',
                   'buy this domain', 'domain is parked', 'welcome to nginx',
                   'apache2 ubuntu default', '404 not found', 'page not found',
                   'site not found', 'account suspended', 'bad gateway',
                   '503 service', 'gateway timeout', 'no longer available')

STOP_REQUESTED = False
START_TIME = time.time()

SUBDIRS = ['low', 'medium', 'high']
os.makedirs(OUTPUT_DIR, exist_ok=True)
for subdir in SUBDIRS:
    os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)

os.makedirs(os.path.join(DATASET_GEN_DIR, 'images_clean'), exist_ok=True)
os.makedirs(os.path.join(DATASET_GEN_DIR, 'images_verification'), exist_ok=True)
os.makedirs(os.path.join(DATASET_GEN_DIR, 'labels'), exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


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
        f_judol = count['judol'] / N * 100   # persentase kehadiran pada judol
        f_non = count['non'] / N * 100        # persentase kehadiran pada non-judol
        ratio = f_judol / (f_non + 1)         # Frequency Ratio + Laplace smoothing
        final_weights[label] = round(ratio * SCALING_FACTOR, 2)
    return final_weights


VISUAL_WEIGHTS_FREQ = calculate_frequency_weights()


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


WEIGHTS_CONFIG = {
    'visual': VISUAL_WEIGHTS_FREQ,
    'text': calculate_text_weights(),
    'safety': ['polisi', 'ditangkap', 'hukum', 'berita', 'edukasi',
               'pemerintah', 'learning', 'education', 'repository'],
    'safety_penalty': 80
}

logging.getLogger("ultralytics").setLevel(logging.ERROR)

try:
    import torch
    if os.environ.get('DEVICE'):
        DEVICE = os.environ['DEVICE']
    else:
        DEVICE = 0 if torch.cuda.is_available() else 'cpu'
except ImportError:
    DEVICE = 'cpu'

try:
    model = YOLO(MODEL_PATH)
    model.to(DEVICE)
    MODEL_NAMES = model.names
    NAME_TO_ID = {v: k for k, v in MODEL_NAMES.items()}
except Exception as e:
    sys.exit(1)


def input_listener():
    global STOP_REQUESTED
    while True:
        try:
            user_input = input()
            if user_input.strip().lower() == 'q':
                STOP_REQUESTED = True
                break
        except Exception:
            break


if sys.stdin is not None and sys.stdin.isatty():
    t = threading.Thread(target=input_listener)
    t.daemon = True
    t.start()

json_log_lock = threading.Lock()


def append_to_json_log(entry):
    with json_log_lock:
        log_data = []
        if os.path.exists(JSON_LOG_FILE):
            try:
                with open(JSON_LOG_FILE, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
            except Exception:
                log_data = []
        log_data.append(entry)
        with open(JSON_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)


csv_lock = threading.Lock()


def append_to_csv_realtime(result):
    if result['status'] != 'SUCCESS':
        return
    
    sd = result['score_data']
    category = sd['risk_level']
    fname = f"report_{category.lower()}.csv"
    headers = ['URL', 'Visual Tags', 'Keywords', 'Visual Score', 'Text Score', 'Final Score', 'Timestamp']
    row = [
        result['url'],
        "; ".join(sd['breakdown']['visual_items']),
        "; ".join([f"{k}:{v}" for k, v in sd['breakdown']['text_keywords'].items()]),
        sd['breakdown']['visual_norm'],
        sd['breakdown']['text_norm'],
        sd['final_score'],
        result.get('timestamp', '')
    ]
    
    with csv_lock:
        file_exists = os.path.exists(fname)
        with open(fname, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow(row)


def generate_dataset_entry(img_clean, img_annotated, detections, filename_base):
    if img_clean is None:
        return
    img_h, img_w = img_clean.shape[:2]
    cv2.imwrite(os.path.join(DATASET_GEN_DIR, 'images_clean', f"{filename_base}.jpg"), img_clean)
    cv2.imwrite(os.path.join(DATASET_GEN_DIR, 'images_verification', f"{filename_base}.jpg"), img_annotated)
    if detections:
        with open(os.path.join(DATASET_GEN_DIR, 'labels', f"{filename_base}.txt"), 'w') as f:
            for det in detections:
                cls_id = NAME_TO_ID.get(det['label'])
                if cls_id is None:
                    continue
                x1, y1, x2, y2 = det['box']
                dw, dh = 1. / img_w, 1. / img_h
                x_center = (x1 + x2) / 2.0
                y_center = (y1 + y2) / 2.0
                w = x2 - x1
                h = y2 - y1
                f.write(f"{cls_id} {x_center*dw:.6f} {y_center*dh:.6f} {w*dw:.6f} {h*dh:.6f}\n")


def process_image_yolo(filepath, filename_base):
    img = cv2.imread(filepath)
    if img is None:
        return [], None
    results = model.predict(img, conf=CONFIDENCE_THRESHOLD, verbose=False, device=DEVICE)
    detections = []
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = results[0].names[cls_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            if label in VISUAL_WEIGHTS_FREQ:
                detections.append({'label': label, 'conf': conf, 'box': xyxy})

    annotated_img = img.copy()
    for det in detections:
        label = det['label']
        conf = det['conf']
        x1, y1, x2, y2 = map(int, det['box'])
        is_high = VISUAL_WEIGHTS_FREQ.get(label, 0) > 50
        color = (0, 0, 255) if is_high else (0, 255, 255)
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_img, f"{label} {conf:.0%}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if GEN_DATASET:
        generate_dataset_entry(img, annotated_img, detections, filename_base)
    return detections, annotated_img


def calculate_risk_score(yolo_detections, text_content):
    raw_visual = 0
    visual_details = []
    for item in yolo_detections:
        label = item['label']
        conf = item['conf']
        weight = WEIGHTS_CONFIG['visual'].get(label, 0)
        raw_visual += weight * conf
        visual_details.append(f"{label}:{conf:.2f}")
    norm_visual = min(raw_visual, 100)

    raw_text = 0
    text_lower = text_content.lower()
    found_keywords = {}
    for word, score in WEIGHTS_CONFIG['text'].items():
        count = text_lower.count(word)
        if count > 0:
            raw_text += score
            found_keywords[word] = count
    for safe in WEIGHTS_CONFIG['safety']:
        if safe in text_lower:
            raw_text -= WEIGHTS_CONFIG['safety_penalty']
    norm_text = max(min(raw_text, 100), 0)

    final_score = round((norm_visual * 0.5) + (norm_text * 0.5), 2)

    THRESH_HIGH = 50
    THRESH_LOW = 25
    if final_score >= THRESH_HIGH:
        risk_level = 'HIGH' if norm_visual > 0 else 'MEDIUM'
    elif final_score > THRESH_LOW:
        risk_level = 'MEDIUM'
    else:
        risk_level = 'LOW'

    return {
        'final_score': final_score,
        'risk_level': risk_level,
        'breakdown': {
            'visual_norm': round(norm_visual, 2),
            'text_norm': norm_text,
            'visual_items': visual_details,
            'text_keywords': found_keywords
        }
    }


_STEALTH = None
if USE_STEALTH:
    try:
        from playwright_stealth import Stealth
        _STEALTH = Stealth()
    except Exception:
        try:
            from playwright_stealth import stealth_async as _stealth_fn
            _STEALTH = _stealth_fn
        except Exception:
            pass


async def apply_stealth(page):
    if not _STEALTH:
        return
    try:
        if hasattr(_STEALTH, 'apply_stealth_async'):
            await _STEALTH.apply_stealth_async(page)
        elif callable(_STEALTH):
            await _STEALTH(page)
    except Exception:
        pass


async def classify_access(page):
    try:
        final_url = page.url or ''
        title = (await page.title()) or ''
        body = await page.evaluate("document.body ? document.body.innerText.slice(0,4000) : ''")
    except Exception:
        return 'OK', '', ''
    blob = f"{final_url} {title} {body}".lower()
    if any(sig in blob for sig in BLOCK_SIGNATURES):
        return 'BLOCKED_KOMDIGI', final_url, title
    if any(sig in blob for sig in DEAD_SIGNATURES):
        return 'DEAD', final_url, title
    if any(sig in blob for sig in CHALLENGE_SIGNATURES):
        return 'BOT_CHALLENGE', final_url, title
    if ('cloudflare' in blob or 'ray id' in blob) and len((body or '').strip()) < 600:
        return 'BOT_CHALLENGE', final_url, title
    return 'OK', final_url, title


def _err_to_category(err_msg):
    e = (err_msg or '').lower()
    if 'ssl' in e or 'name_not_resolved' in e or 'unreachable' in e \
       or 'timeout' in e or 'blank' in e or 'empty' in e or 'connection' in e:
        return 'DEAD'
    return 'ERROR'


async def _process_page(page, url, scan_result):
    full_url = url if url.startswith('http') else f'https://{url}'

    if STOP_REQUESTED:
        raise Exception("Stop Requested")

    try:
        await page.goto(full_url, timeout=TIMEOUT_MS, wait_until='domcontentloaded')
    except PlaywrightTimeoutError:
        try:
            content_len = await page.evaluate("document.body.innerText.length")
            if content_len < 100:
                raise Exception("Timeout & Page Empty")
        except Exception:
            raise Exception("Timeout & Unreachable")

    if STOP_REQUESTED:
        raise Exception("Stop Requested")

    category, final_url, title = await classify_access(page)
    if category == 'BOT_CHALLENGE':
        deadline = time.time() + CF_WAIT_MAX / 1000.0
        while category == 'BOT_CHALLENGE' and time.time() < deadline and not STOP_REQUESTED:
            await asyncio.sleep(2)
            category, final_url, title = await classify_access(page)
            
    scan_result['access_category'] = category
    scan_result['final_url'] = final_url
    scan_result['page_title'] = title[:120]

    try:
        await page.wait_for_load_state('networkidle', timeout=NETWORKIDLE_MS)
    except PlaywrightTimeoutError:
        pass
    await asyncio.sleep(SLEEP_LOAD)

    try:
        vp = page.viewport_size
        width = vp['width'] if vp else 1366
        await page.mouse.click(50, 50)
        await asyncio.sleep(0.2)
        await page.mouse.click(width - 50, 50)
    except Exception:
        pass

    await asyncio.sleep(SLEEP_SETTLE)

    page_text = await page.evaluate("document.body.innerText + ' ' + Array.from(document.images).map(i=>i.src).join(' ')")
    if len(page_text.strip()) < 50:
        raise Exception("Page Blank")

    safe_name = full_url.replace('https://', '').replace('http://', '').replace('/', '_').replace(':', '')[:50]
    temp_filepath = os.path.join(OUTPUT_DIR, f"{safe_name}_temp.jpg")
    await page.screenshot(path=temp_filepath, full_page=False, timeout=20000)

    detections, annotated_img = await asyncio.to_thread(process_image_yolo, temp_filepath, safe_name)
    math_result = calculate_risk_score(detections, page_text)

    risk_category = math_result['risk_level'].lower()
    final_filepath = os.path.join(OUTPUT_DIR, risk_category, f"{safe_name}.jpg")
    cv2.imwrite(final_filepath, annotated_img)
    if os.path.exists(temp_filepath):
        os.remove(temp_filepath)

    scan_result.update({
        'status': 'SUCCESS',
        'score_data': math_result,
        'screenshot': final_filepath,
    })


async def scan_url(sem, context, url, results_list):
    if STOP_REQUESTED:
        return

    async with sem:
        if STOP_REQUESTED:
            return

        page = await context.new_page()
        scan_result = {
            'url': url, 'status': 'FAILED',
            'score_data': None, 'error': None,
            'timestamp': datetime.now().isoformat(),
            'duration_sec': 0
        }
        url_start = time.time()

        try:
            await apply_stealth(page)
            await asyncio.wait_for(_process_page(page, url, scan_result), timeout=PER_URL_TIMEOUT / 1000.0)
        except asyncio.TimeoutError:
            scan_result['error'] = 'Hard Timeout'
            scan_result.setdefault('access_category', 'DEAD')
        except Exception as e:
            if str(e) != "Stop Requested":
                err_msg = str(e).split('\n')[0][:60]
                scan_result['error'] = err_msg
                scan_result.setdefault('access_category', _err_to_category(err_msg))
        finally:
            scan_result['duration_sec'] = round(time.time() - url_start, 2)
            if scan_result['status'] == 'SUCCESS' or (scan_result['error'] and scan_result['error'] != "Stop Requested"):
                results_list.append(scan_result)
                append_to_json_log(scan_result)
                append_to_csv_realtime(scan_result)
            try:
                await page.close()
            except Exception:
                pass


def generate_validation_report(results):
    success = [r for r in results if r['status'] == 'SUCCESS']
    failed  = [r for r in results if r['status'] == 'FAILED']

    if not success:
        return

    kategori_count = Counter(r['score_data']['risk_level'] for r in success)
    total_success  = len(success)
    total_all      = len(results)

    all_scores        = [r['score_data']['final_score'] for r in success]
    all_visual_scores = [r['score_data']['breakdown']['visual_norm'] for r in success]
    all_text_scores   = [r['score_data']['breakdown']['text_norm'] for r in success]
    all_durations     = [r['duration_sec'] for r in results if r.get('duration_sec')]

    all_visual_labels = []
    all_keywords      = []
    for r in success:
        for item in r['score_data']['breakdown']['visual_items']:
            all_visual_labels.append(item.split(':')[0])
        for kw in r['score_data']['breakdown']['text_keywords'].keys():
            all_keywords.append(kw)

    label_counter   = Counter(all_visual_labels)
    keyword_counter = Counter(all_keywords)

    total_duration = time.time() - START_TIME
    summary = {
        'tanggal_run': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_url_input': total_all,
        'total_berhasil': total_success,
        'total_gagal': len(failed),
        'tingkat_keberhasilan_persen': round(total_success / total_all * 100, 2),
        'distribusi_klasifikasi': dict(kategori_count),
        'distribusi_persen': {k: round(v / total_success * 100, 2) for k, v in kategori_count.items()},
        'skor_final': {
            'rata_rata': round(sum(all_scores) / len(all_scores), 2),
            'minimum': round(min(all_scores), 2),
            'maksimum': round(max(all_scores), 2)
        },
        'skor_visual': {
            'rata_rata': round(sum(all_visual_scores) / len(all_visual_scores), 2),
            'minimum': round(min(all_visual_scores), 2),
            'maksimum': round(max(all_visual_scores), 2)
        },
        'skor_tekstual': {
            'rata_rata': round(sum(all_text_scores) / len(all_text_scores), 2),
            'minimum': round(min(all_text_scores), 2),
            'maksimum': round(max(all_text_scores), 2)
        },
        'durasi_total_detik': round(total_duration, 2),
        'durasi_total_menit': round(total_duration / 60, 2),
        'rata_rata_per_url_detik': round(sum(all_durations) / len(all_durations), 2) if all_durations else 0,
        'top_10_label_visual': label_counter.most_common(10),
        'top_10_keyword_tekstual': keyword_counter.most_common(10),
        'top_5_error': Counter(r.get('error', 'Unknown') for r in failed).most_common(5)
    }

    summary_path = os.path.join(REPORT_DIR, 'summary_statistik.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if not MATPLOTLIB_AVAILABLE:
        return summary

    fig, ax = plt.subplots(figsize=(7, 7))
    labels  = list(kategori_count.keys())
    sizes   = list(kategori_count.values())
    colors  = {'HIGH': '#d32f2f', 'MEDIUM': '#f57c00', 'LOW': '#388e3c'}
    clrs    = [colors.get(l, '#9e9e9e') for l in labels]
    explode = [0.05] * len(labels)

    ax.pie(sizes, labels=labels, colors=clrs, explode=explode, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 13, 'fontweight': 'bold'})
    ax.set_title(f'Distribusi Klasifikasi Risiko Sistem GATE\n(n = {total_success} situs berhasil diproses)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'grafik_distribusi_klasifikasi.png'), dpi=150, bbox_inches='tight')
    plt.close()

    top_labels = label_counter.most_common(9)
    if top_labels:
        fig, ax = plt.subplots(figsize=(10, 5))
        lbl_names = [x[0] for x in top_labels]
        lbl_cnts  = [x[1] for x in top_labels]
        bars = ax.bar(lbl_names, lbl_cnts, color='#1565c0', edgecolor='white', linewidth=0.7)
        for bar, cnt in zip(bars, lbl_cnts):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, str(cnt), ha='center', va='bottom', fontsize=10)
        ax.set_title('Frekuensi Deteksi Label Visual per Kelas', fontsize=13, fontweight='bold')
        ax.set_xlabel('Kelas Objek Visual')
        ax.set_ylabel('Jumlah Kemunculan')
        ax.set_xticklabels(lbl_names, rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, 'grafik_label_visual.png'), dpi=150, bbox_inches='tight')
        plt.close()

    top_kws = keyword_counter.most_common(10)
    if top_kws:
        fig, ax = plt.subplots(figsize=(10, 5))
        kw_names = [x[0] for x in top_kws]
        kw_cnts  = [x[1] for x in top_kws]
        bars = ax.bar(kw_names, kw_cnts, color='#6a1b9a', edgecolor='white', linewidth=0.7)
        for bar, cnt in zip(bars, kw_cnts):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, str(cnt), ha='center', va='bottom', fontsize=10)
        ax.set_title('Frekuensi Kemunculan Kata Kunci Tekstual Teratas', fontsize=13, fontweight='bold')
        ax.set_xlabel('Kata Kunci')
        ax.set_ylabel('Jumlah Kemunculan')
        ax.set_xticklabels(kw_names, rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, 'grafik_keyword_tekstual.png'), dpi=150, bbox_inches='tight')
        plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_scores, bins=20, color='#0277bd', edgecolor='white', linewidth=0.7)
    ax.axvline(x=25, color='orange', linestyle='--', linewidth=2, label='Batas LOW/MEDIUM (25)')
    ax.axvline(x=50, color='red',    linestyle='--', linewidth=2, label='Batas MEDIUM/HIGH (50)')
    ax.set_title('Distribusi Skor Final Seluruh URL', fontsize=13, fontweight='bold')
    ax.set_xlabel('Skor Final')
    ax.set_ylabel('Jumlah URL')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'grafik_distribusi_skor.png'), dpi=150, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 8))
    colors_map = {'HIGH': '#d32f2f', 'MEDIUM': '#f57c00', 'LOW': '#388e3c'}
    for kat in ['LOW', 'MEDIUM', 'HIGH']:
        subset = [r for r in success if r['score_data']['risk_level'] == kat]
        if subset:
            xs = [r['score_data']['breakdown']['visual_norm'] for r in subset]
            ys = [r['score_data']['breakdown']['text_norm']   for r in subset]
            ax.scatter(xs, ys, c=colors_map[kat], label=kat, alpha=0.6, edgecolors='white', linewidth=0.5, s=40)
    ax.axvline(x=0,  color='gray',   linestyle=':', linewidth=1)
    ax.axhline(y=25, color='orange', linestyle='--', linewidth=1.5, label='Batas Skor Tekstual 25')
    ax.axvline(x=25, color='blue',   linestyle='--', linewidth=1.5, label='Batas Skor Visual 25')
    ax.set_title('Sebaran Skor Visual vs Skor Tekstual per Kategori', fontsize=13, fontweight='bold')
    ax.set_xlabel('Skor Visual (Ternormalisasi)')
    ax.set_ylabel('Skor Tekstual (Ternormalisasi)')
    ax.legend()
    ax.set_xlim(-2, 105)
    ax.set_ylim(-2, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'grafik_scatter_visual_vs_tekstual.png'), dpi=150, bbox_inches='tight')
    plt.close()

    generate_sample_collage('high', max_samples=20, title='Sampel 20 URL Kategori HIGH')
    generate_sample_collage('medium', max_samples=12, title='Sampel URL Kategori MEDIUM')
    generate_sample_collage('low', max_samples=12, title='Sampel URL Kategori LOW')
    generate_csv_reports(results)
    return summary


def generate_sample_collage(category, max_samples=20, title=''):
    folder = os.path.join(OUTPUT_DIR, category)
    if not os.path.exists(folder):
        return

    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not files:
        return

    sampled = random.sample(files, min(max_samples, len(files)))
    n_cols  = 4
    n_rows  = (len(sampled) + n_cols - 1) // n_cols
    thumb_w, thumb_h = 300, 200

    canvas = np.zeros((n_rows * thumb_h + 60, n_cols * thumb_w, 3), dtype=np.uint8)
    canvas[:60, :] = (30, 30, 30)

    cv2.putText(canvas, title, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    for idx, fname in enumerate(sampled):
        img_path = os.path.join(folder, fname)
        img      = cv2.imread(img_path)
        if img is None:
            continue
        thumb = cv2.resize(img, (thumb_w, thumb_h))
        row   = idx // n_cols
        col   = idx % n_cols
        y1    = row * thumb_h + 60
        y2    = y1 + thumb_h
        x1    = col * thumb_w
        x2    = x1 + thumb_w
        canvas[y1:y2, x1:x2] = thumb

        url_label = os.path.splitext(fname)[0][:30]
        cv2.putText(canvas, url_label, (x1 + 4, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 255, 200), 1)

    out_path = os.path.join(REPORT_DIR, f'collage_sampel_{category}.jpg')
    cv2.imwrite(out_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])


def generate_csv_reports(results):
    buckets = {'HIGH': [], 'MEDIUM': [], 'LOW': [], 'FAILED': []}

    for res in results:
        if res['status'] == 'SUCCESS':
            sd = res['score_data']
            row = [
                res['url'],
                "; ".join(sd['breakdown']['visual_items']),
                "; ".join([f"{k}:{v}" for k, v in sd['breakdown']['text_keywords'].items()]),
                sd['breakdown']['visual_norm'],
                sd['breakdown']['text_norm'],
                sd['final_score'],
                res.get('timestamp', ''),
                res.get('duration_sec', '')
            ]
            buckets[sd['risk_level']].append(row)
        else:
            buckets['FAILED'].append([
                res['url'], "ERROR", res.get('error', 'Unknown'), 0, 0, 0,
                res.get('timestamp', ''), res.get('duration_sec', '')
            ])

    headers = ['URL', 'Visual Tags', 'Keywords', 'Visual Score', 'Text Score', 'Final Score', 'Timestamp', 'Duration (s)']

    for cat, rows in buckets.items():
        if rows:
            fname = os.path.join(REPORT_DIR, f"report_final_{cat.lower()}.csv")
            with open(fname, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers if cat != 'FAILED' else ['URL', 'Status', 'Error', 'V', 'T', 'F', 'Timestamp', 'Duration (s)'])
                writer.writerows(rows)


async def main():
    if not os.path.exists(INPUT_FILE):
        return

    targets = []
    async with aiofiles.open(INPUT_FILE, mode='r') as f:
        async for line in f:
            if line.strip():
                targets.append(line.strip().split()[0])

    if RESUME and os.path.exists(JSON_LOG_FILE):
        try:
            with open(JSON_LOG_FILE, 'r', encoding='utf-8') as f:
                done = json.load(f)

            def _norm(u):
                u = u.strip().lower()
                for pre in ('https://', 'http://'):
                    if u.startswith(pre):
                        u = u[len(pre):]
                return u[4:].rstrip('/') if u.startswith('www.') else u.rstrip('/')

            done_set = {_norm(d['url']) for d in done}
            targets = [t for t in targets if _norm(t) not in done_set]
        except Exception:
            pass

    if MAX_TARGETS > 0:
        targets = targets[:MAX_TARGETS]

    if not targets:
        return

    results_data = []
    launch_args = ['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu', '--disable-blink-features=AutomationControlled']
    
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(headless=True, channel=BROWSER_CHANNEL, args=launch_args)
        except Exception:
            browser = await p.chromium.launch(headless=True, args=launch_args)

        context = await browser.new_context(
            viewport={'width': 1366, 'height': 2500},
            ignore_https_errors=True,
            user_agent=USER_AGENT,
            locale='id-ID',
            timezone_id='Asia/Jakarta',
        )

        if BLOCK_HEAVY:
            async def _route(route):
                if route.request.resource_type in ('media', 'font'):
                    await route.abort()
                else:
                    await route.continue_()
            await context.route('**/*', _route)

        sem   = asyncio.Semaphore(CONCURRENCY)
        tasks = [scan_url(sem, context, url, results_data) for url in targets]
        await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await browser.close()
        except Exception:
            pass

    generate_validation_report(results_data)


def run_screening():
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass
    return asyncio.run(main())


if __name__ == '__main__':
    try:
        run_screening()
    except KeyboardInterrupt:
        pass