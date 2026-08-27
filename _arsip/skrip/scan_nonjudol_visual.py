"""
Scan YOLO untuk memperluas sampel kalibrasi NON-JUDOL pada FR visual.

Situs non-judol (508 game + 608 berita) sebelumnya di-scrape teks saja; skrip ini
mengambil screenshot tiap situs dan menjalankan YOLO untuk mencatat label visual
yang terdeteksi. Hasilnya dipakai untuk menghitung frekuensi non-judol yang lebih
besar & bersih (terpisah dari data uji) pada pembobotan Frequency Ratio visual.

Jalankan di Colab (CPU cukup) ATAU lokal. Output: nonjudol_visual.jsonl
Satu baris JSON per URL: {url, access_category, visual_items:[ "casino:0.42", ... ]}
Resume-aware.

=== CARA PAKAI DI COLAB ===
  Sel 1:  !pip -q install playwright ultralytics
          !playwright install chromium
          !playwright install-deps chromium
  Sel 2:  from google.colab import drive; drive.mount('/content/drive')
          Pastikan di MyDrive/GATESystem/ ada: best-4.pt, dataset_negatif_game.txt,
          dataset_nonjudol_training.txt, dan skrip ini.
  Sel 3:  !python /content/drive/MyDrive/GATESystem/scan_nonjudol_visual.py

=== CARA PAKAI LOKAL ===
  python3 scan_nonjudol_visual.py        (butuh laptop menyala; pakai caffeinate -is)
"""
import os
import sys
import json
import asyncio
from datetime import datetime
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

IS_COLAB = 'google.colab' in sys.modules
if IS_COLAB:
    if not os.path.exists('/content/drive/MyDrive'):
        from google.colab import drive
        drive.mount('/content/drive')
    BASE = '/content/drive/MyDrive/GATESystem'
else:
    BASE = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'

MODEL_PATH = os.path.join(BASE, 'best-4.pt')
URL_FILES = [os.path.join(BASE, 'dataset_negatif_game.txt'),
             os.path.join(BASE, 'dataset_nonjudol_training.txt')]
OUT = os.path.join(BASE, 'nonjudol_visual.jsonl')
CONCURRENCY = int(os.environ.get('CONCURRENCY', 4))
TIMEOUT_MS = 30000
PER_URL_TIMEOUT = 70
CONF = 0.25
USER_AGENT = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
              '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36')

import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)
from ultralytics import YOLO
model = YOLO(MODEL_PATH)
NAMES = model.names

WEIGHTED = {'zeus', 'mahjong', 'pragmatic', 'casino', 'koin', 'vulgar', 'pool', 'pg', 'mahjong_card'}


def load_urls():
    urls = []
    for p in URL_FILES:
        if os.path.exists(p):
            with open(p) as f:
                urls += [l.strip() for l in f if l.strip()]
    # dedup preserving order
    seen = set(); out = []
    for u in urls:
        if u not in seen:
            seen.add(u); out.append(u)
    return out


def load_done():
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                try:
                    done.add(json.loads(line)['url'])
                except Exception:
                    pass
    return done


def detect(img_path):
    res = model.predict(img_path, conf=CONF, verbose=False)
    items = []
    if len(res[0].boxes) > 0:
        for box in res[0].boxes:
            label = res[0].names[int(box.cls[0])]
            if label in WEIGHTED:
                items.append(f"{label}:{float(box.conf[0]):.2f}")
    return items


write_lock = asyncio.Lock()


async def scan_one(sem, context, url, stats):
    async with sem:
        full = url if url.startswith('http') else f'https://{url}'
        page = await context.new_page()
        entry = {'url': url, 'access_category': None, 'visual_items': [],
                 'error': None, 'timestamp': datetime.now().isoformat()}
        tmp = os.path.join(BASE, f"_tmp_{abs(hash(url)) % 10**8}.jpg")
        try:
            async def _go():
                try:
                    await page.goto(full, timeout=TIMEOUT_MS, wait_until='domcontentloaded')
                except PlaywrightTimeoutError:
                    pass
                await asyncio.sleep(2)
                body = await page.evaluate("document.body ? document.body.innerText.length : 0")
                if not body or body < 50:
                    entry['access_category'] = 'DEAD'; return
                entry['access_category'] = 'OK'
                await page.screenshot(path=tmp, full_page=False, timeout=20000)
                entry['visual_items'] = await asyncio.to_thread(detect, tmp)
            await asyncio.wait_for(_go(), timeout=PER_URL_TIMEOUT)
        except asyncio.TimeoutError:
            entry['error'] = 'Timeout'; entry['access_category'] = entry['access_category'] or 'DEAD'
        except Exception as e:
            entry['error'] = str(e).split('\n')[0][:80]; entry['access_category'] = entry['access_category'] or 'ERROR'
        finally:
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except Exception: pass
            try: await page.close()
            except Exception: pass
        async with write_lock:
            with open(OUT, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        stats['done'] += 1
        tag = ('VIS:' + ','.join(i.split(':')[0] for i in entry['visual_items'])) if entry['visual_items'] else entry['access_category']
        print(f"[{stats['done']}/{stats['total']}] {url[:55]:55s} => {tag}", flush=True)


async def main():
    urls = load_urls()
    done = load_done()
    todo = [u for u in urls if u not in done]
    print(f"[INFO] Total {len(urls)} non-judol | sisa {len(todo)} | selesai {len(done)}")
    if not todo:
        return
    stats = {'done': 0, 'total': len(todo)}
    args = ['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu']
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(headless=True, channel='chrome', args=args)
        except Exception:
            browser = await p.chromium.launch(headless=True, args=args)
        context = await browser.new_context(viewport={'width': 1366, 'height': 2500},
                                            ignore_https_errors=True, user_agent=USER_AGENT,
                                            locale='id-ID', timezone_id='Asia/Jakarta')
        async def _route(route):
            if route.request.resource_type in ('media', 'font'):
                await route.abort()
            else:
                await route.continue_()
        await context.route('**/*', _route)
        sem = asyncio.Semaphore(CONCURRENCY)
        await asyncio.gather(*(scan_one(sem, context, u, stats) for u in todo), return_exceptions=True)
        try: await browser.close()
        except Exception: pass
    print("[INFO] Selesai. Output:", OUT)


if __name__ == '__main__':
    asyncio.run(main())
