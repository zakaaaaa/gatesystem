"""
Re-scrape teks halaman untuk kebutuhan cascade IndoBERT.

Dua kelompok target (dibangun otomatis dari scan_log.json + dataset eval):
  1. eval_medium : 254 URL kategori MEDIUM pada dataset evaluasi
                   -> HANYA untuk evaluasi cascade, DILARANG dipakai training (data leakage)
  2. train_judol : URL access OK di scan_log yang BUKAN bagian dataset evaluasi
                   -> kandidat kelas "promosi judol" untuk training IndoBERT

Output: teks_dataset/eval_medium.jsonl dan teks_dataset/train_judol.jsonl
Satu baris JSON per URL: url, group, is_judol, june_risk, access_category,
final_url, page_title, text (innerText, maks 20000 char), img_srcs, timestamp.

Mendukung RESUME: URL yang sudah ada di file output dilewati.
"""
import os
import sys
import json
import time
import asyncio
from datetime import datetime
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
SCAN_LOG = os.path.join(BASE_DIR, 'scan_log.json')
OUT_DIR = os.path.join(BASE_DIR, 'teks_dataset')
os.makedirs(OUT_DIR, exist_ok=True)

CONCURRENCY = int(os.environ.get('CONCURRENCY', 6))
MAX_TARGETS = int(os.environ.get('MAX_TARGETS', 0))
TIMEOUT_MS = int(os.environ.get('TIMEOUT_MS', 30000))
PER_URL_TIMEOUT = int(os.environ.get('PER_URL_TIMEOUT', 90000))
CF_WAIT_MAX = int(os.environ.get('CF_WAIT_MAX', 20000))
MAX_TEXT_CHARS = 20000
USER_AGENT = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
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

_STEALTH = None
try:
    from playwright_stealth import Stealth
    _STEALTH = Stealth()
except Exception:
    try:
        from playwright_stealth import stealth_async as _STEALTH
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


def load_list(path):
    with open(os.path.join(BASE_DIR, path)) as f:
        return set(line.strip().split()[0] for line in f if line.strip())


def build_targets():
    with open(SCAN_LOG, 'r', encoding='utf-8') as f:
        log = json.load(f)
    by_url = {e['url']: e for e in log}

    judol = load_list('dataset_judol_aktif.txt')
    non = load_list('dataset_nonjudol_300.txt')
    eval_set = judol | non

    targets = []
    for u in sorted(eval_set):
        e = by_url.get(u)
        if e and e.get('score_data') and e['score_data']['risk_level'] == 'MEDIUM':
            targets.append({'url': u, 'group': 'eval_medium',
                            'is_judol': u in judol,
                            'june_risk': 'MEDIUM'})
    for e in log:
        u = e['url']
        if u in eval_set or not e.get('score_data') or e.get('access_category') != 'OK':
            continue
        targets.append({'url': u, 'group': 'train_judol',
                        'is_judol': True,
                        'june_risk': e['score_data']['risk_level']})
    return targets


def out_path(group):
    return os.path.join(OUT_DIR, f'{group}.jsonl')


def load_done():
    done = set()
    for group in ('eval_medium', 'train_judol'):
        p = out_path(group)
        if not os.path.exists(p):
            continue
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    done.add(json.loads(line)['url'])
                except Exception:
                    pass
    return done


write_lock = asyncio.Lock()


async def save_entry(entry):
    async with write_lock:
        with open(out_path(entry['group']), 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


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


async def _grab(page, target, entry):
    url = target['url']
    full_url = url if url.startswith('http') else f'https://{url}'
    await page.goto(full_url, timeout=TIMEOUT_MS, wait_until='domcontentloaded')

    category, final_url, title = await classify_access(page)
    if category == 'BOT_CHALLENGE':
        deadline = time.time() + CF_WAIT_MAX / 1000.0
        while category == 'BOT_CHALLENGE' and time.time() < deadline:
            await asyncio.sleep(2)
            category, final_url, title = await classify_access(page)

    entry['access_category'] = category
    entry['final_url'] = final_url
    entry['page_title'] = (title or '')[:200]

    if category != 'OK':
        return

    try:
        await page.wait_for_load_state('networkidle', timeout=8000)
    except PlaywrightTimeoutError:
        pass
    await asyncio.sleep(2)

    text = await page.evaluate("document.body ? document.body.innerText : ''")
    img_srcs = await page.evaluate("Array.from(document.images).map(i=>i.src).slice(0,100)")

    if len(text.strip()) < 50:
        entry['access_category'] = 'DEAD'
        entry['error'] = 'Page Blank'
        return

    entry['text'] = text[:MAX_TEXT_CHARS]
    entry['text_len'] = len(text)
    entry['img_srcs'] = img_srcs


async def scrape_one(sem, context, target, stats):
    async with sem:
        page = await context.new_page()
        entry = dict(target)
        entry.update({'access_category': None, 'final_url': '', 'page_title': '',
                      'text': None, 'text_len': 0, 'img_srcs': [],
                      'error': None, 'timestamp': datetime.now().isoformat()})
        try:
            await apply_stealth(page)
            await asyncio.wait_for(_grab(page, target, entry), timeout=PER_URL_TIMEOUT / 1000.0)
        except asyncio.TimeoutError:
            entry['error'] = 'Hard Timeout'
            entry['access_category'] = entry['access_category'] or 'DEAD'
        except Exception as e:
            entry['error'] = str(e).split('\n')[0][:80]
            entry['access_category'] = entry['access_category'] or 'ERROR'
        finally:
            try:
                await page.close()
            except Exception:
                pass

        await save_entry(entry)
        stats['done'] += 1
        cat = entry['access_category']
        ok = 'TEKS' if entry['text'] else cat
        print(f"[{stats['done']}/{stats['total']}] {target['group']:11s} {target['url'][:45]:45s} => {ok}", flush=True)


async def main():
    targets = build_targets()
    done = load_done()
    targets = [t for t in targets if t['url'] not in done]
    if MAX_TARGETS > 0:
        targets = targets[:MAX_TARGETS]

    n_med = sum(1 for t in targets if t['group'] == 'eval_medium')
    print(f"[INFO] Target tersisa: {len(targets)} (eval_medium {n_med}, train_judol {len(targets)-n_med}); sudah selesai: {len(done)}")
    if not targets:
        return

    stats = {'done': 0, 'total': len(targets)}
    launch_args = ['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu',
                   '--disable-blink-features=AutomationControlled']
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(headless=True, channel='chrome', args=launch_args)
        except Exception:
            browser = await p.chromium.launch(headless=True, args=launch_args)
        context = await browser.new_context(
            viewport={'width': 1366, 'height': 2500},
            ignore_https_errors=True,
            user_agent=USER_AGENT,
            locale='id-ID',
            timezone_id='Asia/Jakarta',
        )

        async def _route(route):
            if route.request.resource_type in ('media', 'font', 'image'):
                await route.abort()
            else:
                await route.continue_()
        await context.route('**/*', _route)

        sem = asyncio.Semaphore(CONCURRENCY)
        # eval_medium diproses lebih dulu: paling penting & paling rawan domain mati
        targets.sort(key=lambda t: 0 if t['group'] == 'eval_medium' else 1)
        await asyncio.gather(*(scrape_one(sem, context, t, stats) for t in targets),
                             return_exceptions=True)
        try:
            await browser.close()
        except Exception:
            pass

    print("[INFO] Selesai. Output di", OUT_DIR)


if __name__ == '__main__':
    asyncio.run(main())
