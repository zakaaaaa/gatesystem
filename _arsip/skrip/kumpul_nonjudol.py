"""
Kumpulkan hard negative untuk training IndoBERT cascade: artikel berita/edukasi
berbahasa Indonesia yang MEMBAHAS judi online tapi bukan promosi.

Fase 1: kunjungi halaman pencarian portal berita untuk kueri seputar judol,
        ekstrak tautan artikel -> dataset_nonjudol_training.txt
Fase 2: scrape teks tiap artikel -> teks_dataset/train_nonjudol.jsonl
        (skema sama dengan ambil_teks.py; group='train_nonjudol', is_judol=False)

Anti-leakage: domain yang muncul di dataset evaluasi (judol maupun non-judol 300)
dikecualikan. Resume-aware di kedua fase.
"""
import os
import json
import asyncio
import re
from datetime import datetime
from urllib.parse import urlparse
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
OUT_DIR = os.path.join(BASE_DIR, 'teks_dataset')
URL_LIST_FILE = os.path.join(BASE_DIR, 'dataset_nonjudol_training.txt')
OUT_JSONL = os.path.join(OUT_DIR, 'train_nonjudol.jsonl')
os.makedirs(OUT_DIR, exist_ok=True)

CONCURRENCY = int(os.environ.get('CONCURRENCY', 3))
TIMEOUT_MS = 30000
PER_URL_TIMEOUT = 60
MAX_TEXT_CHARS = 20000
MAX_PER_SEARCH_PAGE = 15
USER_AGENT = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
              '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36')

QUERIES = [
    'judi online',
    'situs judi online diblokir',
    'polisi tangkap judi online',
    'bahaya judi slot online',
    'kecanduan judi online',
    'komdigi blokir judol',
    'slot gacor penipuan',
    'bandar judi online ditangkap',
    'dampak judi online masyarakat',
    'pemerintah berantas judi online',
]

# (nama, template URL pencarian, pola URL artikel)
SEARCH_SOURCES = [
    ('detik', 'https://www.detik.com/search/searchall?query={q}',
     r'https?://[a-z]+\.detik\.com/[a-z-]+/d-\d+/'),
    ('cnnindonesia', 'https://www.cnnindonesia.com/search/?query={q}',
     r'https?://www\.cnnindonesia\.com/[a-z-]+/\d{8,}'),
    ('kompas', 'https://search.kompas.com/search/?q={q}',
     r'https?://[a-z]+\.kompas\.com/read/\d{4}/'),
    ('antara', 'https://www.antaranews.com/search?q={q}',
     r'https?://www\.antaranews\.com/berita/\d+/'),
    ('liputan6', 'https://www.liputan6.com/search?q={q}',
     r'https?://www\.liputan6\.com/[a-z-]+/read/\d+/'),
]

# halaman tag: padat artikel relevan, tanpa perlu kueri
TAG_PAGES = [
    ('detik-tag', 'https://www.detik.com/tag/{t}',
     r'https?://[a-z]+\.detik\.com/[a-z-]+/d-\d+/'),
    ('cnn-tag', 'https://www.cnnindonesia.com/tag/{t}',
     r'https?://www\.cnnindonesia\.com/[a-z-]+/\d{8,}'),
    ('antara-tag', 'https://www.antaranews.com/tag/{t}',
     r'https?://www\.antaranews\.com/berita/\d+/'),
    ('liputan6-tag', 'https://www.liputan6.com/tag/{t}',
     r'https?://www\.liputan6\.com/[a-z-]+/read/\d+/'),
    ('tempo-tag', 'https://www.tempo.co/tag/{t}',
     r'https?://www\.tempo\.co/[a-z-]+/[a-z0-9-]+-\d{6,}'),
]
TAGS = ['judi-online', 'judol', 'judi-slot', 'slot-online', 'pemberantasan-judi-online']

# seed statis: situs legit yang kosakatanya beririsan dgn judol (game, social casino,
# fintech dgn istilah deposit/withdraw, edukasi/pemerintah) — jenis hard negative
# yang sama dengan profil FP evaluasi; domain di dataset eval otomatis tersaring
STATIC_SEEDS = [
    'https://poki.com/id', 'https://www.crazygames.co.id/',
    'https://www.agame.com/games/mahjong', 'https://www.arkadium.com/games/mahjongg-solitaire/',
    'https://cardgames.io/', 'https://www.chess.com/id',
    'https://mahjong-game.com/', 'https://www.solitr.com/',
    'https://www.zynga.com/games/zynga-poker/', 'https://www.playtika.com/',
    'https://www.huuugegames.com/',
    'https://www.dana.id/', 'https://www.gopay.co.id/',
    'https://ovo.id/', 'https://www.bca.co.id/id/individu/produk/simpanan/deposito-berjangka',
    'https://www.ojk.go.id/id/berita-dan-kegiatan/publikasi/Pages/Waspada-Investasi-dan-Judi-Online.aspx',
    'https://www.komdigi.go.id/berita', 'https://aduankonten.id/',
    'https://pusiknas.polri.go.id/', 'https://www.djp.go.id/',
]


def _domain(u):
    try:
        d = urlparse(u if u.startswith('http') else f'https://{u}').netloc.lower()
        return d[4:] if d.startswith('www.') else d
    except Exception:
        return ''


def load_excluded_domains():
    """Isolasi test set cascade: domain 254 URL eval MEDIUM (yang diputus ulang
    IndoBERT) dilarang muncul di data training. Domain judol dikecualikan juga
    sebagai pengaman salah label. Baris eval non-MEDIUM tidak pernah menyentuh
    classifier sehingga bukan jalur kebocoran (portal berita kontrol = LOW)."""
    def _load(fname):
        p = os.path.join(BASE_DIR, fname)
        with open(p) as f:
            return set(line.strip().split()[0] for line in f if line.strip())

    judol = _load('dataset_judol_aktif.txt')
    non = _load('dataset_nonjudol_300.txt')
    excluded = {_domain(u) for u in judol}

    with open(os.path.join(BASE_DIR, 'scan_log.json'), encoding='utf-8') as f:
        by_url = {e['url']: e for e in json.load(f)}
    for u in judol | non:
        e = by_url.get(u)
        if e and e.get('score_data') and e['score_data']['risk_level'] == 'MEDIUM':
            excluded.add(_domain(u))
    return excluded


async def collect_links(context):
    excluded = load_excluded_domains()
    found = []
    seen = set()
    if os.path.exists(URL_LIST_FILE):
        with open(URL_LIST_FILE) as f:
            found = [l.strip() for l in f if l.strip()]
        seen = set(found)
        print(f"[INFO] Melanjutkan daftar yang ada: {len(found)} URL", flush=True)

    listing_pages = [(name, tmpl.format(q=q.replace(' ', '+')), pat)
                     for name, tmpl, pat in SEARCH_SOURCES for q in QUERIES]
    listing_pages += [(name, tmpl.format(t=t), pat)
                      for name, tmpl, pat in TAG_PAGES for t in TAGS]

    page = await context.new_page()
    for name, url, pat in listing_pages:
        rx = re.compile(pat)
        try:
            await page.goto(url, timeout=TIMEOUT_MS, wait_until='domcontentloaded')
            # tunggu sampai minimal satu link artikel muncul (hasil pencarian dirender JS)
            try:
                await page.wait_for_function(
                    "pat => Array.from(document.querySelectorAll('a[href]')).some(a => new RegExp(pat).test(a.href))",
                    arg=pat, timeout=15000)
            except PlaywrightTimeoutError:
                pass
            await asyncio.sleep(1.5)
            hrefs = await page.evaluate("Array.from(document.querySelectorAll('a[href]')).map(a=>a.href)")
        except Exception as e:
            print(f"[GAGAL] {name} {url[:60]}: {str(e)[:60]}", flush=True)
            continue
        n_new = 0
        for h in hrefs:
            if n_new >= MAX_PER_SEARCH_PAGE:
                break
            if not rx.match(h):
                continue
            h = h.split('?')[0].split('#')[0]
            if h in seen or _domain(h) in excluded:
                continue
            seen.add(h)
            found.append(h)
            n_new += 1
        print(f"[LINK] {name:13s} {url[:55]:55s}: +{n_new} (total {len(found)})", flush=True)
    await page.close()

    for h in STATIC_SEEDS:
        if h not in seen and _domain(h) not in excluded:
            seen.add(h)
            found.append(h)

    with open(URL_LIST_FILE, 'w') as f:
        f.write('\n'.join(found) + '\n')
    print(f"[INFO] {len(found)} URL tersimpan di {URL_LIST_FILE}", flush=True)
    return found


def load_done():
    done = set()
    if os.path.exists(OUT_JSONL):
        with open(OUT_JSONL) as f:
            for line in f:
                try:
                    done.add(json.loads(line)['url'])
                except Exception:
                    pass
    return done


write_lock = asyncio.Lock()


async def scrape_article(sem, context, url, stats):
    async with sem:
        page = await context.new_page()
        entry = {'url': url, 'group': 'train_nonjudol', 'is_judol': False,
                 'june_risk': None, 'access_category': None, 'final_url': '',
                 'page_title': '', 'text': None, 'text_len': 0, 'img_srcs': [],
                 'error': None, 'timestamp': datetime.now().isoformat()}
        try:
            async def _go():
                try:
                    await page.goto(url, timeout=TIMEOUT_MS, wait_until='domcontentloaded')
                except PlaywrightTimeoutError:
                    pass  # halaman berat (iklan) sering timeout padahal konten sudah ada
                await asyncio.sleep(2)
                entry['final_url'] = page.url
                entry['page_title'] = ((await page.title()) or '')[:200]
                text = await page.evaluate("document.body ? document.body.innerText : ''")
                if len(text.strip()) < 200:
                    entry['access_category'] = 'DEAD'
                    entry['error'] = 'Page Blank/Short'
                    return
                entry['access_category'] = 'OK'
                entry['text'] = text[:MAX_TEXT_CHARS]
                entry['text_len'] = len(text)
            await asyncio.wait_for(_go(), timeout=PER_URL_TIMEOUT)
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

        async with write_lock:
            with open(OUT_JSONL, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        stats['done'] += 1
        ok = 'TEKS' if entry['text'] else entry['access_category']
        print(f"[{stats['done']}/{stats['total']}] {url[:70]:70s} => {ok}", flush=True)


async def main():
    launch_args = ['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu']
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(headless=True, channel='chrome', args=launch_args)
        except Exception:
            browser = await p.chromium.launch(headless=True, args=launch_args)
        context = await browser.new_context(
            viewport={'width': 1366, 'height': 2500},
            user_agent=USER_AGENT, locale='id-ID', timezone_id='Asia/Jakarta',
        )

        async def _route(route):
            if route.request.resource_type in ('media', 'font', 'image'):
                await route.abort()
            else:
                await route.continue_()
        await context.route('**/*', _route)

        urls = await collect_links(context)
        done = load_done()
        todo = [u for u in urls if u not in done]
        print(f"[INFO] Fase 2: scrape {len(todo)} artikel (sudah selesai: {len(done)})")

        stats = {'done': 0, 'total': len(todo)}
        sem = asyncio.Semaphore(CONCURRENCY)
        await asyncio.gather(*(scrape_article(sem, context, u, stats) for u in todo),
                             return_exceptions=True)
        try:
            await browser.close()
        except Exception:
            pass
    print("[INFO] Selesai. Output di", OUT_JSONL)


if __name__ == '__main__':
    asyncio.run(main())
