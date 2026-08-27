"""
Kumpulkan hard negative kedua untuk cascade IndoBERT: situs GAME dan CASINO-ADJACENT
yang legal namun kaya kosakata judi (slot, casino, poker, mahjong, jackpot). Ini
profil false-positive sebenarnya pada zona ambigu (situs game), yang kurang
terwakili pada kumpulan negatif pertama (train_nonjudol.jsonl, didominasi berita).

Fase 1: kunjungi halaman listing/tag portal game untuk istilah gambling-adjacent,
        ekstrak tautan halaman game individual -> dataset_negatif_game.txt
Fase 2: scrape teks tiap halaman -> teks_dataset/train_neg_game.jsonl
        (skema sama; group='train_nonjudol', is_judol=False)

Anti-leakage: domain judol dan domain 254 URL zona MEDIUM dataset evaluasi
(mis. ign.com, playstation.com) dikecualikan. Resume-aware di kedua fase.
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
URL_LIST_FILE = os.path.join(BASE_DIR, 'dataset_negatif_game.txt')
OUT_JSONL = os.path.join(OUT_DIR, 'train_neg_game.jsonl')
os.makedirs(OUT_DIR, exist_ok=True)

CONCURRENCY = int(os.environ.get('CONCURRENCY', 4))
TIMEOUT_MS = 30000
PER_URL_TIMEOUT = 60
MAX_TEXT_CHARS = 20000
MAX_PER_LISTING = 25
USER_AGENT = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
              '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36')

TERMS = ['slot', 'casino', 'poker', 'mahjong', 'blackjack', 'roulette',
         'bingo', 'jackpot', 'solitaire', 'card-game', 'baccarat']

# (nama, template listing/tag, regex halaman game individual)
LISTING_SOURCES = [
    ('y8', 'https://www.y8.com/tags/{t}', r'https?://www\.y8\.com/games/[a-z0-9_]+'),
    ('crazygames', 'https://www.crazygames.com/t/{t}', r'https?://www\.crazygames\.com/game/[a-z0-9-]+'),
    ('poki', 'https://poki.com/en/tag/{t}', r'https?://poki\.com/en/g/[a-z0-9-]+'),
    ('kongregate', 'https://www.kongregate.com/games?tag={t}', r'https?://www\.kongregate\.com/games/[A-Za-z0-9_]+/[a-z0-9-]+'),
    ('itchio', 'https://itch.io/games/tag-{t}', r'https?://[a-z0-9-]+\.itch\.io/[a-z0-9-]+'),
    ('addictinggames', 'https://www.addictinggames.com/search?query={t}', r'https?://www\.addictinggames\.com/[a-z-]+/[a-z0-9-]+'),
    ('gamedistribution', 'https://gamedistribution.com/games/?search={t}', r'https?://gamedistribution\.com/games/[a-z0-9-]+'),
]

# Seed statis: kasino sosial (legal, penuh kata slot/casino/free coins) + portal game
# + situs kartu/mahjong legit + media game. JENIS hard negative paling menyerupai judol.
STATIC_SEEDS = [
    # kasino sosial (bukan judi uang asli)
    'https://www.slotomania.com/', 'https://www.houseoffun.com/',
    'https://www.doubledowncasino.com/', 'https://www.gsngames.com/',
    'https://www.jackpotpartycasino.com/', 'https://www.worldwinner.com/',
    'https://www.bigfishgames.com/games/genres/12/casino.html',
    'https://www.pogo.com/games/casino', 'https://www.gametwist.com/en/',
    'https://www.doubleu-casino.com/', 'https://www.myvegas.com/',
    'https://www.zynga.com/games/hit-it-rich-slots/',
    # portal game umum & bertema kartu/slot
    'https://www.miniclip.com/games/en/', 'https://www.kizi.com/',
    'https://www.friv.com/', 'https://www.coolmathgames.com/',
    'https://www.gamesgames.com/games/mahjong', 'https://www.solitaire.org/',
    'https://cardgames.io/', 'https://www.247solitaire.com/',
    'https://www.mahjonggames.com/', 'https://freeslots.com/',
    'https://www.vegasslotsonline.com/free/',   # DEMO slot gratis, bukan uang asli
    'https://www.free-slots.games/', 'https://slotomania.fandom.com/',
    # media/database game
    'https://www.pcgamer.com/', 'https://www.gamespot.com/',
    'https://www.metacritic.com/game/', 'https://www.mobygames.com/',
    'https://www.pokerstars.com/en/how-to-play/',  # halaman edukasi aturan poker
    # game & media game Indonesia
    'https://duniagames.co.id/', 'https://www.gamebrott.com/',
    'https://oneesports.id/', 'https://gamefinity.id/',
]


def _domain(u):
    try:
        d = urlparse(u if u.startswith('http') else f'https://{u}').netloc.lower()
        return d[4:] if d.startswith('www.') else d
    except Exception:
        return ''


def load_excluded_domains():
    """Domain judol + domain 254 URL zona MEDIUM (test set cascade) dikecualikan."""
    def _load(fname):
        with open(os.path.join(BASE_DIR, fname)) as f:
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
    found, seen = [], set()
    if os.path.exists(URL_LIST_FILE):
        with open(URL_LIST_FILE) as f:
            found = [l.strip() for l in f if l.strip()]
        seen = set(found)
        print(f"[INFO] Melanjutkan daftar: {len(found)} URL", flush=True)

    listing_pages = [(name, tmpl.format(t=t), pat)
                     for name, tmpl, pat in LISTING_SOURCES for t in TERMS]
    page = await context.new_page()
    for name, url, pat in listing_pages:
        rx = re.compile(pat)
        try:
            await page.goto(url, timeout=TIMEOUT_MS, wait_until='domcontentloaded')
            try:
                await page.wait_for_function(
                    "pat => Array.from(document.querySelectorAll('a[href]')).some(a => new RegExp(pat).test(a.href))",
                    arg=pat, timeout=12000)
            except PlaywrightTimeoutError:
                pass
            await asyncio.sleep(1.5)
            hrefs = await page.evaluate("Array.from(document.querySelectorAll('a[href]')).map(a=>a.href)")
        except Exception as e:
            print(f"[GAGAL] {name} {url[:55]}: {str(e)[:50]}", flush=True)
            continue
        n_new = 0
        for h in hrefs:
            if n_new >= MAX_PER_LISTING:
                break
            if not rx.match(h):
                continue
            h = h.split('?')[0].split('#')[0]
            if h in seen or _domain(h) in excluded:
                continue
            seen.add(h); found.append(h); n_new += 1
        print(f"[LINK] {name:14s} {url[:50]:50s}: +{n_new} (total {len(found)})", flush=True)
    await page.close()

    for h in STATIC_SEEDS:
        if h not in seen and _domain(h) not in excluded:
            seen.add(h); found.append(h)

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


async def scrape_one(sem, context, url, stats):
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
                    pass
                await asyncio.sleep(2)
                entry['final_url'] = page.url
                entry['page_title'] = ((await page.title()) or '')[:200]
                text = await page.evaluate("document.body ? document.body.innerText : ''")
                if len(text.strip()) < 150:
                    entry['access_category'] = 'DEAD'; entry['error'] = 'Blank/Short'; return
                entry['access_category'] = 'OK'
                entry['text'] = text[:MAX_TEXT_CHARS]; entry['text_len'] = len(text)
            await asyncio.wait_for(_go(), timeout=PER_URL_TIMEOUT)
        except asyncio.TimeoutError:
            entry['error'] = 'Hard Timeout'; entry['access_category'] = entry['access_category'] or 'DEAD'
        except Exception as e:
            entry['error'] = str(e).split('\n')[0][:80]; entry['access_category'] = entry['access_category'] or 'ERROR'
        finally:
            try:
                await page.close()
            except Exception:
                pass
        async with write_lock:
            with open(OUT_JSONL, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        stats['done'] += 1
        print(f"[{stats['done']}/{stats['total']}] {url[:60]:60s} => {'TEKS' if entry['text'] else entry['access_category']}", flush=True)


async def main():
    launch_args = ['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu']
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(headless=True, channel='chrome', args=launch_args)
        except Exception:
            browser = await p.chromium.launch(headless=True, args=launch_args)
        context = await browser.new_context(
            viewport={'width': 1366, 'height': 2500}, user_agent=USER_AGENT,
            locale='en-US', timezone_id='Asia/Jakarta')

        async def _route(route):
            if route.request.resource_type in ('media', 'font', 'image'):
                await route.abort()
            else:
                await route.continue_()
        await context.route('**/*', _route)

        urls = await collect_links(context)
        done = load_done()
        todo = [u for u in urls if u not in done]
        print(f"[INFO] Fase 2: scrape {len(todo)} halaman game (selesai: {len(done)})")
        stats = {'done': 0, 'total': len(todo)}
        sem = asyncio.Semaphore(CONCURRENCY)
        await asyncio.gather(*(scrape_one(sem, context, u, stats) for u in todo),
                             return_exceptions=True)
        try:
            await browser.close()
        except Exception:
            pass
    print("[INFO] Selesai. Output di", OUT_JSONL)


if __name__ == '__main__':
    asyncio.run(main())
