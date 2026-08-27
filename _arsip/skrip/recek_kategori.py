"""
recek_kategori.py
=================
Cek-ulang KATEGORI AKSES untuk entri yang sebelumnya 'OK' tapi skor LOW.
Tujuan: menangkap halaman ERROR/MATI Cloudflare (web server down 521,
origin DNS 1016, dll) & parkir yang lolos jadi OK -> menggelembungkan FN.

Fetch ringan (TANPA YOLO/screenshot ulang), pakai signature & stealth
yang sama dgn screening_gate. Meng-update access_category di scan_log.json
dan menghapus screenshot terkait di folder low.

Pakai:  python recek_kategori.py
"""
import os, sys, json, asyncio, time
from collections import Counter

import screening_gate as G   # reuse classify_access, signatures, stealth, konstanta

SCAN_LOG = G.JSON_LOG_FILE
LOWDIR   = os.path.join(G.OUTPUT_DIR, 'low')
CONC     = int(os.environ.get('CONCURRENCY', 6))


def norm(u):
    u = u.strip().lower()
    for p in ('https://', 'http://'):
        if u.startswith(p):
            u = u[len(p):]
    return u[4:].rstrip('/') if u.startswith('www.') else u.rstrip('/')


async def recheck(sem, context, entry, updates):
    async with sem:
        page = await context.new_page()
        try:
            full = entry['url'] if entry['url'].startswith('http') else 'https://' + entry['url']
            await G.apply_stealth(page)
            try:
                await page.goto(full, timeout=G.TIMEOUT_MS, wait_until='domcontentloaded')
            except Exception:
                updates[entry['url']] = 'DEAD'      # tak bisa diakses sekarang
                return
            await asyncio.sleep(2)
            cat, _, _ = await G.classify_access(page)
            if cat == 'BOT_CHALLENGE':              # beri waktu tantangan selesai
                dl = time.time() + G.CF_WAIT_MAX / 1000.0
                while cat == 'BOT_CHALLENGE' and time.time() < dl:
                    await asyncio.sleep(2)
                    cat, _, _ = await G.classify_access(page)
            updates[entry['url']] = cat
            print(f"  {entry['url'][:48]:<48} -> {cat}")
        except Exception:
            updates[entry['url']] = 'DEAD'
        finally:
            try:
                await page.close()
            except Exception:
                pass


async def main():
    if not os.path.exists(SCAN_LOG):
        print(f"[ERROR] {SCAN_LOG} tidak ada."); sys.exit(1)

    data = json.load(open(SCAN_LOG, encoding='utf-8'))
    # Suspect: SUCCESS + skor LOW + saat ini OK (error page lolos sbg OK)
    suspects = [d for d in data
                if d.get('status') == 'SUCCESS'
                and (d.get('score_data') or {}).get('risk_level') == 'LOW'
                and (d.get('access_category') or 'OK') == 'OK']
    print(f"[INFO] Suspect LOW-OK untuk dicek ulang: {len(suspects)}")
    if not suspects:
        print("Tidak ada yang perlu dicek."); return

    updates = {}
    launch_args = ['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu',
                   '--disable-blink-features=AutomationControlled']
    async with G.async_playwright() as p:
        try:
            browser = await p.chromium.launch(headless=True, channel=G.BROWSER_CHANNEL, args=launch_args)
        except Exception:
            browser = await p.chromium.launch(headless=True, args=launch_args)
        context = await browser.new_context(
            viewport={'width': 1366, 'height': 2500}, ignore_https_errors=True,
            user_agent=G.USER_AGENT, locale='id-ID', timezone_id='Asia/Jakarta')
        sem = asyncio.Semaphore(CONC)
        await asyncio.gather(*[recheck(sem, context, e, updates) for e in suspects],
                             return_exceptions=True)
        try:
            await browser.close()
        except Exception:
            pass

    # Terapkan update ke scan_log + hapus screenshot yang jadi non-OK
    retag = 0
    deleted = 0
    for d in data:
        nc = updates.get(d['url'])
        if nc and nc != 'OK':
            d['access_category'] = nc
            retag += 1
            sp = d.get('screenshot')
            if not sp or not os.path.exists(sp):
                safe = (('https://' + d['url']) if not d['url'].startswith('http') else d['url'])
                safe = safe.replace('https://', '').replace('http://', '').replace('/', '_').replace(':', '')[:50]
                sp = os.path.join(LOWDIR, safe + '.jpg')
            if os.path.exists(sp) and os.path.normpath('hasil_analisa_judol/low') in os.path.normpath(sp):
                os.remove(sp); deleted += 1

    json.dump(data, open(SCAN_LOG, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("  HASIL CEK-ULANG")
    print("=" * 50)
    print("  Distribusi kategori baru:", dict(Counter(updates.values())))
    print(f"  Di-retag jadi non-OK : {retag}")
    print(f"  Screenshot low dihapus: {deleted}")
    print(f"  Sisa file di low     : {len(os.listdir(LOWDIR))}")
    print("=" * 50)
    print("Lanjut:  python bangun_dataset_judol.py 2000  &&  python evaluasi_sistem.py eval")


if __name__ == '__main__':
    asyncio.run(main())
