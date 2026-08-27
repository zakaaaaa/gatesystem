"""Ambil ulang screenshot BERSIH untuk test set independen confusion matrix.

Membaca cm_test_set/kandidat_domain.tsv (kategori<TAB>domain), mengunjungi tiap
domain dengan viewport yang sama seperti pipeline GATE (1366x2500), dan menyimpan
screenshot polos (tanpa anotasi) ke cm_test_set/images/ sampai kuota per kategori
terpenuhi: HIGH 50, MEDIUM 25, LOW 25.
"""
import asyncio, os, sys
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

BASE = os.path.dirname(os.path.abspath(__file__))
TSV = os.path.join(BASE, "cm_test_set", "kandidat_domain.tsv")
OUT = os.path.join(BASE, "cm_test_set", "images")
QUOTA = {"high": 50, "medium": 25, "low": 25}
TIMEOUT_MS = 25000
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

captured = {k: 0 for k in QUOTA}
lock = asyncio.Lock()


async def grab(sem, context, cat, domain):
    async with lock:
        if captured[cat] >= QUOTA[cat]:
            return
    async with sem:
        async with lock:
            if captured[cat] >= QUOTA[cat]:
                return
        page = await context.new_page()
        try:
            await page.goto(f"https://{domain}", timeout=TIMEOUT_MS,
                            wait_until="domcontentloaded")
            try:
                await page.wait_for_load_state("networkidle", timeout=8000)
            except PWTimeout:
                pass
            await asyncio.sleep(2)
            try:
                vp = page.viewport_size
                await page.mouse.click(50, 50)
                await asyncio.sleep(0.2)
                await page.mouse.click((vp["width"] if vp else 1366) - 50, 50)
            except Exception:
                pass
            await asyncio.sleep(1)
            text = await page.evaluate("document.body.innerText")
            if len(text.strip()) < 50:
                raise Exception("blank")
            path = os.path.join(OUT, f"{domain}.jpg")
            await page.screenshot(path=path, full_page=False, timeout=15000)
            async with lock:
                if captured[cat] < QUOTA[cat]:
                    captured[cat] += 1
                    print(f"OK [{cat} {captured[cat]}/{QUOTA[cat]}] {domain}",
                          flush=True)
                else:
                    os.remove(path)
        except Exception as e:
            print(f"GAGAL {domain}: {str(e).splitlines()[0][:50]}", flush=True)
        finally:
            try:
                await page.close()
            except Exception:
                pass


async def main():
    rows = []
    with open(TSV) as f:
        for line in f:
            cat, domain = line.strip().split("\t")
            rows.append((cat, domain))
    sem = asyncio.Semaphore(6)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1366, "height": 2500},
            user_agent=UA, locale="id-ID")
        await asyncio.gather(*(grab(sem, context, c, d) for c, d in rows))
        await browser.close()
    print("SELESAI:", captured, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
