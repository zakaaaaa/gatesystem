import os
import sys
import cv2
import csv
import glob
import numpy as np
import asyncio
import argparse
import subprocess
from datetime import datetime
from collections import deque
from playwright.async_api import async_playwright

import screening as gate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, 'dataset_eval_combined.txt')

DIR_EVIDENCE = os.path.join(BASE_DIR, 'evidence_demo')
DIR_DATASET = os.path.join(BASE_DIR, 'dataset_demo')
RESULTS_CSV = os.path.join(BASE_DIR, 'hasil_klasifikasi_demo.csv')

for cat in ['high', 'medium', 'low']:
    os.makedirs(os.path.join(DIR_EVIDENCE, cat), exist_ok=True)
for sub in ['images_clean', 'images_verification', 'labels']:
    os.makedirs(os.path.join(DIR_DATASET, sub), exist_ok=True)

SLEEP_SETTLE = 1.5

WINDOW = 'GATE - Pipeline Screening'
SOUND_DIR = '/System/Library/Sounds'
SOUNDS = {
    'open':     'Morse',
    'shutter':  'Pop',
    'detect':   'Tink',
    'keyword':  'Bottle',
    'penalty':  'Funk',
    'high':     'Sosumi',
    'medium':   'Ping',
    'low':      'Glass',
    'fail':     'Basso',
}
RISK_COLOR = {'high': (60, 60, 235), 'medium': (30, 150, 240), 'low': (90, 180, 70)}
C_VISUAL, C_TEXT, C_FINAL = (235, 170, 60), (120, 220, 120), (240, 240, 240)
C_BG, C_PANEL, C_MUTED = 22, (34, 34, 34), (150, 150, 150)

CANVAS_W, CANVAS_H = 1400, 880
HEAD_H = 64
LEFT_X, LEFT_W = 18, 462
RIGHT_X = 512
RIGHT_W = CANVAS_W - RIGHT_X - 18
F = cv2.FONT_HERSHEY_SIMPLEX

_procs = deque()


def play(key):
    if not CFG.sound:
        return
    path = os.path.join(SOUND_DIR, f'{SOUNDS.get(key, "")}.aiff')
    if not os.path.exists(path):
        return
    while _procs and _procs[0].poll() is not None:
        _procs.popleft()
    _procs.append(subprocess.Popen(
        ['afplay', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))


def wait(ms):
    return cv2.waitKey(max(1, int(ms * CFG.speed))) == 27


def score_breakdown(detections, text_content):
    text_lower = text_content.lower()

    visual_hits = [{'label': d['label'],
                    'conf': d['conf'],
                    'weight': gate.VISUAL_WEIGHTS.get(d['label'], 0),
                    'points': gate.VISUAL_WEIGHTS.get(d['label'], 0) * d['conf'],
                    'box': d['box']} for d in detections]
    raw_visual = sum(h['points'] for h in visual_hits)
    norm_visual = min(raw_visual, 100)

    text_hits = [{'word': w, 'points': s}
                 for w, s in gate.TEXT_WEIGHTS.items() if w in text_lower]
    safety_hits = [{'word': w, 'points': -gate.SAFETY_PENALTY}
                   for w in gate.SAFETY_WORDS if w in text_lower]
    raw_text = sum(h['points'] for h in text_hits) + sum(h['points'] for h in safety_hits)
    norm_text = max(min(raw_text, 100), 0)

    final = round((norm_visual * 0.5) + (norm_text * 0.5), 2)
    if final >= 50:
        risk = 'high' if norm_visual > 0 else 'medium'
    elif final > 25:
        risk = 'medium'
    else:
        risk = 'low'

    return {'visual_hits': visual_hits, 'raw_visual': raw_visual, 'norm_visual': norm_visual,
            'text_hits': text_hits, 'safety_hits': safety_hits,
            'raw_text': raw_text, 'norm_text': norm_text,
            'final': final, 'risk': risk}


def verify_breakdown(bd, detections, text_content):
    risk, score = gate.calculate_score(detections, text_content)
    if risk != bd['risk'] or abs(score - bd['final']) > 1e-6:
        raise AssertionError(
            f"rincian demo menyimpang dari screening.calculate_score(): "
            f"demo=({bd['risk']}, {bd['final']}) vs produksi=({risk}, {score})")
    return risk, score


def new_canvas():
    return np.full((CANVAS_H, CANVAS_W, 3), C_BG, np.uint8)


def text(c, s, x, y, scale=0.5, color=(230, 230, 230), thick=1):
    cv2.putText(c, s, (x, y), F, scale, color, thick, cv2.LINE_AA)


def header(c, url, stage, color=(52, 52, 52)):
    cv2.rectangle(c, (0, 0), (CANVAS_W, HEAD_H), color, -1)
    text(c, url[:78], 16, 26, 0.58, (255, 255, 255))
    text(c, stage, 16, 50, 0.52, (140, 235, 190))


def place_shot(c, shot):
    h, w = shot.shape[:2]
    y0 = HEAD_H + 16
    c[y0:y0 + h, LEFT_X:LEFT_X + w] = shot
    cv2.rectangle(c, (LEFT_X - 1, y0 - 1), (LEFT_X + w, y0 + h), (80, 80, 80), 1)
    return y0


def panel(c, title, color=(200, 200, 200)):
    cv2.rectangle(c, (RIGHT_X, HEAD_H + 16), (RIGHT_X + RIGHT_W, CANVAS_H - 18), C_PANEL, -1)
    text(c, title, RIGHT_X + 20, HEAD_H + 52, 0.62, color, 2)
    cv2.line(c, (RIGHT_X + 20, HEAD_H + 66), (RIGHT_X + RIGHT_W - 20, HEAD_H + 66),
             (70, 70, 70), 1)


def frame(url, stage, shot, title, title_color=(200, 200, 200)):
    c = new_canvas()
    header(c, url, stage)
    place_shot(c, shot)
    panel(c, title, title_color)
    return c


def bar(c, x, y, w, h, value, vmax, color, label, suffix=''):
    cv2.rectangle(c, (x, y), (x + w, y + h), (58, 58, 58), -1)
    fill = int(w * min(value / vmax, 1.0))
    if fill > 0:
        cv2.rectangle(c, (x, y), (x + fill, y + h), color, -1)
    text(c, label, x, y - 10, 0.5, (205, 205, 205))
    text(c, f"{value:.2f}{suffix}", x + w + 14, y + h - 6, 0.58, color, 2)


def fit_shot(img):
    h, w = img.shape[:2]
    scale = min(LEFT_W / w, (CANVAS_H - HEAD_H - 34) / h)
    return cv2.resize(img, (int(w * scale), int(h * scale))), scale


def wrap(s, width):
    out, line = [], ''
    for word in s.split():
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


def stage_popup(url, before, after, ov_before, ov_after, points, pscale, robust_used):
    stage = 'Tahap 1/4 - Penanganan popup iklan'
    title = 'PENANGANAN POPUP'

    def info(c, lines):
        y = HEAD_H + 110
        for s, col in lines:
            text(c, s, RIGHT_X + 20, y, 0.52, col)
            y += 30

    for _ in range(2):
        c = frame(url, stage, before, title, (200, 200, 255))
        info(c, [(f"Popup terdeteksi : {'YA' if ov_before else 'TIDAK'}",
                  (120, 120, 255) if ov_before else C_MUTED)]
             + ([(f"  {ov_before[0]['desc'][:46]}", C_MUTED)] if ov_before else []))
        cv2.imshow(WINDOW, c)
        if wait(500):
            return True

    for i, (px, py) in enumerate(points, 1):
        for r in (10, 20, 30):
            c = frame(url, stage, before, title, (200, 200, 255))
            info(c, [(f"Popup terdeteksi : {'YA' if ov_before else 'TIDAK'}",
                      (120, 120, 255) if ov_before else C_MUTED),
                     (f"Klik sudut {'kiri' if i == 1 else 'kanan'} atas ({px}, {py})",
                      (120, 220, 255))])
            y0 = HEAD_H + 16
            sx, sy = LEFT_X + int(px * pscale), y0 + int(py * pscale)
            cv2.circle(c, (sx, sy), r, (120, 220, 255), 2)
            cv2.drawMarker(c, (sx, sy), (120, 220, 255), cv2.MARKER_CROSS, 14, 2)
            cv2.imshow(WINDOW, c)
            if wait(70):
                return True

    if not ov_before:
        verdict, col = 'Tidak ada popup yang perlu ditutup.', C_MUTED
    elif not ov_after:
        verdict, col = 'Popup berhasil ditutup.', (120, 220, 120)
    else:
        verdict, col = 'Popup MASIH ADA - ikut terpotret.', (120, 120, 255)

    c = frame(url, stage, after, title, (200, 200, 255))
    info(c, [(f"Popup terdeteksi : {'YA' if ov_before else 'TIDAK'}",
              (120, 120, 255) if ov_before else C_MUTED),
             (f"Klik sudut kiri atas  ({points[0][0]}, {points[0][1]})", C_MUTED),
             (f"Klik sudut kanan atas ({points[1][0]}, {points[1][1]})", C_MUTED),
             (f"Jeda settle {SLEEP_SETTLE}s", C_MUTED),
             ('', C_MUTED),
             (verdict, col)]
         + ([('Mode --popup-robust aktif (di luar laporan).', (120, 220, 255))]
            if robust_used else []))
    cv2.imshow(WINDOW, c)
    return wait(1300)


def stage_capture(url, shot):
    c = frame(url, 'Screenshot diterima - menyiapkan inferensi', shot,
              'MENUNGGU ANALISA')
    text(c, 'Screenshot bersih tersimpan.', RIGHT_X + 20, HEAD_H + 110, 0.52, C_MUTED)
    text(c, 'Elemen demo dihapus sebelum capture,', RIGHT_X + 20, HEAD_H + 140, 0.52, C_MUTED)
    text(c, 'sehingga citra identik dengan pipeline produksi.',
         RIGHT_X + 20, HEAD_H + 170, 0.52, C_MUTED)
    cv2.imshow(WINDOW, c)
    return wait(700)


def stage_scan(url, shot):
    h = shot.shape[0]
    for y in range(0, h, max(5, int(h / 40))):
        s = shot.copy()
        cv2.rectangle(s, (0, max(0, y - 55)), (s.shape[1], y), (0, 80, 50), -1)
        s = cv2.addWeighted(s, 0.75, shot, 0.25, 0)
        cv2.line(s, (0, y), (s.shape[1], y), (140, 255, 60), 2)
        c = frame(url, 'Tahap 2/4 - Inferensi YOLO berjalan', s, 'ANALISA VISUAL (YOLO)', C_VISUAL)
        text(c, 'Memindai citra...', RIGHT_X + 20, HEAD_H + 110, 0.55, C_MUTED)
        cv2.imshow(WINDOW, c)
        if wait(13):
            return True
    return False


def stage_visual(url, shot, bd, scale):
    hits = bd['visual_hits']
    marked = shot.copy()
    running = 0.0
    stage = 'Tahap 2/4 - Analisa visual (YOLO)'

    if not hits:
        c = frame(url, stage, marked, 'ANALISA VISUAL (YOLO)', C_VISUAL)
        text(c, 'Tidak ada objek judol terdeteksi.', RIGHT_X + 20, HEAD_H + 110, 0.55, C_MUTED)
        bar(c, RIGHT_X + 20, CANVAS_H - 110, RIGHT_W - 130, 26, 0.0, 100, C_VISUAL,
            'Skor visual (norm_visual)')
        cv2.imshow(WINDOW, c)
        return wait(800), marked

    for i, hit in enumerate(hits, 1):
        x1, y1, x2, y2 = (int(v * scale) for v in hit['box'])
        running += hit['points']
        play('detect')

        for t in range(4):
            s = marked.copy()
            cv2.rectangle(s, (x1, y1), (x2, y2), (60, 60, 255), 5 - t)
            c = frame(url, stage, s, 'ANALISA VISUAL (YOLO)', C_VISUAL)
            draw_visual_list(c, hits[:i], running, bd)
            cv2.imshow(WINDOW, c)
            if wait(50):
                return True, marked

        cv2.rectangle(marked, (x1, y1), (x2, y2), (60, 60, 255), 2)
        lbl = f"{hit['label']} {hit['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(lbl, F, 0.42, 1)
        cv2.rectangle(marked, (x1, max(0, y1 - th - 7)), (x1 + tw + 6, y1), (60, 60, 255), -1)
        text(marked, lbl, x1 + 3, max(9, y1 - 5), 0.42, (255, 255, 255))

        c = frame(url, stage, marked, 'ANALISA VISUAL (YOLO)', C_VISUAL)
        draw_visual_list(c, hits[:i], running, bd)
        cv2.imshow(WINDOW, c)
        if wait(230):
            return True, marked

    c = frame(url, stage, marked, 'ANALISA VISUAL (YOLO)', C_VISUAL)
    draw_visual_list(c, hits, bd['raw_visual'], bd, show_norm=True)
    cv2.imshow(WINDOW, c)
    return wait(900), marked


def draw_visual_list(c, hits, running, bd, show_norm=False):
    text(c, 'label            bobot  x  conf   =   poin', RIGHT_X + 20, HEAD_H + 100, 0.46, C_MUTED)
    y = HEAD_H + 130
    for h in hits[-14:]:
        text(c, f"{h['label']:<14}", RIGHT_X + 20, y, 0.5, (225, 225, 225))
        text(c, f"{h['weight']:>6.2f} x {h['conf']:.2f}", RIGHT_X + 175, y, 0.5, C_MUTED)
        text(c, f"= {h['points']:>6.2f}", RIGHT_X + 320, y, 0.5, C_VISUAL)
        y += 26
    text(c, f"raw_visual = {running:.2f}", RIGHT_X + 20, CANVAS_H - 150, 0.55, (225, 225, 225))
    if show_norm:
        text(c, f"min(raw_visual, 100) -> {bd['norm_visual']:.2f}",
             RIGHT_X + 250, CANVAS_H - 150, 0.5, C_MUTED)
    bar(c, RIGHT_X + 20, CANVAS_H - 110, RIGHT_W - 130, 26,
        min(running, 100), 100, C_VISUAL, 'Skor visual (norm_visual)')


def stage_text(url, shot, bd, page_text):
    stage = 'Tahap 3/4 - Analisa teks halaman'
    snippet = ' '.join(page_text.split())[:260]
    lines = wrap(snippet, 62)

    shown = 0
    total_chars = sum(len(l) for l in lines)
    while shown <= total_chars:
        c = frame(url, stage, shot, 'ANALISA TEKS', C_TEXT)
        text(c, 'Teks hasil scraping:', RIGHT_X + 20, HEAD_H + 100, 0.48, C_MUTED)
        left, y = shown, HEAD_H + 128
        for line in lines:
            if left <= 0:
                break
            text(c, line[:left], RIGHT_X + 20, y, 0.44, (200, 220, 200))
            left -= len(line)
            y += 22
        cv2.imshow(WINDOW, c)
        if wait(16):
            return True
        shown += 14

    hits = bd['text_hits'] + bd['safety_hits']
    running = 0.0
    for i, hit in enumerate(hits, 1):
        running += hit['points']
        play('penalty' if hit['points'] < 0 else 'keyword')
        c = frame(url, stage, shot, 'ANALISA TEKS', C_TEXT)
        draw_text_list(c, hits[:i], running, bd)
        cv2.imshow(WINDOW, c)
        if wait(150):
            return True

    c = frame(url, stage, shot, 'ANALISA TEKS', C_TEXT)
    draw_text_list(c, hits, bd['raw_text'], bd, show_norm=True)
    cv2.imshow(WINDOW, c)
    return wait(900)


def draw_text_list(c, hits, running, bd, show_norm=False):
    text(c, 'kata kunci ditemukan', RIGHT_X + 20, HEAD_H + 100, 0.46, C_MUTED)
    col_x, y0 = [RIGHT_X + 20, RIGHT_X + 300, RIGHT_X + 580], HEAD_H + 130
    per_col = 17
    for i, h in enumerate(hits[-(per_col * 3):]):
        cx = col_x[i // per_col]
        cy = y0 + (i % per_col) * 25
        neg = h['points'] < 0
        text(c, f"{h['word'][:15]:<16}", cx, cy, 0.48, (255, 190, 120) if neg else (225, 225, 225))
        text(c, f"{h['points']:>+7.2f}", cx + 150, cy, 0.48,
             (90, 160, 255) if neg else C_TEXT)
    if bd['safety_hits']:
        text(c, f"safety word: -{gate.SAFETY_PENALTY} per kata",
             RIGHT_X + 20, CANVAS_H - 178, 0.46, (255, 190, 120))
    text(c, f"raw_text = {running:.2f}", RIGHT_X + 20, CANVAS_H - 150, 0.55, (225, 225, 225))
    if show_norm:
        text(c, f"clamp(raw_text, 0, 100) -> {bd['norm_text']:.2f}",
             RIGHT_X + 250, CANVAS_H - 150, 0.5, C_MUTED)
    bar(c, RIGHT_X + 20, CANVAS_H - 110, RIGHT_W - 130, 26,
        max(min(running, 100), 0), 100, C_TEXT, 'Skor teks (norm_text)')


def stage_fusion(url, shot, bd):
    stage = 'Tahap 4/4 - Penggabungan skor'
    nv, nt, final = bd['norm_visual'], bd['norm_text'], bd['final']
    risk = bd['risk']
    color = RISK_COLOR[risk]
    bx, bw = RIGHT_X + 30, RIGHT_W - 150

    def base(t):
        c = frame(url, stage, shot, 'PENGGABUNGAN SKOR', C_FINAL)
        bar(c, bx, HEAD_H + 130, bw, 24, nv * t, 100, C_VISUAL, 'norm_visual')
        bar(c, bx, HEAD_H + 210, bw, 24, nt * t, 100, C_TEXT, 'norm_text')
        return c

    for step in range(0, 21):
        c = base(step / 20)
        cv2.imshow(WINDOW, c)
        if wait(22):
            return True

    for step in range(0, 21):
        t = step / 20
        c = base(1.0)
        text(c, f"x 0.5  ->  {nv * 0.5:.2f}", bx + bw - 170, HEAD_H + 118, 0.5, C_VISUAL)
        text(c, f"x 0.5  ->  {nt * 0.5:.2f}", bx + bw - 170, HEAD_H + 198, 0.5, C_TEXT)
        text(c, 'skor_akhir = (norm_visual x 0.5) + (norm_text x 0.5)',
             bx, HEAD_H + 280, 0.5, C_MUTED)
        bar(c, bx, HEAD_H + 320, bw, 34, final * t, 100, C_FINAL, 'SKOR AKHIR')
        for thr, name in ((25, 'ambang 25'), (50, 'ambang 50')):
            tx = bx + int(bw * thr / 100)
            cv2.line(c, (tx, HEAD_H + 314), (tx, HEAD_H + 360), (120, 120, 120), 1)
            text(c, name, tx - 26, HEAD_H + 378, 0.4, C_MUTED)
        cv2.imshow(WINDOW, c)
        if wait(24):
            return True

    play(risk)
    c = base(1.0)
    text(c, f"x 0.5  ->  {nv * 0.5:.2f}", bx + bw - 170, HEAD_H + 118, 0.5, C_VISUAL)
    text(c, f"x 0.5  ->  {nt * 0.5:.2f}", bx + bw - 170, HEAD_H + 198, 0.5, C_TEXT)
    text(c, 'skor_akhir = (norm_visual x 0.5) + (norm_text x 0.5)',
         bx, HEAD_H + 280, 0.5, C_MUTED)
    bar(c, bx, HEAD_H + 320, bw, 34, final, 100, C_FINAL, 'SKOR AKHIR')
    for thr, name in ((25, 'ambang 25'), (50, 'ambang 50')):
        tx = bx + int(bw * thr / 100)
        cv2.line(c, (tx, HEAD_H + 314), (tx, HEAD_H + 360), (120, 120, 120), 1)
        text(c, name, tx - 26, HEAD_H + 378, 0.4, C_MUTED)

    py = CANVAS_H - 190
    cv2.rectangle(c, (RIGHT_X + 20, py), (RIGHT_X + RIGHT_W - 20, CANVAS_H - 40), (28, 28, 28), -1)
    cv2.rectangle(c, (RIGHT_X + 20, py), (RIGHT_X + 30, CANVAS_H - 40), color, -1)
    text(c, f"{risk.upper()}", RIGHT_X + 52, py + 62, 1.5, color, 3)
    text(c, f"skor {final}", RIGHT_X + 300, py + 62, 1.0, (235, 235, 235), 2)
    rule = ('skor >= 50 dan ada objek visual -> HIGH' if risk == 'high' else
            'skor >= 50 tanpa objek visual -> MEDIUM' if final >= 50 else
            'skor > 25 -> MEDIUM' if risk == 'medium' else 'skor <= 25 -> LOW')
    text(c, rule, RIGHT_X + 52, py + 100, 0.5, C_MUTED)
    text(c, f"bukti disimpan di evidence_demo/{risk}/", RIGHT_X + 52, py + 128, 0.48, C_MUTED)
    cv2.imshow(WINDOW, c)
    return wait(1800)


JS_BANNER = """() => {
  const d = document.createElement('div');
  d.id = '__gate_banner';
  d.textContent = 'GATE - MENGAMBIL SCREENSHOT';
  Object.assign(d.style, {position:'fixed', top:'0', left:'0', right:'0',
    zIndex:'2147483647', background:'#111', color:'#0f6', font:'600 15px monospace',
    padding:'10px 16px', letterSpacing:'2px', textAlign:'center'});
  document.documentElement.appendChild(d);
}"""

JS_CLEAN = """() => {
  ['__gate_banner','__gate_flash','__gate_result']
    .forEach(id => document.getElementById(id)?.remove());
}"""

JS_FLASH = """() => {
  const d = document.createElement('div');
  d.id = '__gate_flash';
  Object.assign(d.style, {position:'fixed', inset:'0', zIndex:'2147483647',
    background:'#fff', opacity:'1', pointerEvents:'none',
    transition:'opacity .45s ease-out'});
  document.documentElement.appendChild(d);
  requestAnimationFrame(() => { d.style.opacity = '0'; });
  setTimeout(() => d.remove(), 600);
}"""

JS_HIGHLIGHT = """(words) => {
  const re = new RegExp('(' + words.map(w =>
    w.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&')).join('|') + ')', 'gi');
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const targets = [];
  let n, count = 0;
  while ((n = walker.nextNode()) && count < 400) {
    if (!n.nodeValue.trim()) continue;
    const p = n.parentNode;
    if (!p || /SCRIPT|STYLE|NOSCRIPT/.test(p.nodeName)) continue;
    if (re.test(n.nodeValue)) { targets.push(n); count++; }
    re.lastIndex = 0;
  }
  targets.forEach(node => {
    const span = document.createElement('span');
    span.className = '__gate_hl';
    span.innerHTML = node.nodeValue.replace(re,
      '<mark style="background:#ffe600;color:#000;padding:0 2px">$1</mark>');
    node.parentNode.replaceChild(span, node);
  });
}"""

JS_DETECT_OVERLAY = """() => {
  const out = [];
  document.querySelectorAll('*').forEach(el => {
    const cs = getComputedStyle(el);
    if (!['fixed','absolute'].includes(cs.position)) return;
    if (cs.display === 'none' || cs.visibility === 'hidden' || cs.opacity === '0') return;
    const r = el.getBoundingClientRect();
    if (r.width < innerWidth * 0.3 || r.height < 200) return;
    if ((parseInt(cs.zIndex) || 0) < 5) return;
    out.push({desc: el.tagName.toLowerCase()
                    + (el.id ? '#' + el.id : '')
                    + (el.className ? '.' + el.className.toString().trim()
                                        .split(/\\s+/).slice(0,3).join('.') : ''),
              z: parseInt(cs.zIndex) || 0});
  });
  return out;
}"""

JS_RESULT = """([risk, score, color]) => {
  const d = document.createElement('div');
  d.id = '__gate_result';
  d.textContent = risk + '  -  Skor ' + score;
  Object.assign(d.style, {position:'fixed', top:'16px', right:'16px',
    zIndex:'2147483647', background:color, color:'#fff',
    font:'700 18px system-ui', padding:'12px 20px', borderRadius:'8px',
    boxShadow:'0 6px 24px rgba(0,0,0,.4)'});
  document.documentElement.appendChild(d);
}"""


async def safe_eval(page, js, arg=None):
    try:
        if arg is None:
            return await page.evaluate(js)
        return await page.evaluate(js, arg)
    except Exception:
        return None


async def shot_bytes(page):
    try:
        buf = await page.screenshot(full_page=False)
        return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


async def handle_popup(page):
    ov_before = await safe_eval(page, JS_DETECT_OVERLAY) or []

    vp = page.viewport_size
    width = vp['width'] if vp else 1366
    points = [(50, 50), (width - 50, 50)]

    try:
        await page.mouse.click(50, 50)
        await asyncio.sleep(0.2)
        await page.mouse.click(width - 50, 50)
    except Exception:
        pass

    await asyncio.sleep(SLEEP_SETTLE)
    ov_after = await safe_eval(page, JS_DETECT_OVERLAY) or []

    robust_used = False
    if CFG.popup_robust and not ov_after:
        deadline = asyncio.get_event_loop().time() + 12
        while asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.5)
            if await safe_eval(page, JS_DETECT_OVERLAY):
                robust_used = True
                try:
                    await page.keyboard.press('Escape')
                except Exception:
                    pass
                await asyncio.sleep(SLEEP_SETTLE)
                ov_after = await safe_eval(page, JS_DETECT_OVERLAY) or []
                break

    return ov_before, ov_after, points, robust_used


def process_image(img_path, filename_base):
    img = cv2.imread(img_path)
    if img is None:
        return [], None, None

    img_h, img_w = img.shape[:2]
    results = gate.model.predict(img, conf=0.25, verbose=False)

    detections = []
    annotated = img.copy()
    for box in results[0].boxes:
        label = results[0].names[int(box.cls[0])]
        conf = float(box.conf[0])
        if label not in gate.VISUAL_WEIGHTS:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        detections.append({'label': label, 'conf': conf, 'box': (x1, y1, x2, y2)})
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1 - 10),
                    F, 0.5, (0, 0, 255), 2)

    cv2.imwrite(os.path.join(DIR_DATASET, 'images_clean', f"{filename_base}.jpg"), img)
    cv2.imwrite(os.path.join(DIR_DATASET, 'images_verification', f"{filename_base}.jpg"), annotated)

    with open(os.path.join(DIR_DATASET, 'labels', f"{filename_base}.txt"), 'w') as f:
        for det in detections:
            x1, y1, x2, y2 = det['box']
            cls_id = gate.NAME_TO_ID.get(det['label'], 0)
            xc, yc = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            w, h = (x2 - x1), (y2 - y1)
            f.write(f"{cls_id} {xc/img_w:.6f} {yc/img_h:.6f} {w/img_w:.6f} {h/img_h:.6f}\n")

    return detections, annotated, img


async def scan_url(context, url, idx, total, results):
    full_url = url if url.startswith('http') else f'https://{url}'
    safe_name = url.replace('https://', '').replace('http://', '').replace('/', '_')[:40]
    temp_path = os.path.join(BASE_DIR, f"{safe_name}_demo_temp.jpg")

    print(f"[{idx}/{total}] MEMBUKA  {full_url}", flush=True)
    play('open')
    page = await context.new_page()

    try:
        await page.goto(full_url, timeout=30000, wait_until='domcontentloaded')
        await asyncio.sleep(5)

        png_before = await shot_bytes(page)
        ov_before, ov_after, points, robust_used = await handle_popup(page)
        png_after = await shot_bytes(page)
        if png_before is not None and png_after is not None:
            b_shot, pscale = fit_shot(png_before)
            a_shot, _ = fit_shot(png_after)
            if stage_popup(full_url, b_shot, a_shot, ov_before, ov_after,
                           points, pscale, robust_used):
                return 'stop'

        page_text = await page.evaluate(
            "document.body.innerText + ' ' + "
            "Array.from(document.images).map(i=>i.src).join(' ')")

        await safe_eval(page, JS_BANNER)
        await asyncio.sleep(0.7 * CFG.speed)
        await safe_eval(page, JS_CLEAN)

        await page.screenshot(path=temp_path, full_page=False)
        play('shutter')
        print(f"[{idx}/{total}] SCREENSHOT diambil", flush=True)
        await safe_eval(page, JS_FLASH)

        detections, annotated, raw = process_image(temp_path, safe_name)
        if raw is None:
            raise RuntimeError('screenshot tidak terbaca')

        bd = score_breakdown(detections, page_text)
        risk, score = verify_breakdown(bd, detections, page_text)

        shot, scale = fit_shot(raw)
        if stage_capture(full_url, shot):
            return 'stop'
        if stage_scan(full_url, shot):
            return 'stop'
        stop, marked = stage_visual(full_url, shot, bd, scale)
        if stop:
            return 'stop'

        if bd['text_hits']:
            await safe_eval(page, JS_HIGHLIGHT, [h['word'] for h in bd['text_hits']])
        if stage_text(full_url, marked, bd, page_text):
            return 'stop'
        if stage_fusion(full_url, marked, bd):
            return 'stop'

        cv2.imwrite(os.path.join(DIR_EVIDENCE, risk, f"{safe_name}.jpg"), annotated)
        await safe_eval(page, JS_RESULT,
                        [risk.upper(), score,
                         {'high': '#c0182b', 'medium': '#d97706', 'low': '#15803d'}[risk]])
        await asyncio.sleep(0.8 * CFG.speed)

        print(f"[{idx}/{total}] SELESAI  {url} => {risk.upper()} (Skor: {score} | "
              f"visual {bd['norm_visual']:.2f}, teks {bd['norm_text']:.2f} | "
              f"{len(detections)} objek, {len(bd['text_hits'])} kata kunci)", flush=True)

        results.append({
            'url': url,
            'kategori': risk.upper(),
            'skor': score,
            'skor_visual': round(bd['norm_visual'], 2),
            'skor_tekstual': round(bd['norm_text'], 2),
            'jumlah_objek': len(detections),
            'jumlah_kata_kunci': len(bd['text_hits']),
            'bukti': os.path.join('evidence_demo', risk, f"{safe_name}.jpg"),
        })

    except AssertionError as e:
        print(f"\n[FATAL] {url}: {e}\n", flush=True)
        return 'stop'
    except Exception as e:
        play('fail')
        print(f"[{idx}/{total}] GAGAL    {url} => {type(e).__name__}", flush=True)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        try:
            await page.close()
        except Exception:
            pass
    return 'ok'


async def run_screening(state):
    """UC2 + UC3 — menjalankan skrining massal secara visual dan mencatat hasil."""
    url_file = state['url_file']
    if not os.path.exists(url_file):
        print(f"[ERROR] Berkas {url_file} tidak ditemukan. Gunakan menu [1] dahulu.")
        return

    with open(url_file) as f:
        urls = [line.strip().split()[0] for line in f if line.strip()]

    urls = urls[CFG.start:]
    if CFG.limit:
        urls = urls[:CFG.limit]
    total = len(urls)
    if total == 0:
        print("[ERROR] Daftar URL kosong.")
        return

    results = []
    print(f"\n[INFO] {total} URL akan diproses dari: {os.path.basename(url_file)}")
    print("[INFO] Output bukti -> evidence_demo/ , dataset -> dataset_demo/")
    print("[INFO] Tekan ESC pada jendela OpenCV untuk menghentikan proses.\n")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, CANVAS_W, CANVAS_H)

    viewport = ({'width': 1366, 'height': 900} if CFG.viewport_demo
                else {'width': 1366, 'height': 2500})

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, args=['--start-maximized'])
        context = await browser.new_context(viewport=viewport, ignore_https_errors=True)
        for i, url in enumerate(urls, 1):
            if await scan_url(context, url, i, total, results) == 'stop':
                print("\n[INFO] Dihentikan oleh pengguna.")
                break
        await browser.close()

    cv2.destroyAllWindows()

    if results:
        with open(RESULTS_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        state['results'] = results
        print(f"\n[INFO] {len(results)} hasil klasifikasi disimpan ke:")
        print(f"       {RESULTS_CSV}")
    print("[INFO] Skrining selesai. Kembali ke menu utama.\n")


# ---------------------------------------------------------------------------
# ANTARMUKA MENU — mengekspos kelima use case dalam satu kali menjalankan skrip
# ---------------------------------------------------------------------------

def _banner():
    active = state.get('url_file', '')
    label = os.path.basename(active) if active and os.path.exists(active) else "(belum dipilih)"
    print("\n" + "=" * 54)
    print("   SISTEM GATE - Gambling Activity Tracing Engine")
    print("=" * 54)
    print(f"   Daftar URL aktif : {label}")
    print("-" * 54)
    print("  [1] Siapkan daftar URL target")
    print("  [2] Jalankan skrining massal")
    print("  [3] Lihat hasil klasifikasi risiko")
    print("  [4] Verifikasi bukti tangkapan layar")
    print("  [0] Keluar")
    print("=" * 54)


def uc1_siapkan_daftar():
    """UC1 — memilih berkas daftar URL target."""
    print("\n--- [UC1] Menyiapkan Daftar URL Target ---")
    files = sorted(glob.glob(os.path.join(BASE_DIR, '*.txt')))
    if files:
        print("Berkas .txt yang tersedia di direktori kerja:")
        for i, f in enumerate(files, 1):
            try:
                n = sum(1 for _ in open(f, errors='ignore'))
            except OSError:
                n = '?'
            print(f"  [{i:2d}] {os.path.basename(f):45s} ({n} baris)")
    else:
        print("(Tidak ada berkas .txt di direktori kerja.)")
    print("  [P] Tempel path berkas secara manual")
    print("  [B] Kembali")
    choice = input("Pilih berkas: ").strip()

    low = choice.lower()
    if low == 'b':
        return
    if low == 'p':
        path = input("Masukkan path berkas .txt: ").strip().strip('"\'')
        if os.path.exists(path):
            state['url_file'] = os.path.abspath(path)
            print(f"[OK] Daftar URL diset ke: {state['url_file']}")
        else:
            print("[ERROR] Berkas tidak ditemukan.")
        return
    if choice.isdigit() and 1 <= int(choice) <= len(files):
        state['url_file'] = files[int(choice) - 1]
        print(f"[OK] Daftar URL diset ke: {os.path.basename(state['url_file'])}")
    else:
        print("[ERROR] Pilihan tidak valid.")


def uc4_lihat_hasil():
    """UC4 — menampilkan ringkasan klasifikasi dan lokasi berkas CSV."""
    print("\n--- [UC4] Hasil Klasifikasi Risiko ---")
    if not os.path.exists(RESULTS_CSV):
        print("[INFO] Belum ada hasil. Jalankan skrining (menu [2]) terlebih dahulu.")
        return
    rows = list(csv.DictReader(open(RESULTS_CSV)))
    if not rows:
        print("[INFO] Berkas hasil kosong.")
        return
    counts = {'HIGH': [], 'MEDIUM': [], 'LOW': []}
    for r in rows:
        counts.setdefault(r['kategori'], []).append(r)
    print(f"Total situs terklasifikasi : {len(rows)}")
    for cat in ('HIGH', 'MEDIUM', 'LOW'):
        items = counts.get(cat, [])
        print(f"\n  {cat:6s} : {len(items)} situs")
        for r in sorted(items, key=lambda x: float(x['skor']), reverse=True)[:5]:
            print(f"      - {r['url']:40s} (skor {r['skor']})")
        if len(items) > 5:
            print(f"      ... dan {len(items) - 5} situs lainnya")
    print("\n[INFO] Laporan lengkap tersimpan pada berkas CSV:")
    print(f"       {RESULTS_CSV}")


def uc5_verifikasi_bukti():
    """Verifikasi bukti — menampilkan path bukti yang dapat diklik untuk preview."""
    print("\n--- Verifikasi Bukti Tangkapan Layar ---")
    cats = {'1': 'high', '2': 'medium', '3': 'low'}
    print("  [1] Kategori HIGH")
    print("  [2] Kategori MEDIUM")
    print("  [3] Kategori LOW")
    print("  [B] Kembali")
    choice = input("Pilih kategori: ").strip().lower()
    if choice not in cats:
        return
    cat = cats[choice]
    folder = os.path.join(DIR_EVIDENCE, cat)
    imgs = sorted(glob.glob(os.path.join(folder, '*.jpg')))
    if not imgs:
        print(f"[INFO] Belum ada bukti pada kategori {cat.upper()}. Jalankan skrining dahulu.")
        return
    print(f"\n[INFO] {len(imgs)} bukti kategori {cat.upper()} "
          "(klik path untuk membuka & preview):\n")
    for i, path in enumerate(imgs, 1):
        link = 'file://' + os.path.abspath(path)
        print(f"  {i:3d}. {link}")
    print(f"\n[INFO] Folder bukti: {os.path.abspath(folder)}")


def menu_loop():
    while True:
        _banner()
        choice = input("Pilih menu: ").strip()
        if choice == '1':
            uc1_siapkan_daftar()
        elif choice == '2':
            asyncio.run(run_screening(state))
        elif choice == '3':
            uc4_lihat_hasil()
        elif choice == '4':
            uc5_verifikasi_bukti()
        elif choice == '0':
            print("\n[INFO] Terima kasih telah menggunakan Sistem GATE.")
            break
        else:
            print("[ERROR] Menu tidak dikenal.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Sistem GATE - demo visual pipeline screening')
    ap.add_argument('--limit', type=int, default=0, help='batasi jumlah URL (0 = semua)')
    ap.add_argument('--start', type=int, default=0, help='mulai dari URL ke-N')
    ap.add_argument('--fast', action='store_true', help='percepat animasi 3x')
    ap.add_argument('--no-sound', dest='sound', action='store_false', help='matikan efek suara')
    ap.add_argument('--viewport-demo', action='store_true',
                    help='viewport 1366x900 (seukuran layar) alih-alih 1366x2500 seperti pipeline')
    ap.add_argument('--popup-robust', action='store_true',
                    help='tambahan DI LUAR laporan: tunggu popup muncul lalu tutup dengan Escape')
    CFG = ap.parse_args()
    CFG.speed = 0.33 if CFG.fast else 1.0

    if sys.platform != 'darwin':
        CFG.sound = False

    state = {'url_file': INPUT_FILE, 'results': []}
    menu_loop()
