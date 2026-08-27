"""
Evaluasi cascade IndoBERT pada zona ambigu (MEDIUM) sistem GATE.

Menerapkan aturan PRE-REGISTERED (aturan_cascade.md):
  - Zona = kategori MEDIUM saja.
  - p = P(promosi judol | teks) dari IndoBERT (indobert_gate/).
  - p >= 0,5 -> HIGH (terdeteksi); p < 0,5 -> LOW (tidak terdeteksi).
  - MEDIUM tak-terakses saat re-scrape -> tetap MEDIUM ("tidak dapat diverifikasi"),
    dihitung positif (sesuai reduksi biner 4.3.6).
  - Sensitivitas dilaporkan pada 0,3 dan 0,7; ambang utama tetap 0,5.

Ground truth (independen dari skor sistem):
  judol aktif = positif; non-judol kontrol = negatif.
Reduksi biner: HIGH & MEDIUM = terdeteksi (positif); LOW = negatif.

Output: laporan_validasi/hasil_cascade.json + ringkasan ke stdout.
"""
import os
import json
import torch
import warnings
from collections import Counter

warnings.filterwarnings('ignore')
BASE = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
MODEL_DIR = os.path.join(BASE, 'indobert_gate')
EVAL_MEDIUM = os.path.join(BASE, 'teks_dataset', 'eval_medium.jsonl')
SCAN_LOG = os.path.join(BASE, 'scan_log.json')
OUT = os.path.join(BASE, 'laporan_validasi', 'hasil_cascade.json')
MAX_LENGTH = 512


def load_urls(fname):
    with open(os.path.join(BASE, fname)) as f:
        return set(l.strip().split()[0] for l in f if l.strip())


def metrics(tp, fn, fp, tn):
    acc = (tp + tn) / (tp + tn + fp + fn) * 100
    prec = tp / (tp + fp) * 100 if (tp + fp) else 0.0
    rec = tp / (tp + fn) * 100 if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn,
            'accuracy': round(acc, 2), 'precision': round(prec, 2),
            'recall': round(rec, 2), 'f1': round(f1, 2)}


def main():
    judol = load_urls('dataset_judol_aktif.txt')
    non = load_urls('dataset_nonjudol_300.txt')

    # --- baseline dari scan_log (kondisi 4.3.6) ---
    with open(SCAN_LOG, encoding='utf-8') as f:
        by_url = {e['url']: e for e in json.load(f)}

    def baseline_risk(u):
        e = by_url.get(u)
        if e and e.get('score_data'):
            return e['score_data']['risk_level']       # HIGH/MEDIUM/LOW
        return 'INACCESSIBLE'                            # non-judol tak terakses dsb.

    base = {u: baseline_risk(u) for u in (judol | non)}

    def confusion(risk_map):
        tp = sum(1 for u in judol if risk_map[u] in ('HIGH', 'MEDIUM'))
        fn = sum(1 for u in judol if risk_map[u] not in ('HIGH', 'MEDIUM'))
        fp = sum(1 for u in non if risk_map[u] in ('HIGH', 'MEDIUM'))
        tn = sum(1 for u in non if risk_map[u] not in ('HIGH', 'MEDIUM'))
        return tp, fn, fp, tn

    tp0, fn0, fp0, tn0 = confusion(base)
    print('=== BASELINE (sebelum cascade) ===')
    print(' ', metrics(tp0, fn0, fp0, tn0))

    # --- muat teks zona MEDIUM ---
    rows = [json.loads(l) for l in open(EVAL_MEDIUM, encoding='utf-8') if l.strip()]
    with_text = [r for r in rows if r.get('text')]
    no_text = [r for r in rows if not r.get('text')]
    print(f'\nZona MEDIUM: {len(rows)} URL | terverifikasi(teks): {len(with_text)} | '
          f'tak-terakses: {len(no_text)}')

    # --- inferensi IndoBERT ---
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    torch.set_num_threads(os.cpu_count() or 8)

    probs = {}
    print('inferensi IndoBERT pada zona ambigu ...', flush=True)
    with torch.no_grad():
        for i, r in enumerate(with_text):
            enc = tok(r['text'], truncation=True, max_length=MAX_LENGTH,
                      padding='max_length', return_tensors='pt')
            logits = model(**enc).logits
            probs[r['url']] = torch.softmax(logits, dim=-1)[0, 1].item()
            if (i + 1) % 25 == 0:
                print(f'  {i+1}/{len(with_text)}', flush=True)

    # --- terapkan aturan cascade utk beberapa ambang ---
    results = {'baseline': metrics(tp0, fn0, fp0, tn0),
               'zona': {'total': len(rows), 'terverifikasi': len(with_text),
                        'tak_terakses': len(no_text)},
               'per_ambang': {}}

    for thr in (0.5, 0.3, 0.7):
        risk = dict(base)
        flips = {'MEDIUM->HIGH': 0, 'MEDIUM->LOW': 0}
        flip_correct = 0
        flip_wrong = 0
        for r in with_text:
            u = r['url']
            new = 'HIGH' if probs[u] >= thr else 'LOW'
            if new != 'MEDIUM':
                # semua baseline di zona ini = MEDIUM
                key = f'MEDIUM->{new}'
                flips[key] += 1
                is_judol = u in judol
                # perubahan benar bila: judol->HIGH (tetap terdeteksi benar) atau
                # non->LOW (dibuang benar). judol->LOW atau non->HIGH = salah.
                correct = (is_judol and new == 'HIGH') or ((not is_judol) and new == 'LOW')
                flip_correct += int(correct)
                flip_wrong += int(not correct)
            risk[u] = new
        # URL tak-terakses tetap MEDIUM (sudah = base, tidak diubah)
        tp, fn, fp, tn = confusion(risk)
        results['per_ambang'][str(thr)] = {
            'metrics': metrics(tp, fn, fp, tn),
            'naik_HIGH': flips['MEDIUM->HIGH'],
            'turun_LOW': flips['MEDIUM->LOW'],
            'perubahan_benar': flip_correct,
            'perubahan_salah': flip_wrong,
        }

    # --- ringkasan zona pada ambang utama 0,5 ---
    thr = 0.5
    zj = [r for r in with_text if r['url'] in judol]
    zn = [r for r in with_text if r['url'] not in judol]
    zj_high = sum(1 for r in zj if probs[r['url']] >= thr)
    zn_low = sum(1 for r in zn if probs[r['url']] < thr)
    results['zona_detail_0.5'] = {
        'judol_terverifikasi': len(zj), 'judol_dipertahankan_HIGH': zj_high,
        'judol_jadi_FN': len(zj) - zj_high,
        'non_terverifikasi': len(zn), 'non_dibuang_ke_LOW': zn_low,
        'non_tetap_FP': len(zn) - zn_low,
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # --- cetak ringkas ---
    print('\n=== HASIL CASCADE ===')
    for thr in ('0.5', '0.3', '0.7'):
        d = results['per_ambang'][thr]
        m = d['metrics']
        tag = ' (UTAMA)' if thr == '0.5' else ''
        print(f"\nAmbang {thr}{tag}: naik->HIGH {d['naik_HIGH']}, turun->LOW {d['turun_LOW']}, "
              f"benar {d['perubahan_benar']}/{d['perubahan_benar']+d['perubahan_salah']}")
        print(f"  {m}")
    print('\nDetail zona (ambang 0,5):', results['zona_detail_0.5'])
    print(f'\nDisimpan ke {OUT}')


if __name__ == '__main__':
    main()
