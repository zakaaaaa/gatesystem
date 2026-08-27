#!/bin/bash
# ============================================================
# Orchestrator OTONOM pembangunan dataset judol aktif + evaluasi.
# - Resumable (aman kalau putus: jalankan lagi, lanjut otomatis)
# - Self-healing (kalau scan crash, ronde berikutnya lanjut via RESUME)
# Jalankan di bawah caffeinate agar tahan sleep. Biarkan LID TERBUKA.
# ============================================================
cd "$(dirname "$0")" || exit 1

TARGET=${TARGET:-2000}
export CONCURRENCY=${CONCURRENCY:-6}
export USE_STEALTH=1
export GEN_DATASET=0          # hemat disk: tak menulis dataset-candidate
export RESUME=1

ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo "===== ORCHESTRATOR MULAI $(ts) | target=$TARGET concurrency=$CONCURRENCY ====="

# Pool awal besar (banyak yang akan mati/terblokir)
python3 siapkan_pool_judol.py 8000

ROUND=1
MAXROUND=10
while true; do
  echo "----- RONDE $ROUND: SCAN POOL $(ts) -----"
  INPUT_FILE=pool_judol.txt python3 screening_gate.py

  echo "----- RONDE $ROUND: SARING AKTIF $(ts) -----"
  if python3 bangun_dataset_judol.py "$TARGET"; then
    echo "===== TARGET $TARGET TERCAPAI $(ts) ====="
    break
  fi

  ROUND=$((ROUND+1))
  if [ "$ROUND" -gt "$MAXROUND" ]; then
    echo "[STOP] $MAXROUND ronde tercapai (CNS mungkin habis). Lanjut dgn yang ada."
    # paksa bangun dgn target = jumlah aktif saat ini bila perlu (manual nanti)
    break
  fi
  echo "----- TAMBAH POOL (ronde berikutnya) $(ts) -----"
  python3 siapkan_pool_judol.py 5000
done

echo "===== SCAN 300 SITUS KONTROL NON-JUDOL $(ts) ====="
INPUT_FILE=dataset_nonjudol_300.txt python3 screening_gate.py

echo "===== HITUNG METRIK EVALUASI $(ts) ====="
python3 evaluasi_sistem.py eval

echo "===== ORCHESTRATOR SELESAI $(ts) ====="
