#!/usr/bin/env bash
# Setup environment Sistem GATE di PC baru.
# Pakai:  bash setup.sh
set -e

PYTHON=${PYTHON:-python3}

echo "==> Python yang dipakai: $($PYTHON --version)"
echo "    (disarankan Python 3.10.x — dikembangkan di 3.10.5)"

echo "==> Membuat virtual environment .venv"
$PYTHON -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrade pip"
pip install --upgrade pip

echo "==> Install dependency dari requirements.txt"
pip install -r requirements.txt

echo "==> Download browser Chromium untuk Playwright"
python -m playwright install chromium
# Di Linux, lengkapi dependency sistem dengan:
#   python -m playwright install-deps chromium

echo
echo "==> Selesai. Cek model:"
[ -f best-4.pt ] && echo "    best-4.pt   OK" || echo "    best-4.pt   HILANG — salin manual (±50 MB)"
[ -d indobert_gate ] && echo "    indobert_gate/  OK" || echo "    indobert_gate/  tidak ada — hanya perlu utk eksperimen cascade IndoBERT"
echo
echo "Aktifkan lagi environment nanti dengan:  source .venv/bin/activate"
