#!/usr/bin/env bash
# Quickstart script for Groww Mutual Fund FAQ Assistant

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

if [ ! -d "venv" ]; then
    echo "[*] Creating virtual environment..."
    python3 -m venv venv
    ./venv/bin/pip install --upgrade pip
    ./venv/bin/pip install -r requirements.txt
fi

if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

echo "[+] Starting Groww Mutual Fund FAQ Assistant..."
./venv/bin/streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true
