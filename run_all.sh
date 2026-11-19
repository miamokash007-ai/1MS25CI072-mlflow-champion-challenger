#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

if ! curl -s -f http://127.0.0.1:5000 > /dev/null; then
    mlflow server --host 127.0.0.1 --port 5000 &
    sleep 5
fi

python prepare_data.py
python evaluate.py
python register_champion.py
python predict_from_mlflow.py
python predict_from_mlflow.py --alias challenger
python capture_screenshots.py
python write_report.py

echo "Pipeline execution finished successfully."
