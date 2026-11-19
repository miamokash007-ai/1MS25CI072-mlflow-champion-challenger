"""Load the Champion/Challenger model from the MLflow Model Registry and run inference.

Evaluates predictions on novel sentences from data/new_data.csv using pure MLflow pyfunc.
Usage:
    python predict_from_mlflow.py                    # Evaluates @champion
    python predict_from_mlflow.py --alias challenger # Evaluates @challenger
Student: M K Mohammed Kashif | USN: 1MS25CI072
"""
import argparse
import os
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
REGISTERED_NAME = "sentiment-model"

parser = argparse.ArgumentParser(description="Predict sentiment using registered MLflow model aliases.")
parser.add_argument("--alias", default="champion", choices=["champion", "challenger"], help="Model alias to load")
args = parser.parse_args()

mlflow.set_tracking_uri(TRACKING_URI)
model_uri = f"models:/{REGISTERED_NAME}@{args.alias}"

client = MlflowClient()
mv = client.get_model_version_by_alias(REGISTERED_NAME, args.alias)
model = mlflow.pyfunc.load_model(model_uri)

new_data_path = Path("data/new_data.csv")
if not new_data_path.exists():
    raise SystemExit(f"Input file '{new_data_path}' not found.")

new_data = pd.read_csv(new_data_path)
pred = model.predict(new_data[["text"]])

out = pd.DataFrame(
    {
        "text": new_data["text"],
        "prediction": pred["prediction"],
        "score": pred["score"],
        "expected": new_data["expected"],
    }
)
out["match"] = out["prediction"] == out["expected"]

Path("outputs").mkdir(exist_ok=True)
out_file = "outputs/new_data_predictions.csv" if args.alias == "champion" else f"outputs/new_data_predictions_{args.alias}.csv"
out.to_csv(out_file, index=False)

show = out.copy()
show["text"] = show["text"].str.slice(0, 58)
lines = [
    f"Model URI          : {model_uri}",
    f"Registered Version : {mv.version}",
    f"Hugging Face Model : {mv.tags.get('hf_model_id')}",
    f"Source Run         : {mv.tags.get('source_run_name')} (run_id {mv.run_id})",
    "",
    show[["text", "prediction", "score", "expected"]].to_string(index=False),
    "",
    f"{int(out['match'].sum())}/{len(out)} predictions match the expected sentiment labels",
    f"Saved output to {out_file}",
]
text = "\n".join(lines)
print(text)

command = "python predict_from_mlflow.py" + ("" if args.alias == "champion" else f" --alias {args.alias}")
term_file = out_file.replace(".csv", "_terminal.txt")
Path(term_file).write_text(f"$ {command}\n{text}\n", encoding="utf-8")
