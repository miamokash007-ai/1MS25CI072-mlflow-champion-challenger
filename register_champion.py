"""Register Model A and Model B under one registered model and assign aliases.

The champion decision is resolved dynamically from the latest compare_sentiment run,
ensuring reproducibility without hardcoding run IDs.
Student: M K Mohammed Kashif | USN: 1MS25CI072
"""
import os
import mlflow
from mlflow import MlflowClient

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT = "hf-sentiment-champion-challenger"
REGISTERED_NAME = "sentiment-model"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

exp = client.get_experiment_by_name(EXPERIMENT)
if exp is None:
    raise SystemExit(f"Experiment '{EXPERIMENT}' not found. Run evaluate.py first.")

summary_runs = client.search_runs(
    [exp.experiment_id],
    filter_string="attributes.run_name = 'compare_sentiment' AND attributes.status = 'FINISHED'",
    order_by=["attributes.start_time DESC"],
    max_results=1,
)
if not summary_runs:
    raise SystemExit("No completed 'compare_sentiment' summary run found. Run evaluate.py first.")

summary = summary_runs[0]
winner = summary.data.tags["winner_letter"]
print(f"Summary Run ID : {summary.info.run_id}")
print(f"Decision       : {summary.data.tags['winner']}")
print(f"Reason         : {summary.data.tags['decision_reason']}\n")

children = client.search_runs(
    [exp.experiment_id],
    filter_string=f"tags.mlflow.parentRunId = '{summary.info.run_id}'"
)
by_letter = {r.data.tags["model_letter"]: r for r in children}

versions = {}
for letter in ("A", "B"):
    run = by_letter[letter]
    mv = mlflow.register_model(run.data.tags["model_uri"], REGISTERED_NAME)
    tag_data = {
        "hf_model_id": run.data.params["hf_model_id"],
        "model_letter": letter,
        "source_run_name": run.info.run_name,
        "macro_f1": f"{run.data.metrics['macro_f1']:.4f}",
        "accuracy": f"{run.data.metrics['accuracy']:.4f}",
        "avg_latency_ms": f"{run.data.metrics['avg_latency_ms']:.1f}",
        "student_usn": "1MS25CI072",
        "student_name": "M K Mohammed Kashif",
    }
    for key, value in tag_data.items():
        client.set_model_version_tag(REGISTERED_NAME, mv.version, key, value)
    versions[letter] = mv.version
    print(f"Registered {REGISTERED_NAME} v{mv.version} <- {run.info.run_name} ({run.data.params['hf_model_id']})")

loser = "B" if winner == "A" else "A"
client.set_registered_model_alias(REGISTERED_NAME, "champion", versions[winner])
client.set_registered_model_alias(REGISTERED_NAME, "challenger", versions[loser])
client.update_registered_model(
    REGISTERED_NAME,
    description="Sentiment analysis model (SST-2). @champion assigned based on champion_rule.json evaluation; @challenger is alternative candidate.",
)

print(f"\nModel Registry Aliases Assigned:")
print(f"  @champion   -> Version {versions[winner]} (Model {winner})")
print(f"  @challenger -> Version {versions[loser]} (Model {loser})")
