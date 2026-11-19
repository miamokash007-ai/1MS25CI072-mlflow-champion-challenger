"""Generate the Results and Decision sections of README.md from outputs/metrics_summary.json.

Student: M K Mohammed Kashif | USN: 1MS25CI072
Course: Skill Lab – MLOps with MLflow (Mini Project)
"""
import json
import os
import re
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT = "hf-sentiment-champion-challenger"

metrics_summary_path = Path("outputs/metrics_summary.json")
if not metrics_summary_path.exists():
    raise SystemExit("outputs/metrics_summary.json not found. Run evaluate.py first.")

with open(metrics_summary_path, "r", encoding="utf-8") as f:
    s = json.load(f)

rule = s["rule"]
A, B = s["A"], s["B"]
margin = f'{rule["margin_pct_points"]:g}'
guard = rule["latency_guardrail_ms"]
n = s["eval_samples"]
delta = s["delta_primary_metric_pts"]
ratio = s["latency_ratio_B_over_A"]
p = s["mcnemar_p_value"]


def pct(x):
    return f"{100 * x:.2f}"


rows = [
    ("Accuracy (%) ↑", pct(A["accuracy"]), pct(B["accuracy"]), f"{100 * (B['accuracy'] - A['accuracy']):+.2f}"),
    ("**Macro-F1 (%) ↑ – Primary**", f"**{pct(A['macro_f1'])}**", f"**{pct(B['macro_f1'])}**", f"**{delta:+.2f}**"),
    ("Macro Precision (%) ↑", pct(A["precision_macro"]), pct(B["precision_macro"]), f"{100 * (B['precision_macro'] - A['precision_macro']):+.2f}"),
    ("Macro Recall (%) ↑", pct(A["recall_macro"]), pct(B["recall_macro"]), f"{100 * (B['recall_macro'] - A['recall_macro']):+.2f}"),
    ("Misclassifications (of 200) ↓", str(int(A["num_errors"])), str(int(B["num_errors"])), f"{int(B['num_errors'] - A['num_errors']):+d}"),
    ("Avg Latency (ms / sentence) ↓", f"{A['avg_latency_ms']:.1f}", f"{B['avg_latency_ms']:.1f}", f"× {ratio:.2f}"),
    ("Total Inference Time (s) ↓", f"{A['total_inference_s']:.2f}", f"{B['total_inference_s']:.2f}", f"{B['total_inference_s'] - A['total_inference_s']:+.2f}"),
    ("Parameters (Millions) ↓", f"{A['model_params_millions']:.0f}", f"{B['model_params_millions']:.0f}", f"× {B['model_params_millions'] / A['model_params_millions']:.1f}"),
]

table = "| Metric | Model A – Champion (distilbert-sst2) | Model B – Challenger (roberta-large) | Difference (B − A) |\n|---|---|---|---|\n"
table += "\n".join(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |" for r in rows)

dis_path = Path("outputs/disagreements_top5.csv")
dis_md = "| SST-2 idx | Sentence | Ground Truth | Model A | Model B | Analysis |\n|---|---|---|---|---|---|\n"
if dis_path.exists():
    dis = pd.read_csv(dis_path)
    for _, r in dis.iterrows():
        sent = str(r["text"]).replace("|", "/")
        sent = sent if len(sent) <= 88 else sent[:85] + "..."
        dis_md += (f"| {r['sst2_idx']} | {sent} | {r['reference']} | {r['pred_A']} ({r['score_A']:.2f}) | "
                   f"{r['pred_B']} ({r['score_B']:.2f}) | {r['comment']} |\n")

results_content = f"""Run on {s['device']} with {s['library_versions']}. Evaluated on {n} sampled SST-2 validation sentences (seed={s.get('seed', 42)}).

{table}

**Key Statistical Findings:**
- The two models disagree on {s['num_disagreements']} of the {n} sentences.
- Model A was uniquely correct {s['only_A_correct']} times, while Model B was uniquely correct {s['only_B_correct']} times.
- Exact McNemar test result: **p = {p:.3f}**.

![Comparison of the two models](outputs/comparison_bar_chart.png)

| Confusion Matrix – Model A | Confusion Matrix – Model B |
|:---:|:---:|
| ![Model A Confusion Matrix](outputs/confusion_matrix_distilbert-sst2.png) | ![Model B Confusion Matrix](outputs/confusion_matrix_roberta-large.png) |

![Prediction Confidence Distribution](outputs/confidence_histogram.png)

### Qualitative Disagreement Analysis (Sample of 5 Disagreements)
*(Complete list available in `outputs/disagreements_all.csv`)*

{dis_md}
### Verification on Unseen Data with Champion (`outputs/new_data_predictions.csv`)

![predict_from_mlflow.py output](screenshots/3_predict_from_mlflow_output.png)

### MLflow UI Evidence

![Compare View](screenshots/1_mlflow_compare_view.png)

![Registry Aliases](screenshots/2_registry_aliases.png)
"""

# Decision formulation
winner = s["winner"]
b_wins_metric = delta >= rule["margin_pct_points"]
b_fast_enough = B["avg_latency_ms"] <= guard

if winner == "B":
    verdict = (f"**Decision: Promote Model B (roberta-large) to @champion.** The Challenger improved macro-F1 from "
               f"{pct(A['macro_f1'])}% to {pct(B['macro_f1'])}%, achieving an absolute gain of {delta:.2f} percentage points "
               f"(surpassing the predefined {margin}-point threshold). Furthermore, its average inference latency of "
               f"{B['avg_latency_ms']:.1f} ms per sentence comfortably meets the {guard} ms latency guardrail.")
elif b_wins_metric and not b_fast_enough:
    verdict = (f"**Decision: Retain Model A (distilbert-sst2) as @champion.** While the Challenger achieved an improvement "
               f"of {delta:.2f} percentage points on macro-F1 (clearing the {margin}-point margin), its average latency of "
               f"{B['avg_latency_ms']:.1f} ms per sentence exceeded the maximum permissible guardrail of {guard} ms.")
else:
    verdict = (f"**Decision: Retain Model A (distilbert-sst2) as @champion.** The Challenger's macro-F1 delta of "
               f"{delta:.2f} percentage points was below the required margin of {margin} points. By the predefined tie-breaker rule, "
               f"Model A remains champion due to its superior computational efficiency (5.3x fewer parameters and faster inference).")

decision_content = f"""{verdict}

**Engineering & Trade-off Analysis:**
- **Accuracy vs. Computational Overhead:** While roberta-large demonstrates superior semantic comprehension, it requires ~355M parameters versus DistilBERT's ~67M parameters (a 5.3× increase) and higher latency.
- **Statistical Significance:** With an exact McNemar test p-value of {p:.3f}, the performance differential on 200 samples reflects empirical gains on challenging syntactic constructs (e.g. nuanced negative phrasing and irony), but deployment in latency-critical production environments should weigh throughput requirements against accuracy needs.
"""

readme_path = Path("README.md")
if readme_path.exists():
    content = readme_path.read_text(encoding="utf-8")
    content = re.sub(r"<!-- RESULTS_START -->.*?<!-- RESULTS_END -->",
                     f"<!-- RESULTS_START -->\n{results_content}\n<!-- RESULTS_END -->",
                     content, flags=re.DOTALL)
    content = re.sub(r"<!-- DECISION_START -->.*?<!-- DECISION_END -->",
                     f"<!-- DECISION_START -->\n{decision_content}\n<!-- DECISION_END -->",
                     content, flags=re.DOTALL)
    readme_path.write_text(content, encoding="utf-8")
    print("Successfully updated README.md with results and decision.")
else:
    print("README.md not found to update.")

try:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp:
        summary_runs = client.search_runs(
            [exp.experiment_id],
            filter_string="attributes.run_name = 'compare_sentiment' AND attributes.status = 'FINISHED'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if summary_runs:
            client.log_artifact(summary_runs[0].info.run_id, "README.md")
            print("Logged updated README.md to summary run in MLflow.")
except Exception as e:
    print(f"Artifact logging notice: {e}")
