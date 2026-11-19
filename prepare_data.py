"""Create data/eval_inputs.csv: a fixed, seeded subset of the SST-2 validation set.

Both Champion and Challenger models are evaluated on this identical dataset.
Student: M K Mohammed Kashif | USN: 1MS25CI072
"""
from pathlib import Path
from datasets import load_dataset

SEED = 42
N_SAMPLES = 200
OUT = Path("data/eval_inputs.csv")

if not OUT.exists():
    print(f"Loading SST-2 validation set (seed={SEED}, n={N_SAMPLES})...")
    ds = load_dataset("stanfordnlp/sst2", split="validation")
    subset = ds.shuffle(seed=SEED).select(range(N_SAMPLES)).to_pandas()

    df = subset.rename(columns={"idx": "sst2_idx", "sentence": "text"})
    df["reference"] = df["label"].map({0: "NEGATIVE", 1: "POSITIVE"})
    df = df[["sst2_idx", "text", "reference"]]
    df["text"] = df["text"].str.strip()

    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Successfully saved {len(df)} rows to {OUT}")
else:
    print(f"Using existing dataset {OUT}")
