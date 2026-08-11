"""Generate COMP662 class predictions from a saved model and a CSV file."""
import sys
import joblib
import pandas as pd


if len(sys.argv) != 3:
    raise SystemExit("Usage: python predict.py input.csv output.csv")
bundle = joblib.load("models/1173808_Assignment1_final.joblib")
data = pd.read_csv(sys.argv[1])
missing = set(bundle["features"]) - set(data.columns)
if missing:
    raise SystemExit(f"Missing columns: {', '.join(sorted(missing))}")
pd.DataFrame({"Class": bundle["model"].predict(data[bundle["features"]])}).to_csv(sys.argv[2], index=False)
