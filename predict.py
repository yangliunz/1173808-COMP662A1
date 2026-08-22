"""Generate COMP662 class predictions from a saved model pipeline and a CSV file."""
from pathlib import Path
import sys
import joblib
import pandas as pd


if len(sys.argv) != 3:
    raise SystemExit("Usage: python predict.py input.csv output.csv")

model_path = Path(__file__).resolve().parent / "models" / "1173808_Assignment1_final.joblib"
if not model_path.exists():
    raise SystemExit(f"Model file not found at {model_path}")

bundle = joblib.load(model_path)
data = pd.read_csv(sys.argv[1])

missing = set(bundle["features"]) - set(data.columns)
if missing:
    raise SystemExit(f"Missing columns: {', '.join(sorted(missing))}")

# The saved model bundle contains an end-to-end Pipeline (StandardScaler + RandomForestClassifier)
# which automatically and consistently handles feature scaling.
input_features = data[bundle["features"]]
predictions = bundle["model"].predict(input_features)

output_path = Path(sys.argv[2])
output_path.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"Class": predictions}).to_csv(output_path, index=False)
print(f"Successfully generated {len(predictions)} predictions -> {output_path}")
