import sys
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import json

param = yaml.safe_load(open("params.yaml"))["eval"]     

model_path = sys.argv[2]
test_csv = sys.argv[1]
metrics_path = Path(sys.argv[3])
metrics_path.parent.mkdir(parents=True, exist_ok=True)


model = joblib.load(model_path)
test_df = pd.read_csv(test_csv)

X = test_df.drop(columns=param["target"])
y_true = test_df[param["target"]]
y_pred = model.predict(X)

metrics = {
    "mae":  float(mean_absolute_error(y_true, y_pred)),
    "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    "r2":   float(r2_score(y_true, y_pred)),
}

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(json.dumps(metrics, indent=2))