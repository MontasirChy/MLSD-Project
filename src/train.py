import sys 
import pandas as pd
import os
import yaml
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib


params = yaml.safe_load(open("params.yaml"))["train"]

train_csv = sys.argv[1]
model_out = Path(sys.argv[2])
model_out.parent.mkdir(parents=True, exist_ok=True)


train_df = pd.read_csv(train_csv)

X_train = train_df.drop(columns=params["target"])
y_train = train_df[params["target"]]

model = XGBRegressor(
    n_estimators=params["n_estimators"], 
    random_state=params["random_state"])
model.fit(X_train, y_train)


joblib.dump(model, model_out)
print(f"Saved model -> {model_out}")