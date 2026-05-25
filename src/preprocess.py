import sys 
import pandas as pd
import os
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , LabelEncoder
from pathlib import Path

params = yaml.safe_load(open("params.yaml"))["preprocess"]

input_path = sys.argv[1]
output_path = Path(sys.argv[2])

output_path.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_path)

cat = df.select_dtypes(include="object").columns
num = df.select_dtypes(exclude="object").columns
num = num.drop(columns = params["target"], errors="ignore")

#  converting the categorical features to numerical 
for col in cat:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# converting the numerical features to standard scale 
scaler = StandardScaler()
df[num] = scaler.fit_transform(df[num])


df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

train_df, test_df = train_test_split(
    df,
    test_size=params["test_size"],
    random_state=params["random_state"],
)

train_df.to_csv(output_path / "train.csv", index=False)
test_df.to_csv(output_path / "test.csv", index=False)


print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")