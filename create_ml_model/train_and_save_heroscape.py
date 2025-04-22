#!/usr/bin/env python3
# train_and_save_heroscape_v6.py  –  TF‑IDF + Random Forest
# ---------------------------------------------------------
#   python train_and_save_heroscape_v6.py heroscape_characters.csv
# Produces heroscape_point_model_v6.pkl

import argparse, re, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, r2_score

# ---------- maps & helpers ----------
TYPE_MAP = {
    0: "Unique Hero", 1: "Common Hero",
    2: "Unique Squad", 3: "Common Squad",
    4: "Event Hero"
}

def join_powers(row: pd.Series) -> str:
    return " ".join(str(row[c]) for c in ("SP1","SP2","SP3") if pd.notna(row[c]))

# ---------- CLI ----------
parser = argparse.ArgumentParser(description="HeroScape v6")
parser.add_argument("csv", help="heroscape_characters.csv")
parser.add_argument("--out", default="heroscape_point_model.pkl")
args = parser.parse_args()

# ---------- load & preprocess ----------
df = pd.read_csv(args.csv)
if df["Type"].dtype != object:
    df["Type"] = df["Type"].map(TYPE_MAP)

df["Abilities"] = df.apply(join_powers, axis=1).fillna("")
df["logHeight"] = np.log1p(df["Height"])
df["CostPerFig"] = df["Pts"] / df["NumUnits"]

numeric_cols = ["Life","MVE","RGE","ATK","DEF","logHeight","NumUnits"]
categorical_cols = ["Army","Race","Class","Size","Type"]
text_col = "Abilities"

X = df[numeric_cols + categorical_cols + [text_col]]
y = df["CostPerFig"]

# ---------- column transformer ----------
prep = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("txt", TfidfVectorizer(lowercase=True,
                            stop_words="english",
                            max_features=3000), text_col)
])

# ---------- sparse‑friendly regressor ----------
rf = RandomForestRegressor(
        n_estimators=1000,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42
)

model = Pipeline([
    ("prep", prep),
    ("reg", TransformedTargetRegressor(
               regressor=rf,
               func=np.log1p,
               inverse_func=np.expm1))
])

# ---------- train / evaluate ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
preds = model.predict(X_test)

print("v6 TF‑IDF + RF  —  MAE:", round(mean_absolute_error(y_test, preds),2),
      "|  R²:", round(r2_score(y_test, preds),3))

joblib.dump(model, args.out)
print("Model saved →", args.out)
