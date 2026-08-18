import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
sys.path.append(BASE_DIR)

from src.ml.mlflow_utils import log_model_run

print("==================================================")
print("   FAIR COMPARISON: RandomForest on units_sold     ")
print("   (same target, same split as XGBoost)            ")
print("==================================================\n")

master_path = os.path.join(PROCESSED_DIR, "master_features.csv")
if not os.path.exists(master_path):
    print(f"Error: {master_path} missing! Run merge_data.py first.")
    sys.exit(1)

df = pd.read_csv(master_path)

# ---- FIX: real, always-populated numeric target, not the broken fallback ----
target_col = "units_sold"

# ---- Same feature set XGBoost uses, so any difference in results is
#      actually due to the algorithm, not different inputs ----
feature_cols = [
    "current_price", "cost_price", "inventory_ratio",
    "price_to_cost_ratio", "inventory_urgency_score", "sensitivity_encoded"
]
feature_cols = [c for c in feature_cols if c in df.columns]

print(f"Target: '{target_col}'")
print(f"Features: {feature_cols}\n")

df = df.dropna(subset=feature_cols + [target_col])

# ---- FIX: same date-based split as demand_model.py, not a random split.
#      A random split would be an easier problem (no genuine "future"
#      data held out), so it wouldn't be a fair comparison. ----
date_col = "date" if "date" in df.columns else None
if date_col:
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
else:
    print("Warning: no date column found, falling back to index order.")

split_idx = int(len(df) * (0.70 / 0.85))  # same ratio used in demand_model.py

X_train = df[feature_cols].iloc[:split_idx]
X_test = df[feature_cols].iloc[split_idx:]
y_train = df[target_col].iloc[:split_idx]
y_test = df[target_col].iloc[split_idx:]

print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}\n")

hyperparams = {
    "n_estimators": 500,   # matched to XGBoost's n_estimators for a fair comparison
    "max_depth": 6,        # matched to XGBoost's max_depth
    "random_state": 42
}

print("Training RandomForest...")
model = RandomForestRegressor(**hyperparams, n_jobs=-1)
model.fit(X_train, y_train)

preds = model.predict(X_test)
train_preds = model.predict(X_train)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
mape = np.mean(np.abs((y_test - preds) / (y_test + 1))) * 100
train_mape = np.mean(np.abs((y_train - train_preds) / (y_train + 1))) * 100

print(f"\nRandomForest results (same target/split as XGBoost):")
print(f"  Test MAPE:  {mape:.1f}%")
print(f"  Train MAPE: {train_mape:.1f}%")
print(f"  MAE: {mae:.1f} units | RMSE: {rmse:.1f}")

print(f"\nFor comparison, XGBoost's validated result was: 18.6% test MAPE")
if mape < 18.6:
    print(f"  -> RandomForest is BETTER on this run ({mape:.1f}% < 18.6%)")
elif mape > 18.6:
    print(f"  -> XGBoost is BETTER on this run ({mape:.1f}% > 18.6%)")
else:
    print(f"  -> Essentially tied")

metrics_payload = {
    "mae": round(float(mae), 4),
    "rmse": round(float(rmse), 4),
    "test_mape": round(float(mape), 2),
    "train_mape": round(float(train_mape), 2),
}

log_model_run(
    model=model,
    model_name="pricing_random_forest_units_sold",  # new experiment name — doesn't overwrite the old broken run
    params=hyperparams,
    metrics=metrics_payload
)

print("\n==================================================")
print("Logged to MLflow as 'pricing_random_forest_units_sold'")
print("Compare against 'pricing_engine_demand_forecasting_xgboost' in the MLflow UI")
print("==================================================")