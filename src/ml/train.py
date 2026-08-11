import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Establish paths correctly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # D:\pricing_engine\src\ml
BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))  # D:\pricing_engine
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# 2. Append the ROOT project directory to sys.path so Python can find 'ml'
sys.path.append(BASE_DIR)

# 3. Import using the folder name directly
# To this
from src.ml.mlflow_utils import log_model_run

print("==================================================")
print("       SECTION 6: PRODUCTION MODEL TRAINING       ")
print("==================================================\n")

# 4. Load your master feature dataset
master_path = os.path.join(PROCESSED_DIR, "master_features.csv")
if not os.path.exists(master_path):
    print(f"Error: {master_path} missing! Run merge_data.py first.")
    sys.exit(1)

print("Loading integrated feature matrices...")
df = pd.read_csv(master_path)

# 5. Select Features & Target
target_col = 'elasticity' if 'elasticity' in df.columns else df.columns[-1] # Default fallback
feature_cols = [
    'total_units_sold', 'inventory_urgency_score', 'price_to_cost_ratio', 
    'profit_margin_pct', 'demand_volatility', 'moving_avg_7day'
]

# Ensure all targeted features exist in the data layout
feature_cols = [col for col in feature_cols if col in df.columns]

if not feature_cols:
    print("Error: No matching numerical features found to train the model!")
    sys.exit(1)

print(f"Target variable selected: '{target_col}'")
print(f"Training features chosen: {feature_cols}")

# --- SAFETY FIX: Force all training columns to be strictly numeric ---
X = pd.DataFrame()
for col in feature_cols:
    # errors='coerce' forces any non-numeric text into NaN
    X[col] = pd.to_numeric(df[col], errors='coerce')

# Clean missing entries safely for the mathematical arrays
X = X.fillna(0)
y = pd.to_numeric(df[target_col], errors='coerce').fillna(0)
# --------------------------------------------------------------------

# 6. Train/Test Split (Fixed the argument name typo here as well)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Model Configuration & Training
print("\n🌲 Training production RandomForest pricing brain...")
hyperparams = {
    "n_estimators": 100,
    "max_depth": 6,
    "random_state": 42
}

model = RandomForestRegressor(**hyperparams)
model.fit(X_train, y_train)

# 8. Model Performance Metrics Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

metrics_payload = {
    "mae": round(mae, 4),
    "r2_score": round(r2, 4)
}

print("Model training complete. Evaluation metrics finalized.")

# 9. Push Structured Flight Records to Local MLflow Server
print("Logging model metrics, hyperparams, and artifacts to MLflow...")
log_model_run(
    model=model,
    model_name="pricing_random_forest",
    params=hyperparams,
    metrics=metrics_payload
)

print("\n==================================================")