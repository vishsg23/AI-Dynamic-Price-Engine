import os
import sys
import pickle
import joblib
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(BASE_DIR)

from src.ml.demand_model import train_xgboost_demand, train_prophet_per_product

MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MASTER_PATH = os.path.join(BASE_DIR, "data", "processed", "master_features.csv")


def main():
    print("=" * 60)
    print("NIGHTLY RETRAIN: XGBoost + Prophet (per product)")
    print("=" * 60)

    if not os.path.exists(MASTER_PATH):
        print(f"Missing {MASTER_PATH} — run merge_data.py first.")
        sys.exit(1)

    master_df = pd.read_csv(MASTER_PATH)

    # ------------------------------------------------------------
    # 1. XGBoost — predicts demand, price is one of its features,
    #    so it already learns how demand responds to price on its own.
    # ------------------------------------------------------------
    print("\n1) Training XGBoost demand model...")
    xgb_model = train_xgboost_demand(master_df)
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgb_model.pkl"))
    print("   Saved models/xgb_model.pkl")

    # ------------------------------------------------------------
    # 2. Prophet — one model per product, seasonal factor only
    #    (today's forecast ÷ average forecast), cheap to use at
    #    prediction time since it's precomputed here, not live.
    # ------------------------------------------------------------
    print("\n2) Training Prophet per product...")
    product_ids = master_df["product_id"].unique()
    prophet_cache = {}
    failed = []

    for i, pid in enumerate(product_ids):
        try:
            _, forecast = train_prophet_per_product(master_df, product_id=pid)
            avg_yhat = forecast["yhat"].mean()
            today_yhat = forecast.iloc[0]["yhat"]
            factor = today_yhat / avg_yhat if avg_yhat > 0 else 1.0
            prophet_cache[pid] = max(0.5, min(2.0, factor))  # clip extreme swings
        except Exception:
            failed.append(pid)
        if (i + 1) % 50 == 0:
            print(f"   ...{i + 1}/{len(product_ids)} products processed")

    with open(os.path.join(MODELS_DIR, "prophet_cache.pkl"), "wb") as f:
        pickle.dump(prophet_cache, f)

    print(f"\n   Prophet trained for {len(prophet_cache)}/{len(product_ids)} products")
    if failed:
        shown = failed[:10]
        print(f"   Skipped {len(failed)} (insufficient history / fit error): {shown}{'...' if len(failed) > 10 else ''}")
    print("   Saved models/prophet_cache.pkl")

    print("\nRetrain complete.")


if __name__ == "__main__":
    main()