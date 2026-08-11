import os
import sys
import pickle
import joblib
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(BASE_DIR)

MODELS_DIR = os.path.join(BASE_DIR, "models")

_xgb_model = None
_xgb_features = None
_prophet_cache = None


def load_models():
    """Call this ONCE at startup — before optimization.py's main block runs,
    and before stream_processor.py starts streaming."""
    global _xgb_model, _xgb_features, _prophet_cache

    xgb_path = os.path.join(MODELS_DIR, "xgb_model.pkl")
    prophet_path = os.path.join(MODELS_DIR, "prophet_cache.pkl")

    if not os.path.exists(xgb_path):
        raise FileNotFoundError(
            f"{xgb_path} not found — run `python src/ml/retrain_all.py` first."
        )

    _xgb_model = joblib.load(xgb_path)
    _xgb_features = _xgb_model.get_booster().feature_names

    if os.path.exists(prophet_path):
        with open(prophet_path, "rb") as f:
            _prophet_cache = pickle.load(f)
    else:
        print(" No prophet_cache.pkl — seasonal adjustment disabled.")
        _prophet_cache = {}

    print(
        f"prediction_service ready — xgb_features={len(_xgb_features)}, "
        f"prophet_products={len(_prophet_cache)}"
    )


def predict_demand(candidate_price, product_row):
    """
    XGBoost predicts baseline demand AT base_price (fixed reference —
    the training data has ZERO price variance per product, so XGBoost
    cannot reliably learn a price response on its own; using it at the
    candidate price would just reflect cross-product correlation, not
    a real within-product effect).

    Prophet applies a seasonal multiplier on top.

    price_elasticity (a real, verified column already present in
    master_features.csv — NOT avg_price_elasticity, which is 25%+
    zero-filled and unreliable) drives how demand responds as
    candidate_price moves away from base_price.
    """
    if _xgb_model is None:
        raise RuntimeError("Call load_models() once before predict_demand().")

    pid = product_row.get("product_id")
    base_price = product_row.get("base_price", product_row.get("current_price", candidate_price))

    # ---- XGBoost baseline, evaluated at base_price (fixed) ----
    xgb_feat = {f: product_row.get(f, 0) for f in _xgb_features}
    xgb_feat["current_price"] = base_price
    cost_price = product_row.get("cost_price", 0)
    if cost_price > 0:
        xgb_feat["price_to_cost_ratio"] = base_price / cost_price
    xgb_vec = np.array([xgb_feat.get(f, 0) for f in _xgb_features]).reshape(1, -1)
    baseline_demand = max(0, _xgb_model.predict(xgb_vec)[0])

    # ---- Prophet seasonal factor (precomputed, cheap) ----
    seasonal_factor = _prophet_cache.get(pid, 1.0) if _prophet_cache else 1.0
    seasonal_demand = baseline_demand * seasonal_factor

    # ---- Price response from the dataset's own price_elasticity column ----
    elasticity = product_row.get("price_elasticity", -1.0)
    elasticity = min(-0.1, max(-5.0, elasticity))  # matches this column's real observed range

    if base_price:
        pct_price_change = (candidate_price - base_price) / base_price
        final_demand = seasonal_demand * (1 + elasticity * pct_price_change)
    else:
        final_demand = seasonal_demand

    return max(0, round(final_demand))