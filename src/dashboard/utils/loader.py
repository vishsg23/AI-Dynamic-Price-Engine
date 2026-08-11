import time
import pandas as pd
import streamlit as st
import numpy as np

_batch_info = {
    "last_updated": None,
    "processing_time": None,
    "batch_id": None,
    "rows": None,
}


def get_batch_info():
    return dict(_batch_info)


@st.cache_data(ttl=300)
def load_results():
    _start = time.time()

    df = pd.read_csv("data/outputs/optimization_results.csv")
    df.columns = df.columns.str.strip()

    numeric_cols = [
        "current_price", "recommended_price", "profit_uplift", "current_profit",
        "predicted_profit", "price_change_pct", "predicted_demand", "inventory_ratio",
        "cost_price", "profit_uplift_pct", "spoilage_savings_estimate",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Only fill NaN on columns where NaN genuinely means "zero/unknown".
    # profit_uplift / profit_uplift_pct are intentionally NaN for
    # expiry_discount rows (not comparable to full-price profit) — do
    # NOT fillna those, or the distinction we fixed gets silently erased.
    safe_to_zero = ["inventory_ratio", "predicted_demand", "spoilage_savings_estimate"]
    for col in safe_to_zero:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    df["display_label"] = df["product_name"] + " (#" + df["product_id"].astype(str) + ")"

    if "profit_uplift_pct" not in df.columns or df["profit_uplift_pct"].isna().all():
        df["profit_uplift_pct"] = np.where(
            (df["current_profit"].notna()) & (df["current_profit"] != 0),
            (df["profit_uplift"] / df["current_profit"]) * 100,
            np.nan,
        )

    df["price_difference"] = df["recommended_price"] - df["current_price"]
    df["recommendation"] = np.where(
        df["price_difference"] > 0, "Increase",
        np.where(df["price_difference"] < 0, "Decrease", "No Change")
    )

    if "inventory_ratio" in df.columns:
        df["inventory_health"] = pd.cut(
            df["inventory_ratio"], bins=[-100, 0.3, 0.6, 100],
            labels=["Critical", "Warning", "Healthy"]
        )

    # Flag used everywhere a chart needs to exclude expiry_discount rows
    # from profit-based metrics, instead of repeating the filter inline.
    df["is_profit_comparable"] = df["optimization_method"] != "expiry_discount"

    _batch_info["last_updated"] = pd.Timestamp.now()
    _batch_info["processing_time"] = round(time.time() - _start, 2)
    _batch_info["batch_id"] = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    _batch_info["rows"] = len(df)

    return df


@st.cache_resource
def load_product_lookup():
    """Full-feature per-product table (base_price, price_elasticity, etc.)
    used by the live price simulator — optimization_results.csv alone
    doesn't carry every feature prediction_service needs."""
    import sys, os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.append(BASE_DIR)
    from src.ml import prediction_service
    from src.ml.optimization import build_product_level_df

    prediction_service.load_models()

    master_features_path = os.path.join(BASE_DIR, "data", "processed", "master_features.csv")
    try:
        master_df = pd.read_csv(master_features_path)
    except (pd.errors.ParserError, MemoryError):
        master_df = pd.read_csv(master_features_path, engine="python")

    lookup = build_product_level_df(master_df)
    if "base_price" not in lookup.columns:
        lookup["base_price"] = lookup["current_price"]
    if "cost_price" not in lookup.columns:
        lookup["cost_price"] = lookup["current_price"] * 0.7
    lookup = lookup.set_index("product_id")
    return lookup, prediction_service