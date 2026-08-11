import os
import sys
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from prophet import Prophet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(BASE_DIR)

from src.ml.mlflow_utils import log_model_run


def train_prophet_per_product(sales_df, product_id):
    """
    Train Prophet for a single product.
    Prophet requires columns: 'ds' (date) and 'y' (target value).
    """
    product_data = sales_df[sales_df['product_id'] == product_id].copy()

    if 'date' in product_data.columns:
        date_col = 'date'
    elif 'forecast_date' in product_data.columns:
        date_col = 'forecast_date'
    else:
        date_col = None
        for col in product_data.columns:
            if col.lower() == 'date':
                date_col = col
                break

    if not date_col:
        product_data['mock_date'] = pd.date_range(start='2026-01-01', periods=len(product_data), freq='D')
        date_col = 'mock_date'
        print("No real date column found — Prophet using a synthetic date range.")

    prophet_df = product_data[[date_col, 'units_sold']].rename(
        columns={date_col: 'ds', 'units_sold': 'y'}
    )
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

    model = Prophet(
        seasonality_mode='multiplicative',
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.1
    )

    if 'festival_flag' in product_data.columns:
        model.add_regressor('festival_flag')
        prophet_df['festival_flag'] = product_data['festival_flag'].values

    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=30, freq='D')
    if 'festival_flag' in product_data.columns:
        future['festival_flag'] = 0

    forecast = model.predict(future)
    return model, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)


FEATURES = [
    'current_price', 'cost_price', 'inventory_ratio',
    'price_to_cost_ratio', 'inventory_urgency_score', 'sensitivity_encoded'
]

def train_xgboost_demand(master_df):
    available_features = [f for f in FEATURES if f in master_df.columns]
    df = master_df.dropna(subset=available_features + ['units_sold'])

    if 'date' in df.columns:
        date_col = 'date'
    else:
        date_col = None
        for col in df.columns:
            if col.lower() == 'date':
                date_col = col
                break

    if date_col:
        print(f"Sorting time-series data by column: '{date_col}'")
        df_sorted = df.copy()
        df_sorted[date_col] = pd.to_datetime(df_sorted[date_col])
        df_sorted = df_sorted.sort_values(date_col)
    else:
        print("Warning: No explicit date column found. Sorting by index fallback.")
        df_sorted = df.sort_index()

    split_idx = int(len(df_sorted) * (0.70 / 0.85))  # yields 70/15/15 overall once streaming_holdout.csv is removed

    X_train = df_sorted[available_features].iloc[:split_idx]
    X_test = df_sorted[available_features].iloc[split_idx:]
    y_train = df_sorted['units_sold'].iloc[:split_idx]
    y_test = df_sorted['units_sold'].iloc[split_idx:]

    params = {
        'n_estimators': 500,   # raised from 300 — early_stopping_rounds still protects against overfitting
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }

    model = xgb.XGBRegressor(
        **params,
        random_state=42,
        early_stopping_rounds=20,
        eval_metric='rmse'
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = np.mean(np.abs((y_test - preds) / (y_test + 1))) * 100

    print(f'\nMAE: {mae:.1f} units | RMSE: {rmse:.1f} | MAPE: {mape:.1f}%')
    print(f'Best iteration: {model.best_iteration} (out of {params["n_estimators"]} max)')

    print("\n" + "=" * 60)
    print("VALIDATION CHECKS")
    print("=" * 60)

    importances = pd.Series(model.feature_importances_, index=available_features).sort_values(ascending=False)
    print("\n1) Feature importance:")
    print(importances)
    top_share = importances.iloc[0] / importances.sum() * 100
    if top_share > 55:
        print(f"Top feature '{importances.index[0]}' holds {top_share:.0f}% of importance — investigate for leakage.")
    else:
        print(f" No single feature dominates (top = {top_share:.0f}%).")

    train_preds = model.predict(X_train)
    train_mape = np.mean(np.abs((y_train - train_preds) / (y_train + 1))) * 100
    print(f"\n2) Train MAPE: {train_mape:.1f}%  vs  Test MAPE: {mape:.1f}%")
    if mape - train_mape > 5:
        print(" Test MAPE notably worse than train — possible overfitting.")
    else:
        print("Train/test MAPE reasonably close.")

    product_means_train = df_sorted.iloc[:split_idx].groupby('product_id')['units_sold'].mean()
    test_product_ids = df_sorted['product_id'].iloc[split_idx:]
    naive_preds = test_product_ids.map(product_means_train).fillna(y_train.mean())
    naive_mape = np.mean(np.abs((y_test.values - naive_preds.values) / (y_test.values + 1))) * 100
    print(f"\n3) Naive (per-product mean, TRAIN-only) MAPE: {naive_mape:.1f}%  vs  Model MAPE: {mape:.1f}%")
    if mape >= naive_mape:
        print("Model does NOT clearly beat a naive per-product average — limited real value.")
    else:
        print(f" Model beats naive baseline by {naive_mape - mape:.1f} points.")

    # 4. Per-product MAPE — now also names the worst offenders, not just counts them
    test_df = df_sorted.iloc[split_idx:].copy()
    test_df['pred'] = preds
    test_df['abs_pct_err'] = np.abs((test_df['units_sold'] - test_df['pred']) / (test_df['units_sold'] + 1)) * 100

    if 'product_name' in test_df.columns:
        per_product = test_df.groupby('product_id').agg(
            mape=('abs_pct_err', 'mean'),
            product_name=('product_name', 'first'),
            n_test_rows=('abs_pct_err', 'count')
        )
    else:
        per_product = test_df.groupby('product_id').agg(
            mape=('abs_pct_err', 'mean'),
            n_test_rows=('abs_pct_err', 'count')
        )

    bad_products = per_product[per_product['mape'] > 30].sort_values('mape', ascending=False)
    print(f"\n4) Per-product MAPE distribution:")
    print(per_product['mape'].describe())
    print(f"Products with per-product MAPE > 30%: {len(bad_products)} / {len(per_product)}")
    print("\nTop 15 worst-predicted products (check for low n_test_rows = low volume = expected noise):")
    print(bad_products.head(15))

    print("=" * 60 + "\n")

    # NEW: save the same validation numbers to JSON so the dashboard's
    # Model Health tab (render_model_health() in dashboard_charts.py)
    # has data to display instead of showing "no file found".
    validation_metrics = {
        "test_mape": round(float(mape), 2),
        "train_mape": round(float(train_mape), 2),
        "naive_mape": round(float(naive_mape), 2),
        "feature_importance": {k: float(v) for k, v in importances.to_dict().items()},
        "per_product_mape_stats": {k: float(v) for k, v in per_product["mape"].describe().to_dict().items()},
        "best_iteration": int(model.best_iteration),
        "n_estimators_max": int(params["n_estimators"]),
    }
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    validation_path = os.path.join(models_dir, "validation_metrics.json")
    with open(validation_path, "w") as f:
        json.dump(validation_metrics, f, indent=2)
    print(f"Saved {validation_path}")

    log_model_run(model, 'demand_forecasting_xgboost', params, {'mae': mae, 'rmse': rmse, 'mape': mape})

    return model


def predict_demand_ensemble(product_id, features_row, xgb_model, prophet_forecast):
    """Kept for reference — not used by prediction_service.py."""
    features_tensor = np.array(features_row).reshape(1, -1)
    xgb_pred = xgb_model.predict(features_tensor)[0]
    prophet_pred = prophet_forecast['yhat'].mean()
    ensemble = 0.6 * xgb_pred + 0.4 * prophet_pred
    return max(0, round(ensemble))


if __name__ == "__main__":
    processed_path = os.path.join(BASE_DIR, "data", "processed", "master_features.csv")

    if not os.path.exists(processed_path):
        print(f" Missing data file: {processed_path}. Please build data layers first.")
        sys.exit(1)

    master_df = pd.read_csv(processed_path)

    print("Initiating Section 7 Engine...")
    print("\n1️Training Global XGBoost Model...")
    xgb_model = train_xgboost_demand(master_df)

    if 'product_id' in master_df.columns and 33627 in master_df['product_id'].values:
        print("\n2️Training Localized Prophet Model for Product 33627 (Mix Popcorn)...")
        prophet_model, prophet_fcst = train_prophet_per_product(master_df, product_id=33627)
        available_features = [f for f in FEATURES if f in master_df.columns]
        sample_row = master_df[master_df['product_id'] == 33627][available_features].fillna(0).iloc[0].values
        predicted_demand = predict_demand_ensemble(33627, sample_row, xgb_model, prophet_fcst)
        print(f"\n🔮 Predicted demand for Mix Popcorn: {predicted_demand} units")
    else:
        print("\n Note: Product ID 33627 not found in this dataset.")
        print("Model pipelines compiled successfully!")