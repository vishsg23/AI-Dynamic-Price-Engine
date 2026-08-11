import os
import sys
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(BASE_DIR)

from src.ml import prediction_service

print("==================================================")
print("     SECTION 9: PRICE OPTIMIZATION ENGINE        ")
print("==================================================\n")


def optimize_price(product_row):
    """Find the price that maximizes profit for one product."""
    cost_price = product_row['cost_price']
    base_price = product_row['base_price']
    current_price = product_row.get('current_price', base_price)

    sensitivity_category = product_row.get(
        'price_sensitivity_category',
        product_row.get('price_sensitivity', 'Medium')
    )

    if sensitivity_category == 'Low':
        candidate_price = round(base_price * 1.10, 2)
        candidate_price = max(candidate_price, cost_price * 1.05)

        candidate_demand = prediction_service.predict_demand(candidate_price, product_row)
        candidate_profit = (candidate_price - cost_price) * candidate_demand

        current_demand = prediction_service.predict_demand(current_price, product_row)
        current_profit_est = (current_price - cost_price) * current_demand

        if candidate_profit < current_profit_est:
            return round(current_price, 2), 'inelastic_no_change'

        return candidate_price, 'inelastic_markup'

    if product_row.get('expiry_risk_flag', 0) == 1:
        return round(max(cost_price * 1.02, base_price * 0.75), 2), 'expiry_discount'

    def neg_profit(price):
        demand = prediction_service.predict_demand(price, product_row)
        profit = (price - cost_price) * demand
        return -profit

    lower_bound = cost_price * 1.05
    upper_bound = base_price * 1.50
    if lower_bound >= upper_bound:
        upper_bound = lower_bound * 1.2

    result = minimize_scalar(neg_profit, bounds=(lower_bound, upper_bound), method='bounded')
    optimal_price = result.x

    if product_row.get('stock_urgency_category') == 'Critical':
        optimal_price = min(optimal_price, base_price * 1.15)

    optimal_price = max(optimal_price, cost_price * 1.05)
    optimal_price = min(optimal_price, base_price * 1.50)

    return round(optimal_price, 2), 'elasticity_optimized'


def build_product_level_df(master_df):
    """Collapse master_df (possibly many rows per product) to one row per product."""
    if 'product_id' not in master_df.columns:
        raise ValueError("master_df must contain a 'product_id' column")

    numeric_cols = master_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'product_id' in numeric_cols:
        numeric_cols.remove('product_id')

    non_numeric_cols = [c for c in master_df.columns if c not in numeric_cols and c != 'product_id']

    agg_dict = {c: 'mean' for c in numeric_cols}
    for c in non_numeric_cols:
        agg_dict[c] = lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]

    return master_df.groupby('product_id', as_index=False).agg(agg_dict)


def run_full_optimization(master_df):
    product_df = build_product_level_df(master_df)
    print(f"Collapsed {len(master_df)} rows down to {len(product_df)} unique products")

    if 'base_price' not in product_df.columns:
        product_df['base_price'] = product_df['current_price']
    if 'cost_price' not in product_df.columns:
        product_df['cost_price'] = product_df['current_price'] * 0.7
    if 'inventory_level' not in product_df.columns:
        product_df['inventory_level'] = 0  # fallback if this column isn't present

    print(f"Processing optimize loops across {len(product_df)} unique products...")

    results = []
    for idx, row in product_df.iterrows():
        row_dict = row.to_dict()
        row_dict['product_id'] = row_dict.get('product_id', idx)

        optimal_price, method = optimize_price(row_dict)
        pred_demand = prediction_service.predict_demand(optimal_price, row_dict)
        pred_profit = round((optimal_price - row_dict['cost_price']) * pred_demand, 2)

        pred_demand_at_current = prediction_service.predict_demand(row_dict['current_price'], row_dict)
        curr_profit = round((row_dict['current_price'] - row_dict['cost_price']) * pred_demand_at_current, 2)

        if method == 'expiry_discount':
            # NOT comparable to full-price profit — the point of this branch
            # is avoiding a total write-off, not beating normal sales.
            # profit_uplift is left as NaN (not a misleading negative number).
            profit_uplift_val = np.nan
            profit_uplift_pct_val = np.nan
            spoilage_savings = round(row_dict['inventory_level'] * optimal_price, 2)
        else:
            profit_uplift_val = round(pred_profit - curr_profit, 2)
            profit_uplift_pct_val = round((profit_uplift_val / curr_profit) * 100, 1) if curr_profit != 0 else 0.0
            spoilage_savings = np.nan

        results.append({
            'product_id': row_dict['product_id'],
            'product_name': row_dict.get('product_name', f"Product_{row_dict['product_id']}"),
            'department': row_dict.get('department', 'General'),
            'cost_price': row_dict['cost_price'],
            'current_price': row_dict['current_price'],
            'recommended_price': optimal_price,
            'price_change_pct': round(((optimal_price / row_dict['current_price']) - 1) * 100, 1)
                if row_dict['current_price'] else 0.0,
            'predicted_demand': pred_demand,
            'predicted_profit': pred_profit,
            'current_profit': curr_profit,
            'profit_uplift': profit_uplift_val,
            'profit_uplift_pct': profit_uplift_pct_val,
            'spoilage_savings_estimate': spoilage_savings,
            'optimization_method': method,
            'inventory_ratio': row_dict.get('inventory_ratio', 0),
            'stock_urgency_category': row_dict.get('stock_urgency_category', 'Normal')
        })

    results_df = pd.DataFrame(results)

    dup_count = results_df['product_id'].duplicated().sum()
    if dup_count > 0:
        print(f"Warning: {dup_count} duplicate product_id rows still found in output!")
    else:
        print("Confirmed: every product_id appears exactly once in the results")

    output_dir = os.path.join(BASE_DIR, "data", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "optimization_results.csv")
    results_df.to_csv(output_path, index=False)

    print(f"\nOptimization metrics dumped cleanly to: {output_path}")
    print("\nSample Optimization Recommendations (Top 10):")
    print(results_df[['product_id', 'product_name', 'recommended_price', 'profit_uplift']].head(10).to_string(index=False))

    return results_df


if __name__ == "__main__":
    processed_path = os.path.join(BASE_DIR, "data", "processed", "master_features.csv")

    if not os.path.exists(processed_path):
        print(f"Missing core master features at {processed_path}")
        sys.exit(1)

    master_df = pd.read_csv(processed_path)

    print("Loading trained models (XGBoost + Prophet)...")
    prediction_service.load_models()

    print("\n Running Full Engine Optimization...")
    run_full_optimization(master_df)
    print("\n==================================================")
    print("CHECKPOINT REACHED: Optimization Pipeline Complete!")


    