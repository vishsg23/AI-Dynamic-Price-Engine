from fastapi import APIRouter, HTTPException

from src.api.schemas import PriceRequest, PriceResponse
from src.api.gemini_explainer import generate_price_explanation
from src.ml.optimization import optimize_price
from src.ml import prediction_service

router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "healthy", "model_version": "1.0.0"}


def _get_product_or_404(product_id: int, product_lookup) -> dict:
    if product_id not in product_lookup.index:
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found")

    row = product_lookup.loc[product_id].to_dict()
    row["product_id"] = product_id
    return row


def _score_current_and_optimal_price(product: dict):
    """
    Runs the pricing model twice: once at the recommended price, once at
    today's price, using the same demand model both times so the two
    profit numbers are actually comparable.
    """
    optimal_price, method = optimize_price(product)
    predicted_demand = prediction_service.predict_demand(optimal_price, product)
    predicted_profit = round((optimal_price - product["cost_price"]) * predicted_demand, 2)

    demand_at_current_price = prediction_service.predict_demand(product["current_price"], product)
    current_profit = round((product["current_price"] - product["cost_price"]) * demand_at_current_price, 2)

    price_change_pct = (
        round(((optimal_price / product["current_price"]) - 1) * 100, 1)
        if product["current_price"] else 0.0
    )

    # An expiry-driven discount is about avoiding a write-off, not
    # beating normal sales, so "profit uplift" isn't a meaningful
    # number for these — leave it blank rather than showing something misleading.
    profit_uplift = None if method == "expiry_discount" else round(predicted_profit - current_profit, 2)

    return optimal_price, method, predicted_demand, predicted_profit, price_change_pct, profit_uplift


@router.get("/price/{product_id}", response_model=PriceResponse)
def get_price_recommendation(product_id: int):
    from src.api.main import app_state

    product = _get_product_or_404(product_id, app_state["product_lookup"])

    optimal_price, method, predicted_demand, predicted_profit, price_change_pct, profit_uplift = (
        _score_current_and_optimal_price(product)
    )

    product_name = product.get("product_name", f"Product_{product_id}")

    ai_result = generate_price_explanation({
        "product_name": product_name,
        "department": product.get("department", "General"),
        "current_price": product["current_price"],
        "recommended_price": optimal_price,
        "price_change_pct": price_change_pct,
        "price_sensitivity_category": product.get("price_sensitivity_category", "Medium"),
        "stock_urgency_category": product.get("stock_urgency_category", "Normal"),
        "profit_uplift": profit_uplift,
    })

    return PriceResponse(
        product_id=product_id,
        product_name=product_name,
        cost_price=product["cost_price"],
        current_price=product["current_price"],
        recommended_price=optimal_price,
        price_change_pct=price_change_pct,
        predicted_demand=predicted_demand,
        predicted_profit=predicted_profit,
        profit_uplift=profit_uplift,
        optimization_method=method,
        ai_explanation=ai_result["explanation"],
        ai_confidence_label=ai_result["confidence_label"],
        ai_confidence_pct=ai_result["confidence_pct"],
        ai_reasoning=ai_result["reasoning"],
    )