from pydantic import BaseModel
from typing import Optional

# Defines structure for incoming requests (if sending payload)
class PriceRequest(BaseModel):
    product_id: int
    current_inventory: Optional[float] = None
    festival_flag: int = 0
    is_weekend: int = 0

# Defines structure for outgoing JSON responses
class PriceResponse(BaseModel):
    product_id: int
    product_name: str
    cost_price: float
    current_price: float
    recommended_price: float
    price_change_pct: float
    predicted_demand: int
    predicted_profit: float
    profit_uplift: Optional[float] = None  # None for expiry_discount rows — not comparable to full-price profit
    optimization_method: str
    ai_explanation: Optional[str] = None
    ai_confidence_label: Optional[str] = None
    ai_confidence_pct: Optional[int] = None
    ai_reasoning: Optional[str] = None