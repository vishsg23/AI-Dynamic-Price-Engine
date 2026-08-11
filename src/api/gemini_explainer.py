import os
import json
import re
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# If Gemini is down, unreachable, or returns something we can't parse,
# we still owe the frontend a usable explanation. This builds one from
# plain math instead of AI text, so the app never shows a blank field.
def _fallback_explanation(product_data: dict) -> dict:
    direction = "reduced" if product_data.get("price_change_pct", 0) < 0 else "increased"
    change_amount = abs(product_data.get("price_change_pct", 0))
    uplift = product_data.get("profit_uplift", 0)

    return {
        "explanation": (
            f"Price {direction} by {change_amount:.1f}% to maximize weekly profit. "
            f"Expected profit uplift of ${uplift:.2f}."
        ),
        "confidence_label": "Unavailable",
        "confidence_pct": None,
        "reasoning": "AI service unavailable — this is a rule-based fallback summary, not a Gemini-generated explanation.",
    }


def _strip_markdown_fences(text: str) -> str:
    """Gemini sometimes wraps its JSON in ```json ... ``` even when told not to. Strip that off before parsing."""
    text = re.sub(r"^```(json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    return text.strip()


def _build_prompt(product_data: dict) -> str:
    return f"""
You are a pricing analyst at a retail company.
Respond with ONLY a JSON object, no markdown fences, no extra text, in exactly this shape:

{{
  "explanation": "2-3 sentence plain-English explanation for a business manager, no jargon, no bullet points",
  "confidence_label": "High" or "Medium" or "Low",
  "confidence_pct": an integer from 0 to 100 reflecting how confident you are in this recommendation given the signals below,
  "reasoning": "one short sentence naming the SPECIFIC signals driving this recommendation (e.g. demand trend, inventory level, price sensitivity) — not a restatement of the price change itself"
}}

Signals for this product:
Product: {product_data.get('product_name')}
Department: {product_data.get('department', 'General')}
Current Price: ${product_data.get('current_price', 0):.2f}
Recommended Price: ${product_data.get('recommended_price', 0):.2f}
Price Change: {product_data.get('price_change_pct', 0):+.1f}%
Price Sensitivity: {product_data.get('price_sensitivity_category', 'Medium')}
Stock Urgency: {product_data.get('stock_urgency_category', 'Normal')}
Profit Uplift: ${product_data.get('profit_uplift', 0):.2f}

Respond with the JSON object now:
"""


def generate_price_explanation(product_data: dict) -> dict:
    """
    Asks Gemini to explain, in plain English, why a product's price is
    being changed. Always returns the same shape, whether Gemini
    cooperates or not:

      {
        "explanation": str,
        "confidence_label": str,   # "High" | "Medium" | "Low" | "Unavailable"
        "confidence_pct": int | None,
        "reasoning": str,
      }
    """
    model = genai.GenerativeModel("gemini-flash-latest")
    prompt = _build_prompt(product_data)

    try:
        response = model.generate_content(prompt)
        raw_text = _strip_markdown_fences(response.text)
        parsed = json.loads(raw_text)

        explanation = str(parsed.get("explanation", "")).strip()
        if not explanation:
            explanation = _fallback_explanation(product_data)["explanation"]

        reasoning = str(parsed.get("reasoning", "")).strip()
        if not reasoning:
            reasoning = "No specific reasoning was returned."

        confidence_pct = parsed.get("confidence_pct")

        return {
            "explanation": explanation,
            "confidence_label": str(parsed.get("confidence_label", "Medium")),
            "confidence_pct": int(confidence_pct) if confidence_pct is not None else None,
            "reasoning": reasoning,
        }

    except (json.JSONDecodeError, ValueError, KeyError):
        # Gemini responded, but not with valid JSON — show the raw text
        # rather than throwing it away, so it's at least visible.
        try:
            return {
                "explanation": response.text.strip(),
                "confidence_label": "Unavailable",
                "confidence_pct": None,
                "reasoning": "The AI response could not be parsed into a structured format.",
            }
        except Exception:
            return _fallback_explanation(product_data)

    except Exception:
        # Covers network errors, API outages, missing API key, etc.
        return _fallback_explanation(product_data)