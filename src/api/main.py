import traceback
import pandas as pd
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.router import router
from src.ml import prediction_service
from src.ml.optimization import build_product_level_df

# Holds the trained models and the per-product lookup table for as long
# as the app is running, so every request can reuse them instead of
# reloading from disk each time. Filled in by lifespan() below, cleared
# on shutdown.
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")
    prediction_service.load_models()

    master_df = pd.read_csv("data/processed/master_features.csv")
    product_lookup = build_product_level_df(master_df)

    # Not every build of master_features.csv is guaranteed to have these
    # columns, so fill in a reasonable default rather than crashing startup.
    if "base_price" not in product_lookup.columns:
        product_lookup["base_price"] = product_lookup["current_price"]
    if "cost_price" not in product_lookup.columns:
        product_lookup["cost_price"] = product_lookup["current_price"] * 0.7

    product_lookup = product_lookup.set_index("product_id")
    app_state["product_lookup"] = product_lookup

    print(f"Ready. {len(product_lookup)} unique products loaded.")
    yield
    app_state.clear()


app = FastAPI(
    title="Dynamic Pricing Engine API",
    description="AI-powered price recommendations",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def debug_exception_handler(request: Request, exc: Exception):
    # This project is still under active development, so it's more
    # useful to see the full traceback in the response than to hide
    # errors behind a generic 500 — swap this for a quieter handler
    # before shipping to real users.
    tb = traceback.format_exc()
    print("=" * 60)
    print("FULL TRACEBACK:")
    print(tb)
    print("=" * 60)
    return JSONResponse(status_code=500, content={"error": str(exc), "traceback": tb})


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")