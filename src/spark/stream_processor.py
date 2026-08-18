import os
import sys
import time
import traceback
import pandas as pd

os.environ["HADOOP_HOME"] = r"D:\pricing_engine\.venv\hadoop"
os.environ["hadoop.home.dir"] = r"D:\pricing_engine\.venv\hadoop"
os.environ["PATH"] += os.pathsep + r"D:\pricing_engine\.venv\hadoop\bin"

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_timestamp, window, sum, avg, count
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(BASE_DIR)

DELTA_PATH = os.path.join(BASE_DIR, "data", "delta", "processed_features")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "data", "delta", "checkpoints")
MASTER_FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "master_features.csv")
OPT_OUTPUT_PATH = os.path.join(BASE_DIR, "data", "outputs", "optimization_results.csv")
STREAMING_LOG_PATH = os.path.join(BASE_DIR, "data", "processed", "streaming_features_log.csv")

os.makedirs(DELTA_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
os.makedirs(os.path.dirname(OPT_OUTPUT_PATH), exist_ok=True)

print("=" * 60)
print("        PYSPARK REAL-TIME STREAM PROCESSOR")
print("=" * 60)

if not os.path.exists(MASTER_FEATURES_PATH):
    print(f" Missing {MASTER_FEATURES_PATH} — run merge_data.py first.")
    sys.exit(1)

from src.ml import prediction_service
from src.ml.optimization import optimize_price, build_product_level_df

# ----------------------------------------------------------------
# Model / lookup state is now reloadable, not load-once-at-startup.
# A dict (not separate globals) so reload_if_stale() can swap the
# whole thing atomically without partial-update bugs.
# ----------------------------------------------------------------
_state = {"product_lookup": None, "mtime": None, "last_checked": 0}

RELOAD_CHECK_INTERVAL_SECONDS = 60  # how often to check if the file changed


def build_lookup():
    print("Loading master_features.csv...")
    master_df = pd.read_csv(MASTER_FEATURES_PATH)

    print("Building static per-product lookup table...")
    lookup = build_product_level_df(master_df)

    if "base_price" not in lookup.columns:
        lookup["base_price"] = lookup["current_price"]
    if "cost_price" not in lookup.columns:
        lookup["cost_price"] = lookup["current_price"] * 0.7
    if "inventory_level" not in lookup.columns:
        lookup["inventory_level"] = 0

    lookup = lookup.set_index("product_id")
    print(f"Ready. {len(lookup)} products in lookup table.")
    return lookup


def reload_if_stale():
    """
    Checks (at most once every RELOAD_CHECK_INTERVAL_SECONDS, to avoid a
    disk stat() on every single micro-batch) whether master_features.csv
    has changed since it was last loaded — e.g. because Airflow's nightly
    retrain + update_master_data task ran while this stream was already
    running. If so, reloads both the trained models and the product
    lookup table, so a running stream picks up new products and a freshly
    retrained model WITHOUT needing to be manually restarted.
    """
    now = time.time()
    if now - _state["last_checked"] < RELOAD_CHECK_INTERVAL_SECONDS:
        return
    _state["last_checked"] = now

    current_mtime = os.path.getmtime(MASTER_FEATURES_PATH)
    if _state["mtime"] is not None and current_mtime == _state["mtime"]:
        return  # file hasn't changed since last load

    print("\n" + "=" * 60)
    print("master_features.csv changed on disk — reloading models + lookup table")
    print("=" * 60)
    prediction_service.load_models()  # re-reads models/xgb_model.pkl + prophet_cache.pkl
    _state["product_lookup"] = build_lookup()
    _state["mtime"] = current_mtime
    print("Reload complete. Stream continues with updated models/products.\n")


# ---- Initial load, same as before ----
print("Loading trained models (XGBoost + Prophet)...")
prediction_service.load_models()
_state["product_lookup"] = build_lookup()
_state["mtime"] = os.path.getmtime(MASTER_FEATURES_PATH)
_state["last_checked"] = time.time()

spark = (
    SparkSession.builder
    .appName("DynamicPricingStream")
    .master("local[2]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.executor.heartbeatInterval", "60s")
    .config("spark.network.timeout", "300s")
    .config("spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,io.delta:delta-spark_2.12:3.2.0")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
spark.conf.set("spark.sql.shuffle.partitions", "4")
print("Spark Session Started")

event_schema = StructType([
    StructField("product_id", IntegerType()),
    StructField("units_sold", IntegerType()),
    StructField("current_price", DoubleType()),
    StructField("inventory_ratio", DoubleType()),
    StructField("timestamp", StringType())
])

print("Connecting to Kafka...")

kafka_stream = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "localhost:9092")
    .option("subscribe", "orders_topic")
    .option("startingOffsets", "latest")
    .option("failOnDataLoss", "false")
    .load()
)

parsed = (
    kafka_stream
    .select(from_json(col("value").cast("string"), event_schema).alias("data"))
    .select("data.*")
    .filter(col("product_id").isNotNull())
    .withColumn("event_time", to_timestamp(col("timestamp")))
    .filter(col("event_time").isNotNull())
)

print("Kafka Connected")

windowed = (
    parsed
    .withWatermark("event_time", "15 seconds")
    .groupBy(window("event_time", "10 seconds"), col("product_id"))
    .agg(
        sum("units_sold").alias("hourly_demand"),
        avg("current_price").alias("avg_price"),
        avg("inventory_ratio").alias("avg_inventory"),
        count("*").alias("orders")
    )
)


def build_fallback_product(pid, agg_row):
    """
    Builds a rough product profile for a product_id that isn't in the
    lookup table yet — i.e. a genuinely new product that hasn't gone
    through a batch pipeline run. Uses only what the streaming event
    itself tells us, plus conservative generic defaults everywhere else.

    This is deliberately a ROUGH ESTIMATE, not a validated price — every
    row produced this way is tagged is_new_product=True so it's never
    confused with a normal, fully-informed recommendation downstream.
    """
    current_price = float(agg_row["avg_price"]) if agg_row["avg_price"] else 0.0
    cost_price = round(current_price * 0.7, 2) if current_price else 0.0

    return {
        "product_id": pid,
        "product_name": f"Unknown_Product_{pid}",
        "department": "Unclassified",
        "current_price": current_price,
        "base_price": current_price,
        "cost_price": cost_price,
        "price_to_cost_ratio": round(current_price / cost_price, 2) if cost_price else 1.0,
        "inventory_ratio": float(agg_row["avg_inventory"]) if agg_row["avg_inventory"] is not None else 0.5,
        "inventory_urgency_score": 0.5,     # neutral — we genuinely don't know
        "sensitivity_encoded": 1,           # neutral/"Medium" — no elasticity data yet
        "price_elasticity": -1.0,           # prediction_service's own default
        "units_sold": int(agg_row["hourly_demand"]) if agg_row["hourly_demand"] else 0,
        "inventory_level": 0,
        "stock_urgency_category": "Unknown",
    }


def score_one_product(pid, agg_row, product_lookup):
    """
    Returns (scored_dict, training_row_dict). If the product is known,
    prices it normally. If it's NOT in the lookup table (a genuinely new
    product), builds a rough fallback profile instead of silently
    dropping the event — flagged clearly as is_new_product so it's never
    mistaken for a fully-informed recommendation.
    """
    is_new_product = pid not in product_lookup.index

    if is_new_product:
        print(f"  [new product] product_id {pid} not in lookup table — pricing with fallback defaults")
        live_product = build_fallback_product(pid, agg_row)
    else:
        live_product = product_lookup.loc[pid].to_dict()
        live_product["product_id"] = pid
        live_product["current_price"] = agg_row["avg_price"]
        live_product["units_sold"] = agg_row["hourly_demand"]
        live_product["inventory_ratio"] = agg_row["avg_inventory"]

    try:
        optimal_price, method = optimize_price(live_product)
        pred_demand = prediction_service.predict_demand(optimal_price, live_product)
        pred_profit = round((optimal_price - live_product["cost_price"]) * pred_demand, 2)

        pred_demand_at_current = prediction_service.predict_demand(
            live_product["current_price"], live_product
        )
        curr_profit = round(
            (live_product["current_price"] - live_product["cost_price"]) * pred_demand_at_current, 2
        )

        if method == "expiry_discount":
            profit_uplift = None
            profit_uplift_pct = None
            spoilage_savings = round(live_product.get("inventory_level", 0) * optimal_price, 2)
        else:
            profit_uplift = round(pred_profit - curr_profit, 2)
            profit_uplift_pct = round((profit_uplift / curr_profit) * 100, 1) if curr_profit != 0 else 0.0
            spoilage_savings = None

        price_change_pct = (
            round(((optimal_price / live_product["current_price"]) - 1) * 100, 1)
            if live_product["current_price"] else 0.0
        )
    except Exception:
        # Fallback profiles are rougher and more likely to hit an edge case
        # (e.g. cost_price of 0) — don't let one bad new-product event kill
        # the whole micro-batch.
        print(f"  [new product] pricing failed for product_id {pid}, skipping this event:")
        traceback.print_exc()
        return None, None

    scored = {
        "product_id": pid,
        "product_name": live_product.get("product_name", f"Product_{pid}"),
        "department": live_product.get("department", "General"),
        "cost_price": live_product["cost_price"],
        "current_price": live_product["current_price"],
        "recommended_price": optimal_price,
        "price_change_pct": price_change_pct,
        "predicted_demand": pred_demand,
        "predicted_profit": pred_profit,
        "current_profit": curr_profit,
        "profit_uplift": profit_uplift,
        "profit_uplift_pct": profit_uplift_pct,
        "spoilage_savings_estimate": spoilage_savings,
        "optimization_method": method,
        "inventory_ratio": live_product["inventory_ratio"],
        "stock_urgency_category": live_product.get("stock_urgency_category", "Normal"),
        "is_new_product": is_new_product,
    }

    # New products still get logged for training feedback — this is how
    # they eventually stop being "new": once merge_data.py / the next
    # batch run picks them up properly, they'll have real cost/elasticity
    # data instead of these fallback defaults.
    training_row = {
        "product_id": pid,
        "date": pd.Timestamp.now().normalize(),
        "units_sold": live_product["units_sold"],
        "current_price": live_product["current_price"],
        "cost_price": live_product["cost_price"],
        "inventory_ratio": live_product["inventory_ratio"],
        "price_to_cost_ratio": live_product.get(
            "price_to_cost_ratio",
            live_product["current_price"] / live_product["cost_price"] if live_product["cost_price"] else 0
        ),
        "inventory_urgency_score": live_product.get("inventory_urgency_score", 0),
        "sensitivity_encoded": live_product.get("sensitivity_encoded", 0),
    }

    return scored, training_row


def process_batch(batch_df, batch_id):
    reload_if_stale()  # cheap check, most calls return immediately

    if batch_df.isEmpty():
        print(f"[batch {batch_id}] no data this trigger")
        return

    batch_df.write.format("delta").mode("append").save(DELTA_PATH)

    batch_as_pandas = batch_df.toPandas()

    results = []
    training_rows = []
    for _, row in batch_as_pandas.iterrows():
        pid = row["product_id"]
        try:
            scored, training_row = score_one_product(pid, row, _state["product_lookup"])
            if scored is not None:
                results.append(scored)
                training_rows.append(training_row)
        except Exception:
            print(f"[batch {batch_id}] error on product {pid}:")
            traceback.print_exc()

    if not results:
        print(f"[batch {batch_id}] no scoreable products this batch")
        return

    out_df = pd.DataFrame(results)
    out_df.to_csv(OPT_OUTPUT_PATH, mode="a", header=not os.path.exists(OPT_OUTPUT_PATH), index=False)
    new_product_count = out_df["is_new_product"].sum()
    print(f"[batch {batch_id}] wrote {len(out_df)} optimized rows"
          + (f" ({new_product_count} were new/unrecognized products)" if new_product_count else ""))

    training_df = pd.DataFrame(training_rows)
    training_df.to_csv(
        STREAMING_LOG_PATH,
        mode="a",
        header=not os.path.exists(STREAMING_LOG_PATH),
        index=False
    )
    print(f"[batch {batch_id}] logged {len(training_df)} rows to streaming_features_log.csv")


query = (
    windowed.writeStream
    .outputMode("update")
    .foreachBatch(process_batch)
    .option("checkpointLocation", CHECKPOINT_PATH)
    .trigger(processingTime="10 seconds")
    .start()
)

print()
print("=" * 60)
print("STREAMING STARTED")
print(f"Listening for Kafka events... (auto-reload check every {RELOAD_CHECK_INTERVAL_SECONDS}s)")
print("=" * 60)
print()

query.awaitTermination()