import os
import sys
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
# Everything below this point runs ONCE, before the stream starts:
# load the trained models and build a small in-memory lookup table
# so we're not hitting disk or re-loading models for every event.
# ----------------------------------------------------------------
print("Loading master_features.csv...")
master_df = pd.read_csv(MASTER_FEATURES_PATH)

print("Loading trained models (XGBoost + Prophet)...")
prediction_service.load_models()

print("Building static per-product lookup table...")
product_lookup = build_product_level_df(master_df)

# Not every product row is guaranteed to have these columns depending
# on how master_features.csv was built, so fall back to something
# sane rather than crashing the whole stream over a missing column.
if "base_price" not in product_lookup.columns:
    product_lookup["base_price"] = product_lookup["current_price"]
if "cost_price" not in product_lookup.columns:
    product_lookup["cost_price"] = product_lookup["current_price"] * 0.7
if "inventory_level" not in product_lookup.columns:
    product_lookup["inventory_level"] = 0

product_lookup = product_lookup.set_index("product_id")
print(f"Ready. {len(product_lookup)} products in lookup table.")

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

# Shape of the JSON events coming off the Kafka topic.
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

# Kafka gives us raw bytes in a "value" column — unpack the JSON out of
# it, then drop anything that's missing a product id or has a
# timestamp we can't parse, since those rows are useless downstream.
parsed = (
    kafka_stream
    .select(from_json(col("value").cast("string"), event_schema).alias("data"))
    .select("data.*")
    .filter(col("product_id").isNotNull())
    .withColumn("event_time", to_timestamp(col("timestamp")))
    .filter(col("event_time").isNotNull())
)

print("Kafka Connected")

# Roll individual order events up into 1-minute buckets per product.
# The watermark tells Spark how long to wait for late-arriving events
# before it considers a window "closed" and safe to emit.
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


def score_one_product(pid, agg_row, product_lookup):
    """
    Takes one row of windowed Kafka data (already averaged/summed per
    product for this 1-minute window) and runs it through the pricing
    model. Returns a dict ready to go into the output CSV, or None if
    the product isn't one we have a model/lookup entry for.
    """
    if pid not in product_lookup.index:
        return None

    # Start from the product's static profile (department, cost,
    # base price, etc.) and overlay it with what's actually happening
    # right now in the stream.
    live_product = product_lookup.loc[pid].to_dict()
    live_product["product_id"] = pid
    live_product["current_price"] = agg_row["avg_price"]
    live_product["units_sold"] = agg_row["hourly_demand"]
    live_product["inventory_ratio"] = agg_row["avg_inventory"]

    optimal_price, method = optimize_price(live_product)
    pred_demand = prediction_service.predict_demand(optimal_price, live_product)
    pred_profit = round((optimal_price - live_product["cost_price"]) * pred_demand, 2)

    # For "current profit," ask the model what it thinks demand would be
    # at today's price — same model, same basis optimize_price already
    # used — rather than trusting the raw streaming count. Comparing
    # profit at two different measurement bases would make the uplift
    # number meaningless.
    pred_demand_at_current = prediction_service.predict_demand(
        live_product["current_price"], live_product
    )
    curr_profit = round(
        (live_product["current_price"] - live_product["cost_price"]) * pred_demand_at_current, 2
    )

    # An expiry-driven discount isn't trying to beat normal sales — it's
    # trying to avoid a total write-off on stock that's about to go bad.
    # So "profit uplift" doesn't mean anything for these rows; report
    # how much spoilage we're avoiding instead of a confusing negative number.
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

    return {
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
    }


def process_batch(batch_df, batch_id):
    """
    Runs once per micro-batch (every 10 seconds, per the trigger below).
    Archives the raw windowed aggregates to Delta for audit/history,
    then scores whatever products showed up and appends the results
    to the output CSV.
    """
    if batch_df.isEmpty():
        print(f"[batch {batch_id}] no data this trigger")
        return

    batch_df.write.format("delta").mode("append").save(DELTA_PATH)

    # This batch is one row per product per 1-minute window — small
    # enough that pulling it into pandas is cheap, and it's the only
    # way to call the existing pandas-based model code.
    batch_as_pandas = batch_df.toPandas()

    results = []
    for _, row in batch_as_pandas.iterrows():
        pid = row["product_id"]
        try:
            scored = score_one_product(pid, row, product_lookup)
            if scored is not None:
                results.append(scored)
        except Exception:
            print(f"[batch {batch_id}] error on product {pid}:")
            traceback.print_exc()

    if not results:
        print(f"[batch {batch_id}] no matching products in lookup")
        return

    out_df = pd.DataFrame(results)
    out_df.to_csv(OPT_OUTPUT_PATH, mode="a", header=not os.path.exists(OPT_OUTPUT_PATH), index=False)
    print(f"[batch {batch_id}] wrote {len(out_df)} optimized rows")


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
print("Listening for Kafka events...")
print("=" * 60)
print()

query.awaitTermination()