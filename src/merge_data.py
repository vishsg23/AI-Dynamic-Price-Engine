import os
import sys
import shutil

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ==========================================================
# WINDOWS HADOOP CONFIGURATION (same as every other Spark
# script in this project)
# ==========================================================
os.environ["HADOOP_HOME"] = r"D:\pricing_engine\.venv\hadoop"
os.environ["hadoop.home.dir"] = r"D:\pricing_engine\.venv\hadoop"
os.environ["PATH"] += os.pathsep + r"D:\pricing_engine\.venv\hadoop\bin"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

print("==================================================")
print("     STEP 4: BUILDING CONSOLIDATED FEATURE VIEW   ")
print("           (PySpark version)                      ")
print("==================================================\n")

required_files = ['catalog_clean.csv', 'sales_clean.csv', 'elasticity_clean.csv', 'forecast_clean.csv', 'optimal_clean.csv']
for file_name in required_files:
    if not os.path.exists(os.path.join(PROCESSED_DIR, file_name)):
        print(f"Error: {file_name} is missing! Please run clean_data.py first.")
        sys.exit(1)

spark = (
    SparkSession.builder
    .appName("MergeDataBatch")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print("Spark Session Started\n")


def save_as_one_clean_csv(spark_dataframe, final_file_path):
    """
    Spark writes output as a FOLDER of small files by default, not one
    clean CSV. This function forces Spark to write just one file, then
    renames it to the exact file path we actually want — so every other
    script in the project can still open it normally with pandas.
    """
    temp_folder = final_file_path + "_spark_temp"

    spark_dataframe.coalesce(1).write.mode("overwrite").option("header", True).csv(temp_folder)

    files_written = os.listdir(temp_folder)
    spark_output_file = None
    for file_name in files_written:
        if file_name.startswith("part-") and file_name.endswith(".csv"):
            spark_output_file = os.path.join(temp_folder, file_name)

    if os.path.exists(final_file_path):
        os.remove(final_file_path)
    shutil.move(spark_output_file, final_file_path)
    shutil.rmtree(temp_folder)


# ==========================================
# 1. LOAD ALL 5 CLEANED FILES
# ==========================================
print("Loading all 5 cleaned processing matrices...")
catalog = spark.read.csv(os.path.join(PROCESSED_DIR, 'catalog_clean.csv'), header=True, inferSchema=True)
sales = spark.read.csv(os.path.join(PROCESSED_DIR, 'sales_clean.csv'), header=True, inferSchema=True)
elasticity = spark.read.csv(os.path.join(PROCESSED_DIR, 'elasticity_clean.csv'), header=True, inferSchema=True)
forecast = spark.read.csv(os.path.join(PROCESSED_DIR, 'forecast_clean.csv'), header=True, inferSchema=True)
optimal = spark.read.csv(os.path.join(PROCESSED_DIR, 'optimal_clean.csv'), header=True, inferSchema=True)

# ==========================================
# 2. ROLLING 7-DAY FEATURES
# ==========================================
# "Window functions" in Spark let us calculate something like a rolling
# average WITHOUT collapsing the data down to one row per product. Each
# row keeps its own moving average, calculated by looking at that same
# product's previous 6 rows plus itself.
print("Engineering rolling time-series demand structures...")

sales = sales.orderBy("product_id", "date")

seven_day_window = Window.partitionBy("product_id").orderBy("date").rowsBetween(-6, 0)

sales = sales.withColumn("moving_avg_7day", F.avg("units_sold").over(seven_day_window))
sales = sales.withColumn("demand_volatility", F.stddev_samp("units_sold").over(seven_day_window))
sales = sales.na.fill({"demand_volatility": 0.0})

# ==========================================
# 3. JOIN CATALOG INTO SALES
# ==========================================
# Sales stays at its REAL per-transaction grain the whole way through —
# this is the important fix from before. We never shrink sales down to
# one row per product; every join below adds columns onto each real row.
print("Building master dataset at REAL daily transaction grain...")

# Some columns exist in both `sales` and `catalog` (like product_name).
# In Spark, if we join without handling this, we'd end up with two
# columns that have the exact same name, which causes errors later.
# So before joining, we rename catalog's overlapping columns to add
# "_cat" at the end — matching what the old pandas suffixes=('', '_cat')
# used to do automatically.
sales_column_names = sales.columns

catalog_renamed = catalog
for column_name in catalog.columns:
    if column_name in sales_column_names and column_name != "product_id":
        catalog_renamed = catalog_renamed.withColumnRenamed(column_name, column_name + "_cat")

master = sales.join(catalog_renamed, on="product_id", how="left")

# ==========================================
# 4. JOIN ELASTICITY DATA
# ==========================================
master_column_names = master.columns

elasticity_renamed = elasticity
for column_name in elasticity.columns:
    if column_name in master_column_names and column_name != "product_id":
        elasticity_renamed = elasticity_renamed.withColumnRenamed(column_name, column_name + "_elas")

master = master.join(elasticity_renamed, on="product_id", how="left")

# ==========================================
# 5. JOIN OPTIMAL PRICING DATA (joined on product_name, not product_id)
# ==========================================
master_column_names = master.columns

optimal_renamed = optimal
for column_name in optimal.columns:
    if column_name in master_column_names and column_name != "product_name":
        optimal_renamed = optimal_renamed.withColumnRenamed(column_name, column_name + "_opt")

master = master.join(optimal_renamed, on="product_name", how="left")

# ==========================================
# 6. ADD FORECAST DATA (summarized, not joined row-by-row)
# ==========================================
# forecast_clean.csv is a 30-day-ahead PREDICTION, not real historical
# daily data — it doesn't share real dates with `sales`. Joining it
# row-by-row would just copy one forecast value across many rows,
# which is the exact mistake that broke this pipeline before. Instead,
# we summarize it down to one average per product first.
print("Summarizing forecast data per product...")

forecast_summary = forecast.groupBy("product_id").agg(
    F.avg("predicted_units").alias("predicted_units_avg"),
    F.avg("predicted_revenue").alias("predicted_revenue_avg"),
    F.avg("confidence_pct").alias("confidence_pct_avg"),
)

master = master.join(forecast_summary, on="product_id", how="left")

# ==========================================
# 7. TOTAL UNITS SOLD PER PRODUCT
# ==========================================
# This adds up total units sold for each product, and repeats that same
# total on every row belonging to that product (this is what
# .transform('sum') did in pandas).
product_total_window = Window.partitionBy("product_id")
master = master.withColumn("total_units_sold", F.sum("units_sold").over(product_total_window))

# ==========================================
# 8. BUSINESS LOGIC FEATURES
# ==========================================
print("Injecting intelligent optimization business features...")

master = master.withColumn("inventory_urgency_score", F.lit(1) - F.col("inventory_ratio"))

master = master.withColumn("price_to_cost_ratio", F.col("current_price") / F.col("cost_price"))

if "is_perishable" in master.columns and "shelf_life_days" in master.columns:
    master = master.withColumn(
        "expiry_risk_flag",
        F.when((F.col("is_perishable") == 1) & (F.col("shelf_life_days") < 14), 1).otherwise(0)
    )
else:
    master = master.withColumn("expiry_risk_flag", F.lit(0))

# ---- Weekend demand boost ----
# Step A: average units sold on weekends, per product
weekend_sales_avg = sales.filter(F.col("is_weekend") == 1) \
    .groupBy("product_id") \
    .agg(F.avg("units_sold").alias("weekend_avg"))

# Step B: average units sold on regular weekdays, per product
weekday_sales_avg = sales.filter(F.col("is_weekend") == 0) \
    .groupBy("product_id") \
    .agg(F.avg("units_sold").alias("weekday_avg"))

master = master.join(weekend_sales_avg, on="product_id", how="left")
master = master.join(weekday_sales_avg, on="product_id", how="left")

# If weekday_avg is 0, use 1 instead so we don't divide by zero.
master = master.withColumn(
    "weekday_avg_safe",
    F.when(F.col("weekday_avg") == 0, 1).otherwise(F.col("weekday_avg"))
)
master = master.withColumn("weekend_demand_boost", F.col("weekend_avg") / F.col("weekday_avg_safe"))
master = master.drop("weekend_avg", "weekday_avg", "weekday_avg_safe")

# ---- Festival demand multiplier ----
festival_sales_avg = sales.filter(F.col("festival_flag") == 1) \
    .groupBy("product_id") \
    .agg(F.avg("units_sold").alias("festival_avg"))

normal_sales_avg = sales.filter(F.col("festival_flag") == 0) \
    .groupBy("product_id") \
    .agg(F.avg("units_sold").alias("normal_avg"))

master = master.join(festival_sales_avg, on="product_id", how="left")
master = master.join(normal_sales_avg, on="product_id", how="left")

master = master.withColumn(
    "normal_avg_safe",
    F.when(F.col("normal_avg") == 0, 1).otherwise(F.col("normal_avg"))
)
master = master.withColumn("festival_demand_multiplier", F.col("festival_avg") / F.col("normal_avg_safe"))
master = master.na.fill({"festival_demand_multiplier": 1.0})
master = master.drop("festival_avg", "normal_avg", "normal_avg_safe")

# ---- Profit margin percentage ----
master = master.withColumn(
    "profit_margin_pct",
    ((F.col("current_price") - F.col("cost_price")) / F.col("current_price")) * 100
)

# ---- Competitor price gap ----
if "recommended_optimal_price" in master.columns:
    master = master.withColumn("competitor_price_gap", F.col("recommended_optimal_price") - F.col("current_price"))
else:
    master = master.withColumn("competitor_price_gap", F.lit(0.0))

# ---- Stock urgency category ----
# Turns the numeric inventory_ratio into a simple label anyone can read.
master = master.withColumn(
    "stock_urgency_category",
    F.when(F.col("inventory_ratio").isNull(), "Normal")
     .when(F.col("inventory_ratio") < 0.2, "Critical")
     .when(F.col("inventory_ratio") > 0.7, "Low")
     .otherwise("Normal")
)

# ==========================================
# 9. SAVE THE FINAL FILE
# ==========================================
output_path = os.path.join(PROCESSED_DIR, "master_features.csv")
save_as_one_clean_csv(master, output_path)

row_count = master.count()
column_count = len(master.columns)

print(f"\nSuccess! Created complete consolidated feature matrix view.")
print(f"   └── Structure: {row_count} rows x {column_count} features")
print(f"   └── Location: data/processed/master_features.csv")
print("==================================================")

spark.stop()