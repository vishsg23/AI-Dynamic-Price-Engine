import os
import sys
import shutil

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

os.environ["HADOOP_HOME"] = r"D:\pricing_engine\.venv\hadoop"
os.environ["hadoop.home.dir"] = r"D:\pricing_engine\.venv\hadoop"
os.environ["PATH"] += os.pathsep + r"D:\pricing_engine\.venv\hadoop\bin"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

print("==================================================")
print("     SECTION 3: RUNNING SAFE DATA CLEANING        ")
print("           (PySpark version)                      ")
print("==================================================\n")

spark = (
    SparkSession.builder
    .appName("CleanDataBatch")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print("Spark Session Started\n")


def save_as_one_clean_csv(spark_dataframe, final_file_path):
    """
    Spark normally writes output as a FOLDER containing several small
    partitioned files, not one clean CSV. Since every other script in
    this project (train_xgboost_demand, optimization.py, etc.) expects
    to open ONE exact file with pandas, this function:
      1. Tells Spark to write everything as a single file (coalesce(1))
      2. Finds that one file inside the temporary folder Spark created
      3. Moves and renames it to the exact filename we actually want
      4. Deletes the leftover temporary folder
    This keeps every downstream script working exactly as before,
    even though the cleaning itself now happens in Spark.
    """
    temp_folder = final_file_path + "_spark_temp"

    spark_dataframe.coalesce(1).write.mode("overwrite").option("header", True).csv(temp_folder)

    written_files = [f for f in os.listdir(temp_folder) if f.startswith("part-") and f.endswith(".csv")]
    spark_generated_file = os.path.join(temp_folder, written_files[0])

    if os.path.exists(final_file_path):
        os.remove(final_file_path)
    shutil.move(spark_generated_file, final_file_path)
    shutil.rmtree(temp_folder)


# ==========================================
# 1. CLEANING: sales_transactions.csv
# ==========================================
sales_path = os.path.join(RAW_DIR, "sales_transactions.csv")
if os.path.exists(sales_path):
    print("Cleaning sales_transactions.csv...")

    sales = spark.read.csv(sales_path, header=True, inferSchema=True)

    sales = sales.filter(F.col("units_sold") >= 0)
    sales = sales.filter(F.col("current_price") >= F.col("cost_price"))

    sales = sales.na.fill({"department": "Unknown"})

    sales = sales.withColumn("date", F.to_date("date"))
    sales = sales.withColumn("date_only", F.col("date"))
    sales = sales.withColumn("year_month", F.date_format("date", "yyyy-MM"))

    save_as_one_clean_csv(sales, os.path.join(PROCESSED_DIR, "sales_clean.csv"))
    print("  Saved cleanly to data/processed/sales_clean.csv")
print("-" * 50) #here we call the function 

# ==========================================
# 2. CLEANING: price_elasticity.csv
# ==========================================
elasticity_path = os.path.join(RAW_DIR, "price_elasticity.csv")
if os.path.exists(elasticity_path):
    print("Cleaning price_elasticity.csv...")

    elasticity = spark.read.csv(elasticity_path, header=True, inferSchema=True)
    column_names = elasticity.columns

    if "recommended_optimal_price" in column_names:
        elasticity = elasticity.withColumn(
            "recommended_optimal_price",
            F.when(
                F.col("recommended_optimal_price").isin(float("inf"), float("-inf")),
                None
            ).otherwise(F.col("recommended_optimal_price"))
        )

        if "avg_price" in column_names:
            price_column = "avg_price"
        elif "cost_price" in column_names:
            price_column = "cost_price"
        else:
            price_column = column_names[3]

        department_median_prices = (
            elasticity.groupBy("department")
            .agg(F.expr(f"percentile_approx({price_column}, 0.5)").alias("department_median_price"))
        )
        elasticity = elasticity.join(department_median_prices, on="department", how="left")
        elasticity = elasticity.withColumn(
            "recommended_optimal_price",
            F.coalesce(F.col("recommended_optimal_price"), F.col("department_median_price"))
        ).drop("department_median_price")

    sensitivity_columns = [c for c in column_names if "sensitivity" in c and "category" not in c]
    if sensitivity_columns:
        sensitivity_column = sensitivity_columns[0]
        elasticity = elasticity.withColumn(
            sensitivity_column,
            F.when(
                F.col(sensitivity_column).isin(float("inf"), float("-inf")),
                0
            ).otherwise(F.col(sensitivity_column))
        )

    if "department" in column_names:
        elasticity = elasticity.na.fill({"department": "Unknown"})

    category_columns = [c for c in column_names if "category" in c]
    if category_columns:
        category_column = category_columns[0]
        elasticity = elasticity.withColumn(
            "sensitivity_encoded",
            F.when(F.col(category_column) == "High", 3)
             .when(F.col(category_column) == "Medium", 2)
             .when(F.col(category_column) == "Low", 1)
             .otherwise(1)
        )

    save_as_one_clean_csv(elasticity, os.path.join(PROCESSED_DIR, "elasticity_clean.csv"))
    print("   Saved cleanly to data/processed/elasticity_clean.csv")
print("-" * 50)

# ==========================================
# 3. CLEANING: demand_forecast.csv
# ==========================================
forecast_path = os.path.join(RAW_DIR, "demand_forecast.csv")
if os.path.exists(forecast_path):
    print("🧹 Cleaning demand_forecast.csv...")

    forecast = spark.read.csv(forecast_path, header=True, inferSchema=True)
    column_names = forecast.columns

    date_columns = [c for c in column_names if "date" in c]
    if date_columns:
        forecast = forecast.withColumn(date_columns[0], F.to_date(F.col(date_columns[0])))

    low_columns = [c for c in column_names if "low" in c]
    mid_columns = [c for c in column_names if "mid" in c or "expected" in c or "mean" in c]
    high_columns = [c for c in column_names if "high" in c]

    if low_columns and mid_columns and high_columns:
        low_col, mid_col, high_col = low_columns[0], mid_columns[0], high_columns[0]

        forecast = forecast.filter(
            (F.col(low_col) <= F.col(mid_col)) & (F.col(mid_col) <= F.col(high_col))
        )

        forecast = forecast.filter(
            ~((F.col(low_col) == 0) & (F.col(mid_col) == 0) & (F.col(high_col) == 0))
        )

    if "festival_flag" in column_names:
        forecast = forecast.na.fill({"festival_flag": 0})
        forecast = forecast.withColumn("festival_flag", F.col("festival_flag").cast("int"))

    save_as_one_clean_csv(forecast, os.path.join(PROCESSED_DIR, "forecast_clean.csv"))
    print("  Saved cleanly to data/processed/forecast_clean.csv")
print("-" * 50)

# ==========================================
# 4. CLEANING: product_catalog.csv
# ==========================================
catalog_path = os.path.join(RAW_DIR, "product_catalog.csv")
if os.path.exists(catalog_path):
    print("Cleaning product_catalog.csv...")

    catalog = spark.read.csv(catalog_path, header=True, inferSchema=True)
    if "department" in catalog.columns:
        catalog = catalog.na.fill({"department": "Unknown"})

    save_as_one_clean_csv(catalog, os.path.join(PROCESSED_DIR, "catalog_clean.csv"))
    print(" Saved cleanly to data/processed/catalog_clean.csv")
print("-" * 50)

# ==========================================
# 5. CLEANING: optimal_pricing_output.csv
# ==========================================
optimal_path = os.path.join(RAW_DIR, "optimal_pricing_output.csv")
if os.path.exists(optimal_path):
    print("Cleaning optimal_pricing_output.csv...")

    optimal = spark.read.csv(optimal_path, header=True, inferSchema=True)

    save_as_one_clean_csv(optimal, os.path.join(PROCESSED_DIR, "optimal_clean.csv"))
    print(" Saved cleanly to data/processed/optimal_clean.csv")

print("\n==================================================")

spark.stop()