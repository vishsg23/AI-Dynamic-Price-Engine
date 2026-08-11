import os
import subprocess
import sys

REQUIRED_PACKAGES = ['pyspark']
for package_name in REQUIRED_PACKAGES:
    try:
        __import__(package_name)
    except ImportError:
        print(f"{package_name} is missing! Installing it automatically...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"{package_name} successfully installed!")

from pyspark.sql import SparkSession

os.environ["HADOOP_HOME"] = r"D:\pricing_engine\.venv\hadoop"
os.environ["hadoop.home.dir"] = r"D:\pricing_engine\.venv\hadoop"
os.environ["PATH"] += os.pathsep + r"D:\pricing_engine\.venv\hadoop\bin"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

print("==================================================")
print("     SECTION 2: COMPLETE DATA PIPELINE AUDIT      ")
print("           (PySpark version)                      ")
print("==================================================\n")

spark = (
    SparkSession.builder
    .appName("DataAuditBatch")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print(" Spark Session Started\n")


def ingest_and_audit(file_name):
    """
    Opens one raw data file using Spark and prints a quick summary:
    how many rows, how many columns, and the first few column names.
    This is just a health-check — it doesn't clean or change anything.
    """
    target_path = os.path.join(RAW_DATA_DIR, file_name)

    if not os.path.exists(target_path):
        print(f" Connection Broken: '{file_name}' is missing.")
        print(f"   Expected Location: {target_path}\n")
        return None

    try:
        if file_name.endswith(".xlsx"):
            try:
                spark_dataframe = spark.read.format("com.crealytics.spark.excel") \
                    .option("header", "true") \
                    .load(target_path)
            except Exception:
                import pandas as pd
                pandas_dataframe = pd.read_excel(target_path)
                spark_dataframe = spark.createDataFrame(pandas_dataframe)
        else:
            spark_dataframe = spark.read.csv(target_path, header=True, inferSchema=True)

        row_count = spark_dataframe.count()
        column_count = len(spark_dataframe.columns)
        first_few_columns = spark_dataframe.columns[:4]

        print(f"Successfully Ingested: '{file_name}'")
        print(f"   ├── Structure: {row_count} rows x {column_count} columns")
        print(f"   └── Sample Fields: {first_few_columns}...")
        print("-" * 50)
        return spark_dataframe

    except Exception as e:
        print(f" Error reading '{file_name}': {str(e)}\n")
        return None


my_datasets = [
    "product_catalog.csv",
    "sales_transactions.csv",
    "price_elasticity.csv",
    "demand_features.csv",
    "demand_forecast.csv",
    "optimal_pricing_output.csv",
    "Grocery_Pricing_Optimization_Dataset.xlsx"
]

for file_name in my_datasets:
    ingest_and_audit(file_name)

spark.stop()