import os
import sys

os.environ["HADOOP_HOME"] = r"D:\pricing_engine\.venv\hadoop"
os.environ["hadoop.home.dir"] = r"D:\pricing_engine\.venv\hadoop"
os.environ["PATH"] += os.pathsep + r"D:\pricing_engine\.venv\hadoop\bin"

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

master_path = os.path.join(PROCESSED_DIR, "master_features.csv")
if not os.path.exists(master_path):
    print("❌ Error: master_features.csv not found! Please run merge_data.py first.")
    sys.exit(1)

spark = (
    SparkSession.builder
    .appName("ValidateDataBatch")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

df = spark.read.csv(master_path, header=True, inferSchema=True)

print("==================================================")
print("    SECTION 4: CORE PIPELINE DATA VALIDATION     ")
print("           (PySpark version)                      ")
print("==================================================\n")

failed_checks = []
total_expectations = 7


def check_expectation(description, success_condition, failure_details=""):
    """
    Records a pass/fail for one data-quality rule. Doesn't raise or stop
    anything on its own — just logs a failure so we can report everything
    that's wrong at once, instead of stopping at the first problem.
    """
    if success_condition:
        return True
    failed_checks.append(f"❌ FAILED: {description}\n   └── Detail: {failure_details}")
    return False


columns = df.columns

# 1. product_id must always be present
null_ids = df.filter(F.col("product_id").isNull()).count()
check_expectation("product_id must not be null", null_ids == 0, f"Found {null_ids} missing IDs")

# 2. Prices must be within realistic retail range
if "current_price" in columns:
    price_col = "current_price"
elif "current_base_price" in columns:
    price_col = "current_base_price"
else:
    price_col = None

if price_col:
    invalid_prices = df.filter((F.col(price_col) < 0.50) | (F.col(price_col) > 500.00)).count()
    check_expectation(
        f"{price_col} within range ($0.50 - $500.00)",
        invalid_prices == 0,
        f"Found {invalid_prices} rows outside valid bounds"
    )
else:
    check_expectation("Price column check", False, "Neither 'current_price' nor 'current_base_price' found")

# 3. Cost price must be positive
if "cost_price" in columns:
    invalid_costs = df.filter(F.col("cost_price") < 0.01).count()
    check_expectation("cost_price must be positive", invalid_costs == 0, f"Found {invalid_costs} rows with zero/negative cost")
else:
    check_expectation("cost_price presence", False, "Column 'cost_price' missing")

# 4. Units sold must be >= 0
if "total_units_sold" in columns:
    invalid_units = df.filter(F.col("total_units_sold") < 0).count()
    check_expectation("total_units_sold must be >= 0", invalid_units == 0, f"Found {invalid_units} rows with negative sales")

# 5. Inventory ratio must be between 0 and 1 (if present)
if "inventory_ratio" in columns:
    invalid_inv = df.filter((F.col("inventory_ratio") < 0.0) | (F.col("inventory_ratio") > 1.0)).count()
    check_expectation("inventory_ratio between 0.0 and 1.0", invalid_inv == 0, f"Found {invalid_inv} rows outside bounds")

# 6. Sensitivity category must be valid (if present)
if "price_sensitivity_category" in columns:
    invalid_cats = df.filter(~F.col("price_sensitivity_category").isin(["High", "Medium", "Low"])).count()
    check_expectation(
        "price_sensitivity_category in ['High', 'Medium', 'Low']",
        invalid_cats == 0,
        f"Found {invalid_cats} unrecognized categories"
    )

# 7. Department must not be null
if "department" in columns:
    null_depts = df.filter(F.col("department").isNull()).count()
    check_expectation("department must not be null", null_depts == 0, f"Found {null_depts} missing departments")

# --- Final Evaluation Summary ---
passed_expectations = total_expectations - len(failed_checks)
validation_success = len(failed_checks) == 0

print(f"Validation passed: {validation_success}")
print(f"Total expectations: {total_expectations}")
print(f"Passed: {passed_expectations} / {total_expectations}")

if not validation_success:
    print("\nValidation failures detected:")
    for failure in failed_checks:
        print(failure)
    print("==================================================")
    spark.stop()
    raise ValueError("Data validation FAILED – pipeline stopped")

print("\n All validations PASSED – safe to proceed to pricing strategy!")
print("==================================================")

spark.stop()