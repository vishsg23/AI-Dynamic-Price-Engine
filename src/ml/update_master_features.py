"""
Runs as an Airflow task, BEFORE retrain_all.py.

WHAT IT DOES:
  Appends any rows accumulated in streaming_features_log.csv (written
  throughout the day by stream_processor.py's process_batch()) onto
  master_features.csv, then deletes the log so the same rows aren't
  added again tomorrow.

  This is the step that makes "daily retraining" actually mean something:
  without it, retrain_all.py would keep training on the exact same
  historical data every single day, since nothing else updates
  master_features.csv.

SAFE TO RUN WHEN THERE'S NO NEW DATA:
  If no streaming happened (log file missing or empty), this exits
  cleanly and retrain_all.py just trains on the data it already has.
"""
import os
import sys
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(BASE_DIR)

MASTER_PATH = os.path.join(BASE_DIR, "data", "processed", "master_features.csv")
LOG_PATH = os.path.join(BASE_DIR, "data", "processed", "streaming_features_log.csv")


def main():
    print("=" * 60)
    print("UPDATE MASTER FEATURES FROM STREAMING LOG")
    print("=" * 60)

    if not os.path.exists(LOG_PATH):
        print(f"No streaming log found at {LOG_PATH} — nothing new to add today.")
        print("(This is normal if the stream wasn't running, or ran with zero matched events.)")
        return

    new_rows = pd.read_csv(LOG_PATH)

    if new_rows.empty:
        print("Streaming log exists but is empty — nothing to add.")
        os.remove(LOG_PATH)
        print(f"Cleared empty {LOG_PATH}")
        return

    if not os.path.exists(MASTER_PATH):
        print(f"ERROR: {MASTER_PATH} not found — run merge_data.py first.")
        sys.exit(1)

    master_df = pd.read_csv(MASTER_PATH)

    before_count = len(master_df)
    updated_df = pd.concat([master_df, new_rows], ignore_index=True)
    updated_df.to_csv(MASTER_PATH, index=False)
    after_count = len(updated_df)

    print(f"master_features.csv: {before_count} rows -> {after_count} rows "
          f"(+{len(new_rows)} rows from today's streaming data)")

    # Clear the log so tomorrow's retrain doesn't re-add the same rows again
    os.remove(LOG_PATH)
    print(f"Cleared {LOG_PATH} (rows already folded into master_features.csv)")

    print("\nDone. retrain_all.py will now train on data that includes today's streamed events.")


if __name__ == "__main__":
    main()