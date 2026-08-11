import os
import shutil
import pandas as pd

# ---- Paths ----
# This script lives in src/, alongside merge_data.py / clean_data.py /
# generate_datasets.py, so go up one level to reach the project root.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MASTER_PATH = os.path.join(PROCESSED_DIR, "master_features.csv")
HOLDOUT_PATH = os.path.join(PROCESSED_DIR, "streaming_holdout.csv")
BACKUP_PATH = os.path.join(PROCESSED_DIR, "master_features_FULL_BACKUP.csv")

# ---- Split ratio ----
HOLDOUT_FRACTION = 0.15   # newest 15% of rows -> streaming_holdout.csv
                           # remaining 85% stays in master_features.csv
                           # (demand_model.py then splits that 85% into 70/15
                           #  using split_idx = int(len(df) * (0.70/0.85)))


def main():
    # ---- Safety checks ----
    if not os.path.exists(MASTER_PATH):
        raise SystemExit(f"ERROR: {MASTER_PATH} not found.")

    if os.path.exists(HOLDOUT_PATH):
        raise SystemExit(
            f"ERROR: {HOLDOUT_PATH} already exists.\n"
            f"This script has probably already been run once.\n"
            f"Delete {HOLDOUT_PATH} manually first if you really want to re-split."
        )

    # ---- Load ----
    df = pd.read_csv(MASTER_PATH)
    print(f"Loaded {MASTER_PATH}: {len(df)} total rows")

    # ---- Find the date column ----
    date_col = None
    for col in df.columns:
        if col.lower() == "date":
            date_col = col
            break
    if date_col is None:
        raise SystemExit("ERROR: No 'date' column found — cannot do a chronological split.")

    # ---- Sort chronologically (oldest first) ----
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # ---- Cut at the 85% mark ----
    split_idx = int(len(df) * (1 - HOLDOUT_FRACTION))
    remaining_df = df.iloc[:split_idx].copy()   # oldest 85% -> stays as master_features.csv
    holdout_df = df.iloc[split_idx:].copy()     # newest 15% -> streaming_holdout.csv

    # ---- Print exact numbers, no ambiguity ----
    total = len(df)
    print("\n--- SPLIT RESULT ---")
    print(f"Total rows:            {total}")
    print(f"Remaining (train/test): {len(remaining_df)} rows "
          f"({len(remaining_df)/total:.1%}) "
          f"[{remaining_df[date_col].min().date()} to {remaining_df[date_col].max().date()}]")
    print(f"Streaming holdout:      {len(holdout_df)} rows "
          f"({len(holdout_df)/total:.1%}) "
          f"[{holdout_df[date_col].min().date()} to {holdout_df[date_col].max().date()}]")

    # ---- Back up original file before overwriting ----
    shutil.copy(MASTER_PATH, BACKUP_PATH)
    print(f"\nOriginal file backed up to: {BACKUP_PATH}")

    # ---- Write outputs ----
    remaining_df.to_csv(MASTER_PATH, index=False)
    holdout_df.to_csv(HOLDOUT_PATH, index=False)
    print(f"Overwrote:  {MASTER_PATH}")
    print(f"Created:    {HOLDOUT_PATH}")

    print("\nDONE.")
    print("Next steps:")
    print("  1. In producer.py, change the data_path to point to streaming_holdout.csv")
    print("  2. demand_model.py's split ratio (0.70/0.85) already matches this split — no change needed there")


if __name__ == "__main__":
    main()