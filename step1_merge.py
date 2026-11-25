import os
import shutil
import pandas as pd

# Kaggle paths (automatically falls back to local workspace if not running on Kaggle)
KAGGLE_BASE = "/kaggle/input/datasets/sanjeedaafrose/dataset"
if os.path.exists(KAGGLE_BASE):
    P1_IMG_DIR = f"{KAGGLE_BASE}/Phase1-20261125T190005Z-1-001/Phase1"
    P2_IMG_DIR = f"{KAGGLE_BASE}/Phase2-20261125T190139Z-1-001/Phase2"
    P1_EXCEL   = f"{KAGGLE_BASE}/phase1.xlsx"
    P2_EXCEL   = f"{KAGGLE_BASE}/Phase2.xlsx"
    OUT_DIR    = "/kaggle/working/merged_dataset"
else:
    LOCAL_BASE = r"f:\Priya Apu\Dataset"
    P1_IMG_DIR = os.path.join(LOCAL_BASE, r"Phase1-20261125T190005Z-1-001\Phase1")
    P2_IMG_DIR = os.path.join(LOCAL_BASE, r"Phase2-20261125T190139Z-1-001\Phase2")
    P1_EXCEL   = os.path.join(LOCAL_BASE, "phase1.xlsx")
    P2_EXCEL   = os.path.join(LOCAL_BASE, "Phase2.xlsx")
    OUT_DIR    = os.path.join(LOCAL_BASE, "merged_dataset")

OUT_IMG_DIR = os.path.join(OUT_DIR, "images")
OUT_EXCEL   = os.path.join(OUT_DIR, "merged_labels.xlsx")

if os.path.exists(OUT_IMG_DIR):
    shutil.rmtree(OUT_IMG_DIR)
os.makedirs(OUT_IMG_DIR, exist_ok=True)

# 1. Load both Excel files and unify column names
df1 = pd.read_excel(P1_EXCEL).rename(columns={
    "Img Reference": "orig_image",
    "ES Reference": "es_code",
    "Status": "status",
})
df1["phase"] = "Phase1"

df2 = pd.read_excel(P2_EXCEL).rename(columns={
    "Photo": "orig_image",
    "ES Checkpoint": "es_code",
    "Status": "status",
})
df2["phase"] = "Phase2"

# 2. Standardize class names across the ENTIRE dataset
STATUS_MAP = {
    "acceptable": "Acceptable",
    "unacceptable": "Unacceptable",
    "onsite verification required": "Onsite Verification",
    "need onsite verification": "Onsite Verification",
    "onsite verification": "Onsite Verification",
}

def process_phase(df, src_dir, prefix):
    # Case-insensitive filename lookup (safe for Kaggle Linux)
    disk_files = {
        f.strip().lower(): f
        for f in os.listdir(src_dir)
        if os.path.isfile(os.path.join(src_dir, f))
    }

    df = df.copy()
    df["orig_clean"] = df["orig_image"].astype(str).str.strip()
    df["es_code"] = df["es_code"].astype(str).str.strip().str.upper()
    df["status_raw"] = df["status"].astype(str).str.strip()
    df["status"] = df["status_raw"].str.lower().map(STATUS_MAP)

    unmapped = df[df["status"].isna()]["status_raw"].unique()
    if len(unmapped) > 0:
        raise ValueError(f"Unmapped status values found: {unmapped}")

    # Match rows whose image actually exists in the folder
    df["real_fname"] = df["orig_clean"].str.lower().map(disk_files)
    missing_rows = int(df["real_fname"].isna().sum())
    df = df.dropna(subset=["real_fname"]).copy()

    # Standardized unique filename in merged folder (e.g., p1_0001.jpg, p2_01.jpg)
    df["image_name"] = df["real_fname"].apply(
        lambda x: f"{prefix}_{os.path.splitext(x)[0].lower()}.jpg"
    )

    # Copy matched images to single merged folder
    unique_imgs = df[["real_fname", "image_name"]].drop_duplicates()
    for real_f, new_f in unique_imgs.itertuples(index=False):
        shutil.copy2(os.path.join(src_dir, real_f), os.path.join(OUT_IMG_DIR, new_f))

    print(
        f"{prefix.upper()}: copied {len(unique_imgs)} unique images ({len(df)} rows); "
        f"dropped {missing_rows} rows with missing image files."
    )
    return df[["image_name", "es_code", "status", "phase"]]

clean1 = process_phase(df1, P1_IMG_DIR, "p1")
clean2 = process_phase(df2, P2_IMG_DIR, "p2")

merged_df = pd.concat([clean1, clean2], ignore_index=True)

# 3. Deduplicate repeated (image_name, es_code) pairs using safety priority:
#    Unacceptable > Onsite Verification > Acceptable
priority = {"Unacceptable": 0, "Onsite Verification": 1, "Acceptable": 2}
merged_df["_prio"] = merged_df["status"].map(priority)
rows_before = len(merged_df)
merged_df = (
    merged_df.sort_values("_prio")
    .drop_duplicates(subset=["image_name", "es_code"], keep="first")
    .drop(columns=["_prio"])
    .sort_values(["image_name", "es_code"])
    .reset_index(drop=True)
)
print(f"Deduplicated (image_name, es_code): {rows_before} -> {len(merged_df)} rows")

# 4. Save single merged Excel file
merged_df.to_excel(OUT_EXCEL, index=False)

print("\n=== MERGED DATASET SUMMARY ===")
print("Merged folder path :", OUT_IMG_DIR)
print("Merged Excel path  :", OUT_EXCEL)
print("Images in folder   :", len(os.listdir(OUT_IMG_DIR)))
print("Unique imgs (Excel):", merged_df["image_name"].nunique())
print("Total rows (Excel) :", len(merged_df))
print("Unique ES codes    :", merged_df["es_code"].nunique())
print("\nUnified Class Counts (Full Dataset):")
print(merged_df["status"].value_counts().to_string())
print("\nFirst 8 rows:")
print(merged_df.head(8).to_string(index=False))
