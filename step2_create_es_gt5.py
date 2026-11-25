import os
import shutil
import pandas as pd

KAGGLE_BASE = "/kaggle/input/datasets/sanjeedaafrose/dataset"
if os.path.exists(KAGGLE_BASE):
    P1_IMG_DIR = f"{KAGGLE_BASE}/Phase1-20261125T190005Z-1-001/Phase1"
    P2_IMG_DIR = f"{KAGGLE_BASE}/Phase2-20261125T190139Z-1-001/Phase2"
    P1_EXCEL   = f"{KAGGLE_BASE}/phase1.xlsx"
    P2_EXCEL   = f"{KAGGLE_BASE}/Phase2.xlsx"
    NEW_DIR    = "/kaggle/working/dataset_es_gt5"
else:
    LOCAL_BASE = r"f:\Priya Apu\Dataset"
    P1_IMG_DIR = os.path.join(LOCAL_BASE, r"Phase1-20261125T190005Z-1-001\Phase1")
    P2_IMG_DIR = os.path.join(LOCAL_BASE, r"Phase2-20261125T190139Z-1-001\Phase2")
    P1_EXCEL   = os.path.join(LOCAL_BASE, "phase1.xlsx")
    P2_EXCEL   = os.path.join(LOCAL_BASE, "Phase2.xlsx")
    NEW_DIR    = os.path.join(LOCAL_BASE, "dataset_es_gt5")

NEW_IMG_DIR = os.path.join(NEW_DIR, "images")
NEW_EXCEL   = os.path.join(NEW_DIR, "labels_es_gt5.xlsx")

if os.path.exists(NEW_IMG_DIR):
    shutil.rmtree(NEW_IMG_DIR)
os.makedirs(NEW_IMG_DIR, exist_ok=True)

STATUS_MAP = {
    "acceptable": "Acceptable",
    "unacceptable": "Unacceptable",
    "onsite verification required": "Onsite Verification",
    "need onsite verification": "Onsite Verification",
    "onsite verification": "Onsite Verification",
}

def load_matched(excel_path, img_dir, img_col, es_col):
    disk_map = {
        f.strip().lower(): f
        for f in os.listdir(img_dir)
        if os.path.isfile(os.path.join(img_dir, f))
    }
    df = pd.read_excel(excel_path)
    df["Image"] = df[img_col].astype(str).str.strip().str.lower().map(disk_map)
    df = df.dropna(subset=["Image"]).copy()  # Keep exact original filename on disk (no renaming)
    df["ES"] = df[es_col].astype(str).str.strip().str.upper()
    df["Status"] = df["Status"].astype(str).str.strip().str.lower().map(STATUS_MAP)
    df["src_dir"] = img_dir
    return df[["Image", "ES", "Status", "src_dir"]]

df_all = pd.concat([
    load_matched(P1_EXCEL, P1_IMG_DIR, "Img Reference", "ES Reference"),
    load_matched(P2_EXCEL, P2_IMG_DIR, "Photo", "ES Checkpoint"),
], ignore_index=True)

# Deduplicate exact/repeated (Image, ES) pairs
priority = {"Unacceptable": 0, "Onsite Verification": 1, "Acceptable": 2}
df_all["_prio"] = df_all["Status"].map(priority)
df_all = (
    df_all.sort_values("_prio")
    .drop_duplicates(subset=["Image", "ES"], keep="first")
    .sort_index()
    .drop(columns=["_prio"])
)

# Filter to ES codes that have MORE THAN 5 unique images (> 5)
es_img_counts = df_all.groupby("ES")["Image"].nunique()
valid_es = es_img_counts[es_img_counts > 5].index
filtered_df = df_all[df_all["ES"].isin(valid_es)].copy()

# Copy exact original images as-is (no renaming, no resizing/rotation changes)
unique_imgs = filtered_df[["Image", "src_dir"]].drop_duplicates()
for fname, sdir in unique_imgs.itertuples(index=False):
    shutil.copy2(os.path.join(sdir, fname), os.path.join(NEW_IMG_DIR, fname))

# Save clean Excel without extra helper columns
out_df = filtered_df[["Image", "ES", "Status"]].reset_index(drop=True)
out_df.to_excel(NEW_EXCEL, index=False)

print("=== NEW DATASET (ES > 5 IMAGES) ===")
print("Images folder      :", NEW_IMG_DIR)
print("Excel file         :", NEW_EXCEL)
print("Selected ES codes  :", out_df["ES"].nunique(), "(all have > 5 images)")
print("Images in folder   :", len(os.listdir(NEW_IMG_DIR)))
print("Unique imgs (Excel):", out_df["Image"].nunique())
print("Total rows (Excel) :", len(out_df))
print("\nSample rows (original image names preserved):")
print(out_df.head(10).to_string(index=False))
