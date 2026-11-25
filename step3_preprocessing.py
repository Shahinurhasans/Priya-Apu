import os
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedShuffleSplit

# ==========================================
# Section 1: Set Paths & Load Filtered Dataset
# ==========================================
if os.path.exists("/kaggle/working/dataset_es_gt5"):
    BASE_DIR = "/kaggle/working/dataset_es_gt5"
else:
    BASE_DIR = r"f:\Priya Apu\Dataset\dataset_es_gt5"

RAW_IMG_DIR  = os.path.join(BASE_DIR, "images")
EXCEL_PATH   = os.path.join(BASE_DIR, "labels_es_gt5.xlsx")
PREP_DIR     = os.path.join(BASE_DIR, "preprocessed")
PREP_IMG_DIR = os.path.join(PREP_DIR, "images_224")
os.makedirs(PREP_IMG_DIR, exist_ok=True)

df = pd.read_excel(EXCEL_PATH)
print(f"Loaded Excel: {len(df)} rows | {df['Image'].nunique()} unique images | {df['ES'].nunique()} ES codes")

# ==========================================
# Section 2: Clean Text & Check Missing Values
# ==========================================
df["Image"]  = df["Image"].astype(str).str.strip()
df["ES"]     = df["ES"].astype(str).str.strip().str.upper()
df["Status"] = df["Status"].astype(str).str.strip()

assert df.isnull().sum().sum() == 0, "Missing values found in dataset!"

# ==========================================
# Section 3: Image Preprocessing (EXIF Fix, RGB, Resize 224x224)
# ==========================================
IMG_SIZE = (224, 224)
unique_images = df["Image"].unique()

for fname in unique_images:
    src_path = os.path.join(RAW_IMG_DIR, fname)
    dst_path = os.path.join(PREP_IMG_DIR, fname)
    with Image.open(src_path) as img:
        img = ImageOps.exif_transpose(img)  # Auto-correct camera EXIF rotation
        img = img.convert("RGB")            # Ensure 3-channel RGB format
        img = img.resize(IMG_SIZE, Image.Resampling.BILINEAR)
        img.save(dst_path, quality=95)

print(f"Preprocessed {len(unique_images)} images -> RGB {IMG_SIZE} saved in: {PREP_IMG_DIR}")

# ==========================================
# Section 4: Encode Labels (Status ID, ES ID & Multi-Label ES Matrix)
# ==========================================
STATUS_TO_ID = {"Acceptable": 0, "Unacceptable": 1, "Onsite Verification": 2}
ES_LIST      = sorted(df["ES"].unique())
ES_TO_ID     = {es: idx for idx, es in enumerate(ES_LIST)}

df["status_id"] = df["Status"].map(STATUS_TO_ID)
df["es_id"]     = df["ES"].map(ES_TO_ID)

# Build per-image multi-label binary matrix (1 row per unique image, 23 ES columns)
es_multilabel = (
    pd.crosstab(df["Image"], df["ES"])
    .reindex(columns=ES_LIST, fill_value=0)
    .clip(upper=1)
    .reset_index()
)

# Compute worst-case status per image for stratified splitting
prio_map = {"Unacceptable": 0, "Onsite Verification": 1, "Acceptable": 2}
img_worst_status = (
    df.assign(_p=df["Status"].map(prio_map))
    .sort_values("_p")
    .drop_duplicates("Image")[["Image", "Status"]]
    .rename(columns={"Status": "img_status"})
)
es_multilabel = es_multilabel.merge(img_worst_status, on="Image", how="left")

# ==========================================
# Section 5: Leak-Free Train / Test Split by Unique Image (80% / 20%)
# ==========================================
# Find a stratified image split where every ES code appears in both Train and Test
selected_split = None
for seed in range(42, 200):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    tr_idx, te_idx = next(splitter.split(es_multilabel, es_multilabel["img_status"]))
    tr_es_counts = es_multilabel.iloc[tr_idx][ES_LIST].sum()
    te_es_counts = es_multilabel.iloc[te_idx][ES_LIST].sum()
    if (tr_es_counts >= 4).all() and (te_es_counts >= 1).all():
        selected_split = (tr_idx, te_idx, seed)
        break

tr_idx, te_idx, used_seed = selected_split
train_imgs = set(es_multilabel.iloc[tr_idx]["Image"])
test_imgs  = set(es_multilabel.iloc[te_idx]["Image"])

# Ensure zero image leakage between Train and Test
assert len(train_imgs & test_imgs) == 0, "Data leakage detected!"

train_es_df = es_multilabel[es_multilabel["Image"].isin(train_imgs)].reset_index(drop=True)
test_es_df  = es_multilabel[es_multilabel["Image"].isin(test_imgs)].reset_index(drop=True)

train_pairs_df = df[df["Image"].isin(train_imgs)].reset_index(drop=True)
test_pairs_df  = df[df["Image"].isin(test_imgs)].reset_index(drop=True)

# ==========================================
# Section 6: Save Preprocessed Splits & Print Summary
# ==========================================
train_es_df.to_csv(os.path.join(PREP_DIR, "train_es_multilabel.csv"), index=False)
test_es_df.to_csv(os.path.join(PREP_DIR, "test_es_multilabel.csv"), index=False)
train_pairs_df.to_csv(os.path.join(PREP_DIR, "train_pairs.csv"), index=False)
test_pairs_df.to_csv(os.path.join(PREP_DIR, "test_pairs.csv"), index=False)

print(f"\n=== PREPROCESSING & SPLIT SUMMARY (seed={used_seed}) ===")
print(f"Train: {len(train_es_df)} unique images | {len(train_pairs_df)} (Image, ES) rows")
print(f"Test : {len(test_es_df)} unique images | {len(test_pairs_df)} (Image, ES) rows")
print("Image overlap between Train & Test:", len(train_imgs & test_imgs))
print("ES codes present in Train / Test  :", int((train_es_df[ES_LIST].sum() > 0).sum()), "/", int((test_es_df[ES_LIST].sum() > 0).sum()))
print("\nTrain Status distribution:\n", train_pairs_df["Status"].value_counts().to_string())
print("\nTest Status distribution:\n", test_pairs_df["Status"].value_counts().to_string())
