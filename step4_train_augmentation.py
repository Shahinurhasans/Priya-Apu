import os
import shutil
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance

# ==========================================
# Section 1: Set Paths & Load Train/Test Splits
# ==========================================
if os.path.exists("/kaggle/working/dataset_es_gt5"):
    NEW_DIR = "/kaggle/working/dataset_es_gt5"
else:
    NEW_DIR = r"f:\Priya Apu\Dataset\dataset_es_gt5"

PREP_DIR     = os.path.join(NEW_DIR, "preprocessed")
PREP_IMG_DIR = os.path.join(PREP_DIR, "images_224")

train_es_df    = pd.read_csv(os.path.join(PREP_DIR, "train_es_multilabel.csv"))
test_es_df     = pd.read_csv(os.path.join(PREP_DIR, "test_es_multilabel.csv"))
train_pairs_df = pd.read_csv(os.path.join(PREP_DIR, "train_pairs.csv"))
test_pairs_df  = pd.read_csv(os.path.join(PREP_DIR, "test_pairs.csv"))

# ==========================================
# Section 2: Train-Only Data Augmentation (Never Touch Test Set)
# ==========================================
TRAIN_AUG_DIR = os.path.join(PREP_DIR, "train_aug_images_224")
if os.path.exists(TRAIN_AUG_DIR):
    shutil.rmtree(TRAIN_AUG_DIR)
os.makedirs(TRAIN_AUG_DIR, exist_ok=True)

aug_es_rows = []
aug_pair_rows = []

for idx, row in train_es_df.iterrows():
    fname = row["Image"]
    base_name, ext = os.path.splitext(fname)
    img_pairs = train_pairs_df[train_pairs_df["Image"] == fname]

    with Image.open(os.path.join(PREP_IMG_DIR, fname)) as img:
        # 1. Original train image
        orig_name = fname
        img.save(os.path.join(TRAIN_AUG_DIR, orig_name), quality=95)

        # 2. Augmentation 1: Horizontal flip + mild brightness adjustment
        aug1_name = f"aug1_{base_name}{ext}"
        aug1 = ImageEnhance.Brightness(ImageOps.mirror(img)).enhance(1.12)
        aug1.save(os.path.join(TRAIN_AUG_DIR, aug1_name), quality=95)

        # 3. Augmentation 2: Mild rotation (+/- 8 deg) + mild contrast adjustment
        aug2_name = f"aug2_{base_name}{ext}"
        angle = 8 if idx % 2 == 0 else -8
        aug2 = ImageEnhance.Contrast(img.rotate(angle, resample=Image.Resampling.BILINEAR)).enhance(1.12)
        aug2.save(os.path.join(TRAIN_AUG_DIR, aug2_name), quality=95)

    for new_fname in [orig_name, aug1_name, aug2_name]:
        new_es_row = row.copy()
        new_es_row["Image"] = new_fname
        aug_es_rows.append(new_es_row)

        new_pairs = img_pairs.copy()
        new_pairs["Image"] = new_fname
        aug_pair_rows.append(new_pairs)

train_es_aug_df    = pd.DataFrame(aug_es_rows).reset_index(drop=True)
train_pairs_aug_df = pd.concat(aug_pair_rows, ignore_index=True)

# ==========================================
# Section 3: Save Augmented Train Tables & Print Summary
# ==========================================
train_es_aug_df.to_csv(os.path.join(PREP_DIR, "train_es_multilabel_aug.csv"), index=False)
train_pairs_aug_df.to_csv(os.path.join(PREP_DIR, "train_pairs_aug.csv"), index=False)

print(f"Train before augmentation : {len(train_es_df)} images ({len(train_pairs_df)} rows)")
print(f"Train after augmentation  : {len(train_es_aug_df)} images ({len(train_pairs_aug_df)} rows)")
print(f"Test (strictly untouched) : {len(test_es_df)} images ({len(test_pairs_df)} rows)")
print("\nAugmented Train Status counts:\n", train_pairs_aug_df["Status"].value_counts().to_string())
