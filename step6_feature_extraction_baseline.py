import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# ==========================================
# Section 1: Load Augmented Train & Untouched Test Splits
# ==========================================
if os.path.exists("/kaggle/working/dataset_es_gt5"):
    NEW_DIR = "/kaggle/working/dataset_es_gt5"
else:
    NEW_DIR = r"f:\Priya Apu\Dataset\dataset_es_gt5"

PREP_DIR      = os.path.join(NEW_DIR, "preprocessed")
PREP_IMG_DIR  = os.path.join(PREP_DIR, "images_224")
TRAIN_AUG_DIR = os.path.join(PREP_DIR, "train_aug_images_224")
OUT_DIR       = os.path.join(NEW_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

train_es_aug_df    = pd.read_csv(os.path.join(PREP_DIR, "train_es_multilabel_aug.csv"))
test_es_df         = pd.read_csv(os.path.join(PREP_DIR, "test_es_multilabel.csv"))
train_pairs_aug_df = pd.read_csv(os.path.join(PREP_DIR, "train_pairs_aug.csv"))
test_pairs_df      = pd.read_csv(os.path.join(PREP_DIR, "test_pairs.csv"))

ES_LIST = sorted(train_pairs_aug_df["ES"].unique())

# ==========================================
# Section 2: Extract 512-Dim Features with Frozen Pretrained ResNet-18
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
feature_extractor = nn.Sequential(*list(resnet.children())[:-1]).to(device).eval()

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_names, folder_path, batch_size=64):
    feats = []
    with torch.no_grad():
        for i in range(0, len(image_names), batch_size):
            batch_files = image_names[i : i + batch_size]
            tensors = [
                img_transform(Image.open(os.path.join(folder_path, f)).convert("RGB"))
                for f in batch_files
            ]
            out = feature_extractor(torch.stack(tensors).to(device)).flatten(1).cpu().numpy()
            feats.append(out)
    return np.vstack(feats)

X_train_es = extract_features(train_es_aug_df["Image"].tolist(), TRAIN_AUG_DIR)
X_test_es  = extract_features(test_es_df["Image"].tolist(), PREP_IMG_DIR)
Y_train_es = train_es_aug_df[ES_LIST].values
Y_test_es  = test_es_df[ES_LIST].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_es)
X_test_scaled  = scaler.transform(X_test_es)

# ==========================================
# Section 3: Train Simple Baseline Model 1 (ES Number Predictor)
# ==========================================
base_groups = train_es_aug_df["Image"].str.replace(r"^aug[12]_", "", regex=True)
oof_probs = np.zeros_like(Y_train_es, dtype=float)

for tr_i, val_i in GroupKFold(n_splits=4).split(X_train_scaled, Y_train_es, groups=base_groups):
    tmp = MultiOutputClassifier(LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=42))
    tmp.fit(X_train_scaled[tr_i], Y_train_es[tr_i])
    oof_probs[val_i] = np.column_stack([p[:, 1] for p in tmp.predict_proba(X_train_scaled[val_i])])

best_th, best_oof = 0.5, 0.0
for th in np.arange(0.30, 0.85, 0.05):
    sc = f1_score(Y_train_es, (oof_probs >= th).astype(int), average="micro", zero_division=0)
    if sc > best_oof:
        best_oof, best_th = sc, th

es_baseline = MultiOutputClassifier(LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=42))
es_baseline.fit(X_train_scaled, Y_train_es)

test_es_probs = np.column_stack([p[:, 1] for p in es_baseline.predict_proba(X_test_scaled)])
Y_test_es_pred = (test_es_probs >= best_th).astype(int)
for i in range(len(Y_test_es_pred)):
    if Y_test_es_pred[i].sum() == 0:
        Y_test_es_pred[i, np.argmax(test_es_probs[i])] = 1

# ==========================================
# Section 4: Train Simple Baseline Model 2 (Status Classifier)
# ==========================================
train_feat_map = dict(zip(train_es_aug_df["Image"], X_train_es))
test_feat_map  = dict(zip(test_es_df["Image"], X_test_es))

def build_pair_features(pairs_df, feat_map):
    img_vecs  = scaler.transform(np.vstack([feat_map[f] for f in pairs_df["Image"]]))
    es_onehot = np.eye(len(ES_LIST))[pairs_df["es_id"].values] * 3.0
    return np.hstack([img_vecs, es_onehot])

status_baseline = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=42)
status_baseline.fit(build_pair_features(train_pairs_aug_df, train_feat_map), train_pairs_aug_df["status_id"].values)
y_test_st_pred = status_baseline.predict(build_pair_features(test_pairs_df, test_feat_map))

# ==========================================
# Section 5: Compute & Save Baseline Metrics
# ==========================================
tp, fp, fn = 0, 0, 0
for i, fname in enumerate(test_es_df["Image"]):
    sub = test_pairs_df[test_pairs_df["Image"] == fname]
    true_set = set(zip(sub["es_id"], sub["status_id"]))
    pred_set = set()
    for eid in np.where(Y_test_es_pred[i] == 1)[0]:
        pf = np.hstack([X_test_scaled[i : i + 1], np.eye(len(ES_LIST))[eid : eid + 1] * 3.0])
        pred_set.add((int(eid), int(status_baseline.predict(pf)[0])))
    tp += len(true_set & pred_set)
    fp += len(pred_set - true_set)
    fn += len(true_set - pred_set)

baseline_summary = pd.DataFrame([{
    "Model": "Two-Stage Baseline (Frozen ResNet-18 + LogReg)",
    "ES_Precision": precision_score(Y_test_es, Y_test_es_pred, average="micro", zero_division=0),
    "ES_Recall": recall_score(Y_test_es, Y_test_es_pred, average="micro", zero_division=0),
    "ES_Micro_F1": f1_score(Y_test_es, Y_test_es_pred, average="micro", zero_division=0),
    "ES_Macro_F1": f1_score(Y_test_es, Y_test_es_pred, average="macro", zero_division=0),
    "Status_Acc": accuracy_score(test_pairs_df["status_id"].values, y_test_st_pred),
    "Status_Macro_F1": f1_score(test_pairs_df["status_id"].values, y_test_st_pred, average="macro"),
    "EndToEnd_Pair_F1": 2 * tp / (2 * tp + fp + fn),
}])

baseline_csv = os.path.join(OUT_DIR, "baseline_metrics.csv")
baseline_summary.to_csv(baseline_csv, index=False)

print("=== TWO-STAGE BASELINE RESULTS ===")
print(baseline_summary.round(4).to_string(index=False))
