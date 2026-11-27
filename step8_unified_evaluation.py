import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, classification_report, confusion_matrix
)

# ==========================================
# Section 1: Load Unseen Test Set & Saved Unified Model Checkpoint
# ==========================================
if os.path.exists("/kaggle/working/dataset_es_gt5"):
    NEW_DIR = "/kaggle/working/dataset_es_gt5"
else:
    NEW_DIR = r"f:\Priya Apu\Dataset\dataset_es_gt5"

PREP_DIR     = os.path.join(NEW_DIR, "preprocessed")
PREP_IMG_DIR = os.path.join(PREP_DIR, "images_224")
OUT_DIR      = os.path.join(NEW_DIR, "outputs")

test_es_df    = pd.read_csv(os.path.join(PREP_DIR, "test_es_multilabel.csv"))
test_pairs_df = pd.read_csv(os.path.join(PREP_DIR, "test_pairs.csv"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(os.path.join(OUT_DIR, "unified_audit_net.pth"), map_location=device)
ES_LIST     = ckpt["es_list"]
CLASS_NAMES = ckpt["class_names"]
best_th_uni = ckpt["threshold"]

class UnifiedAuditNet(nn.Module):
    def __init__(self, num_es, num_status=3, emb_dim=64):
        super().__init__()
        base = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.es_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, num_es))
        self.es_emb = nn.Embedding(num_es, emb_dim)
        self.status_head = nn.Sequential(
            nn.Linear(512 + emb_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_status)
        )
        self.num_es = num_es

    def forward(self, x):
        feat = self.backbone(x).flatten(1)
        es_logits = self.es_head(feat)
        es_ids = torch.arange(self.num_es, device=x.device)
        emb = self.es_emb(es_ids).unsqueeze(0).expand(x.size(0), -1, -1)
        feat_exp = feat.unsqueeze(1).expand(-1, self.num_es, -1)
        status_logits = self.status_head(torch.cat([feat_exp, emb], dim=-1))
        return es_logits, status_logits

model = UnifiedAuditNet(num_es=len(ES_LIST)).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# ==========================================
# Section 2: Run Single Forward Pass on All 45 Unseen Test Images
# ==========================================
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

X_te_img = torch.stack([
    img_transform(Image.open(os.path.join(PREP_IMG_DIR, f)).convert("RGB"))
    for f in test_es_df["Image"]
])

Y_te_es_np = test_es_df[ES_LIST].values.astype(int)
Y_te_st_np = np.full((len(test_es_df), len(ES_LIST)), -100, dtype=np.int64)
img_to_idx = {img: i for i, img in enumerate(test_es_df["Image"])}
for row in test_pairs_df.itertuples():
    Y_te_st_np[img_to_idx[row.Image], row.es_id] = row.status_id

with torch.no_grad():
    te_es_logits, te_st_logits = model(X_te_img.to(device))
    te_es_probs = torch.sigmoid(te_es_logits).cpu().numpy()
    te_st_preds = te_st_logits.argmax(dim=-1).cpu().numpy()

Y_te_es_pred = (te_es_probs >= best_th_uni).astype(int)
for i in range(len(Y_te_es_pred)):
    if Y_te_es_pred[i].sum() == 0:
        Y_te_es_pred[i, np.argmax(te_es_probs[i])] = 1

# ==========================================
# Section 3: Evaluate Output 1 (ES Numbers), Output 2 (Status) & End-to-End Pair F1
# ==========================================
print("=== UNIFIED MODEL — OUTPUT 1: ES NUMBER PREDICTION ===")
print(f"Micro Precision : {precision_score(Y_te_es_np, Y_te_es_pred, average='micro', zero_division=0):.4f}")
print(f"Micro Recall    : {recall_score(Y_te_es_np, Y_te_es_pred, average='micro', zero_division=0):.4f}")
print(f"Micro F1-Score  : {f1_score(Y_te_es_np, Y_te_es_pred, average='micro', zero_division=0):.4f}")
print(f"Macro F1-Score  : {f1_score(Y_te_es_np, Y_te_es_pred, average='macro', zero_division=0):.4f}")

valid_mask = Y_te_st_np != -100
y_true_st  = Y_te_st_np[valid_mask]
y_pred_st  = te_st_preds[valid_mask]

print("\n=== UNIFIED MODEL — OUTPUT 2: STATUS CLASSIFICATION ===")
print(f"Accuracy        : {accuracy_score(y_true_st, y_pred_st):.4f}")
print(f"Macro F1-Score  : {f1_score(y_true_st, y_pred_st, average='macro'):.4f}")
print(classification_report(y_true_st, y_pred_st, target_names=CLASS_NAMES, digits=4))

tp, fp, fn = 0, 0, 0
for i in range(len(Y_te_es_np)):
    true_pairs = {(j, int(Y_te_st_np[i, j])) for j in np.where(Y_te_es_np[i] == 1)[0]}
    pred_pairs = {(j, int(te_st_preds[i, j])) for j in np.where(Y_te_es_pred[i] == 1)[0]}
    tp += len(true_pairs & pred_pairs)
    fp += len(pred_pairs - true_pairs)
    fn += len(true_pairs - pred_pairs)

uni_e2e_f1 = 2 * tp / (2 * tp + fp + fn)
print(f"=== END-TO-END (IMAGE ONLY -> ES + STATUS) PAIR F1: {uni_e2e_f1:.4f} ===")

# ==========================================
# Section 4: Save Confusion Matrix & Baseline-vs-Unified Comparison Excel
# ==========================================
cm = confusion_matrix(y_true_st, y_pred_st)
plt.figure(figsize=(5.5, 4.5))
plt.imshow(cm, cmap="Blues")
plt.title("Unified Model Status Confusion Matrix")
plt.xticks(range(3), CLASS_NAMES, rotation=15)
plt.yticks(range(3), CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("Ground Truth")
for r in range(3):
    for c in range(3):
        plt.text(c, r, str(cm[r, c]), ha="center", va="center", fontweight="bold",
                 color="white" if cm[r, c] > cm.max() / 2 else "black")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "unified_confusion_matrix.png"), dpi=200)
plt.show()

uni_row = pd.DataFrame([{
    "Model": "Proposed Unified Multi-Task Net (End-to-End)",
    "ES_Precision": precision_score(Y_te_es_np, Y_te_es_pred, average="micro", zero_division=0),
    "ES_Recall": recall_score(Y_te_es_np, Y_te_es_pred, average="micro", zero_division=0),
    "ES_Micro_F1": f1_score(Y_te_es_np, Y_te_es_pred, average="micro", zero_division=0),
    "ES_Macro_F1": f1_score(Y_te_es_np, Y_te_es_pred, average="macro", zero_division=0),
    "Status_Acc": accuracy_score(y_true_st, y_pred_st),
    "Status_Macro_F1": f1_score(y_true_st, y_pred_st, average="macro"),
    "EndToEnd_Pair_F1": uni_e2e_f1,
}])

baseline_csv = os.path.join(OUT_DIR, "baseline_metrics.csv")
if os.path.exists(baseline_csv):
    comp_df = pd.concat([pd.read_csv(baseline_csv), uni_row], ignore_index=True).round(4)
else:
    comp_df = uni_row.round(4)

comp_excel = os.path.join(OUT_DIR, "model_comparison_results.xlsx")
comp_df.to_excel(comp_excel, index=False)
print("\n=== MODEL COMPARISON TABLE ===")
print(comp_df.to_string(index=False))
