import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import f1_score

torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# Section 1: Set Paths & Load Preprocessed Splits
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
train_pairs_aug_df = pd.read_csv(os.path.join(PREP_DIR, "train_pairs_aug.csv"))

ES_LIST     = sorted(train_pairs_aug_df["ES"].unique())
CLASS_NAMES = ["Acceptable", "Unacceptable", "Onsite Verification"]
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ==========================================
# Section 2: Build Unified Multi-Task Training Tensors
# ==========================================
def build_unified_tensors(es_df, pairs_df, img_folder):
    imgs = torch.stack([
        img_transform(Image.open(os.path.join(img_folder, f)).convert("RGB"))
        for f in es_df["Image"]
    ])
    y_es = torch.tensor(es_df[ES_LIST].values, dtype=torch.float32)

    # -100 masks out non-applicable ES slots in CrossEntropyLoss
    y_st_np = np.full((len(es_df), len(ES_LIST)), -100, dtype=np.int64)
    img_to_idx = {img: i for i, img in enumerate(es_df["Image"])}
    for row in pairs_df.itertuples():
        y_st_np[img_to_idx[row.Image], row.es_id] = row.status_id

    return imgs, y_es, torch.tensor(y_st_np, dtype=torch.long)

X_tr_img, Y_tr_es, Y_tr_st = build_unified_tensors(train_es_aug_df, train_pairs_aug_df, TRAIN_AUG_DIR)
train_loader = DataLoader(TensorDataset(X_tr_img, Y_tr_es, Y_tr_st), batch_size=32, shuffle=True)

# ==========================================
# Section 3: Define Single Unified Multi-Task Network (Option B)
# ==========================================
class UnifiedAuditNet(nn.Module):
    def __init__(self, num_es, num_status=3, emb_dim=64):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Freeze early general-feature layers; fine-tune deeper layers (layer3, layer4)
        for name, param in base.named_parameters():
            if not (name.startswith("layer3") or name.startswith("layer4")):
                param.requires_grad = False
        self.backbone = nn.Sequential(*list(base.children())[:-1])

        # Output Branch 1: Applicable ES number logits (B, 23)
        self.es_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, num_es),
        )

        # Output Branch 2: Per-ES 3-class Status logits (B, 23, 3)
        self.es_emb = nn.Embedding(num_es, emb_dim)
        self.status_head = nn.Sequential(
            nn.Linear(512 + emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_status),
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

# ==========================================
# Section 4: Configure Joint Loss & Optimizer
# ==========================================
pos_counts = Y_tr_es.sum(dim=0)
neg_counts = len(Y_tr_es) - pos_counts
bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.sqrt(neg_counts / pos_counts).to(device))

valid_st = Y_tr_st[Y_tr_st != -100].numpy()
st_counts = np.bincount(valid_st, minlength=3)
st_weights = torch.tensor(len(valid_st) / (3.0 * st_counts), dtype=torch.float32).to(device)
ce_loss_fn = nn.CrossEntropyLoss(weight=st_weights, ignore_index=-100)

optimizer = torch.optim.AdamW([
    {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": 1e-4},
    {"params": list(model.es_head.parameters()) + list(model.es_emb.parameters()) + list(model.status_head.parameters()), "lr": 1e-3},
], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# ==========================================
# Section 5: Joint End-to-End Training Loop
# ==========================================
EPOCHS = 20
loss_history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for bx, by_es, by_st in train_loader:
        bx, by_es, by_st = bx.to(device), by_es.to(device), by_st.to(device)
        optimizer.zero_grad()

        es_logits, st_logits = model(bx)
        loss = bce_loss_fn(es_logits, by_es) + ce_loss_fn(st_logits.reshape(-1, 3), by_st.reshape(-1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(bx)

    scheduler.step()
    epoch_loss = running_loss / len(X_tr_img)
    loss_history.append(epoch_loss)
    if epoch in [1, 5, 10, 15, 20]:
        print(f"Epoch {epoch:2d}/{EPOCHS} | Joint Loss: {epoch_loss:.4f}")

# ==========================================
# Section 6: Tune ES Threshold on Train & Save Model Checkpoint + Loss Curve
# ==========================================
model.eval()
with torch.no_grad():
    tr_es_probs = torch.sigmoid(model(X_tr_img.to(device))[0]).cpu().numpy()

best_th_uni, best_tr_f1 = 0.55, 0.0
Y_tr_es_np = Y_tr_es.numpy().astype(int)
for th in np.arange(0.40, 0.85, 0.05):
    sc = f1_score(Y_tr_es_np, (tr_es_probs >= th).astype(int), average="micro", zero_division=0)
    if sc > best_tr_f1:
        best_tr_f1, best_th_uni = sc, th

ckpt_path = os.path.join(OUT_DIR, "unified_audit_net.pth")
torch.save({
    "model_state_dict": model.state_dict(),
    "es_list": ES_LIST,
    "class_names": CLASS_NAMES,
    "threshold": float(best_th_uni),
}, ckpt_path)

plt.figure(figsize=(6, 4))
plt.plot(range(1, EPOCHS + 1), loss_history, marker="o", color="#1f77b4")
plt.title("Unified Multi-Task Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Joint Loss (L_ES + L_Status)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_loss_curve.png"), dpi=200)
plt.show()

print(f"Saved Unified Model Checkpoint: {ckpt_path} (Threshold={best_th_uni:.2f})")
