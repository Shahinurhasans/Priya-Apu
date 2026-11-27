import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torchvision import models, transforms

# ==========================================
# Section 1: Load Trained Unified Multi-Task Model
# ==========================================
if os.path.exists("/kaggle/working/dataset_es_gt5"):
    NEW_DIR = "/kaggle/working/dataset_es_gt5"
else:
    NEW_DIR = r"f:\Priya Apu\Dataset\dataset_es_gt5"

NEW_IMG_DIR = os.path.join(NEW_DIR, "images")
PREP_DIR    = os.path.join(NEW_DIR, "preprocessed")
OUT_DIR     = os.path.join(NEW_DIR, "outputs")

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

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ==========================================
# Section 2: Single-Image Audit Function (Input = Raw Image Path Only)
# ==========================================
def audit_single_image(img_path):
    with Image.open(img_path) as img:
        img_224 = ImageOps.exif_transpose(img).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)

    with torch.no_grad():
        inp = img_transform(img_224).unsqueeze(0).to(device)
        es_logits, st_logits = model(inp)
        es_probs = torch.sigmoid(es_logits)[0].cpu().numpy()
        st_preds = st_logits[0].argmax(dim=-1).cpu().numpy()

    pred_idx = np.where(es_probs >= best_th_uni)[0]
    if len(pred_idx) == 0:
        pred_idx = np.array([int(np.argmax(es_probs))])

    results = [
        (ES_LIST[i], CLASS_NAMES[int(st_preds[i])], float(es_probs[i]))
        for i in pred_idx
    ]
    return results, img_224

# ==========================================
# Section 3: Run & Visualize Demo on 5 Unseen Test Photos
# ==========================================
sample_imgs = test_es_df["Image"].iloc[:5].tolist()
fig, axes = plt.subplots(1, 5, figsize=(22, 5.5))

for ax, fname in zip(axes, sample_imgs):
    gt_sub = test_pairs_df[test_pairs_df["Image"] == fname]
    gt_txt = "\n".join([f"{r.ES}: {r.Status}" for r in gt_sub.itertuples()])

    preds, vis_img = audit_single_image(os.path.join(NEW_IMG_DIR, fname))
    pr_txt = "\n".join([f"{es}: {st} ({conf:.2f})" for es, st, conf in preds])

    print(f"=== {fname} ===")
    print("  Ground Truth :", ", ".join([f"{r.ES}={r.Status}" for r in gt_sub.itertuples()]))
    print("  Predicted    :", ", ".join([f"{es}={st} ({conf:.2f})" for es, st, conf in preds]))

    ax.imshow(vis_img)
    ax.axis("off")
    ax.set_title(
        f"Image: {fname}\n\n[Ground Truth]\n{gt_txt}\n\n[Unified Predicted]\n{pr_txt}",
        fontsize=9,
        loc="left",
    )

plt.tight_layout()
demo_fig_path = os.path.join(OUT_DIR, "demo_5_test_predictions.png")
plt.savefig(demo_fig_path, dpi=200)
plt.show()
print("\nSaved demo visualization:", demo_fig_path)
