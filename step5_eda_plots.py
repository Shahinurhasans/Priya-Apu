import os
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# Section 1: Load Preprocessed Splits & Setup Output Folder
# ==========================================
if os.path.exists("/kaggle/working/dataset_es_gt5"):
    NEW_DIR = "/kaggle/working/dataset_es_gt5"
else:
    NEW_DIR = r"f:\Priya Apu\Dataset\dataset_es_gt5"

PREP_DIR = os.path.join(NEW_DIR, "preprocessed")
OUT_DIR  = os.path.join(NEW_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

train_pairs_df = pd.read_csv(os.path.join(PREP_DIR, "train_pairs.csv"))
test_pairs_df  = pd.read_csv(os.path.join(PREP_DIR, "test_pairs.csv"))
full_pairs_df  = pd.concat([train_pairs_df, test_pairs_df], ignore_index=True)

CLASS_NAMES = ["Acceptable", "Unacceptable", "Onsite Verification"]

# ==========================================
# Section 2: Plot Status Distribution & ES Checkpoint Counts
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: 3-Class Compliance Status Balance
status_counts = full_pairs_df["Status"].value_counts().reindex(CLASS_NAMES)
axes[0].bar(status_counts.index, status_counts.values, color=["#2ca02c", "#d62728", "#ff7f0e"])
axes[0].set_title("Compliance Status Distribution (ES > 5 Dataset)")
axes[0].set_ylabel("Number of (Image, ES) Pairs")
for i, v in enumerate(status_counts.values):
    axes[0].text(i, v + 3, str(v), ha="center", fontweight="bold")

# Plot 2: Image Counts per ES Checkpoint
es_counts = full_pairs_df["ES"].value_counts()
axes[1].bar(es_counts.index, es_counts.values, color="#1f77b4")
axes[1].set_title("Sample Count per ES Checkpoint (23 Checkpoints)")
axes[1].set_ylabel("Number of Images")
axes[1].tick_params(axis="x", rotation=60)

plt.tight_layout()
eda_plot_path = os.path.join(OUT_DIR, "eda_summary.png")
plt.savefig(eda_plot_path, dpi=200)
plt.show()
print("Saved EDA figure:", eda_plot_path)
