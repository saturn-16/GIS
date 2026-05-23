import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ── Dataset path ──────────────────────────────────────────────
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data", "Semantic segmentation dataset")

# ── Class color map (RGB values from the mask PNG files) ──────
CLASS_COLORS = {
    "Building":   (60,  16,  152),   # purple  #3C1098
    "Land":       (132, 41,  246),   # violet  #8429F6
    "Road":       (110, 193, 228),   # blue    #6EC1E4
    "Vegetation": (254, 221,  58),   # yellow  #FEDD3A
    "Water":      (226, 169,  41),   # orange  #E2A929
    "Unlabeled":  (155, 155, 155),   # gray    #9B9B9B
}

# ── Load one image + mask from Tile_1 ─────────────────────────
tile = "Tile 1"
img_folder  = os.path.join(DATASET_PATH, tile, "images")
mask_folder = os.path.join(DATASET_PATH, tile, "masks")

img_files  = sorted(os.listdir(img_folder))
mask_files = sorted(os.listdir(mask_folder))

print(f"Found {len(img_files)} images in {tile}")
print("First few:", img_files[:3])

# Load first pair
img  = np.array(Image.open(os.path.join(img_folder,  img_files[0])))
mask = np.array(Image.open(os.path.join(mask_folder, mask_files[0])))

print(f"\nImage shape : {img.shape}")
print(f"Mask shape  : {mask.shape}")
print(f"Mask unique colors (first 5): {np.unique(mask.reshape(-1, mask.shape[2]), axis=0)[:5]}")

# ── Visualize ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow(img);  axes[0].set_title("Satellite Image", fontsize=14); axes[0].axis("off")
axes[1].imshow(mask); axes[1].set_title("Segmentation Mask", fontsize=14); axes[1].axis("off")
plt.suptitle(f"File: {img_files[0]}", fontsize=12)
plt.tight_layout()
plt.show()

# ── Class pixel distribution ───────────────────────────────────
print("\n── Pixel count per class ──")
mask_flat = mask.reshape(-1, 3)
for name, color in CLASS_COLORS.items():
    count = np.sum(np.all(mask_flat == color, axis=1))
    pct   = count / len(mask_flat) * 100
    print(f"  {name:<12}: {count:>8,} pixels  ({pct:.1f}%)")