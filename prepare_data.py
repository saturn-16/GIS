import os
import numpy as np
from PIL import Image
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
_BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(_BASE_DIR, "data", "Semantic segmentation dataset")
SAVE_PATH    = os.path.join(_BASE_DIR, "patches")
PATCH_SIZE   = 256

os.makedirs(os.path.join(SAVE_PATH, "images"), exist_ok=True)
os.makedirs(os.path.join(SAVE_PATH, "masks"),  exist_ok=True)

# ── Class color → index mapping ────────────────────────────────
COLOR_TO_CLASS = {
    (60,  16,  152): 0,   # Building
    (132, 41,  246): 1,   # Land
    (110, 193, 228): 2,   # Road
    (254, 221,  58): 3,   # Vegetation
    (226, 169,  41): 4,   # Water
    (155, 155, 155): 5,   # Unlabeled
}
NUM_CLASSES = 6

def mask_to_classes(mask_rgb):
    """Convert RGB mask to 2D array of class indices."""
    h, w = mask_rgb.shape[:2]
    class_map = np.zeros((h, w), dtype=np.uint8)
    for color, idx in COLOR_TO_CLASS.items():
        match = np.all(mask_rgb == np.array(color), axis=-1)
        class_map[match] = idx
    return class_map

def extract_patches(img, mask_cls, patch_size, tile_name, img_name, patch_counter):
    """Slice image and mask into non-overlapping patches."""
    h, w = img.shape[:2]
    patches_saved = 0
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            img_patch  = img[y:y+patch_size, x:x+patch_size]
            mask_patch = mask_cls[y:y+patch_size, x:x+patch_size]

            # Skip patches that are mostly unlabeled (>80%)
            unlabeled_ratio = np.sum(mask_patch == 5) / mask_patch.size
            if unlabeled_ratio > 0.8:
                continue

            name = f"{tile_name}_{img_name}_p{patch_counter[0]:04d}"
            Image.fromarray(img_patch).save(
                os.path.join(SAVE_PATH, "images", f"{name}.png"))
            np.save(
                os.path.join(SAVE_PATH, "masks", f"{name}.npy"), mask_patch)

            patch_counter[0] += 1
            patches_saved += 1
    return patches_saved

# ── Loop through all tiles ─────────────────────────────────────
tile_folders = sorted([
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
])

print(f"Found tiles: {tile_folders}\n")

patch_counter = [0]
total_patches = 0

for tile in tile_folders:
    img_dir  = os.path.join(DATASET_PATH, tile, "images")
    mask_dir = os.path.join(DATASET_PATH, tile, "masks")

    if not os.path.exists(img_dir):
        print(f"  Skipping {tile} — no images folder found")
        continue

    if not os.path.exists(mask_dir):
        print(f"  Skipping {tile} — no masks folder found")
        continue

    img_files = sorted(os.listdir(img_dir))
    print(f"Processing {tile} — {len(img_files)} images")

    for img_file in img_files:
        # Match mask file (same name, different extension)
        base     = os.path.splitext(img_file)[0]
        mask_file = base + ".png"
        mask_path = os.path.join(mask_dir, mask_file)

        if not os.path.exists(mask_path):
            print(f"  ⚠ No mask for {img_file}, skipping")
            continue

        img      = np.array(Image.open(os.path.join(img_dir, img_file)).convert("RGB"))
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
        mask_cls = mask_to_classes(mask_rgb)

        saved = extract_patches(img, mask_cls, PATCH_SIZE, tile.replace(" ", ""), base, patch_counter)
        total_patches += saved

print(f"\n✅ Done! Total patches saved: {total_patches}")
print(f"   Images → {SAVE_PATH}\\images")
print(f"   Masks  → {SAVE_PATH}\\masks")

# ── Quick class distribution across all patches ────────────────
print("\n── Class distribution across all patches ──")
CLASS_NAMES = ["Building", "Land", "Road", "Vegetation", "Water", "Unlabeled"]
mask_files  = list(Path(SAVE_PATH, "masks").glob("*.npy"))
counts      = np.zeros(NUM_CLASSES, dtype=np.int64)

for mf in mask_files:
    m = np.load(mf)
    for i in range(NUM_CLASSES):
        counts[i] += np.sum(m == i)

total = counts.sum()
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name:<12}: {counts[i]:>10,}  ({counts[i]/total*100:.1f}%)")