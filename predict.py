import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Config ──────────────────────────────────────────────────────
_BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(_BASE_DIR, "data", "Semantic segmentation dataset")
MODEL_PATH   = os.path.join(_BASE_DIR, "unet_model.keras")
OUTPUT_DIR   = os.path.join(_BASE_DIR, "predictions")
PATCH_SIZE   = 256
NUM_CLASSES  = 6

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Class info ───────────────────────────────────────────────────
CLASS_NAMES  = ["Building", "Land", "Road", "Vegetation", "Water", "Unlabeled"]
CLASS_COLORS = np.array([
    [60,  16,  152],   # Building  — purple
    [132, 41,  246],   # Land      — violet
    [110, 193, 228],   # Road      — blue
    [254, 221,  58],   # Vegetation— yellow
    [226, 169,  41],   # Water     — orange
    [155, 155, 155],   # Unlabeled — gray
], dtype=np.uint8)

# ── Load model ───────────────────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"weighted_sparse_cce": lambda y, p: p}
)
print("Model loaded!\n")

# ── Predict on full image (patch-by-patch) ───────────────────────
def predict_full_image(img_array):
    h, w = img_array.shape[:2]
    pred_map = np.zeros((h, w), dtype=np.uint8)

    for y in range(0, h - PATCH_SIZE + 1, PATCH_SIZE):
        for x in range(0, w - PATCH_SIZE + 1, PATCH_SIZE):
            patch = img_array[y:y+PATCH_SIZE, x:x+PATCH_SIZE] / 255.0
            patch = np.expand_dims(patch, 0).astype(np.float32)
            pred  = model.predict(patch, verbose=0)
            pred_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = np.argmax(pred[0], axis=-1)

    return pred_map

def class_map_to_rgb(class_map):
    """Convert 2D class index map → RGB color image."""
    rgb = CLASS_COLORS[class_map]
    return rgb

def compute_stats(pred_map):
    total = pred_map.size
    print("  ── Predicted class breakdown ──")
    for i, name in enumerate(CLASS_NAMES):
        count = np.sum(pred_map == i)
        print(f"    {name:<12}: {count:>8,}  ({count/total*100:.1f}%)")

# ── Pick images to predict (2 from different tiles) ──────────────
test_cases = [
    ("Tile 1", "image_part_001.jpg"),
    ("Tile 3", "image_part_004.jpg"),
    ("Tile 6", "image_part_007.jpg"),
]

# ── Legend ───────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color=np.array(c)/255, label=n)
    for c, n in zip(CLASS_COLORS, CLASS_NAMES)
]

for tile, img_file in test_cases:
    img_path  = os.path.join(DATASET_PATH, tile, "images", img_file)
    mask_path = os.path.join(DATASET_PATH, tile, "masks",
                             os.path.splitext(img_file)[0] + ".png")

    if not os.path.exists(img_path):
        print(f"Skipping {tile}/{img_file} — file not found")
        continue

    print(f"\nPredicting: {tile} / {img_file}")
    img_array = np.array(Image.open(img_path).convert("RGB"))
    pred_map  = predict_full_image(img_array)
    pred_rgb  = class_map_to_rgb(pred_map)
    compute_stats(pred_map)

    # Load ground truth mask if available
    has_gt = os.path.exists(mask_path)
    ncols  = 3 if has_gt else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 7))

    axes[0].imshow(img_array)
    axes[0].set_title("Satellite Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    if has_gt:
        gt = np.array(Image.open(mask_path).convert("RGB"))
        axes[1].imshow(gt)
        axes[1].set_title("Ground Truth Mask", fontsize=14, fontweight="bold")
        axes[1].axis("off")
        axes[2].imshow(pred_rgb)
        axes[2].set_title("U-Net Prediction", fontsize=14, fontweight="bold")
        axes[2].axis("off")
    else:
        axes[1].imshow(pred_rgb)
        axes[1].set_title("U-Net Prediction", fontsize=14, fontweight="bold")
        axes[1].axis("off")

    fig.legend(handles=legend_patches, loc="lower center",
               ncol=6, fontsize=11, frameon=True,
               bbox_to_anchor=(0.5, -0.04))

    plt.suptitle(f"{tile} — {img_file}", fontsize=13, y=1.01)
    plt.tight_layout()

    out_name = f"{tile.replace(' ', '')}_{os.path.splitext(img_file)[0]}_prediction.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {out_path}")

# ── Per-class IoU score ──────────────────────────────────────────
print("\n\n── Computing IoU on all test images ──")

COLOR_TO_CLASS = {
    (60,16,152):0, (132,41,246):1, (110,193,228):2,
    (254,221,58):3, (226,169,41):4, (155,155,155):5
}

def mask_rgb_to_class(mask_rgb):
    h, w = mask_rgb.shape[:2]
    out  = np.full((h, w), 5, dtype=np.uint8)
    for color, idx in COLOR_TO_CLASS.items():
        match = np.all(mask_rgb == np.array(color), axis=-1)
        out[match] = idx
    return out

iou_sum   = np.zeros(NUM_CLASSES)
iou_count = np.zeros(NUM_CLASSES)

for tile, img_file in test_cases:
    img_path  = os.path.join(DATASET_PATH, tile, "images", img_file)
    mask_path = os.path.join(DATASET_PATH, tile, "masks",
                             os.path.splitext(img_file)[0] + ".png")
    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        continue

    img_array = np.array(Image.open(img_path).convert("RGB"))
    gt_rgb    = np.array(Image.open(mask_path).convert("RGB"))
    gt_cls    = mask_rgb_to_class(gt_rgb)
    pred_map  = predict_full_image(img_array)

    # Crop to common size
    h = min(gt_cls.shape[0], pred_map.shape[0])
    w = min(gt_cls.shape[1], pred_map.shape[1])
    gt_cls   = gt_cls[:h, :w]
    pred_map = pred_map[:h, :w]

    for c in range(NUM_CLASSES):
        intersection = np.sum((pred_map == c) & (gt_cls == c))
        union        = np.sum((pred_map == c) | (gt_cls == c))
        if union > 0:
            iou_sum[c]   += intersection / union
            iou_count[c] += 1

print(f"\n{'Class':<14} {'IoU':>8}")
print("─" * 24)
valid_ious = []
for i, name in enumerate(CLASS_NAMES):
    if iou_count[i] > 0:
        iou = iou_sum[i] / iou_count[i]
        valid_ious.append(iou)
        print(f"{name:<14} {iou:>8.3f}  ({iou*100:.1f}%)")
    else:
        print(f"{name:<14} {'N/A':>8}")

print("─" * 24)
print(f"{'Mean IoU':<14} {np.mean(valid_ious):>8.3f}  ({np.mean(valid_ious)*100:.1f}%)")
print("\n✅ All predictions saved to:", OUTPUT_DIR)