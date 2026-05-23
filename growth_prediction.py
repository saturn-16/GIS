import os
import numpy as np
from PIL import Image
import tensorflow as tf
from scipy.ndimage import distance_transform_edt, uniform_filter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────
_BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(_BASE_DIR, "data", "Semantic segmentation dataset")
MODEL_PATH   = os.path.join(_BASE_DIR, "unet_model.keras")
OUTPUT_DIR   = os.path.join(_BASE_DIR, "growth_output")
PATCH_SIZE   = 256
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Class definitions ────────────────────────────────────────────
CLASS_NAMES  = ["Building", "Land", "Road", "Vegetation", "Water", "Unlabeled"]
COLOR_TO_CLASS = {
    (60,  16,  152): 0,
    (132, 41,  246): 1,
    (110, 193, 228): 2,
    (254, 221,  58): 3,
    (226, 169,  41): 4,
    (155, 155, 155): 5,
}

# Growth probability per class (domain knowledge)
# Land=high, Vegetation=medium-high, Road edge=medium, Building=low, Water=very low
BASE_GROWTH = np.array([0.10, 0.80, 0.20, 0.65, 0.05, 0.30], dtype=np.float32)

# ── Load model ───────────────────────────────────────────────────
print("Loading U-Net model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"weighted_sparse_cce": lambda y, p: p}
)
print("Model ready.\n")

# ── Predict class map from image ─────────────────────────────────
def predict_class_map(img_array):
    h, w = img_array.shape[:2]
    pred = np.zeros((h, w), dtype=np.uint8)
    for y in range(0, h - PATCH_SIZE + 1, PATCH_SIZE):
        for x in range(0, w - PATCH_SIZE + 1, PATCH_SIZE):
            patch = img_array[y:y+PATCH_SIZE, x:x+PATCH_SIZE] / 255.0
            out   = model.predict(np.expand_dims(patch, 0).astype(np.float32), verbose=0)
            pred[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = np.argmax(out[0], axis=-1)
    return pred

# ── Compute spatial features ─────────────────────────────────────
def compute_growth_features(class_map):
    h, w = class_map.shape

    # 1. Base growth probability from class type
    base = BASE_GROWTH[class_map]

    # 2. Distance to nearest road (closer = higher growth)
    road_mask = (class_map == 2).astype(np.float32)
    dist_to_road = distance_transform_edt(1 - road_mask)
    max_dist = dist_to_road.max() + 1e-6
    road_proximity = 1.0 - (dist_to_road / max_dist)  # 1=on road, 0=far

    # 3. Local building density (buildings nearby = urban expansion likely)
    building_mask = (class_map == 0).astype(np.float32)
    building_density = uniform_filter(building_mask, size=40)
    building_density = building_density / (building_density.max() + 1e-6)

    # 4. Distance to nearest building (closer = more likely to urbanize)
    dist_to_building = distance_transform_edt(1 - building_mask)
    building_proximity = 1.0 - (dist_to_building / (dist_to_building.max() + 1e-6))

    # 5. Is this pixel vegetation or bare land? (most growable)
    growable = ((class_map == 1) | (class_map == 3)).astype(np.float32)

    # 6. Water mask — no growth on water
    not_water = (class_map != 4).astype(np.float32)

    return np.stack([
        base,
        road_proximity,
        building_density,
        building_proximity,
        growable,
        not_water,
    ], axis=-1)  # shape: (H, W, 6)

# ── Combine features into a growth score ─────────────────────────
def compute_growth_score(features):
    # Weighted combination of features
    weights = np.array([0.25, 0.30, 0.20, 0.10, 0.10, 0.05])
    score = np.sum(features * weights, axis=-1)

    # Apply growth rules:
    # Zero out water pixels (not_water mask is feature index 5)
    score = score * features[:, :, 5]

    # Normalize to 0-1
    score = (score - score.min()) / (score.max() - score.min() + 1e-6)
    return score.astype(np.float32)

# ── Convert RGB mask to class map ────────────────────────────────
def mask_rgb_to_class(mask_rgb):
    h, w = mask_rgb.shape[:2]
    out = np.full((h, w), 5, dtype=np.uint8)
    for color, idx in COLOR_TO_CLASS.items():
        match = np.all(mask_rgb == np.array(color), axis=-1)
        out[match] = idx
    return out

# ── Growth heatmap colormap (blue=low, yellow=medium, red=high) ──
growth_cmap = mcolors.LinearSegmentedColormap.from_list(
    "growth", ["#0a2f6e", "#1a6bb5", "#f5e642", "#e87c2b", "#c0392b"], N=256
)

# ── Process images ────────────────────────────────────────────────
test_images = [
    ("Tile 1", "image_part_001.jpg"),
    ("Tile 4", "image_part_003.jpg"),
    ("Tile 7", "image_part_006.jpg"),
]

for tile, img_file in test_images:
    img_path  = os.path.join(DATASET_PATH, tile, "images", img_file)
    mask_path = os.path.join(DATASET_PATH, tile, "masks",
                             os.path.splitext(img_file)[0] + ".png")

    if not os.path.exists(img_path):
        print(f"Skipping {tile}/{img_file} — not found")
        continue

    print(f"\nProcessing: {tile} / {img_file}")
    img_array = np.array(Image.open(img_path).convert("RGB"))

    # Step 1: Predict land use classes
    print("  Step 1: Predicting land use classes...")
    pred_class = predict_class_map(img_array)

    # Step 2: Compute spatial features
    print("  Step 2: Computing spatial features...")
    features = compute_growth_features(pred_class)

    # Step 3: Growth score
    print("  Step 3: Computing growth probability...")
    growth_score = compute_growth_score(features)

    # Step 4: Classify into Low / Medium / High
    growth_cat = np.zeros_like(growth_score, dtype=np.uint8)
    growth_cat[growth_score >= 0.40] = 1  # Medium
    growth_cat[growth_score >= 0.65] = 2  # High

    pct_low  = np.mean(growth_cat == 0) * 100
    pct_med  = np.mean(growth_cat == 1) * 100
    pct_high = np.mean(growth_cat == 2) * 100
    print(f"  Growth zones → Low: {pct_low:.1f}%  Medium: {pct_med:.1f}%  High: {pct_high:.1f}%")

    # ── Visualization ─────────────────────────────────────────────
    has_mask = os.path.exists(mask_path)
    fig, axes = plt.subplots(1, 4 if has_mask else 3, figsize=(22, 6))

    # Panel 1: Original image
    axes[0].imshow(img_array)
    axes[0].set_title("Satellite Image", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: U-Net predicted mask
    CLASS_COLORS_MAP = np.array([
        [60,16,152],[132,41,246],[110,193,228],
        [254,221,58],[226,169,41],[155,155,155]
    ], dtype=np.uint8)
    pred_rgb = CLASS_COLORS_MAP[pred_class]
    axes[1].imshow(pred_rgb)
    axes[1].set_title("Predicted Land Use", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    # Panel 3 (optional): Ground truth
    panel = 2
    if has_mask:
        gt = np.array(Image.open(mask_path).convert("RGB"))
        axes[2].imshow(gt)
        axes[2].set_title("Ground Truth", fontsize=13, fontweight="bold")
        axes[2].axis("off")
        panel = 3

    # Panel 4: Growth heatmap
    im = axes[panel].imshow(growth_score, cmap=growth_cmap, vmin=0, vmax=1)
    axes[panel].set_title("Urban Growth Probability", fontsize=13, fontweight="bold")
    axes[panel].axis("off")
    plt.colorbar(im, ax=axes[panel], fraction=0.046, pad=0.04,
                 label="Growth likelihood (0=low, 1=high)")

    # Legend for land use
    legend_patches = [
        mpatches.Patch(color=np.array(c)/255, label=n)
        for c, n in zip(CLASS_COLORS_MAP, CLASS_NAMES)
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=6, fontsize=10, bbox_to_anchor=(0.35, -0.04))

    plt.suptitle(f"Urban Growth Analysis — {tile} / {img_file}",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_name = f"growth_{tile.replace(' ', '')}_{os.path.splitext(img_file)[0]}.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {out_path}")

    # ── Save growth category map as standalone PNG ────────────────
    growth_colors = np.array([
        [26,  114, 182],   # Low  — blue
        [245, 230,  66],   # Medium — yellow
        [192,  57,  43],   # High — red
    ], dtype=np.uint8)
    growth_rgb = growth_colors[growth_cat]
    Image.fromarray(growth_rgb).save(
        os.path.join(OUTPUT_DIR, f"growth_map_{tile.replace(' ', '')}_{os.path.splitext(img_file)[0]}.png")
    )

print("\n\nAll done! Growth maps saved to:", OUTPUT_DIR)
print("\nColor guide for growth map PNG:")
print("  Blue  = Low growth probability (water, existing buildings)")
print("  Yellow = Medium growth (vegetation near urban areas)")
print("  Red   = High growth (bare land adjacent to roads & buildings)")