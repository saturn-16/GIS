import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from scipy.ndimage import distance_transform_edt, uniform_filter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────
_BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(_BASE_DIR, "data", "Semantic segmentation dataset")
MODEL_PATH   = os.path.join(_BASE_DIR, "unet_model.keras")
OUTPUT_DIR   = os.path.join(_BASE_DIR, "env_report")
PATCH_SIZE   = 256
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ["Building", "Land", "Road", "Vegetation", "Water", "Unlabeled"]
CLASS_COLORS_MAP = np.array([
    [60,  16, 152],[132, 41, 246],[110,193,228],
    [254,221,  58],[226,169,  41],[155,155,155]
], dtype=np.uint8)
COLOR_TO_CLASS = {
    (60,16,152):0,(132,41,246):1,(110,193,228):2,
    (254,221,58):3,(226,169,41):4,(155,155,155):5
}
BASE_GROWTH = np.array([0.10,0.80,0.20,0.65,0.05,0.30], dtype=np.float32)

# ── Environmental recommendations database ───────────────────────
RECOMMENDATIONS = {
    "high_growth_land": {
        "title": "High-growth bare land zones",
        "icon": "!",
        "color": "#c0392b",
        "actions": [
            "Plant native tree buffers before construction begins",
            "Mandate green cover ratio of 30% in new developments",
            "Install permeable pavements to reduce runoff",
            "Create pocket parks every 500m in planned zones",
            "Enforce Environmental Impact Assessment before clearing",
        ]
    },
    "high_growth_vegetation": {
        "title": "Vegetation under urban pressure",
        "icon": "!",
        "color": "#e67e22",
        "actions": [
            "Designate protected green corridors — no-build zones",
            "Introduce urban forest policy with 1:3 tree replacement",
            "Map and preserve biodiversity hotspots immediately",
            "Create wildlife crossing points in planned road areas",
            "Incentivize rooftop gardens in adjacent buildings",
        ]
    },
    "existing_buildings": {
        "title": "Existing built-up areas",
        "icon": "i",
        "color": "#2980b9",
        "actions": [
            "Install cool roofs and reflective surfaces to reduce heat island",
            "Add vertical gardens and green facades on dense blocks",
            "Retrofit buildings with solar panels and rainwater harvesting",
            "Create shaded pedestrian pathways with tree canopy",
            "Establish community composting points to cut waste",
        ]
    },
    "road_network": {
        "title": "Road and transport corridors",
        "icon": "i",
        "color": "#8e44ad",
        "actions": [
            "Plant avenue trees along all major roads — minimum 10m spacing",
            "Add bioswales (planted drainage channels) along road edges",
            "Install permeable road shoulders to recharge groundwater",
            "Introduce dedicated cycling lanes to cut vehicle emissions",
            "Use recycled materials in road resurfacing",
        ]
    },
    "water_bodies": {
        "title": "Water bodies and wetlands",
        "icon": "P",
        "color": "#16a085",
        "actions": [
            "Enforce 50m no-construction buffer around all water bodies",
            "Restore degraded wetlands as natural flood barriers",
            "Ban discharge of untreated water into natural water bodies",
            "Create riparian vegetation zones along water edges",
            "Monitor water quality quarterly with citizen science programs",
        ]
    },
    "low_growth_stable": {
        "title": "Stable low-growth zones",
        "icon": "ok",
        "color": "#27ae60",
        "actions": [
            "Maintain existing green cover — avoid unnecessary clearing",
            "Use these zones as carbon sequestration areas",
            "Introduce community gardening and urban agriculture",
            "Monitor for encroachment and enforce zoning laws",
            "Document biodiversity for future conservation planning",
        ]
    },
}

# ── Load model ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading model...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"weighted_sparse_cce": lambda y, p: p}
    )

    # ── Predict class map ────────────────────────────────────────────
    def predict_class_map(img_array):
        h, w = img_array.shape[:2]
        pred = np.zeros((h, w), dtype=np.uint8)
        for y in range(0, h - PATCH_SIZE + 1, PATCH_SIZE):
            for x in range(0, w - PATCH_SIZE + 1, PATCH_SIZE):
                patch = img_array[y:y+PATCH_SIZE, x:x+PATCH_SIZE] / 255.0
                out   = model.predict(np.expand_dims(patch, 0).astype(np.float32), verbose=0)
                pred[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = np.argmax(out[0], axis=-1)
        return pred

    # ── Compute growth score ─────────────────────────────────────────
    def compute_growth_score(class_map):
        base = BASE_GROWTH[class_map]
        road_mask = (class_map == 2).astype(np.float32)
        dist_road = distance_transform_edt(1 - road_mask)
        road_prox = 1.0 - dist_road / (dist_road.max() + 1e-6)
        bld_mask  = (class_map == 0).astype(np.float32)
        bld_dens  = uniform_filter(bld_mask, size=40)
        bld_dens  = bld_dens / (bld_dens.max() + 1e-6)
        dist_bld  = distance_transform_edt(1 - bld_mask)
        bld_prox  = 1.0 - dist_bld / (dist_bld.max() + 1e-6)
        growable  = ((class_map == 1) | (class_map == 3)).astype(np.float32)
        not_water = (class_map != 4).astype(np.float32)
        weights   = np.array([0.25, 0.30, 0.20, 0.10, 0.10, 0.05])
        features  = np.stack([base, road_prox, bld_dens, bld_prox, growable, not_water], axis=-1)
        score     = np.sum(features * weights, axis=-1) * not_water
        score     = (score - score.min()) / (score.max() - score.min() + 1e-6)
        return score.astype(np.float32)

    # ── Analyse zones and pick recommendations ────────────────────────
    def analyse_zones(class_map, growth_score):
        total_px = class_map.size
        high_mask = growth_score >= 0.65
        med_mask  = (growth_score >= 0.40) & (growth_score < 0.65)
        low_mask  = growth_score < 0.40

        stats = {}
        for i, name in enumerate(CLASS_NAMES):
            mask = (class_map == i)
            stats[name] = {
                "total_pct":     np.sum(mask) / total_px * 100,
                "high_growth_pct": np.sum(mask & high_mask) / (np.sum(mask) + 1) * 100,
                "med_growth_pct":  np.sum(mask & med_mask)  / (np.sum(mask) + 1) * 100,
            }

        # Pick relevant recommendation categories
        active_recs = []
        if stats["Land"]["high_growth_pct"] > 15:
            active_recs.append("high_growth_land")
        if stats["Vegetation"]["high_growth_pct"] > 10:
            active_recs.append("high_growth_vegetation")
        if stats["Building"]["total_pct"] > 5:
            active_recs.append("existing_buildings")
        if stats["Road"]["total_pct"] > 5:
            active_recs.append("road_network")
        if stats["Water"]["total_pct"] > 2:
            active_recs.append("water_bodies")
        active_recs.append("low_growth_stable")  # always include

        return stats, active_recs

    # ── Growth heatmap colormap ──────────────────────────────────────
    growth_cmap = mcolors.LinearSegmentedColormap.from_list(
        "growth", ["#0a2f6e","#1a6bb5","#f5e642","#e87c2b","#c0392b"], N=256
    )

    # ── Build full report figure ─────────────────────────────────────
    def build_report(img_array, class_map, growth_score, stats, active_recs, tile, img_file):
        fig = plt.figure(figsize=(22, 26), facecolor="#f8f9fa")
        gs  = gridspec.GridSpec(4, 3, figure=fig,
                                hspace=0.45, wspace=0.3,
                                top=0.93, bottom=0.02,
                                left=0.04, right=0.98)

        # ── Title ────────────────────────────────────────────────────
        fig.text(0.5, 0.965,
                 f"Environmental Impact & Recommendation Report",
                 ha="center", fontsize=20, fontweight="bold", color="#1a1a2e")
        fig.text(0.5, 0.950,
                 f"Location: {tile}  |  Image: {img_file}  |  Analysis: GIS-based U-Net Segmentation",
                 ha="center", fontsize=12, color="#555")

        # ── Row 1: maps ───────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img_array); ax1.set_title("Satellite Image", fontweight="bold", fontsize=13); ax1.axis("off")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(CLASS_COLORS_MAP[class_map])
        ax2.set_title("Predicted Land Use (U-Net)", fontweight="bold", fontsize=13); ax2.axis("off")
        legend_patches = [mpatches.Patch(color=np.array(c)/255, label=n)
                          for c, n in zip(CLASS_COLORS_MAP, CLASS_NAMES)]
        ax2.legend(handles=legend_patches, loc="lower center", ncol=3,
                   fontsize=8, bbox_to_anchor=(0.5, -0.22), framealpha=0.9)

        ax3 = fig.add_subplot(gs[0, 2])
        im = ax3.imshow(growth_score, cmap=growth_cmap, vmin=0, vmax=1)
        ax3.set_title("Urban Growth Probability", fontweight="bold", fontsize=13); ax3.axis("off")
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label="0 = stable  →  1 = high growth")

        # ── Row 2: zone stats bar chart ───────────────────────────────
        ax4 = fig.add_subplot(gs[1, :])
        names   = CLASS_NAMES[:5]
        totals  = [stats[n]["total_pct"] for n in names]
        highs   = [stats[n]["high_growth_pct"] for n in names]
        meds    = [stats[n]["med_growth_pct"] for n in names]
        colors  = ["#3c1098","#8429f6","#6ec1e4","#fedd3a","#e2a929"]

        x = np.arange(len(names))
        w = 0.28
        bars = ax4.bar(x - w, totals, w, label="% of total area", color=colors, alpha=0.85)
        ax4.bar(x,      highs, w, label="% at high growth risk",   color="#c0392b", alpha=0.8)
        ax4.bar(x + w,  meds,  w, label="% at medium growth risk", color="#e67e22", alpha=0.8)

        for bar, val in zip(bars, totals):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax4.set_xticks(x); ax4.set_xticklabels(names, fontsize=12)
        ax4.set_ylabel("Percentage of pixels (%)", fontsize=11)
        ax4.set_title("Land Use Distribution & Growth Risk by Class", fontweight="bold", fontsize=13)
        ax4.legend(fontsize=10, loc="upper right")
        ax4.set_ylim(0, max(totals) * 1.25)
        ax4.grid(axis="y", alpha=0.3)
        ax4.spines[["top","right"]].set_visible(False)

        # ── Rows 3–4: recommendation cards ───────────────────────────
        card_axes = [fig.add_subplot(gs[2, i]) for i in range(3)] + \
                    [fig.add_subplot(gs[3, i]) for i in range(3)]

        for idx, rec_key in enumerate(active_recs[:6]):
            rec = RECOMMENDATIONS[rec_key]
            ax  = card_axes[idx]
            ax.set_facecolor("#ffffff")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.axis("off")

            # Card border
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2)
                spine.set_edgecolor(rec["color"])

            # Header strip
            ax.add_patch(plt.Rectangle((0, 0.82), 1, 0.18,
                         transform=ax.transAxes, color=rec["color"], zorder=2))
            ax.text(0.5, 0.91, rec["title"], transform=ax.transAxes,
                    ha="center", va="center", fontsize=10.5,
                    fontweight="bold", color="white", zorder=3)

            # Action items
            y_pos = 0.74
            for i, action in enumerate(rec["actions"]):
                ax.text(0.06, y_pos, f"{i+1}.", transform=ax.transAxes,
                        fontsize=9, fontweight="bold", color=rec["color"], va="top")
                # Word-wrap manually at ~42 chars
                words = action.split()
                lines, line = [], ""
                for w in words:
                    if len(line) + len(w) + 1 <= 42:
                        line += (" " if line else "") + w
                    else:
                        lines.append(line); line = w
                lines.append(line)

                for j, ln in enumerate(lines):
                    ax.text(0.14, y_pos - j*0.063, ln, transform=ax.transAxes,
                            fontsize=8.5, color="#222", va="top")
                y_pos -= 0.063 * len(lines) + 0.04

        # Hide unused card slots
        for idx in range(len(active_recs), 6):
            card_axes[idx].set_visible(False)

        return fig

    # ── Main: process images ─────────────────────────────────────────
    test_images = [
        ("Tile 1", "image_part_001.jpg"),
        ("Tile 3", "image_part_004.jpg"),
        ("Tile 6", "image_part_007.jpg"),
    ]

    for tile, img_file in test_images:
        img_path = os.path.join(DATASET_PATH, tile, "images", img_file)
        if not os.path.exists(img_path):
            print(f"Skipping {tile}/{img_file}")
            continue

        print(f"\nAnalysing: {tile} / {img_file}")
        img_array    = np.array(Image.open(img_path).convert("RGB"))
        class_map    = predict_class_map(img_array)
        growth_score = compute_growth_score(class_map)
        stats, active_recs = analyse_zones(class_map, growth_score)

        print("  Active recommendations:", active_recs)

        fig = build_report(img_array, class_map, growth_score,
                           stats, active_recs, tile, img_file)

        out = os.path.join(OUTPUT_DIR,
              f"env_report_{tile.replace(' ','')}_{os.path.splitext(img_file)[0]}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#f8f9fa")
        plt.show()
        print(f"  Report saved → {out}")

    print("\n\nAll environmental reports saved to:", OUTPUT_DIR)
