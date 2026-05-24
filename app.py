from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from PIL import Image
import tensorflow as tf
from scipy.ndimage import distance_transform_edt, uniform_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io, base64, os, json, re, warnings
from groq import Groq
from dotenv import load_dotenv
warnings.filterwarnings("ignore")

# Load .env file if present (safe fallback for local dev)
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

app = Flask(__name__, static_folder="static")

# ── Config ────────────────────────────────────────────────────────
_BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(_BASE_DIR, "unet_model.keras")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
PATCH_SIZE   = 256
CLASS_NAMES  = ["Building", "Land", "Road", "Vegetation", "Water", "Unlabeled"]

CLASS_COLORS_MAP = np.array([
    [ 60,  16, 152],
    [132,  41, 246],
    [110, 193, 228],
    [254, 221,  58],
    [226, 169,  41],
    [155, 155, 155],
], dtype=np.uint8)

BASE_GROWTH = np.array([0.10, 0.80, 0.20, 0.65, 0.05, 0.30], dtype=np.float32)

growth_cmap = mcolors.LinearSegmentedColormap.from_list(
    "growth", ["#0a2f6e", "#1a6bb5", "#f5e642", "#e87c2b", "#c0392b"], N=256
)

# ── Groq client (lazy — re-read key on every call so HF Spaces secrets work) ─
def _get_groq_client():
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it as a Secret in your Hugging Face Space settings."
        )
    return Groq(api_key=key)

# ── Lazy-load U-Net model (loaded on first request, not at startup) ─
_model = None

def get_model():
    global _model
    if _model is None:
        print("Loading U-Net model...")
        _model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={"weighted_sparse_cce": lambda y, p: p}
        )
        print("Model ready!")
    return _model

# ── Predict class map ─────────────────────────────────────────────
def predict_class_map(img_array):
    h, w = img_array.shape[:2]
    pred = np.zeros((h, w), dtype=np.uint8)
    ph   = (h // PATCH_SIZE) * PATCH_SIZE
    pw   = (w // PATCH_SIZE) * PATCH_SIZE
    for y in range(0, ph, PATCH_SIZE):
        for x in range(0, pw, PATCH_SIZE):
            patch = img_array[y:y+PATCH_SIZE, x:x+PATCH_SIZE] / 255.0
            out   = get_model().predict(
                np.expand_dims(patch.astype(np.float32), 0), verbose=0
            )
            pred[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = np.argmax(out[0], axis=-1)
    return pred

# ── Compute growth score ──────────────────────────────────────────
def compute_growth_score(class_map):
    base      = BASE_GROWTH[class_map]
    road_mask = (class_map == 2).astype(np.float32)
    dist_road = distance_transform_edt(1 - road_mask)
    road_prox = 1.0 - dist_road / (dist_road.max() + 1e-6)
    bld_mask  = (class_map == 0).astype(np.float32)
    bld_dens  = uniform_filter(bld_mask, size=40)
    bld_dens /= (bld_dens.max() + 1e-6)
    dist_bld  = distance_transform_edt(1 - bld_mask)
    bld_prox  = 1.0 - dist_bld / (dist_bld.max() + 1e-6)
    growable  = ((class_map == 1) | (class_map == 3)).astype(np.float32)
    not_water = (class_map != 4).astype(np.float32)
    features  = np.stack(
        [base, road_prox, bld_dens, bld_prox, growable, not_water], axis=-1
    )
    weights = np.array([0.25, 0.30, 0.20, 0.10, 0.10, 0.05])
    score   = np.sum(features * weights, axis=-1) * not_water
    score   = (score - score.min()) / (score.max() - score.min() + 1e-6)
    return score.astype(np.float32)

# ── Helpers ───────────────────────────────────────────────────────
def arr_to_b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def heatmap_to_b64(score):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(score, cmap=growth_cmap, vmin=0, vmax=1)
    plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04,
                 label="0 = stable  →  1 = high growth")
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()

def img_array_to_b64_jpeg(img_array, max_size=512):
    """Resize and encode image as base64 JPEG for Groq vision."""
    pil_img = Image.fromarray(img_array.astype(np.uint8))
    pil_img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ── Strip markdown fences from Groq JSON responses ────────────────
def _strip_md_fences(raw: str) -> str:
    """Robustly remove ```json ... ``` fences that Groq sometimes wraps around JSON."""
    raw = raw.strip()
    raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()

# ── Build context for Groq ────────────────────────────────────────
def build_analysis_context(class_stats, metrics, growth_zones):
    observations = []
    stats_map    = {s["name"]: s for s in class_stats}

    if metrics["growth_risk"] > 30:
        observations.append(
            f"CRITICAL: {metrics['growth_risk']}% of area faces high urban growth risk"
        )
    if metrics["green_health"] < 10:
        observations.append(
            f"WARNING: Very low green cover at {metrics['green_health']}% — urban heat risk high"
        )
    veg = stats_map.get("Vegetation", {})
    if veg.get("high_risk", 0) > 20:
        observations.append(
            f"Vegetation at {veg['high_risk']}% high-growth risk — biodiversity threat"
        )
    bld = stats_map.get("Building", {})
    if bld.get("pct", 0) > 40:
        observations.append(
            f"Dense urban area — {bld['pct']}% built-up, heat island likely"
        )
    if metrics["water_health"] < 3:
        observations.append(
            "Limited water bodies — drainage and flooding risk needs attention"
        )
    if not observations:
        observations.append(
            "Relatively balanced land use — maintain green-urban equilibrium"
        )

    return {
        "location_type": "Urban/peri-urban satellite imagery",
        "land_use_breakdown": [
            {
                "class":                      s["name"],
                "area_percent":               s["pct"],
                "high_growth_risk_percent":   s["high_risk"],
                "medium_growth_risk_percent": s["med_risk"],
            }
            for s in class_stats
        ],
        "key_metrics": {
            "green_cover_percent":           metrics["green_health"],
            "water_body_percent":            metrics["water_health"],
            "urban_density_percent":         metrics["urban_density"],
            "high_growth_risk_area_percent": metrics["growth_risk"],
        },
        "growth_zone_summary":   growth_zones,
        "critical_observations": observations,
    }

# ── Groq: AI recommendations ──────────────────────────────────────
def get_ai_recommendations(context, img_array):
    img_b64 = img_array_to_b64_jpeg(img_array)

    prompt = f"""You are an expert urban environmental planner and GIS analyst.

I analysed a satellite image using a U-Net deep learning model. Here is the complete GIS analysis data:

{json.dumps(context, indent=2)}

Based on this data AND the satellite image provided, give a comprehensive environmental recommendation report.

Respond ONLY with valid JSON — no markdown fences, no extra text:
{{
  "summary": "2-3 sentence executive summary referencing actual data percentages",
  "urgency_level": "critical|high|moderate|low",
  "sustainability_score": {{
    "overall": 65,
    "green_cover": 40,
    "water_management": 70,
    "urban_heat": 55,
    "biodiversity": 60,
    "reasoning": "brief explanation of scores"
  }},
  "priority_actions": [
    {{
      "rank": 1,
      "title": "Short action title",
      "category": "green_infrastructure|water_management|heat_mitigation|biodiversity|transport|policy",
      "timeframe": "immediate|short_term|long_term",
      "description": "Detailed description referencing actual data",
      "impact": "Expected environmental impact",
      "estimated_benefit": "Quantified benefit if possible"
    }}
  ],
  "environmental_risks": [
    {{
      "risk": "Risk name",
      "severity": "high|medium|low",
      "description": "What the risk is",
      "mitigation": "How to address it"
    }}
  ],
  "zone_recommendations": {{
    "high_growth_zones": ["rec 1", "rec 2", "rec 3"],
    "existing_urban":    ["rec 1", "rec 2", "rec 3"],
    "green_areas":       ["rec 1", "rec 2", "rec 3"],
    "water_bodies":      ["rec 1", "rec 2"]
  }},
  "policy_suggestions": ["policy 1", "policy 2", "policy 3", "policy 4"]
}}"""

    try:
        response = _get_groq_client().chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}"
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
            max_tokens=2000,
            temperature=0.3,
        )
        raw = response.choices[0].message.content.strip()
        return json.loads(_strip_md_fences(raw))

    except Exception as e:
        print(f"Groq vision error: {e} — trying text-only fallback")
        # Text-only fallback if vision model fails
        response = _get_groq_client().chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert urban environmental planner and GIS analyst. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=2000,
            temperature=0.3,
        )
        raw = response.choices[0].message.content.strip()
        return json.loads(_strip_md_fences(raw))

# ── Groq: chat follow-up ──────────────────────────────────────────
def get_ai_chat_response(context, ai_recs, history, question):
    system = f"""You are an urban environmental AI assistant.
GIS Analysis Data: {json.dumps(context)}
Previously generated recommendations: {json.dumps(ai_recs)}
Answer the user's follow-up question specifically using this data.
Be concise, practical and reference actual numbers from the data."""

    messages = [{"role": "system", "content": system}]
    for msg in history[-6:]:
        messages.append({
            "role":    msg["role"],
            "content": msg["content"]
        })
    messages.append({"role": "user", "content": question})

    try:
        response = _get_groq_client().chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=500,
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI error: {str(e)}"

# ── Smart fallback (no API needed) ───────────────────────────────
def get_fallback_recommendations(context):
    metrics    = context["key_metrics"]
    breakdown  = {b["class"]: b for b in context["land_use_breakdown"]}
    green      = metrics["green_cover_percent"]
    water      = metrics["water_body_percent"]
    urban      = metrics["urban_density_percent"]
    risk       = metrics["high_growth_risk_area_percent"]
    veg_risk   = breakdown.get("Vegetation", {}).get("high_growth_risk_percent", 0)
    land_risk  = breakdown.get("Land",       {}).get("high_growth_risk_percent", 0)

    green_score = min(100, int(green * 4))
    water_score = min(100, int(water * 8))
    heat_score  = max(0,   100 - int(urban * 1.5))
    bio_score   = min(100, int(green * 3 + water * 2))
    overall     = int((green_score + water_score + heat_score + bio_score) / 4)
    urgency     = ("critical" if risk > 40 else
                   "high"     if risk > 25 else
                   "moderate" if risk > 10 else "low")

    actions = []
    rank = 1
    if land_risk > 20:
        actions.append({
            "rank": rank, "title": "Protect high-risk bare land",
            "category": "green_infrastructure", "timeframe": "immediate",
            "description": f"{land_risk:.1f}% of bare land faces high urbanisation risk. Plant native tree buffers and enforce green cover ratios before construction.",
            "impact": "Reduce urban heat and preserve biodiversity corridors",
            "estimated_benefit": "Up to 3°C reduction in local temperature"
        }); rank += 1
    if veg_risk > 15:
        actions.append({
            "rank": rank, "title": "Designate vegetation protection zones",
            "category": "biodiversity", "timeframe": "immediate",
            "description": f"{veg_risk:.1f}% of vegetation is at high growth risk. Establish no-build green corridors with 1:3 tree replacement policy.",
            "impact": "Preserve biodiversity and carbon sequestration",
            "estimated_benefit": "Protect 500+ species habitat per km²"
        }); rank += 1
    if urban > 25:
        actions.append({
            "rank": rank, "title": "Urban heat island mitigation",
            "category": "heat_mitigation", "timeframe": "short_term",
            "description": f"With {urban:.1f}% built-up area, install cool roofs, reflective surfaces and vertical gardens on dense blocks.",
            "impact": "Reduce urban heat island effect and energy consumption",
            "estimated_benefit": "15-20% reduction in cooling energy costs"
        }); rank += 1
    if green < 15:
        actions.append({
            "rank": rank, "title": "Emergency urban greening programme",
            "category": "green_infrastructure", "timeframe": "immediate",
            "description": f"Only {green:.1f}% green cover — far below recommended 30%. Launch pocket parks every 500m.",
            "impact": "Improve air quality, reduce flooding and heat stress",
            "estimated_benefit": "30% reduction in stormwater runoff"
        }); rank += 1
    if water < 5:
        actions.append({
            "rank": rank, "title": "Water body conservation",
            "category": "water_management", "timeframe": "short_term",
            "description": f"Limited water at {water:.1f}%. Restore wetlands, enforce 50m buffer zones, install bioswales.",
            "impact": "Natural flood control and groundwater recharge",
            "estimated_benefit": "Reduce flood risk for 10,000+ residents"
        }); rank += 1
    actions.append({
        "rank": rank, "title": "Sustainable transport corridors",
        "category": "transport", "timeframe": "long_term",
        "description": "Plant avenue trees along major roads at 10m spacing. Add dedicated cycling lanes and permeable road shoulders.",
        "impact": "Reduce vehicle emissions and improve air quality",
        "estimated_benefit": "20% reduction in road-side particulate matter"
    })

    obs = context.get("critical_observations", [])
    return {
        "summary": f"Analysis shows {urban:.1f}% urban density with {green:.1f}% green cover and {risk:.1f}% high growth risk. " +
                   (obs[0] if obs else "Balanced land use detected."),
        "urgency_level": urgency,
        "sustainability_score": {
            "overall": overall, "green_cover": green_score,
            "water_management": water_score, "urban_heat": heat_score,
            "biodiversity": bio_score,
            "reasoning": f"Based on {green:.1f}% green, {water:.1f}% water, {urban:.1f}% urban density"
        },
        "priority_actions": actions,
        "environmental_risks": [
            {"risk": "Urban Heat Island",
             "severity": "high" if urban > 30 else "medium",
             "description": f"High built-up density ({urban:.1f}%) causes elevated temperatures.",
             "mitigation": "Install cool roofs and expand tree canopy"},
            {"risk": "Green Cover Loss",
             "severity": "high" if green < 10 else "medium",
             "description": f"Only {green:.1f}% vegetation — below safe urban threshold.",
             "mitigation": "Mandatory green cover ratio in all new developments"},
            {"risk": "Urban Sprawl",
             "severity": "high" if risk > 30 else "medium",
             "description": f"{risk:.1f}% of area at high growth risk — rapid expansion likely.",
             "mitigation": "Zoning laws and protected area designations"},
        ],
        "zone_recommendations": {
            "high_growth_zones": [
                "Mandate 30% green cover in all new construction approvals",
                "Install permeable pavements to reduce stormwater runoff",
                "Create pocket parks every 500m in planned growth areas"
            ],
            "existing_urban": [
                "Retrofit buildings with solar panels and rainwater harvesting",
                "Add vertical gardens and green facades on dense blocks",
                "Establish community composting hubs in residential zones"
            ],
            "green_areas": [
                "Designate as protected zones with no-build enforcement",
                "Document biodiversity baseline for conservation planning",
                "Introduce community gardening and urban agriculture"
            ],
            "water_bodies": [
                "Enforce 50m no-construction buffer around all water bodies",
                "Monitor water quality quarterly with citizen science programs"
            ]
        },
        "policy_suggestions": [
            "Introduce mandatory Environmental Impact Assessment for all projects over 1 hectare",
            "Pass urban green cover ordinance requiring minimum 30% vegetation in new developments",
            "Create city-level biodiversity offset fund for unavoidable green space loss",
            "Implement water-sensitive urban design standards for all new roads"
        ]
    }

# ── Routes ────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/analyse", methods=["POST"])
def analyse():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file      = request.files["image"]
    img_pil   = Image.open(file.stream).convert("RGB")
    img_array = np.array(img_pil)

    # Resize if too large
    h, w = img_array.shape[:2]
    if max(h, w) > 1024:
        scale     = 1024 / max(h, w)
        img_pil   = img_pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        img_array = np.array(img_pil)

    # Crop to patch multiple
    h, w   = img_array.shape[:2]
    h_crop = (h // PATCH_SIZE) * PATCH_SIZE
    w_crop = (w // PATCH_SIZE) * PATCH_SIZE

    if h_crop < PATCH_SIZE or w_crop < PATCH_SIZE:
        return jsonify({
            "error": "Image too small. Please upload at least 256×256 pixels."
        }), 400

    img_array = img_array[:h_crop, :w_crop]

    # Run U-Net
    class_map    = predict_class_map(img_array)
    growth_score = compute_growth_score(class_map)

    total_px  = class_map.size
    high_mask = growth_score >= 0.65
    med_mask  = (growth_score >= 0.40) & (growth_score < 0.65)
    low_mask  = growth_score < 0.40

    # Class statistics
    class_stats = []
    for i, name in enumerate(CLASS_NAMES[:5]):
        mask = (class_map == i)
        px   = int(np.sum(mask))
        class_stats.append({
            "name":      name,
            "pct":       round(px / total_px * 100, 1),
            "high_risk": round(np.sum(mask & high_mask) / (px + 1) * 100, 1),
            "med_risk":  round(np.sum(mask & med_mask)  / (px + 1) * 100, 1),
            "color":     "#%02x%02x%02x" % tuple(CLASS_COLORS_MAP[i]),
        })

    metrics = {
        "green_health":  round(float(np.mean(class_map == 3)) * 100, 1),
        "water_health":  round(float(np.mean(class_map == 4)) * 100, 1),
        "urban_density": round(float(np.mean(class_map == 0)) * 100, 1),
        "growth_risk":   round(float(np.mean(high_mask))      * 100, 1),
    }

    growth_zones = {
        "high_growth_pct":   round(float(np.mean(high_mask)) * 100, 1),
        "medium_growth_pct": round(float(np.mean(med_mask))  * 100, 1),
        "stable_pct":        round(float(np.mean(low_mask))  * 100, 1),
    }

    # Output images
    pred_rgb     = CLASS_COLORS_MAP[class_map]
    growth_cat   = np.zeros_like(growth_score, dtype=np.uint8)
    growth_cat[growth_score >= 0.40] = 1
    growth_cat[growth_score >= 0.65] = 2
    growth_colors = np.array([
        [ 26, 114, 182],
        [245, 230,  66],
        [192,  57,  43],
    ], dtype=np.uint8)
    growth_rgb = growth_colors[growth_cat]

    context = build_analysis_context(class_stats, metrics, growth_zones)

    # Try Groq AI — fallback to rule-based if it fails
    try:
        ai_recs = get_ai_recommendations(context, img_array)
    except Exception as e:
        print(f"Groq failed ({e}) — using smart fallback")
        ai_recs = get_fallback_recommendations(context)

    return jsonify({
        "original":     arr_to_b64(img_array),
        "land_use":     arr_to_b64(pred_rgb),
        "growth_map":   arr_to_b64(growth_rgb),
        "heatmap":      heatmap_to_b64(growth_score),
        "class_stats":  class_stats,
        "metrics":      metrics,
        "growth_zones": growth_zones,
        "context":      context,
        "ai":           ai_recs,
    })

@app.route("/chat", methods=["POST"])
def chat():
    data     = request.get_json()
    context  = data.get("context",  {})
    ai_recs  = data.get("ai_recs",  {})
    history  = data.get("history",  [])
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        reply = get_ai_chat_response(context, ai_recs, history, question)
        return jsonify({"reply": reply})
    except Exception as e:
        print("Chat error:", e)
        return jsonify({"reply": f"AI error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)