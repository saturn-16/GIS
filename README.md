---
title: GIS Urban AI Analyser
emoji: 🛸
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# GIS Urban AI Analyser

A deep-learning powered GIS platform that analyses satellite imagery to:
- **Segment land use** (Buildings, Roads, Vegetation, Water, Bare Land) using a trained U-Net model
- **Predict urban growth zones** (Low / Medium / High risk) using spatial feature analysis
- **Generate AI environmental recommendations** via Groq (Llama 4) with a real-time chat interface

---

## Project Structure

```
gis/
├── app.py                          # Flask web server (main entry point)
├── static/index.html               # Frontend UI (single-page app)
├── train.py                        # U-Net model training script
├── prepare_data.py                 # Dataset preprocessing & patch extraction
├── predict.py                      # Batch prediction + IoU evaluation
├── growth_prediction.py            # Urban growth probability analysis
├── environmental_recommendations.py# PDF/PNG environmental report generator
├── explore.py                      # Dataset exploration utility
├── requirements.txt                # Python dependencies
├── .env.example                    # API key template (copy to .env)
└── data/                           # ← Dataset folder (not committed to git)
    └── Semantic segmentation dataset/
        ├── Tile 1/
        │   ├── images/
        │   └── masks/
        └── ...
```

> **Note:** The trained model (`unet_model.keras`) and dataset (`data/`) are not included in this repository due to file size. See setup instructions below.

---

## Quick Start

### 1. Clone the repo
```bash
git clone <your-repo-url>
cd gis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API key
```bash
# Copy the example env file
cp .env.example .env

# Edit .env and paste your Groq API key
# Get a free key at: https://console.groq.com
```

Your `.env` file should look like:
```
GROQ_API_KEY=gsk_your_actual_key_here
```

### 4. Add the model file
Place `unet_model.keras` in the `gis/` folder (share via Google Drive or similar).

### 5. Run the app
```bash
python app.py
```
Open your browser at **http://localhost:5000** and upload any satellite image.

---

## Training from Scratch

If you want to retrain the model on the dataset:

```bash
# Step 1: Prepare the data (extract 256x256 patches)
python prepare_data.py

# Step 2: Train the U-Net
python train.py

# Step 3: (Optional) Evaluate predictions
python predict.py
```

---

## Dataset

This project uses the [Semantic Segmentation Dataset](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery) from Kaggle.

Place the dataset at:
```
gis/data/Semantic segmentation dataset/
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Flask (Python) |
| Deep Learning | TensorFlow / Keras (U-Net) |
| AI Recommendations | Groq API (Llama 4 Scout) |
| Image Processing | NumPy, Pillow, SciPy |
| Frontend | Vanilla HTML/CSS/JS |

---

## Class Labels

| Class | Colour |
|-------|--------|
| Building | `#3C1098` (purple) |
| Land (bare) | `#8429F6` (violet) |
| Road | `#6EC1E4` (blue) |
| Vegetation | `#FEDD3A` (yellow) |
| Water | `#E2A929` (orange) |
| Unlabeled | `#9B9B9B` (grey) |
