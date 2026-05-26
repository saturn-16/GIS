# GIS Urban AI Analyser

A deep-learning powered GIS platform that analyses satellite imagery using a trained U-Net segmentation model, predicts urban growth zones, and generates AI-powered environmental recommendations.

**Website:** [saturn-16-gis.hf.space](https://saturn-16-gis.hf.space)

**Live Demo:** https://youtu.be/3V_7D5cm4vM

---

## Features

- **Land Use Segmentation** — Classifies satellite images into Buildings, Roads, Vegetation, Water, and Bare Land using a U-Net model trained on real aerial imagery
- **Urban Growth Prediction** — Analyses spatial features to map Low / Medium / High urban growth probability zones
- **AI Environmental Chat** — Real-time recommendations powered by Groq (Llama 4 Scout) based on the analysed image
- **Interactive Visualisations** — Colour-coded segmentation maps and growth heatmaps rendered in the browser

---

## Project Structure

```
GIS/
├── app.py                           # Flask web server (main entry point)
├── static/index.html                # Frontend UI (single-page app)
├── train.py                         # U-Net model training script
├── prepare_data.py                  # Dataset preprocessing & patch extraction
├── predict.py                       # Batch prediction + IoU evaluation
├── growth_prediction.py             # Urban growth probability analysis
├── environmental_recommendations.py # Environmental report generator
├── explore.py                       # Dataset exploration utility
├── Dockerfile                       # Hugging Face Spaces deployment config
├── requirements.txt                 # Python dependencies
├── .env.example                     # API key template (copy to .env)
└── data/                            # Dataset folder (not committed to git)
```

---

## Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/saturn-16/GIS.git
cd GIS
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API key
Copy `.env.example` to `.env` and add your Groq API key (free at [console.groq.com](https://console.groq.com)):
```
GROQ_API_KEY=gsk_your_actual_key_here
```

### 4. Run the app
```bash
python app.py
```
Open **http://localhost:5000** in your browser and upload any satellite image.

---

## Training from Scratch

```bash
# Step 1: Extract 256x256 patches from the dataset
python prepare_data.py

# Step 2: Train the U-Net model
python train.py

# Step 3: (Optional) Evaluate on test tiles
python predict.py
```

---

## Dataset

Uses the [Semantic Segmentation of Aerial Imagery](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery) dataset from Kaggle.

Place at: `GIS/data/Semantic segmentation dataset/`

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Flask (Python) |
| Deep Learning | TensorFlow / Keras (U-Net) |
| AI Chat | Groq API (Llama 4 Scout) |
| Image Processing | NumPy, Pillow, SciPy, OpenCV |
| Frontend | Vanilla HTML / CSS / JavaScript |
| Deployment | Hugging Face Spaces (Docker) |
