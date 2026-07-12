# ── maize_api/app.py ──────────────────────────────────────────────────────────
"""
Maize Disease Detection API
Two-stage pipeline:
  Stage 1 → gate_model.pt      (PyTorch) : is there a crop/leaf in the image?
  Stage 2 → maize_model_v2.h5  (Keras)  : which disease / healthy?

The existing Keras model is used unchanged.
The gate model is copied from the cassava project (gate_model.pt) — no retraining.
Response format is IDENTICAL to the original so ESP32 + dashboard need no changes.
"""

import os, io, base64, logging
from datetime import datetime, timezone
from collections import deque

# ── Keras disease model (unchanged) ──────────────────────────────────────────
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# ── PyTorch gate model ────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

from flask import Flask, request, jsonify, send_from_directory, redirect
from flask_cors import CORS

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(__file__)
GATE_PATH    = os.path.join(BASE_DIR, "gate_model.pt")   # copy from cassava project
DISEASE_PATH = os.path.join(BASE_DIR, "maize_model_v2.h5")

CLASS_NAMES  = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

GATE_SIZE   = 160
GATE_THRESH = 0.55   # slightly lower than cassava (0.6) — maize leaves are narrower
DEVICE      = torch.device("cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Pest thresholds (unchanged from original)
PEST_TEMP_MIN     = 24.0
PEST_TEMP_MAX     = 32.0
PEST_HUMIDITY_MIN = 60.0
GAS_THRESHOLD     = 1027
HISTORY_MAXLEN    = 50

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("maize")

# ── Load models ───────────────────────────────────────────────────────────────
log.info("Loading gate model (PyTorch) ...")
gate_model = torch.jit.load(GATE_PATH, map_location=DEVICE).eval()

log.info("Loading disease model (Keras) ...")
disease_model = load_model(DISEASE_PATH)

log.info("Both models ready.")

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

history        = deque(maxlen=HISTORY_MAXLEN)
latest_reading = None

# ── Gate preprocessing (PyTorch / albumentations) ─────────────────────────────
gate_tf = A.Compose([
    A.Resize(GATE_SIZE, GATE_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

def run_gate(img_bytes):
    """Returns (is_crop: bool, leaf_confidence: float 0-100)."""
    img    = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr    = np.array(img)
    tensor = gate_tf(image=arr)["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(gate_model(tensor), dim=1)[0]
    leaf_conf = probs[1].item()
    return leaf_conf >= GATE_THRESH, round(leaf_conf * 100, 1)

# ── Disease preprocessing (Keras — unchanged from original) ───────────────────
def run_disease(img_bytes):
    """Returns (class_name: str, confidence: float 0-100)."""
    img      = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img      = img.resize((224, 224))
    arr      = np.array(img) / 255.0
    arr      = np.expand_dims(arr, axis=0)
    preds    = disease_model.predict(arr, verbose=0)
    idx      = int(np.argmax(preds[0]))
    conf     = float(np.max(preds[0])) * 100
    return CLASS_NAMES[idx], round(conf, 1)

# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_float(v):
    try:    return float(v)
    except: return None

def assess_pest_risk(temperature, humidity, gas):
    reasons = []
    temp_ok = temperature is not None and PEST_TEMP_MIN <= temperature <= PEST_TEMP_MAX
    hum_ok  = humidity    is not None and humidity >= PEST_HUMIDITY_MIN
    gas_ok  = gas         is not None and gas      >= GAS_THRESHOLD
    if temp_ok: reasons.append(f"Temperature {temperature:.1f}°C is in the pest-favorable range")
    if hum_ok:  reasons.append(f"Humidity {humidity:.1f}% is above the {PEST_HUMIDITY_MIN:.0f}% threshold")
    if gas_ok:  reasons.append(f"Gas reading {gas:.0f} is above the {GAS_THRESHOLD:.0f} threshold")
    return ("HIGH" if (temp_ok and (hum_ok or gas_ok)) else "LOW"), reasons

# ── /predict ──────────────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    global latest_reading

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    img_bytes   = request.files['image'].read()
    temperature = parse_float(request.form.get('temperature'))
    humidity    = parse_float(request.form.get('humidity'))
    gas         = parse_float(request.form.get('gas_raw'))

    pest_risk, pest_reasons = assess_pest_risk(temperature, humidity, gas)
    img_b64 = 'data:image/jpeg;base64,' + base64.b64encode(img_bytes).decode()
    ts      = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        # ── Stage 1: Gate ─────────────────────────────────────────────────────
        is_crop, leaf_conf = run_gate(img_bytes)

        if not is_crop:
            log.info(f"Gate rejected — leaf_conf={leaf_conf}%")
            result = {
                # Exact same keys ESP32 + dashboard already read
                'disease'      : 'No Crop Detected',
                'confidence'   : '0.0%',
                'status'       : 'NO_CROP',
                'temperature'  : temperature,
                'humidity'     : humidity,
                'gas'          : gas,
                'pest_risk'    : pest_risk,
                'pest_reasons' : pest_reasons,
                'timestamp'    : ts,
                'image'        : img_b64,
            }
            latest_reading = result
            history.appendleft(result)
            return jsonify(result)

        # ── Stage 2: Disease (existing Keras model, untouched) ────────────────
        predicted_class, confidence = run_disease(img_bytes)

        log.info(f"Prediction: {predicted_class} ({confidence}%) | "
                 f"T={temperature} H={humidity} Gas={gas} | Pest risk: {pest_risk}")

        result = {
            # Exact same keys as the original app
            'disease'      : predicted_class,
            'confidence'   : f'{confidence:.1f}%',
            'status'       : 'ALERT' if predicted_class != 'Healthy' else 'OK',
            'temperature'  : temperature,
            'humidity'     : humidity,
            'gas'          : gas,
            'pest_risk'    : pest_risk,
            'pest_reasons' : pest_reasons,
            'timestamp'    : ts,
            'image'        : img_b64,
        }

        latest_reading = result
        history.appendleft(result)
        return jsonify(result)

    except Exception as e:
        log.exception("Prediction error")
        return jsonify({'error': str(e)}), 500

# ── Original routes (all unchanged) ──────────────────────────────────────────
@app.route('/latest')
def get_latest():
    if latest_reading is None:
        return jsonify({'error': 'No readings yet'}), 404
    return jsonify(latest_reading)

@app.route('/history')
def get_history():
    return jsonify(list(history))

@app.route('/dashboard')
def dashboard():
    return send_from_directory('static', 'dashboard.html')

@app.route('/')
def home():
    return redirect('/dashboard')

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'models_loaded': True})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)