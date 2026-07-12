from flask import Flask, request, jsonify, send_from_directory, redirect
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import base64
import os
from datetime import datetime
from collections import deque

# ── Setup ─────────────────────────────────────────────
app = Flask(__name__)
# Explicitly allow all origins on every route — the dashboard may be served
# from this same app, from a static host (Netlify/Vercel), or opened as a
# local file:// page while testing, so we don't restrict by origin here.
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the trained model once at startup
model = load_model('maize_model_v2.h5')

# These must match exactly what was used during training
CLASS_NAMES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

# ── Pest-favorable environment thresholds ─────────────
# Based on fall armyworm (Spodoptera frugiperda) studies — the dominant
# maize pest in Nigeria. Optimal development/survival is widely reported
# around 24-32°C, with 52-88% relative humidity supporting rearing/activity.
# These are reasonable defaults for a demo, not a validated field model —
# tune them if your own observations point elsewhere.
PEST_TEMP_MIN = 24.0        # °C, lower bound of favorable range
PEST_TEMP_MAX = 32.0        # °C, upper bound of favorable range
PEST_HUMIDITY_MIN = 60.0    # %RH, above this humidity favors pest activity

# Gas sensor (e.g. MQ135) — elevated readings can indicate decomposing
# plant matter or crop stress that tends to attract pests. There's no
# universally agreed ppm threshold for this, so treat it as a heuristic
# and calibrate GAS_THRESHOLD against your own sensor's baseline readings.
GAS_THRESHOLD = 1027         # raw ADC / ppm-equivalent, calibrate per sensor

# In-memory store of recent readings (most recent first). Fine for a demo;
# swap for a real database if you need this to survive a restart.
HISTORY_MAXLEN = 50
history = deque(maxlen=HISTORY_MAXLEN)
latest_reading = None


# ── Helper Functions ───────────────────────────────────
def prepare_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize((224, 224))          # Match training size
    img_array = np.array(img) / 255.0    # Normalize pixels
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def assess_pest_risk(temperature, humidity, gas):
    """Returns ('HIGH'|'LOW', [reasons]) based on sensor thresholds."""
    reasons = []

    temp_favorable = (temperature is not None and
                       PEST_TEMP_MIN <= temperature <= PEST_TEMP_MAX)
    humidity_favorable = (humidity is not None and humidity >= PEST_HUMIDITY_MIN)
    gas_elevated = (gas is not None and gas >= GAS_THRESHOLD)

    if temp_favorable:
        reasons.append(f'Temperature {temperature:.1f}°C is in the pest-favorable range')
    if humidity_favorable:
        reasons.append(f'Humidity {humidity:.1f}% is above the {PEST_HUMIDITY_MIN:.0f}% threshold')
    if gas_elevated:
        reasons.append(f'Gas reading {gas:.0f} is above the {GAS_THRESHOLD:.0f} threshold')

    # Flag HIGH risk when temperature is favorable AND at least one other
    # signal (humidity or gas) also points the same way.
    is_high_risk = temp_favorable and (humidity_favorable or gas_elevated)

    return ('HIGH' if is_high_risk else 'LOW'), reasons


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ── Main Prediction Route (called by the ESP32-CAM) ──
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    img_file = request.files['image']
    img_bytes = img_file.read()

    # Sensor readings sent alongside the image as form fields
    temperature = parse_float(request.form.get('temperature'))
    humidity = parse_float(request.form.get('humidity'))
    gas = parse_float(request.form.get('gas_raw'))

    # Prepare image and run prediction
    img_array = prepare_image(img_bytes)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(np.max(predictions[0])) * 100

    pest_risk, pest_reasons = assess_pest_risk(temperature, humidity, gas)

    # Build the response
    result = {
        'disease'         : predicted_class,
        'confidence'      : f'{confidence:.1f}%',
        'status'          : 'ALERT' if predicted_class != 'Healthy' else 'OK',
        'temperature'     : temperature,
        'humidity'        : humidity,
        'gas'             : gas,
        'pest_risk'       : pest_risk,
        'pest_reasons'    : pest_reasons,
        'timestamp'       : datetime.utcnow().isoformat() + 'Z',
        'image'           : 'data:image/jpeg;base64,' + base64.b64encode(img_bytes).decode('utf-8'),
    }

    global latest_reading
    latest_reading = result
    history.appendleft(result)

    print(f'Prediction: {predicted_class} ({confidence:.1f}%) | '
          f'T={temperature} H={humidity} Gas={gas} | Pest risk: {pest_risk}')

    return jsonify(result)


# ── Dashboard Polling Routes ───────────────────────────
@app.route('/latest', methods=['GET'])
def get_latest():
    if latest_reading is None:
        return jsonify({'error': 'No readings yet'}), 404
    return jsonify(latest_reading)


@app.route('/history', methods=['GET'])
def get_history():
    return jsonify(list(history))



@app.route('/dashboard', methods=['GET'])
def dashboard():
    return send_from_directory('static', 'dashboard.html')

# ── Health Check Route ────────────────────────────────
@app.route('/', methods=['GET'])
def home():
    return redirect('/dashboard')

# ── Start Server ──────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)