import os, io, base64, logging, requests
from datetime import datetime, timezone
from collections import deque
from flask import Flask, request, jsonify, send_from_directory, redirect
from flask_cors import CORS

# ── Config ────────────────────────────────────────────────────────────────────
# Replace with your actual Hugging Face Space URL trailing with /predict_internal
HF_API_URL = os.environ.get("HF_API_URL", "https://dracer-maize-pest-disease.hf.space/predict_internal")

PEST_TEMP_MIN     = 24.0
PEST_TEMP_MAX     = 32.0
PEST_HUMIDITY_MIN = 60.0
GAS_THRESHOLD     = 1027
HISTORY_MAXLEN    = 50

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("maize")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

history        = deque(maxlen=HISTORY_MAXLEN)
latest_reading = None

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

    # Extract form data
    img_file    = request.files['image']
    img_bytes   = img_file.read()
    temperature = parse_float(request.form.get('temperature'))
    humidity    = parse_float(request.form.get('humidity'))
    gas         = parse_float(request.form.get('gas_raw'))

    pest_risk, pest_reasons = assess_pest_risk(temperature, humidity, gas)
    img_b64 = 'data:image/jpeg;base64,' + base64.b64encode(img_bytes).decode()
    ts      = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        # Reset image file pointer to pass it along to Hugging Face
        img_file.seek(0)
        hf_response = requests.post(HF_API_URL, files={'image': (img_file.filename, img_file.stream, img_file.mimetype)})
        
        if hf_response.status_code != 200:
            return jsonify({'error': f'Hugging Face endpoint returned error: {hf_response.text}'}), 500
        
        hf_data = hf_response.json()

        # ── Stage 1 Verification (From HF Response) ───────────────────────────
        if not hf_data.get('is_crop', False):
            log.info(f"Gate rejected — leaf_conf={hf_data.get('leaf_conf')}%")
            result = {
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

        # ── Stage 2 Extraction ────────────────────────────────────────────────
        predicted_class = hf_data.get('predicted_class')
        confidence      = hf_data.get('confidence')

        log.info(f"Prediction: {predicted_class} ({confidence}%) | "
                 f"T={temperature} H={humidity} Gas={gas} | Pest risk: {pest_risk}")

        result = {
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
        log.exception("Prediction proxy error")
        return jsonify({'error': str(e)}), 500

# ── Original routes (all completely unchanged) ───────────────────────────────
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
    return jsonify({'status': 'ok', 'models_loaded': False, 'proxy_mode': True})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)