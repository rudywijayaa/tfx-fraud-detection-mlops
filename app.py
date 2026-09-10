import os
import base64
import time
from flask import Flask, request, jsonify
import tensorflow as tf
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

# SETUP PROMETHEUS METRICS 
REQUEST_COUNT = Counter(
    'app_request_count',
    'Total HTTP requests',
    ['method', 'endpoint', 'http_status']
)
REQUEST_LATENCY = Histogram(
    'app_request_latency_seconds',
    'Request latency in seconds',
    ['endpoint']
)
FRAUD_DETECTED = Counter(
    'fraud_predictions_total',
    'Total number of positive fraud predictions'
)

# LOAD TFX SAVEDMODEL
# ==========================================
MODEL_BASE_DIR = "serving_model/fraud_detection"
try:
    latest_version = max([d for d in os.listdir(MODEL_BASE_DIR) if os.path.isdir(os.path.join(MODEL_BASE_DIR, d))])
    MODEL_PATH = os.path.join(MODEL_BASE_DIR, latest_version)
    print(f"[*] Loading model dari: {MODEL_PATH}")
    model = tf.saved_model.load(MODEL_PATH)
    infer = model.signatures["serving_default"]
except Exception as e:
    print(f"[!] Gagal memuat model: {e}")
    infer = None

# API ENDPOINTS
@app.route('/')
def index():
    return jsonify({"message": "Fraud Detection API is running. Access /predict for inference and /metrics for Prometheus."})

@app.route('/metrics')
def metrics():
    """Endpoint untuk ditarik (scrape) oleh server Prometheus"""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    
    if infer is None:
        REQUEST_COUNT.labels('POST', '/predict', 500).inc()
        return jsonify({"error": "Model tidak tersedia di server."}), 500
        
    try:
        data = request.get_json()
        
        # Ekstrak string base64 dari payload (format TF Serving)
        b64_string = data['instances'][0]['examples']['b64']
        
        # Decode base64 kembali menjadi bytes
        serialized_example = base64.b64decode(b64_string)
        
        # Ubah menjadi tensor string 1D sesuai ekspektasi TFX signature
        input_tensor = tf.constant([serialized_example], dtype=tf.string)
        
        # Jalankan prediksi
        result = infer(examples=input_tensor)
        
        # Ambil probabilitas output
        output_key = list(result.keys())[0] 
        prediction_score = float(result[output_key].numpy()[0][0])
        
        # Tambah metrik Prometheus jika fraud terdeteksi (threshold > 0.5)
        if prediction_score > 0.5:
            FRAUD_DETECTED.inc()
            
        REQUEST_COUNT.labels('POST', '/predict', 200).inc()
        REQUEST_LATENCY.labels('/predict').observe(time.time() - start_time)
        
        return jsonify({
            "predictions": [[prediction_score]]
        }), 200

    except Exception as e:
        REQUEST_COUNT.labels('POST', '/predict', 400).inc()
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)