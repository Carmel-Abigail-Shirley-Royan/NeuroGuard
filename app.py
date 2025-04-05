from flask import Flask, request, jsonify
import pandas as pd
from email_alert import send_email_alert
import joblib
import traceback
import numpy as np
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load trained model and scaler
try:
    rf_model = joblib.load("seizure_model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("✅ Model and Scaler loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model or scaler: {e}")
    exit(1)

@app.route('/upload', methods=['POST'])
def upload_file():
    print("\n📂 File upload received!")

    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        data = pd.read_csv(file)
        print("\n🟢 Uploaded Data:\n", data.head())

        expected = scaler.n_features_in_
        if data.shape[1] != expected:
            return jsonify({"error": f"Expected {expected} features, got {data.shape[1]}"}), 400

        X_scaled = scaler.transform(data)
        y_pred = rf_model.predict(X_scaled)

        result = ["Seizure Detected" if p == 1 else "No Seizure Detected" for p in y_pred]
        print("\n🧠 Prediction:", result)

        return jsonify({"predictions": result})

    except Exception as e:
        print("\n❌ Processing error:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/emergency', methods=['POST'])
def emergency_alert():
    data = request.get_json()
    user = data.get("user", "Unknown")
    lat = data.get("lat")
    lon = data.get("lon")
    doctor_email = data.get("doctor_email")
    sender_email = data.get("sender_email")
    sender_password = data.get("sender_password")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    maps_link = f"https://www.google.com/maps?q={lat},{lon}"

    print(f"\n🚨 Emergency Alert Received!")
    print(f"👤 User: {user}")
    print(f"📍 Location: Latitude={lat}, Longitude={lon}")
    print(f"✉️ Doctor Email: {doctor_email}")
    print(f"📨 Sender Email: {sender_email}")
    print(f"🕒 Time: {timestamp}")
    print(f"🌍 Maps Link: {maps_link}")

    # 🔔 Send email using user's credentials
    send_email_alert(user, maps_link, doctor_email, sender_email, sender_password)

    return jsonify({
        "status": "Emergency Received",
        "location": {"lat": lat, "lon": lon},
        "time": timestamp,
        "maps_link": maps_link
    })


# Optional model sanity test
test_data = np.array([[60, 36.5, 98, 0.1]])
test_scaled = scaler.transform(test_data)
test_pred = rf_model.predict(test_scaled)
print("\n🔍 Test Prediction (Normal Input):", test_pred)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
