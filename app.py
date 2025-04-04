from flask import Flask, request, jsonify
import pandas as pd
import joblib
import traceback
from twilio.rest import Client
import numpy as np
import os
from dotenv import load_dotenv


from flask_cors import CORS
from datetime import datetime
load_dotenv()

# Optional: Uncomment if using Twilio for SMS
# from twilio.rest import Client

app = Flask(__name__)
CORS(app)  # Allow requests from any frontend

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

        # Check feature count
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
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n🚨 Emergency Alert Received!")
    print(f"👤 User: {user}")
    print(f"📍 Location: Latitude={lat}, Longitude={lon}")
    print(f"🕒 Time: {timestamp}")

    # Optional: Send SMS using Twilio (uncomment and set credentials to use)
    account_sid = os.getenv("TWILIO_SID")
    auth_token = os.getenv("TWILIO_TOKEN")

    if not account_sid or not auth_token:
        print("❌ Missing Twilio credentials in environment variables.")
        return jsonify({"error": "Missing Twilio credentials"}), 500

    

    client = Client(account_sid, auth_token)

    message = client.messages.create(
        body=f"🚨 EMERGENCY: Seizure detected for {user}. Location: https://maps.google.com/?q={lat},{lon}",
        from_='+18782158846',
        to='+916374134569'
    )
    print(f"📲 SMS sent with SID: {message.sid}")
    

    return jsonify({
        "status": "Emergency Received",
        "location": {"lat": lat, "lon": lon},
        "time": timestamp
    })

# Optional internal model test
test_data = np.array([[60, 36.5, 98, 0.1]])
test_scaled = scaler.transform(test_data)
test_pred = rf_model.predict(test_scaled)
print("\n🔍 Test Prediction (Normal Input):", test_pred)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
