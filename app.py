from flask import Flask, request, jsonify
import pandas as pd
import joblib
import traceback
import numpy as np
from flask_cors import CORS  # Import CORS to handle frontend requests

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load the trained model and scaler
try:
    rf_model = joblib.load("seizure_model.pkl")
    scaler = joblib.load("scaler.pkl")  # Ensure the same scaler is used
    print("✅ Model and Scaler loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model or scaler: {e}")
    exit(1)  # Stop execution if models fail to load

@app.route('/upload', methods=['POST'])
def upload_file():
    print("\n📂 New file upload request received!")

    if 'file' not in request.files:
        response = {"error": "No file provided"}
        print("\n🚨 Response Sent:", response)
        return jsonify(response), 400

    file = request.files['file']

    if file.filename == '':
        response = {"error": "No selected file"}
        print("\n🚨 Response Sent:", response)
        return jsonify(response), 400

    try:
        # Read the CSV file
        new_data = pd.read_csv(file)
        print("\n🟢 Uploaded Data (First 5 Rows):\n", new_data.head())

        # Ensure data has the correct number of features
        expected_features = scaler.n_features_in_
        received_features = new_data.shape[1]

        print(f"Expected Features: {expected_features}, Received Features: {received_features}")

        if received_features != expected_features:
            response = {"error": f"Incorrect number of features. Expected {expected_features}, but got {received_features}"}
            print("\n🚨 Response Sent:", response)
            return jsonify(response), 400

        # Preprocess using the same scaler
        X_scaled = scaler.transform(new_data)

        # Predict
        y_pred = rf_model.predict(X_scaled)
        print("\n🔵 Raw Predictions:", y_pred)

        # Convert predictions to readable text
        text_predictions = ["Seizure Detected" if pred == 1 else "No Seizure Detected" for pred in y_pred]
        print("\n🟠 Text Predictions:", text_predictions)

        # Prepare final response
        response = jsonify({"predictions": text_predictions})
        print("\n✅ Response Sent:", response.get_json())  # Log final response
        return response

    except Exception as e:
        print("\n❌ Error during processing:\n", traceback.format_exc())
        response = jsonify({"error": str(e)})
        print("\n🚨 Response Sent:", response.get_json())
        return response, 500

# Debugging: Test if model can predict "No Seizure Detected"
test_data = np.array([[60, 36.5, 98, 0.1]])  # Normal case (non-seizure values)
test_data_scaled = scaler.transform(test_data)
test_prediction = rf_model.predict(test_data_scaled)

print("\n🔴 Test Prediction for Normal Data:", test_prediction)
print("\n🟠 Expected Output: 'No Seizure Detected' if model is balanced")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Allows access from other devices on the same network