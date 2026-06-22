import os
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib

# Suppress TensorFlow logging alerts for clean console execution
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model

# 1. FIND THE EXACT BASE DIRECTORY PATHS Dynamically
base_dir = os.path.dirname(os.path.abspath(__file__))

# Explicitly tell Flask exactly where templates and static folders live
app = Flask(
    __name__,
    template_folder=os.path.join(base_dir, "templates"),
    static_folder=os.path.join(base_dir, "static")
)

# 2. LOAD SERIALIZED ARTIFACTS USING ABSOLUTE PATH CODES
scaler = joblib.load(os.path.join(base_dir, "ANN_scaler.pkl"))
model = load_model(os.path.join(base_dir, "obesity_prediction_ANN_model.keras"))
target_encoder = joblib.load(os.path.join(base_dir, "target_encoder.pkl"))
feature_columns = joblib.load(os.path.join(base_dir, "feature_columns.pkl"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "Missing JSON request body"}), 400

        # Construct the matching patient row structure matrix sequence mapping original training
        patient_row = np.array([
            data["Age"],
            data["Gender"],
            data["Height"],
            data["Weight"],
            data["family_history_with_overweight"],  
            data["FAVC"],
            data["FCVC"],
            data["NCP"],
            data["CAEC"],
            data["SMOKE"],
            data["CH2O"],
            data["SCC"],
            data["FAF"],
            data["TUE"],
            data["CALC"],
            data["MTRANS"]
        ], dtype=object)

        raw_columns = [
            'Age', 'Gender', 'Height', 'Weight', 'family_history_with_overweight',
            'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF',
            'TUE', 'CALC', 'MTRANS'
        ]

        input_df = pd.DataFrame([patient_row], columns=raw_columns)

        # Convert numeric indicators explicitly from generic objects to float variables
        numeric_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        for col in numeric_cols:
            input_df[col] = pd.to_numeric(input_df[col])

        # Define all categorical fields for dummy variable processing
        categorical_cols = [
            'Gender', 'family_history_with_overweight', 'FAVC', 
            'SMOKE', 'SCC', 'CAEC', 'CALC', 'MTRANS'
        ]

        # Apply matching One-Hot encoded alignments 
        input_encoded = pd.get_dummies(input_df, columns=categorical_cols)

        # Force structural order matching with feature columns framework schema
        input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

        # Scale array inputs
        input_scaled = scaler.transform(input_encoded)

        # Route variables into deep layers to extract classifications
        predictions = model.predict(input_scaled, verbose=0)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]

        # Decode arrays back to categorical representation string outputs
        predicted_label = target_encoder.inverse_transform([predicted_class_idx])[0]
        confidence = float(np.max(predictions) * 100)

        return jsonify({
            "status": "success",
            "prediction": str(predicted_label),
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
