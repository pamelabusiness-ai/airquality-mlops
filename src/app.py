# app.py
# Model Deployment Stage
# Flask REST API for Air Quality CO Prediction
# Reference: Flask documentation https://flask.palletsprojects.com/

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import os

app = Flask(__name__)


# Load the trained model
model_path = os.environ.get("MODEL_PATH", "model/model.pkl")
model = joblib.load(model_path)


# Load dataset for stats
data_path = os.environ.get("DATA_PATH", "data/processed.csv")


def get_risk_level(co_value):
    if co_value < 1.0:
        return "LOW"
    elif co_value < 2.5:
        return "MODERATE"
    elif co_value < 4.0:
        return "HIGH"
    else:
        return "VERY HIGH"


@app.route("/")
def home():
    return jsonify(
        {
            "message": "Air Quality Prediction API",
            "model": "Linear Regression",
            "dataset": "UCI Air Quality Dataset",
            "endpoints": {
                "single_prediction": "/predict?temp=VALUE",
                "batch_prediction": "/predict/batch?temps=5,15,25,35",
                "risk_assessment": "/predict/risk?temp=VALUE",
                "dataset_stats": "/stats",
                "health_check": "/health",
            },
        }
    )


@app.route("/predict", methods=["GET"])
def predict():
    try:
        temp = float(request.args.get("temp"))
        prediction = model.predict(np.array([[temp]]))[0]
        return jsonify(
            {
                "temperature": temp,
                "predicted_CO": round(float(prediction), 4),
                "unit": "mg/m3",
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict/batch", methods=["GET"])
def predict_batch():
    try:
        temps_str = request.args.get("temps")
        temps = [float(t) for t in temps_str.split(",")]
        predictions = []
        for temp in temps:
            co = model.predict(np.array([[temp]]))[0]
            predictions.append(
                {
                    "temperature": temp,
                    "predicted_CO": round(float(co), 4),
                    "unit": "mg/m3",
                }
            )
        return jsonify({"batch_size": len(predictions), "predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict/risk", methods=["GET"])
def predict_risk():
    try:
        temp = float(request.args.get("temp"))
        co = model.predict(np.array([[temp]]))[0]
        risk = get_risk_level(float(co))
        return jsonify(
            {
                "temperature": temp,
                "predicted_CO": round(float(co), 4),
                "unit": "mg/m3",
                "risk_level": risk,
                "risk_description": {
                    "LOW": "Air quality is good",
                    "MODERATE": "Air quality is acceptable",
                    "HIGH": "Air quality is unhealthy for sensitive groups",
                    "VERY HIGH": "Air quality is very unhealthy",
                }[risk],
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/stats", methods=["GET"])
def stats():
    try:
        df = pd.read_csv(data_path)
        return jsonify(
            {
                "dataset": "UCI Air Quality Dataset",
                "total_records": len(df),
                "features": list(df.columns),
                "temperature": {
                    "min": round(float(df["T"].min()), 2),
                    "max": round(float(df["T"].max()), 2),
                    "mean": round(float(df["T"].mean()), 2),
                },
                "CO_concentration": {
                    "min": round(float(df["CO(GT)"].min()), 2),
                    "max": round(float(df["CO(GT)"].max()), 2),
                    "mean": round(float(df["CO(GT)"].mean()), 2),
                },
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model": "loaded"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
