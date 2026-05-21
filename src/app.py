# app.py
# Model Deployment Stage
# This script serves the trained model as a REST API using Flask.
# Reference: Flask documentation
# https://flask.palletsprojects.com/

import joblib
import numpy as np
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Load the trained model
model_path = os.environ.get('MODEL_PATH', 'model/model.pkl')
model = joblib.load(model_path)

@app.route('/')
def home():
    return jsonify({
        'message': 'Air Quality Prediction API',
        'usage': '/predict?temp=VALUE',
        'example': '/predict?temp=15.0'
    })

@app.route('/predict', methods=['GET'])
def predict():
    try:
        temp = float(request.args.get('temp'))
        prediction = model.predict(np.array([[temp]]))[0]
        return jsonify({
            'temperature': temp,
            'predicted_CO': round(float(prediction), 4),
            'unit': 'mg/m3'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
