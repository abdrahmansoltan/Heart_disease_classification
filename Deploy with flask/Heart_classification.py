import pickle
import numpy as np

from flask import Flask, request, jsonify

def predict_single(patient, dv, model):
    X = dv.transform([patient])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]


with open('../saved_model/logistic_regression_model_2.pkl', 'rb') as f_in:
    model, dv = pickle.load(f_in)

patient = {'age': 57.0,
            'sex': 0.0,
            'cp': 0.0,
            'trestbps': 140.0,
            'chol': 241.0,
            'fbs': 0.0,
            'restecg': 1.0,
            'thalach': 123.0,
            'exang': 1.0,
            'oldpeak': 0.2,
            'slope': 1.0,
            'ca': 0.0,
            'thal': 3.0}

print('prop of heart disease: ',predict_single(patient, dv, model))


app = Flask('heart_disease')

@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()

    prediction = predict_single(patient, dv, model)
    heart_disease = prediction >= 0.5
    
    result = {
        'heart_disease_probability': float(prediction),
        'heart_disease': bool(heart_disease)
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)