from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Load encoders and scalers
encoder = pickle.load(open('encoder.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['symptoms']
    # Assume the symptoms come as a list
    symptoms = np.array(data).reshape(1, -1)
    symptoms_scaled = scaler.transform(symptoms)
    
    prediction = model.predict(symptoms_scaled)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
