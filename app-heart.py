from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('heart-disease-model.h5')
scaler = StandardScaler()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get data from POST request
        data = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalach']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['ca']),
            float(request.form['thal'])
        ]

        #load the scaler
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        # Reshape and standardize data
        data = scaler.transform(np.array(data).reshape(1, -1))

        # Make prediction
        prediction = model.predict(data)
        result = 'You have a risk of heart disease' if prediction[0] > 0.5 else 'You seem healthy!'

        return render_template('index.html', result=result)

    return render_template('index.html', result=None)

if __name__ == "__main__":
    app.run(debug=True)