# Import necessary libraries
from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model and scaler
model = load_model('palm_yield_model.h5')
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Retrieve input from the user form
        luas_perkebunan = float(request.form['luas_perkebunan'])
        jumlah_pohon = float(request.form['jumlah_pohon'])
        umur_pohon = float(request.form['umur_pohon'])
        produksi_tbs = float(request.form['produksi_tbs'])

        # Preprocess the input
        user_input = np.array([[luas_perkebunan, jumlah_pohon, umur_pohon, produksi_tbs]])
        user_input_scaled = scaler.transform(user_input)

        # Make prediction using the model
        prediction = model.predict(user_input_scaled)
        hasil_panen = prediction[0][0]

        return render_template('index.html', hasil_panen=hasil_panen)

    return render_template('index.html', hasil_panen=None)

if __name__ == "__main__":
    app.run(debug=True)
