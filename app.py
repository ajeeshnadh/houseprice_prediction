# app.py
# from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form
        area = float(request.form.get('area'))
        bedrooms = int(request.form.get('bedrooms'))
        age = int(request.form.get('age'))

        # Prepare input for prediction
        features = np.array([[area, bedrooms, age]])
        prediction = model.predict(features)[0]

        # Show formatted output
        return render_template('index.html',
                               prediction_text=f'Estimated House Price: ₹{prediction:,.2f}')
    except Exception as e:
        import traceback
        return f"<h3>Error:</h3><pre>{traceback.format_exc()}</pre>"

if __name__ == "__main__":
    app.run(debug=True)

