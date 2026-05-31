import os
import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Force Python to look in the exact folder where this app file lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'best_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')

# Safely load the model and scaler
try:
    model = pickle.load(open(model_path, 'rb'))
    scaler = pickle.load(open(scaler_path, 'rb'))
    print("[INFO] Model and Scaler loaded perfectly!")
except Exception as e:
    print(f"[ERROR] Failed to load assets: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Grab all inputs from the web form layout
        form_features = [
            int(request.form['no_of_dependents']),
            int(request.form['education']),
            int(request.form['self_employed']),
            float(request.form['income_annum']),
            float(request.form['loan_amount']),
            float(request.form['loan_term']),
            float(request.form['cibil_score']),
            float(request.form['residential_assets_value']),
            float(request.form['commercial_assets_value']),
            float(request.form['luxury_assets_value']),
            float(request.form['bank_asset_value'])
        ]
        
        # 2. Convert to array and scale using the matched 'scaler' variable
        final_features = np.array([form_features])
        scaled_features = scaler.transform(final_features)
        
        # 3. Predict eligibility
        prediction = model.predict(scaled_features)
        
        if prediction[0] == 1 or str(prediction[0]).strip().lower() in ['1', 'approved', 'yes']:
            result_text = "Loan Status Result: Approved ✅"
        else:
            result_text = "Loan Status Result: Rejected ❌"
            
        return render_template('index.html', prediction_text=result_text)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)