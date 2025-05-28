from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load model and scaler
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.post("/predict")
async def predict(data: dict):
    required_features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 
                        'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 
                        'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 
                        'V28', 'Amount']
    input_data = [data.get(f, 0.0) for f in required_features]
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    return {"fraud": bool(prediction), "fraud_probability": float(probability)}
app = FastAPI()

# Load model and scaler
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.post("/predict")
async def predict(data: dict):
    required_features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 
                        'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 
                        'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 
                        'V28', 'Amount']
    input_data = [data.get(f, 0.0) for f in required_features]
    input_array = np.array(input_data).reshape(1, -1)
    
    # Scale input
    input_scaled = scaler.transform(input_array)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    return {"fraud": bool(prediction), "fraud_probability": float(probability)}



