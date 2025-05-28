from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Fraud Detection API")

# Load model and scaler
try:
    model = joblib.load('fraud_model.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    raise Exception(f"Error loading model or scaler: {e}")

# Define input data model
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.get("/")
async def root():
    return {"message": "Welcome to the Fraud Detection API. Use POST /predict to make predictions."}

@app.post("/predict")
async def predict(transaction: Transaction):
    try:
        # Extract features in correct order
        required_features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 
                            'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 
                            'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 
                            'V28', 'Amount']
        input_data = [getattr(transaction, f) for f in required_features]
        input_array = np.array(input_data).reshape(1, -1)
        
        # Scale input
        input_scaled = scaler.transform(input_array)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        return {"fraud": bool(prediction), "fraud_probability": float(probability)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

