import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import joblib
import uvicorn

app = FastAPI()

# loading model

def load_model():
    try:
        model_data = joblib.load('model_path')
        return model_data['model'], model_data['scaler'] # if model have both model,scaler
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None

model, scaler = load_model()

required_fields = [""]


class ChurnInput(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int

@app.get("/")
async def root():
    return {"message": "Hello world"}

@app.post("/predict")
async def predict(data: ChurnInput):
    try:

        input_data = [getattr(data, field) for field in required_fields]
        input_df = pd.DataFrame([input_data], columns=required_fields)

        scaled_input = scaler.transform(input_df)

        # Prediction
        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)
        confidence = float(prediction_proba[0][prediction[0]] * 100)

        # print(prediction_proba[0][1]) debug area

        churn_proba = prediction_proba[0][1]

        if churn_proba>0.7:
            risk = "high"
        
        elif churn_proba>0.3:
            risk ="medium"
        
        else:
            risk ='low'


        if prediction[0] == 1:
            return {"Churn": int(prediction[0]),
                     "confidence": confidence,
                     "churn_probability": churn_proba,
                     "risk_level":risk}
        else:
            return {"No Churn": int(prediction[0]),
                    "confidence": confidence,
                    "churn_probability": churn_proba,
                    "risk_level":risk}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == '__main__':
    uvicorn.run(debug=True)

#start uvicorn server
# uvicorn app:app --reload
