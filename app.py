from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="Student Placement Prediction")

model = joblib.load("model/placement_prediction_model.pkl")


@app.get("/")
def home():
    return {
        "message": "Student Placement Prediction API is running"
    }


@app.post("/predict")
def predict_student(data: dict):

    student = pd.DataFrame([data])

    prediction = model.predict(student)[0]

    result = {
        "prediction": prediction
    }

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(student)
        result["confidence"] = round(
            float(probability.max() * 100), 2
        )

    return result