from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd

app = FastAPI(title="Student Placement Prediction")

# Load trained model
model = joblib.load("model/placement_prediction_model.pkl")


@app.get("/", response_class=HTMLResponse)
def home():
    with open("templates/index.html", "r", encoding="utf-8") as file:
        return file.read()


@app.post("/predict")
def predict_student(data: dict):

    student = pd.DataFrame([data])

    prediction = model.predict(student)[0]

    probability = model.predict_proba(student)
    confidence = round(float(probability.max() * 100), 2)

    return {
        "prediction": prediction,
        "confidence": confidence
    }