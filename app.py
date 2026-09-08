from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd

app = FastAPI(title="Student Placement Prediction")

# Load model
model = joblib.load("model/placement_prediction_model.pkl")

# Templates folder
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.post("/predict")
def predict_student(data: dict):

    student = pd.DataFrame([data])

    prediction = model.predict(student)[0]

    probability = model.predict_proba(student)

    confidence = round(
        float(probability.max() * 100),
        2
    )

    return {
        "prediction": prediction,
        "confidence": confidence
    }