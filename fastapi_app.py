from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
    vectorizer = pickle.load(f)

class SMSInput(BaseModel):
    message: str

@app.post("/predict")
def predict(input: SMSInput):
    vec = vectorizer.transform([input.message])
    pred = model.predict(vec)[0]
    label = "spam" if pred == 1 else "ham"
    return {"label": label, "prediction": int(pred)}

@app.get("/")
def root():
    return {"message": "SMS Spam API is running"}
