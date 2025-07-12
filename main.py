from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Load model and vectorizer
model = joblib.load("sms_spam_model.pkl")
vectorizer = joblib.load("sms_vectorizer.pkl")

app = FastAPI()

class SMSInput(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"message": "SMS Spam Classifier API is running ✅"}

@app.post("/predict")
def predict_spam(data: SMSInput):
    try:
        vec = vectorizer.transform([data.message])
        prediction = model.predict(vec)[0]
        label = "spam" if prediction == 1 else "ham"
        return {"label": label, "prediction": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
