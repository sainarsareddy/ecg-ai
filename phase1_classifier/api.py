import sys

sys.path.append("..")

import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from phase2_features.extractor import extract_features
from phase3_risk.risk_score import calculate_risk
from phase4_agent.cardiac_agent import get_recent_decisions, make_decision
from phase5_report.report_generator import generate_report

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
model = tf.keras.models.load_model("models/ecg_model.h5")
classes = np.load("models/classes.npy", allow_pickle=True)

SEGMENT_LENGTH = 187

app = FastAPI(
    title="ECG Intelligence API",
    description="ECG Classifier with Clinical Features and Risk Assessment.",
    version="1.0.0"
)

# ── REQUEST SCHEMA ────────────────────────────────────────────────────────────
class ECGInput(BaseModel):
    ecg_signal: list[float]

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "running", "model_version": "v1.0.0", "classes": list(classes)}

@app.get("/agent/recent")
def recent_decisions():
    return {"recent_decisions": get_recent_decisions(5)}

@app.post("/predict")
def predict(data: ECGInput):
    signal = np.array(data.ecg_signal)

    # Validate minimum length
    if len(signal) < 360:
        raise HTTPException(
            status_code=400,
            detail=f"Signal must be at least 360 samples. Got {len(signal)}."
        )

    # Use first 187 samples for model prediction
    predict_signal = signal[:187]
    predict_signal = (predict_signal - np.mean(predict_signal)) / (np.std(predict_signal) + 1e-8)
    predict_signal = predict_signal.reshape(1, SEGMENT_LENGTH, 1)

    # Predict condition
    predictions = model.predict(predict_signal)
    confidence = float(np.max(predictions) * 100)
    condition = classes[np.argmax(predictions)]

    # Extract clinical features from full signal
    features = extract_features(data.ecg_signal)

    # Calculate risk score
    risk = calculate_risk(features, confidence)

    # ACDA Decision
    agent_decision = make_decision(condition, confidence, features, risk)

    report = generate_report(condition, confidence, features, risk, agent_decision)

    return {
        "condition": condition,
        "confidence": round(confidence, 2),
        "all_scores": {cls: round(float(score) * 100, 2) for cls, score in zip(classes, predictions[0])},
        "clinical_features": features,
        "risk_assessment": risk,
        "agent_decision": agent_decision,
        "report": report
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)