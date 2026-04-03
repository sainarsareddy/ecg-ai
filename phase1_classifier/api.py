import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("[STEP 1] imports starting...")
import gdown
import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

print("[STEP 2] core imports loaded")

# ── DOWNLOAD MODEL FILES ──────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

if not os.path.exists("models/ecg_model.h5"):
    print("[INFO] Downloading ecg_model.h5...")
    gdown.download(
        "https://drive.google.com/uc?id=1CgB3tIMCkn1MPuhEFGB8EXeIJbC8WNv3&confirm=t",
        "models/ecg_model.h5",
        quiet=False,
        fuzzy=True
    )
    print("[INFO] ecg_model.h5 downloaded")

if not os.path.exists("models/classes.npy"):
    print("[INFO] Downloading classes.npy...")
    gdown.download(
        "https://drive.google.com/uc?id=1QCuVgdG4kW3yPHIRgN6dfUgN8g6RBMzM&confirm=t",
        "models/classes.npy",
        quiet=False,
        fuzzy=True
    )
    print("[INFO] classes.npy downloaded")

# ── IMPORT PHASE MODULES ──────────────────────────────────────────────────────
try:
    print("[STEP 3] importing phase modules...")
    from phase2_features.extractor import extract_features
    print("[STEP 4] phase2 ok")
    from phase3_risk.risk_score import calculate_risk
    print("[STEP 5] phase3 ok")
    from phase4_agent.cardiac_agent import get_recent_decisions, make_decision
    print("[STEP 6] phase4 ok")
    from phase5_report.report_generator import generate_report
    print("[STEP 7] phase5 ok")
except Exception as e:
    print(f"[ERROR] Failed to import phase modules: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
try:
    print("[STEP 8] Loading model...")
    model = tf.keras.models.load_model("models/ecg_model.h5", compile=False)
    classes = np.load("models/classes.npy", allow_pickle=True)
    print("[STEP 9] Model loaded successfully!")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    traceback.print_exc()
    sys.exit(1)

SEGMENT_LENGTH = 187

# ── FASTAPI APP ───────────────────────────────────────────────────────────────
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

    if len(signal) < 360:
        raise HTTPException(
            status_code=400,
            detail=f"Signal must be at least 360 samples. Got {len(signal)}."
        )

    predict_signal = signal[:187]
    predict_signal = (predict_signal - np.mean(predict_signal)) / (np.std(predict_signal) + 1e-8)
    predict_signal = predict_signal.reshape(1, SEGMENT_LENGTH, 1)

    predictions = model.predict(predict_signal)
    confidence = float(np.max(predictions) * 100)
    condition = classes[np.argmax(predictions)]

    features = extract_features(data.ecg_signal)
    risk = calculate_risk(features, confidence)
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
    uvicorn.run("api:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))