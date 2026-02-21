import json
import os
from datetime import datetime

LOG_FILE = "decisions.log"

def make_decision(condition: str, confidence: float, features: dict, risk: dict) -> dict:

    signal_quality = features.get("signal_quality", 0)
    risk_score = risk.get("risk_score", 0)
    severity = risk.get("severity", "LOW")
    confidence_penalty = risk.get("confidence_penalty_applied", False)
    reasons = risk.get("reasons", [])

    # ── DECISION LOGIC ────────────────────────────────────────────────────────

    # Rule 1 — Signal quality too low
    if signal_quality < 0.05:
        decision = "REJECT"
        decision_reason = "Signal quality is too low to make a reliable diagnosis. Please re-record the ECG."
        triage_urgency = "Routine"

    # Rule 2 — Model not confident enough
    elif confidence < 75 and risk_score <= 4:
        decision = "RECHECK"
        decision_reason = f"Model confidence is only {round(confidence, 1)}%. Signal should be re-recorded for a reliable diagnosis."
        triage_urgency = "Routine"

    # Rule 3 — Critical risk
    elif risk_score > 6 or severity == "CRITICAL":
        decision = "EMERGENCY"
        decision_reason = f"Risk score {risk_score} exceeds critical threshold. Condition '{condition}' with severity {severity} requires immediate intervention."
        triage_urgency = "Immediate"

    # Rule 4 — High risk
    elif risk_score > 4 or severity == "HIGH":
        decision = "ALERT"
        decision_reason = f"Risk score {risk_score} indicates high clinical risk. Specialist review required urgently."
        triage_urgency = "Within 24hrs"

    # Rule 5 — Moderate
    elif severity == "MODERATE":
        decision = "ALERT"
        decision_reason = f"Moderate risk detected. Condition '{condition}' should be reviewed by a GP."
        triage_urgency = "Within 24hrs"

    # Rule 6 — All clear
    else:
        decision = "ACCEPT"
        decision_reason = f"Condition '{condition}' detected with {round(confidence, 1)}% confidence. Low risk. Continue monitoring."
        triage_urgency = "Routine"

    result = {
        "decision": decision,
        "decision_reason": decision_reason,
        "triage_urgency": triage_urgency,
        "condition": condition,
        "confidence": round(confidence, 2),
        "risk_score": risk_score,
        "severity": severity,
        "timestamp": datetime.now().isoformat()
    }

    # ── LOG DECISION ──────────────────────────────────────────────────────────
    log_decision(result)

    return result


def log_decision(result: dict):
    os.makedirs("logs", exist_ok=True)
    with open(f"logs/{LOG_FILE}", "a") as f:
        f.write(json.dumps(result) + "\n")


def get_recent_decisions(n: int = 5) -> list:
    log_path = f"logs/{LOG_FILE}"
    if not os.path.exists(log_path):
        return []
    with open(log_path, "r") as f:
        lines = f.readlines()
    recent = lines[-n:]
    return [json.loads(line) for line in recent]