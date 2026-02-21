def calculate_risk(features: dict, confidence: float) -> dict:

    score = 0
    reasons = []

    hr = features.get("heart_rate")
    qrs = features.get("qrs_duration_ms")
    rr_irregular = features.get("rr_irregular")
    rr_std = features.get("rr_std_ms")
    pr = features.get("pr_interval_ms")
    qt = features.get("qt_interval_ms")

    # ── HEART RATE ────────────────────────────────────────────────────────────
    if hr is not None:
        if hr > 110:
            score += 2
            reasons.append(f"Tachycardia detected — HR {hr} bpm exceeds 110")
        elif hr < 50:
            score += 2
            reasons.append(f"Bradycardia detected — HR {hr} bpm below 50")

    # ── QRS DURATION ──────────────────────────────────────────────────────────
    if qrs is not None:
        if qrs > 120:
            score += 2
            reasons.append(f"Wide QRS — {qrs}ms suggests bundle branch block")

    # ── RR IRREGULARITY ───────────────────────────────────────────────────────
    if rr_irregular:
        score += 3
        reasons.append(f"Irregular RR intervals — RR std {rr_std}ms, possible AFib")

    # ── PR INTERVAL ───────────────────────────────────────────────────────────
    if pr is not None:
        if pr > 200:
            score += 2
            reasons.append(f"Prolonged PR interval — {pr}ms suggests heart block")
        elif pr < 120:
            score += 1
            reasons.append(f"Short PR interval — {pr}ms, possible pre-excitation")

    # ── QT INTERVAL ───────────────────────────────────────────────────────────
    if qt is not None and not (qt != qt):  # nan check
        if qt > 450:
            score += 2
            reasons.append(f"Prolonged QT — {qt}ms, arrhythmia risk")

    # ── CONFIDENCE PENALTY ────────────────────────────────────────────────────
    confidence_penalty = False
    if confidence < 75:
        confidence_penalty = True
        reasons.append(f"Low model confidence — {confidence}%, diagnosis uncertain")

    # ── CONFIDENCE + RISK INTERACTION ─────────────────────────────────────────
    # If uncertain AND risky — escalate
    if confidence_penalty and score >= 5:
        score += 2
        reasons.append("Confidence penalty applied — uncertain diagnosis in high risk case")

    # ── SEVERITY MAPPING ──────────────────────────────────────────────────────
    if score <= 3:
        severity = "LOW"
        action = "Monitor patient. No immediate intervention required."
    elif score <= 6:
        severity = "MODERATE"
        action = "Refer to GP. Follow up within 24 hours."
    elif score <= 9:
        severity = "HIGH"
        action = "Cardiologist referral required. Urgent review needed."
    else:
        severity = "CRITICAL"
        action = "Emergency escalation. Immediate intervention required."

    # Escalate one level if confidence is low and severity is not already critical
    if confidence_penalty and severity != "CRITICAL":
        levels = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
        current_index = levels.index(severity)
        severity = levels[current_index + 1]
        action_map = {
            "MODERATE": "Refer to GP. Follow up within 24 hours.",
            "HIGH": "Cardiologist referral required. Urgent review needed.",
            "CRITICAL": "Emergency escalation. Immediate intervention required."
        }
        action = action_map[severity]
        reasons.append("Severity escalated one level due to low confidence.")

    return {
        "risk_score": score,
        "severity": severity,
        "action": action,
        "reasons": reasons,
        "confidence_penalty_applied": confidence_penalty
    }