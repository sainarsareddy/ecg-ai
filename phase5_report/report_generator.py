import json
import os
import uuid
from datetime import datetime

from dotenv import load_dotenv
from groq import Groq
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_report(condition, confidence, features, risk, agent_decision):

    report_id = str(uuid.uuid4())[:8].upper()

    # ── BUILD PROMPT ──────────────────────────────────────────────────────────
    prompt = f"""
You are a clinical AI assistant. Based on the following ECG analysis results, generate two summaries.

ECG ANALYSIS DATA:
- Condition Detected: {condition}
- Model Confidence: {confidence}%
- Heart Rate: {features.get('heart_rate')} bpm
- RR Interval: {features.get('rr_interval_ms')} ms
- RR Irregular: {features.get('rr_irregular')}
- QRS Duration: {features.get('qrs_duration_ms')} ms
- PR Interval: {features.get('pr_interval_ms')} ms
- QT Interval: {features.get('qt_interval_ms')} ms
- Signal Quality: {features.get('signal_quality')}
- Risk Score: {risk.get('risk_score')} / 12
- Severity: {risk.get('severity')}
- Clinical Reasons: {', '.join(risk.get('reasons', []))}
- Agent Decision: {agent_decision.get('decision')}
- Agent Reason: {agent_decision.get('decision_reason')}
- Triage Urgency: {agent_decision.get('triage_urgency')}

Generate a JSON response with exactly these three keys:
1. "clinical_summary": A 3-4 sentence technical summary for a cardiologist. Use medical terminology. Include all abnormal findings, their clinical significance, and recommended investigations.
2. "patient_summary": A 3-4 sentence plain English summary for the patient and family. No jargon. Explain what the readings mean, whether they should be worried, and what happens next.
3. "suggested_action": One clear sentence stating exactly what should happen next.

Respond with valid JSON only. No extra text. No markdown. No backticks.
"""

    # ── CALL GEMINI ───────────────────────────────────────────────────────────
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.choices[0].message.content.strip()

    # Clean if wrapped in backticks
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    summaries = json.loads(raw)

    # ── GENERATE PDF ──────────────────────────────────────────────────────────
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    pdf_path = os.path.join(reports_dir, f"ECG_Report_{report_id}.pdf")
    generate_pdf(pdf_path, report_id, condition, confidence, features, risk, agent_decision, summaries)
    
    return {
        "report_id": report_id,
        "clinical_summary": summaries.get("clinical_summary"),
        "patient_summary": summaries.get("patient_summary"),
        "suggested_action": summaries.get("suggested_action"),
        "pdf_path": pdf_path
    }


def generate_pdf(path, report_id, condition, confidence, features, risk, agent_decision, summaries):

    doc = SimpleDocTemplate(path, pagesize=letter,
                            rightMargin=inch, leftMargin=inch,
                            topMargin=inch, bottomMargin=inch)

    styles = getSampleStyleSheet()
    elements = []

    title_style = ParagraphStyle('title', fontSize=20, textColor=HexColor('#1A3A5C'),
                                  spaceAfter=6, fontName='Helvetica-Bold')
    sub_style = ParagraphStyle('sub', fontSize=11, textColor=HexColor('#555555'),
                                spaceAfter=20)
    section_style = ParagraphStyle('section', fontSize=13, textColor=HexColor('#2E75B6'),
                                    spaceAfter=8, fontName='Helvetica-Bold')
    body_style = ParagraphStyle('body', fontSize=11, spaceAfter=12, leading=16)

    elements.append(Paragraph("ECG Intelligence System", title_style))
    elements.append(Paragraph(f"Clinical Report  |  ID: {report_id}  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}", sub_style))
    elements.append(Spacer(1, 0.2 * inch))

    # ── DIAGNOSIS SUMMARY ─────────────────────────────────────────────────────
    elements.append(Paragraph("Diagnosis Summary", section_style))
    summary_data = [
        ["Condition", condition],
        ["Confidence", f"{confidence}%"],
        ["Risk Score", f"{risk.get('risk_score')} / 12"],
        ["Severity", risk.get('severity')],
        ["Agent Decision", agent_decision.get('decision')],
        ["Triage Urgency", agent_decision.get('triage_urgency')],
    ]
    t = Table(summary_data, colWidths=[2.5 * inch, 4 * inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), HexColor('#E8F0FE')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [HexColor('#F8FAFF'), HexColor('#FFFFFF')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.2 * inch))

    # ── CLINICAL FEATURES ─────────────────────────────────────────────────────
    elements.append(Paragraph("Clinical Measurements", section_style))
    feature_data = [
        ["Heart Rate", f"{features.get('heart_rate')} bpm"],
        ["RR Interval", f"{features.get('rr_interval_ms')} ms"],
        ["RR Irregular", str(features.get('rr_irregular'))],
        ["QRS Duration", f"{features.get('qrs_duration_ms')} ms"],
        ["PR Interval", f"{features.get('pr_interval_ms')} ms"],
        ["Signal Quality", str(features.get('signal_quality'))],
    ]
    t2 = Table(feature_data, colWidths=[2.5 * inch, 4 * inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), HexColor('#E8F0FE')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [HexColor('#F8FAFF'), HexColor('#FFFFFF')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(t2)
    elements.append(Spacer(1, 0.2 * inch))

    # ── SUMMARIES ─────────────────────────────────────────────────────────────
    elements.append(Paragraph("Clinical Summary (For Doctor)", section_style))
    elements.append(Paragraph(summaries.get("clinical_summary", ""), body_style))
    elements.append(Spacer(1, 0.1 * inch))

    elements.append(Paragraph("Patient Summary (Plain English)", section_style))
    elements.append(Paragraph(summaries.get("patient_summary", ""), body_style))
    elements.append(Spacer(1, 0.1 * inch))

    elements.append(Paragraph("Suggested Action", section_style))
    elements.append(Paragraph(summaries.get("suggested_action", ""), body_style))

    doc.build(elements)
