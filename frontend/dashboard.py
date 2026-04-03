import json
import os
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import wfdb

# ── CONFIG ────────────────────────────────────────────────────────────────────
API_URL = "https://ecg-ai-4xji.onrender.com"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEEDBACK_FILE = os.path.join(ROOT_DIR, "feedback_dataset.csv")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")

st.set_page_config(
    page_title="ECG Intelligence System",
    page_icon="ECG",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
for k, v in [("signal", None), ("result", None), ("feedback_done", False), ("dark_mode", True)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── THEME VARIABLES ───────────────────────────────────────────────────────────
if st.session_state["dark_mode"]:
    T = {
        "app_bg":        "#030712",
        "sidebar_bg":    "#050D1A",
        "sidebar_border": "#0F2236",
        "navbar_bg":     "#050D1A",
        "navbar_border": "#0F2236",
        "card_bg":       "#070F1F",
        "card_border":   "#0F2236",
        "card_hover":    "#0A1628",
        "input_bg":      "#070F1F",
        "input_border":  "#0F2236",
        "text_primary":  "#F1F5F9",
        "text_secondary": "#64748B",
        "text_muted":    "#1E3A5F",
        "text_sidebar":  "#94A3B8",
        "divider":       "#0F2236",
        "plot_grid":     "#0A1628",
        "plot_text":     "#334155",
        "waveform_color": "#00E87A",
        "tab_bg":        "#050D1A",
        "tab_border":    "#0F2236",
        "tab_text":      "#334155",
        "tab_active":    "#38BDF8",
        "measure_bg":    "#070F1F",
        "measure_border": "#0F2236",
        "risk_bg":       "#0F0500",
        "risk_border":   "#431407",
        "risk_text":     "#FED7AA",
        "safe_bg":       "#011F16",
        "safe_border":   "#064E3B",
        "safe_text":     "#A7F3D0",
        "stat_bg":       "#070F1F",
        "stat_border":   "#0F2236",
        "feedback_bg":   "#070F1F",
        "pipe_bg":       "#070F1F",
        "pipe_border":   "#0F2236",
        "empty_bg":      "#070F1F",
        "empty_border":  "#0F2236",
        "scrollbar_track": "#0F172A",
        "scrollbar_thumb": "#1E3A5F",
        "toggle_icon":   "SUN",
        "toggle_label":  "Light Mode",
        "toggle_bg":     "#1E3A5F",
        "toggle_knob":   "#38BDF8",
    }
else:
    T = {
        "app_bg":        "#F0F4F8",
        "sidebar_bg":    "#FFFFFF",
        "sidebar_border": "#E2E8F0",
        "navbar_bg":     "#FFFFFF",
        "navbar_border": "#E2E8F0",
        "card_bg":       "#FFFFFF",
        "card_border":   "#E2E8F0",
        "card_hover":    "#F8FAFC",
        "input_bg":      "#F8FAFC",
        "input_border":  "#E2E8F0",
        "text_primary":  "#0F172A",
        "text_secondary": "#475569",
        "text_muted":    "#94A3B8",
        "text_sidebar":  "#475569",
        "divider":       "#E2E8F0",
        "plot_grid":     "#F1F5F9",
        "plot_text":     "#94A3B8",
        "waveform_color": "#0EA5E9",
        "tab_bg":        "#FFFFFF",
        "tab_border":    "#E2E8F0",
        "tab_text":      "#94A3B8",
        "tab_active":    "#2563EB",
        "measure_bg":    "#FFFFFF",
        "measure_border": "#E2E8F0",
        "risk_bg":       "#FFF7ED",
        "risk_border":   "#FED7AA",
        "risk_text":     "#9A3412",
        "safe_bg":       "#F0FDF4",
        "safe_border":   "#BBF7D0",
        "safe_text":     "#166534",
        "stat_bg":       "#FFFFFF",
        "stat_border":   "#E2E8F0",
        "feedback_bg":   "#FFFFFF",
        "pipe_bg":       "#F8FAFC",
        "pipe_border":   "#E2E8F0",
        "empty_bg":      "#FFFFFF",
        "empty_border":  "#E2E8F0",
        "scrollbar_track": "#F1F5F9",
        "scrollbar_thumb": "#CBD5E1",
        "toggle_icon":   "MOON",
        "toggle_label":  "Dark Mode",
        "toggle_bg":     "#E2E8F0",
        "toggle_knob":   "#2563EB",
    }

dark = st.session_state["dark_mode"]

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
* {{ font-family: 'Inter', sans-serif; box-sizing: border-box; }}

/* ── RESPONSIVE BASE ── */
html {{ font-size: clamp(12px, 1.8vw, 16px); }}

.stApp {{ background: {T['app_bg']} !important; transition: background 0.3s ease; }}

/* Force Streamlit's own theme tokens to match our theme */
:root {{
    --background-color: {T['app_bg']} !important;
    --secondary-background-color: {T['card_bg']} !important;
    --text-color: {T['text_primary']} !important;
    --font: 'Inter', sans-serif !important;
}}

::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: {T['scrollbar_track']}; }}
::-webkit-scrollbar-thumb {{ background: {T['scrollbar_thumb']}; border-radius: 3px; }}




#MainMenu, footer {{ visibility: hidden; }}
header {{ visibility: visible !important; }}
.stDeployButton {{ display: none; }}

/* ── STREAMLIT NATIVE ELEMENTS — LIGHT/DARK OVERRIDE ── */
/* Sidebar */
section[data-testid="stSidebar"] {{
    background-color: {T['sidebar_bg']} !important;
    border-right: 1px solid {T['sidebar_border']} !important;
}}
/* Sidebar text */
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span,
section[data-testid="stSidebar"] .stMarkdown div,
section[data-testid="stSidebar"] label {{
    color: {T['text_secondary']} !important;
}}
/* Main content area */
.main .block-container {{
    background: {T['app_bg']} !important;
}}
/* Selectbox, text inputs */
div[data-testid="stSelectbox"] > div,
div[data-testid="stTextInput"] > div > div,
div[data-testid="stTextArea"] > div {{
    background: {T['input_bg']} !important;
    border-color: {T['input_border']} !important;
    color: {T['text_primary']} !important;
}}
/* Radio buttons and labels */
div[data-testid="stRadio"] > label,
div[data-testid="stSelectbox"] label,
div[data-testid="stTextArea"] label {{
    color: {T['text_secondary']} !important;
}}
/* Dataframe */
div[data-testid="stDataFrame"] {{
    background: {T['card_bg']} !important;
}}
/* Spinner text */
div[data-testid="stSpinner"] p {{ color: {T['text_secondary']} !important; }}
/* Success / error / warning boxes */
div[data-testid="stAlert"] {{ background: {T['card_bg']} !important; }}

/* ── DARK MODE TOGGLE BUTTON (fixed top-right corner) ── */
.theme-toggle-wrap {{
    position: fixed;
    top: 12px;
    right: 16px;
    z-index: 10000;
}}
.theme-toggle-btn {{
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 6px 14px 6px 8px;
    background: {T['card_bg']};
    border: 1px solid {T['card_border']};
    border-radius: 50px;
    cursor: pointer;
    transition: all 0.25s ease;
    box-shadow: 0 2px 12px rgba(0,0,0,{'0.35' if dark else '0.12'});
    text-decoration: none;
    white-space: nowrap;
}}
.theme-toggle-btn:hover {{
    border-color: {'#1E3A5F' if dark else '#BFDBFE'};
    box-shadow: 0 4px 20px rgba(0,0,0,{'0.5' if dark else '0.18'});
    transform: translateY(-1px);
}}
.toggle-track {{
    width: 34px;
    height: 18px;
    background: {T['toggle_bg']};
    border-radius: 9px;
    position: relative;
    flex-shrink: 0;
    transition: background 0.3s ease;
}}
.toggle-knob {{
    width: 13px;
    height: 13px;
    background: {T['toggle_knob']};
    border-radius: 50%;
    position: absolute;
    top: 2.5px;
    left: {'18px' if dark else '3px'};
    transition: left 0.3s ease;
    box-shadow: 0 1px 4px rgba(0,0,0,0.3);
}}
.toggle-text {{
    font-size: 0.68rem;
    font-weight: 700;
    color: {T['text_secondary']};
    letter-spacing: 0.5px;
}}
.toggle-icon-svg {{
    font-size: 0.85rem;
    line-height: 1;
}}

/* ── NAVBAR ── */
.navbar {{
    background: {T['navbar_bg']};
    border-bottom: 1px solid {T['navbar_border']};
    padding: 0 clamp(12px, 2vw, 32px);
    height: clamp(52px, 7vw, 62px);
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: -1rem -1rem 0 -1rem;
    position: sticky;
    top: 0;
    z-index: 999;
    box-shadow: 0 1px 12px rgba(0,0,0,{'0.3' if dark else '0.06'});
    transition: all 0.3s ease;
}}
.navbar-logo {{
    font-size: clamp(0.85rem, 2vw, 1.2rem);
    font-weight: 900;
    background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}}
.navbar-tagline {{
    font-size: clamp(0.5rem, 1.2vw, 0.65rem);
    color: {T['text_muted']};
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    display: none;
}}
@media (min-width: 600px) {{
    .navbar-tagline {{ display: block; }}
}}
.version-badge {{
    padding: 4px 10px;
    background: {T['card_bg']};
    border: 1px solid {T['card_border']};
    border-radius: 6px;
    font-size: clamp(0.58rem, 1.2vw, 0.7rem);
    color: #38BDF8 !important;
    font-weight: 700;
}}
.status-pill {{
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 5px 10px;
    border-radius: 20px;
    font-size: clamp(0.6rem, 1.3vw, 0.72rem);
    font-weight: 600;
}}
.status-online  {{ background: {'#022C22' if dark else '#F0FDF4'}; border: 1px solid {'#065F46' if dark else '#BBF7D0'}; color: #34D399 !important; }}
.status-offline {{ background: {'#1C0202' if dark else '#FEF2F2'}; border: 1px solid {'#7F1D1D' if dark else '#FECACA'}; color: #F87171 !important; }}
.pulse {{ width: 7px; height: 7px; border-radius: 50%; animation: pulse 2s infinite; flex-shrink: 0; }}
.pulse-green {{ background: #34D399; }}
.pulse-red   {{ background: #F87171; }}
@keyframes pulse {{
    0%   {{ box-shadow: 0 0 0 0 rgba(52,211,153,0.5); }}
    70%  {{ box-shadow: 0 0 0 8px rgba(52,211,153,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(52,211,153,0); }}
}}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {{
    background: {T['tab_bg']};
    border-bottom: 1px solid {T['tab_border']};
    padding: 0 4px;
    gap: 0;
    overflow-x: auto;
    flex-wrap: nowrap;
    transition: all 0.3s ease;
}}
.stTabs [data-baseweb="tab"] {{
    background: rgba(0,0,0,0);
    border: none;
    color: {T['tab_text']};
    font-size: clamp(0.68rem, 1.5vw, 0.82rem);
    font-weight: 600;
    padding: clamp(8px, 2vw, 14px) clamp(10px, 2.5vw, 22px);
    border-bottom: 2px solid transparent;
    transition: all 0.2s ease;
    white-space: nowrap;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: {T['text_secondary']};
    background: {'rgba(30,58,95,0.15)' if dark else 'rgba(37,99,235,0.05)'};
}}
.stTabs [aria-selected="true"] {{
    background: rgba(0,0,0,0) !important;
    color: {T['tab_active']} !important;
    border-bottom: 2px solid {T['tab_active']} !important;
}}
.stTabs [data-baseweb="tab-panel"] {{
    padding: clamp(12px, 3vw, 28px) 0 0 0;
    animation: fadeIn 0.35s ease;
}}
@keyframes fadeIn  {{ from {{ opacity:0; transform:translateY(8px); }} to {{ opacity:1; transform:translateY(0); }} }}
@keyframes slideIn {{ from {{ opacity:0; transform:translateX(-8px); }} to {{ opacity:1; transform:translateX(0); }} }}
@keyframes scaleIn {{ from {{ opacity:0; transform:scale(0.97); }} to {{ opacity:1; transform:scale(1); }} }}

/* ── METRIC CARDS ── */
.metric-card {{
    background: {T['card_bg']};
    border: 1px solid {T['card_border']};
    border-radius: clamp(8px, 1.5vw, 14px);
    padding: clamp(10px, 1.8vw, 18px) clamp(10px, 1.8vw, 18px);
    position: relative;
    overflow: hidden;
    transition: all 0.25s ease;
    animation: scaleIn 0.4s ease;
    box-shadow: 0 2px 8px rgba(0,0,0,{'0.2' if dark else '0.04'});
    min-height: clamp(80px, 12vw, 110px);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}}
.metric-card:hover {{
    border-color: {'#1E3A5F' if dark else '#BFDBFE'};
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,{'0.3' if dark else '0.08'});
}}
.metric-card::before {{
    content:''; position:absolute; top:0; left:0; right:0; height:2px; border-radius:inherit;
}}
.mc-blue::before   {{ background: linear-gradient(90deg, #38BDF8, #818CF8); }}
.mc-green::before  {{ background: linear-gradient(90deg, #34D399, #10B981); }}
.mc-amber::before  {{ background: linear-gradient(90deg, #FBBF24, #F59E0B); }}
.mc-red::before    {{ background: linear-gradient(90deg, #F87171, #EF4444); }}
.mc-purple::before {{ background: linear-gradient(90deg, #C084FC, #A855F7); }}
.mc-label {{
    font-size: clamp(0.48rem, 1.1vw, 0.58rem);
    color: {T['text_muted']};
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 700;
    margin-bottom: 6px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.mc-value {{
    font-size: clamp(0.85rem, 2.2vw, 1.5rem);
    font-weight: 800;
    color: {T['text_primary']};
    line-height: 1.1;
    letter-spacing: -0.5px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    display: block;
    width: 100%;
}}
.mc-sub   {{ font-size: clamp(0.48rem, 1vw, 0.62rem); color:{T['text_muted']}; margin-top:6px; font-weight:500; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.mc-icon  {{ position:absolute; top:10px; right:10px; font-size:clamp(0.8rem, 1.8vw, 1.1rem); opacity:0.15; }}

/* ── DECISION BANNER ── */
.decision-banner {{
    border-radius: clamp(8px, 1.5vw, 14px);
    padding: clamp(14px, 3vw, 24px) clamp(16px, 3vw, 28px);
    margin: 20px 0;
    position: relative;
    overflow: hidden;
    animation: fadeIn 0.5s ease;
}}
.decision-banner::before {{
    content:''; position:absolute; top:0; left:0; bottom:0;
    width:4px; border-radius:inherit;
}}
.db-ACCEPT    {{ background: {'linear-gradient(135deg,#011F16,#022C22)' if dark else 'linear-gradient(135deg,#F0FDF4,#DCFCE7)'}; border:1px solid {'#064E3B' if dark else '#BBF7D0'}; }}
.db-ACCEPT::before    {{ background:#34D399; }}
.db-RECHECK   {{ background: {'linear-gradient(135deg,#1A1100,#221700)' if dark else 'linear-gradient(135deg,#FFFBEB,#FEF3C7)'}; border:1px solid {'#78350F' if dark else '#FDE68A'}; }}
.db-RECHECK::before   {{ background:#FBBF24; }}
.db-ALERT     {{ background: {'linear-gradient(135deg,#1A0A00,#220E00)' if dark else 'linear-gradient(135deg,#FFF7ED,#FFEDD5)'}; border:1px solid {'#7C2D12' if dark else '#FED7AA'}; }}
.db-ALERT::before     {{ background:#FB923C; }}
.db-EMERGENCY {{ background: {'linear-gradient(135deg,#1A0202,#220303)' if dark else 'linear-gradient(135deg,#FEF2F2,#FEE2E2)'}; border:1px solid {'#7F1D1D' if dark else '#FECACA'}; }}
.db-EMERGENCY::before {{ background:#F87171; }}
.db-REJECT    {{ background: {'linear-gradient(135deg,#0A0F1A,#0F172A)' if dark else 'linear-gradient(135deg,#F8FAFC,#F1F5F9)'}; border:1px solid {'#1E293B' if dark else '#E2E8F0'}; }}
.db-REJECT::before    {{ background:#64748B; }}
.db-label  {{ font-size:clamp(0.52rem, 1.2vw, 0.62rem); color:{T['text_muted']}; text-transform:uppercase; letter-spacing:2px; font-weight:700; margin-bottom:8px; }}
.db-title  {{ font-size:clamp(1.1rem, 4vw, 1.8rem); font-weight:900; letter-spacing:-1px; margin-bottom:10px; line-height:1; }}
.db-reason {{ font-size:clamp(0.75rem, 1.8vw, 0.88rem); line-height:1.75; max-width:800px; opacity:0.85; }}
.db-badges {{ margin-top:16px; display:flex; gap:6px; flex-wrap:wrap; }}
.db-badge  {{
    padding:4px 10px; border-radius:20px; font-size:clamp(0.6rem, 1.3vw, 0.7rem); font-weight:600;
    background:{'rgba(255,255,255,0.06)' if dark else 'rgba(0,0,0,0.05)'};
    border:1px solid {'rgba(255,255,255,0.1)' if dark else 'rgba(0,0,0,0.08)'};
}}

/* ── SECTION TITLES ── */
.section-title {{
    font-size: clamp(0.52rem, 1.2vw, 0.62rem);
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    color: {T['text_muted']};
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid {T['divider']};
    display: flex;
    align-items: center;
    gap: 8px;
}}
.section-title span {{ color: {'#38BDF8' if dark else '#2563EB'}; }}

/* ── MEASURE ROWS ── */
.measure-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: clamp(8px, 1.5vw, 11px) clamp(10px, 2vw, 16px);
    background: {T['measure_bg']};
    border: 1px solid {T['measure_border']};
    border-radius: 10px;
    margin-bottom: 6px;
    transition: all 0.2s ease;
    animation: slideIn 0.3s ease;
    box-shadow: 0 1px 3px rgba(0,0,0,{'0.15' if dark else '0.03'});
    flex-wrap: wrap;
    gap: 4px;
}}
.measure-row:hover {{
    background: {T['card_hover']};
    border-color: {'#1E3A5F' if dark else '#BFDBFE'};
}}
.m-label   {{ font-size: clamp(0.72rem, 1.5vw, 0.82rem); color:{T['text_muted']}; font-weight:500; }}
.m-value   {{ font-size: clamp(0.75rem, 1.5vw, 0.88rem); color:{T['text_primary']}; font-weight:700; display:flex; align-items:center; gap:6px; }}
.m-normal  {{ color:#34D399; }}
.m-abnormal{{ color:#F87171; }}
.m-na      {{ color:{T['text_muted']}; }}
.indicator {{ width:7px; height:7px; border-radius:50%; flex-shrink:0; }}
.ind-green {{ background:#34D399; box-shadow:0 0 6px rgba(52,211,153,0.5); }}
.ind-red   {{ background:#F87171; box-shadow:0 0 6px rgba(248,113,113,0.5); }}
.ind-amber {{ background:#FBBF24; box-shadow:0 0 6px rgba(251,191,36,0.5); }}
.ind-grey  {{ background:{T['text_muted']}; }}

/* ── RISK / SAFE ── */
.risk-item {{
    padding: clamp(9px, 1.5vw, 12px) clamp(10px, 2vw, 16px);
    background: {T['risk_bg']}; border: 1px solid {T['risk_border']};
    border-left: 3px solid #EA580C; border-radius: 10px; margin-bottom: 8px;
    font-size: clamp(0.72rem, 1.5vw, 0.83rem); color:{T['risk_text']}; line-height:1.6;
    animation: slideIn 0.3s ease;
}}
.safe-item {{
    padding: clamp(9px, 1.5vw, 12px) clamp(10px, 2vw, 16px);
    background: {T['safe_bg']}; border: 1px solid {T['safe_border']};
    border-left: 3px solid #34D399; border-radius: 10px;
    font-size: clamp(0.72rem, 1.5vw, 0.83rem); color:{T['safe_text']};
}}

/* ── SUMMARY CARDS ── */
.summary-card {{
    background: {T['card_bg']}; border: 1px solid {T['card_border']};
    border-radius: clamp(8px, 1.5vw, 14px);
    padding: clamp(14px, 2vw, 20px) clamp(14px, 2vw, 22px);
    margin-bottom: 14px;
    transition: all 0.2s ease; animation: fadeIn 0.4s ease;
    box-shadow: 0 2px 6px rgba(0,0,0,{'0.15' if dark else '0.04'});
}}
.summary-card:hover {{
    border-color: {'#1E3A5F' if dark else '#BFDBFE'};
    box-shadow: 0 6px 20px rgba(0,0,0,{'0.2' if dark else '0.08'});
}}
.s-tag {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 6px;
    font-size: clamp(0.55rem, 1.2vw, 0.62rem);
    font-weight: 800; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 14px;
}}
.s-doctor  {{ background:{'#0F1F3A' if dark else '#EFF6FF'}; color:#38BDF8; border:1px solid {'#1E3A5F' if dark else '#BFDBFE'}; }}
.s-patient {{ background:{'#011F16' if dark else '#F0FDF4'}; color:#34D399; border:1px solid {'#064E3B' if dark else '#BBF7D0'}; }}
.s-action  {{ background:{'#1A1100' if dark else '#FFFBEB'}; color:#FBBF24; border:1px solid {'#78350F' if dark else '#FDE68A'}; }}
.s-text    {{ font-size: clamp(0.78rem, 1.8vw, 0.875rem); color:{T['text_secondary']}; line-height:1.8; }}

/* ── WAVEFORM ── */
.waveform-wrap {{
    background: {T['card_bg']}; border: 1px solid {T['card_border']};
    border-radius: clamp(10px, 1.5vw, 16px);
    padding: 4px; margin-bottom: 20px; overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,{'0.2' if dark else '0.04'});
}}

/* ── STAT BOXES ── */
.stat-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: clamp(8px, 2vw, 14px);
    margin-bottom: 20px;
}}
.stat-box {{
    background: {T['stat_bg']}; border: 1px solid {T['stat_border']};
    border-radius: clamp(8px, 1.5vw, 12px);
    padding: clamp(12px, 2vw, 18px); text-align: center;
    transition: all 0.2s ease;
    box-shadow: 0 1px 4px rgba(0,0,0,{'0.15' if dark else '0.03'});
}}
.stat-box:hover {{ border-color: {'#1E3A5F' if dark else '#BFDBFE'}; }}
.stat-num   {{ font-size: clamp(1.4rem, 4vw, 2rem); font-weight: 900; letter-spacing: -1px; }}
.stat-label {{ font-size: clamp(0.55rem, 1.2vw, 0.62rem); color:{T['text_muted']}; text-transform:uppercase; letter-spacing:1.5px; margin-top:4px; }}

/* ── BUTTONS ── */
.stButton > button {{
    background: linear-gradient(135deg, #1D4ED8, #2563EB) !important;
    color: #FFFFFF !important; border: none !important;
    border-radius: 10px !important;
    padding: clamp(8px, 1.5vw, 11px) clamp(16px, 3vw, 28px) !important;
    font-weight: 700 !important;
    font-size: clamp(0.75rem, 1.6vw, 0.875rem) !important;
    width: 100% !important; transition: all 0.2s ease !important;
}}
.stButton > button:hover {{
    background: linear-gradient(135deg, #2563EB, #3B82F6) !important;
    box-shadow: 0 4px 20px rgba(37,99,235,0.35) !important;
    transform: translateY(-1px) !important;
}}
.stDownloadButton > button {{
    background: linear-gradient(135deg, #064E3B, #065F46) !important;
    color: #FFFFFF !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important; width: 100% !important;
}}
.stDownloadButton > button:hover {{
    background: linear-gradient(135deg, #065F46, #047857) !important;
    box-shadow: 0 4px 16px rgba(6,79,58,0.35) !important;
}}

/* ── FORM ELEMENTS ── */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stTextArea"] > div > textarea {{
    background: {T['input_bg']} !important;
    border: 1px solid {T['input_border']} !important;
    color: {T['text_primary']} !important;
    border-radius: 10px !important;
}}
div[data-testid="stSelectbox"] label,
div[data-testid="stTextArea"] label,
div[data-testid="stRadio"] label,
div[data-testid="stRadio"] p {{
    color: {T['text_muted']} !important;
    font-size: clamp(0.68rem, 1.4vw, 0.75rem) !important;
    font-weight: 600 !important;
}}

/* ── PROGRESS BAR ── */
.stProgress > div > div > div {{
    background: linear-gradient(90deg, #38BDF8, #818CF8) !important;
    border-radius: 4px !important;
}}

/* ── PIPELINE STRIP ── */
.pipeline-strip-outer {{
    background: {T['pipe_bg']};
    border-bottom: 1px solid {T['divider']};
    margin: 0 -1rem 0 -1rem;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
}}
.pipeline-flow {{
    display: flex;
    align-items: center;
    gap: 0;
    padding: clamp(8px, 1.5vw, 10px) clamp(12px, 2vw, 24px);
    min-width: max-content;
}}
.pipe-box {{
    padding: clamp(5px, 1vw, 7px) clamp(8px, 1.5vw, 13px);
    background: {T['pipe_bg']}; border: 1px solid {T['pipe_border']};
    border-radius: 8px;
    font-size: clamp(0.6rem, 1.3vw, 0.72rem);
    color: {'#38BDF8' if dark else '#2563EB'};
    font-weight: 600; white-space: nowrap;
    transition: all 0.2s;
}}
.pipe-box:hover {{ border-color: {'#1E3A5F' if dark else '#BFDBFE'}; }}
.pipe-arrow {{ color: {T['text_muted']}; font-size: 0.9rem; padding: 0 4px; }}

/* ── EMPTY STATES ── */
.empty-state {{
    text-align: center;
    padding: clamp(30px, 8vw, 70px) 20px;
    background: {T['empty_bg']}; border: 2px dashed {T['empty_border']};
    border-radius: clamp(12px, 2vw, 20px); animation: fadeIn 0.5s ease;
}}
.empty-icon  {{ font-size: clamp(1.8rem, 5vw, 3.5rem); margin-bottom: 16px; }}
.empty-title {{ font-size: clamp(0.85rem, 2vw, 1.05rem); font-weight:700; color:{T['text_muted']}; margin-bottom:8px; }}
.empty-sub   {{ font-size: clamp(0.7rem, 1.5vw, 0.83rem); color:{T['text_muted']}; opacity:0.6; }}

/* ── FEEDBACK ── */
.feedback-card {{
    background: {T['feedback_bg']}; border: 1px solid {T['card_border']};
    border-radius: clamp(10px, 1.5vw, 16px);
    padding: clamp(16px, 3vw, 26px);
    box-shadow: 0 2px 8px rgba(0,0,0,{'0.15' if dark else '0.04'});
}}
.feedback-success {{
    padding: clamp(12px, 2vw, 18px) clamp(14px, 2vw, 22px);
    background: {'#011F16' if dark else '#F0FDF4'};
    border: 1px solid {'#065F46' if dark else '#BBF7D0'};
    border-radius: 12px; color: #34D399; font-weight: 600;
    font-size: clamp(0.78rem, 1.8vw, 0.9rem);
}}

/* ── DIVIDER ── */
.divider {{
    height: 1px;
    background: linear-gradient(90deg, rgba(0,0,0,0), {T['divider']}, rgba(0,0,0,0));
    margin: clamp(14px, 3vw, 24px) 0;
}}
hr {{ border-color: {T['divider']} !important; }}
h1,h2,h3,h4,h5 {{ color: {T['text_primary']} !important; }}
p {{ color: {T['text_secondary']}; }}

/* ── ST.TOGGLE THEME STYLING ── */
div[data-testid="stSidebar"] div[data-testid="stToggle"] {{
    background: transparent !important;
    padding: 2px 0 10px 0;
}}
div[data-testid="stSidebar"] div[data-testid="stToggle"] label {{
    color: {T['text_sidebar']} !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px;
}}
/* Track */
div[data-testid="stSidebar"] div[data-testid="stToggle"] span[data-testid="stWidgetLabel"] + div > div {{
    background: {T['toggle_bg']} !important;
}}

/* ── RESPONSIVE COLUMNS ── */
@media (max-width: 640px) {{
    /* Stack columns on very small screens */
    div[data-testid="column"] {{
        min-width: 100% !important;
    }}
    .stat-grid {{
        grid-template-columns: repeat(3, 1fr);
    }}
}}

/* ── REPORT HEADER RESPONSIVE ── */
.report-header {{
    background: {T['card_bg']}; border: 1px solid {T['card_border']};
    border-radius: clamp(8px, 1.5vw, 14px);
    padding: clamp(14px, 2vw, 20px) clamp(16px, 2vw, 26px);
    margin-bottom: 22px;
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,{'0.15' if dark else '0.04'});
}}
.report-header-item {{
    text-align: center;
    min-width: 80px;
}}
.report-header-label {{
    font-size: clamp(0.52rem, 1.2vw, 0.6rem);
    color: {T['text_muted']};
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 700;
    margin-bottom: 6px;
}}
.report-header-value {{
    font-size: clamp(0.9rem, 2.5vw, 1.1rem);
    font-weight: 800;
    color: {T['text_primary']};
}}

/* ── MEASUREMENTS GRID (2-col on small screens) ── */
.measurements-mini-grid {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: clamp(6px, 1.5vw, 10px);
    margin-top: 4px;
}}
.mini-metric {{
    background: {T['app_bg']}; border: 1px solid {T['divider']};
    border-radius: 8px; padding: clamp(8px, 1.5vw, 12px); text-align: center;
}}
.mini-metric-val {{ font-size: clamp(0.95rem, 2.5vw, 1.2rem); font-weight: 800; }}
.mini-metric-lbl {{ font-size: clamp(0.52rem, 1.1vw, 0.6rem); color: {T['text_muted']}; margin-top: 3px; font-weight: 700; letter-spacing: 1px; }}
</style>
""", unsafe_allow_html=True)


# ── HELPERS ───────────────────────────────────────────────────────────────────
def call_api(signal):
    try:
        r = requests.post(f"{API_URL}/predict", json={"ecg_signal": signal}, timeout=60)
        return (r.json(), None) if r.status_code == 200 else (None, f"API {r.status_code}")
    except Exception as e:
        return None, str(e)

def decision_color(d):
    return {"ACCEPT": "#34D399", "RECHECK": "#FBBF24", "ALERT": "#FB923C",
            "EMERGENCY": "#F87171", "REJECT": "#64748B"}.get(d, "#64748B")

def severity_color(s):
    return {"LOW": "#34D399", "MODERATE": "#FBBF24", "HIGH": "#FB923C",
            "CRITICAL": "#F87171"}.get(s, "#64748B")

def resolve_pdf(report):
    if not report: return None
    rid = report.get('report_id', '')
    for candidate in [
        report.get('pdf_path', ''),
        os.path.join(REPORTS_DIR, f"ECG_Report_{rid}.pdf"),
        os.path.join(ROOT_DIR, report.get('pdf_path', '')),
    ]:
        if candidate and os.path.isabs(candidate) and os.path.exists(candidate): return candidate
        if candidate and os.path.exists(candidate): return candidate
    return None

def save_feedback(result, doctor_label, override, reason, notes):
    record = {
        "timestamp": result["agent_decision"].get("timestamp", datetime.now().isoformat()),
        "ai_condition": result["condition"], "ai_confidence": result["confidence"],
        "risk_score": result["risk_assessment"]["risk_score"],
        "severity": result["risk_assessment"]["severity"],
        "agent_decision": result["agent_decision"]["decision"],
        "doctor_label": doctor_label, "override": override,
        "reason": reason, "notes": notes,
        "report_id": result.get("report", {}).get("report_id", "N/A")
    }
    df_new = pd.DataFrame([record])
    df = pd.concat([pd.read_csv(FEEDBACK_FILE), df_new], ignore_index=True) if os.path.exists(FEEDBACK_FILE) else df_new
    df.to_csv(FEEDBACK_FILE, index=False)

def ind(color):
    cls = {"#34D399": "ind-green", "#F87171": "ind-red", "#FBBF24": "ind-amber"}.get(color, "ind-grey")
    return f'<div class="indicator {cls}"></div>'

def measure_row(label, value, unit="", normal_range=None):
    if value is None or (isinstance(value, float) and value != value):
        return f'<div class="measure-row"><span class="m-label">{label}</span><span class="m-value m-na">{ind("#999")} N/A</span></div>'
    display = f"{value} {unit}".strip()
    if normal_range:
        lo, hi = normal_range
        c   = "#34D399" if lo <= value <= hi else "#F87171"
        cls = "m-normal" if lo <= value <= hi else "m-abnormal"
    else:
        c, cls = "#34D399", "m-normal"
    return f'<div class="measure-row"><span class="m-label">{label}</span><span class="m-value {cls}">{ind(c)} {display}</span></div>'

def api_status():
    try:
        h = requests.get(f"{API_URL}/health", timeout=2).json()
        return True, h
    except:
        return False, {}

def plot_layout(height=240, l=50, r=20, t=10, b=45, show_legend=False):
    """
    FIX: showlegend is now a parameter here, NOT duplicated in update_layout calls.
    Always pass showlegend through this function to avoid the duplicate keyword error.
    """
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=height,
        margin=dict(l=l, r=r, t=t, b=b),
        showlegend=show_legend,
        font=dict(family='Inter', size=11, color=T['plot_text'])
    )

def axis_style(title=''):
    return dict(title=title, gridcolor=T['plot_grid'], color=T['plot_text'], zeroline=False)


# ── DARK MODE TOGGLE — placed cleanly in sidebar (see sidebar section below)
# The toggle lives in the sidebar using st.toggle which natively works.
# We also show a small fixed pill in the top-right as a mode indicator only (no button).
mode_label = "Dark" if dark else "Light"
mode_icon_svg = (
    """<svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>"""
    if dark else
    """<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>"""
)
st.markdown(f"""
<div style="position:fixed; top:14px; right:16px; z-index:10000;
            display:flex; align-items:center; gap:6px;
            padding:5px 11px; border-radius:20px;
            background:{T['card_bg']}; border:1px solid {T['card_border']};
            box-shadow:0 2px 10px rgba(0,0,0,{'0.3' if dark else '0.1'});
            color:{T['text_secondary']}; font-size:0.65rem; font-weight:700;
            pointer-events:none; user-select:none;">
    <span style="color:{'#38BDF8' if dark else '#2563EB'}; display:flex; align-items:center;">{mode_icon_svg}</span>
    {mode_label}
</div>
""", unsafe_allow_html=True)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:clamp(14px,3vw,20px) 16px 12px; text-align:center;">
        <div style="margin-bottom:10px;">
            <svg width="38" height="38" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 21.593c-5.63-5.539-11-10.297-11-14.402C1 3.509 3.353 1 6.5 1c1.998 0 3.5 1 4.5 2 1-1 2.502-2 4.5-2C18.647 1 21 3.509 21 7.191c0 4.105-5.37 8.863-11 14.402z"
                      fill="#F87171" opacity="0.9"/>
                <polyline points="6,12 9,8 11,14 13,10 16,12"
                          stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
            </svg>
        </div>
        <div style="font-size:clamp(0.78rem,2vw,0.92rem); font-weight:900;
                    background:linear-gradient(135deg,#38BDF8,#818CF8,#C084FC);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent; letter-spacing:-0.3px;">
            ECG Intelligence
        </div>
        <div style="font-size:clamp(0.52rem,1.2vw,0.6rem); color:{T['text_muted']}; margin-top:4px; letter-spacing:2px; font-weight:700;">
            ADAPTIVE AGENTIC AI
        </div>
    </div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    # ── REAL WORKING THEME TOGGLE ──────────────────────────────────────────
    # st.toggle is the proper Streamlit native toggle — it actually works and triggers rerun.
    new_dark = st.toggle(
        "Dark Mode",
        value=st.session_state["dark_mode"],
        help="Switch between dark and light theme"
    )
    if new_dark != st.session_state["dark_mode"]:
        st.session_state["dark_mode"] = new_dark
        st.rerun()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # API Status
    online, health = api_status()
    if online:
        st.markdown(f"""
        <div style="margin:0 0 16px; padding:10px 14px;
                    background:{'#011F16' if dark else '#F0FDF4'};
                    border:1px solid {'#064E3B' if dark else '#BBF7D0'};
                    border-radius:10px; display:flex; align-items:center; gap:10px;">
            <div class="pulse pulse-green"></div>
            <div>
                <div style="color:#34D399; font-weight:700; font-size:clamp(0.68rem,1.5vw,0.78rem);">API Online</div>
                <div style="color:{'#064E3B' if dark else '#86EFAC'}; font-size:clamp(0.58rem,1.2vw,0.65rem);">{health.get('model_version','v1.0.0')} · {', '.join(health.get('classes',[]))}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="margin:0 0 16px; padding:10px 14px;
                    background:{'#1A0202' if dark else '#FEF2F2'};
                    border:1px solid {'#7F1D1D' if dark else '#FECACA'};
                    border-radius:10px;">
            <div class="pulse pulse-red" style="display:inline-block; margin-right:8px;"></div>
            <span style="color:#F87171; font-weight:700; font-size:clamp(0.68rem,1.5vw,0.78rem);">API Offline</span>
            <div style="color:{'#7F1D1D' if dark else '#FCA5A5'}; font-size:clamp(0.58rem,1.2vw,0.65rem); margin-top:3px;">Run: python api.py</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f'<div style="font-size:0.6rem; color:{T["text_muted"]}; font-weight:700; letter-spacing:2px; text-transform:uppercase; margin-bottom:10px;">Signal Source</div>', unsafe_allow_html=True)
    source = st.radio("", ["Raspberry Pi Live", "MIT-BIH Database", "Upload JSON"], label_visibility="collapsed")

    if source == "Raspberry Pi Live":
        # Info box
        st.markdown(f"""
        <div style="background:{'#0F1F3A' if dark else '#EFF6FF'}; border:1px solid {'#1E3A5F' if dark else '#BFDBFE'};
                    border-radius:10px; padding:11px 13px; margin-bottom:12px; font-size:0.68rem; line-height:1.7; color:{T['text_muted']};">
            <span style="color:#38BDF8; font-weight:700;">Step 1:</span> Run <b>ecg_streamer.py</b> on your Pi<br>
            <span style="color:#38BDF8; font-weight:700;">Step 2:</span> Run <b>ecg_receiver.py</b> on this PC<br>
            <span style="color:#38BDF8; font-weight:700;">Step 3:</span> Click <b>Fetch Signal</b> below
        </div>
        """, unsafe_allow_html=True)

        receiver_url = st.text_input(
            "Receiver URL",
            value="http://localhost:8765",
            help="URL where ecg_receiver.py is running on this PC"
        )

        # Check receiver status
        try:
            import requests as _req
            r = _req.get(f"{receiver_url}/status", timeout=2)
            info = r.json()
            connected_pi = info.get("connected", False)
            ready        = info.get("ready", False)
            buf_len      = info.get("buffer_size", 0)
            pi_mode      = info.get("mode", "unknown")
            pi_sr        = info.get("sample_rate", 360)

            if connected_pi:
                st.markdown(f"""
                <div style="padding:8px 12px; border-radius:8px;
                            background:{'#011F16' if dark else '#F0FDF4'};
                            border:1px solid {'#064E3B' if dark else '#BBF7D0'};
                            font-size:0.7rem; color:#34D399; font-weight:600; margin-bottom:8px;">
                    Pi Connected · {pi_mode} · {pi_sr}Hz · {buf_len} samples buffered
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="padding:8px 12px; border-radius:8px;
                            background:{'#1A0202' if dark else '#FEF2F2'};
                            border:1px solid {'#7F1D1D' if dark else '#FECACA'};
                            font-size:0.7rem; color:#F87171; font-weight:600; margin-bottom:8px;">
                    Receiver running but Pi not connected yet
                </div>
                """, unsafe_allow_html=True)
        except Exception:
            st.markdown(f"""
            <div style="padding:8px 12px; border-radius:8px;
                        background:{'#1A0202' if dark else '#FEF2F2'};
                        border:1px solid {'#7F1D1D' if dark else '#FECACA'};
                        font-size:0.7rem; color:#F87171; font-weight:600; margin-bottom:8px;">
                Receiver offline — run ecg_receiver.py on this PC
            </div>
            """, unsafe_allow_html=True)
            connected_pi = False
            ready        = False

        if st.button("Fetch Signal (15s)", disabled=not ready):
            with st.spinner("Fetching 5400 samples from Pi (15 s window)..."):
                try:
                    import requests as _req
                    r = _req.get(f"{receiver_url}/signal?n=5400", timeout=20)
                    data = r.json()
                    if "error" in data:
                        st.error(data["error"])
                    else:
                        st.session_state["signal"]        = data["ecg_signal"]
                        st.session_state["result"]        = None
                        st.session_state["feedback_done"] = False
                        st.success(f"{len(data['ecg_signal'])} samples loaded from Pi ({data.get('mode','hardware')})")
                        st.rerun()
                except Exception as e:
                    st.error(f"Fetch failed: {e}")

    elif source == "MIT-BIH Database":
        record_id = st.selectbox("Record", ["100", "101", "103", "105", "106", "107", "108"])
        if st.button("Load ECG Signal"):
            with st.spinner("Fetching from PhysioNet..."):
                try:
                    rec = wfdb.rdrecord(record_id, pn_dir='mitdb')
                    st.session_state["signal"] = rec.p_signal[:, 0][:5400].tolist()
                    st.session_state["result"] = None
                    st.session_state["feedback_done"] = False
                    st.success(f"Record {record_id} loaded — 5400 samples")
                except Exception as e:
                    st.error(str(e))

    else:
        f = st.file_uploader("", type=["json"], label_visibility="collapsed")
        if f:
            data = json.load(f)
            st.session_state["signal"] = data.get("ecg_signal", [])
            st.session_state["result"] = None
            st.session_state["feedback_done"] = False
            st.success(f"{len(st.session_state['signal'])} samples loaded")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.6rem; color:{T["text_muted"]}; font-weight:700; letter-spacing:2px; text-transform:uppercase; margin-bottom:12px;">AI Pipeline</div>', unsafe_allow_html=True)

    for code, name, tech, color in [
        ("P1", "ECG Classifier",      "1D CNN · MIT-BIH",        "#38BDF8"),
        ("P2", "Feature Extraction",  "NeuroKit2 · 360Hz",        "#34D399"),
        ("P3", "Risk Engine",         "Rule-Based Scoring",       "#FBBF24"),
        ("P4", "ACDA Agent",          "Autonomous Decision",      "#C084FC"),
        ("P5", "Report Generator",    "Groq · Llama 3.3 70B",    "#FB923C"),
        ("P6", "Feedback Loop",       "Doctor Corrections",       "#F472B6"),
    ]:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:10px; padding:8px 0;
                    border-bottom:1px solid {T['divider']};">
            <div style="width:26px; height:26px; background:{T['card_bg']}; border:1px solid {T['card_border']};
                        border-radius:7px; display:flex; align-items:center; justify-content:center;
                        font-size:0.58rem; font-weight:800; color:{color}; flex-shrink:0;">{code}</div>
            <div style="flex:1; min-width:0;">
                <div style="color:{T['text_sidebar']}; font-size:clamp(0.68rem,1.5vw,0.77rem); font-weight:600;">{name}</div>
                <div style="color:{T['text_muted']}; font-size:clamp(0.55rem,1.2vw,0.63rem);">{tech}</div>
            </div>
            <div style="color:{color}; font-size:0.7rem; font-weight:700;">OK</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="divider"></div>
    <div style="text-align:center; color:{T['text_muted']}; font-size:clamp(0.55rem,1.2vw,0.63rem); padding-bottom:16px; line-height:1.8;">
        ECG Intelligence System v1.0<br>Hackathon 2026 · MIT-BIH · 360Hz
    </div>
    """, unsafe_allow_html=True)


# ── NAVBAR ────────────────────────────────────────────────────────────────────
online, _ = api_status()
status_class = "status-online" if online else "status-offline"
status_text  = "System Online" if online else "API Offline"
pulse_class  = "pulse-green" if online else "pulse-red"

# SVG heart icon for navbar (no emoji)
heart_svg_nav = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M12 21.593c-5.63-5.539-11-10.297-11-14.402C1 3.509 3.353 1 6.5 1c1.998 0 3.5 1 4.5 2 1-1 2.502-2 4.5-2C18.647 1 21 3.509 21 7.191c0 4.105-5.37 8.863-11 14.402z" fill="#F87171" opacity="0.85"/>
  <polyline points="7,12 9.5,8 11.5,14 13.5,10 16,12" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
</svg>"""

st.markdown(f"""
<div class="navbar">
    <div style="display:flex; align-items:center; gap:10px;">
        <!-- Hamburger: always visible, opens sidebar when collapsed -->
        <button id="sidebar-open-btn" onclick="toggleSidebar()" title="Toggle sidebar"
            style="background:{T['card_bg']}; border:1px solid {T['card_border']};
                   border-radius:8px; padding:7px 9px; cursor:pointer;
                   display:flex; flex-direction:column; gap:4px; flex-shrink:0;
                   transition:all 0.2s ease;">
            <span style="display:block;width:16px;height:2px;background:{'#38BDF8' if dark else '#2563EB'};border-radius:2px;transition:all 0.2s;"></span>
            <span style="display:block;width:12px;height:2px;background:{'#38BDF8' if dark else '#2563EB'};border-radius:2px;transition:all 0.2s;"></span>
            <span style="display:block;width:16px;height:2px;background:{'#38BDF8' if dark else '#2563EB'};border-radius:2px;transition:all 0.2s;"></span>
        </button>
        {heart_svg_nav}
        <div>
            <div class="navbar-logo">ECG Intelligence System</div>
            <div class="navbar-tagline">Adaptive Agentic Cardiac AI · Hackathon 2026</div>
        </div>
    </div>
    <div style="display:flex; align-items:center; gap:8px;">
        <div class="status-pill {status_class}">
            <div class="pulse {pulse_class}"></div>
            {status_text}
        </div>
        <div class="version-badge">v1.0.0</div>
    </div>
</div>
<script>
function toggleSidebar() {{
    // Streamlit renders its own collapse/expand button — we just click it
    const btn = document.querySelector('[data-testid="collapsedControl"]') ||
                document.querySelector('button[aria-label="Open sidebar"]') ||
                document.querySelector('button[aria-label="Close sidebar"]') ||
                document.querySelector('[data-testid="stSidebarCollapseButton"] button');
    if (btn) {{ btn.click(); return; }}
    // Fallback: look for any button inside the sidebar toggle wrapper
    const all = document.querySelectorAll('button');
    for (const b of all) {{
        const label = (b.getAttribute('aria-label') || '').toLowerCase();
        if (label.includes('sidebar') || label.includes('collapse') || label.includes('open')) {{
            b.click(); return;
        }}
    }}
}}
// Style the hamburger on hover
document.getElementById('sidebar-open-btn').addEventListener('mouseover', function() {{
    this.style.borderColor = '{'#1E3A5F' if dark else '#BFDBFE'}';
    this.style.background  = '{'#0A1628' if dark else '#F0F9FF'}';
}});
document.getElementById('sidebar-open-btn').addEventListener('mouseout', function() {{
    this.style.borderColor = '{'#0F2236' if dark else '#E2E8F0'}';
    this.style.background  = '{'#070F1F' if dark else '#FFFFFF'}';
}});
</script>
""", unsafe_allow_html=True)

# Pipeline strip
st.markdown(f"""
<div class="pipeline-strip-outer">
    <div class="pipeline-flow">
        <div class="pipe-box">Signal Input</div><div class="pipe-arrow">&#8594;</div>
        <div class="pipe-box">1D CNN Classify</div><div class="pipe-arrow">&#8594;</div>
        <div class="pipe-box">Feature Extract</div><div class="pipe-arrow">&#8594;</div>
        <div class="pipe-box">Risk Score</div><div class="pipe-arrow">&#8594;</div>
        <div class="pipe-box">ACDA Decide</div><div class="pipe-arrow">&#8594;</div>
        <div class="pipe-box">LLM Report</div><div class="pipe-arrow">&#8594;</div>
        <div class="pipe-box">Doctor Review</div><div class="pipe-arrow">&#8594;</div>
        <div class="pipe-box">Retrain</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Analysis",
    "Reports",
    "Feedback",
    "History"
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    if st.session_state["signal"]:
        signal = st.session_state["signal"]

        st.markdown('<div class="section-title"><span>~</span> ECG WAVEFORM</div>', unsafe_allow_html=True)
        st.markdown('<div class="waveform-wrap">', unsafe_allow_html=True)
        fig = go.Figure()
        sr       = 360                                    # samples per second
        n_samples = len(signal)
        time_axis = [i / sr for i in range(n_samples)]   # seconds on x-axis

        fig.add_trace(go.Scatter(
            x=time_axis,
            y=signal,
            mode='lines',
            line=dict(color=T['waveform_color'], width=1.1),
            fill='tozeroy',
            fillcolor=f"rgba({'0,232,122' if dark else '14,165,233'},0.04)",
            hovertemplate='<b>Time:</b> %{x:.3f} s<br><b>Amplitude:</b> %{y:.4f}<extra></extra>'
        ))
        fig.update_layout(
            **plot_layout(260),
            xaxis=dict(
                **axis_style('Time (seconds)'),
                showspikes=True,
                spikecolor=T['divider'],
                tickformat='.1f',
                dtick=1.0,                                # tick every 1 second
                rangeslider=dict(visible=True, thickness=0.06, bgcolor=T['card_bg']),
            ),
            yaxis=dict(**axis_style('Amplitude (mV)')),
            hovermode='x unified'
        )
        dur_s = n_samples / sr
        st.caption(f"Signal: {n_samples} samples · {dur_s:.1f} s · {sr} Hz — drag the range slider below the chart to zoom")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns([2, 1, 2])
        with c2:
            analyse = st.button("Analyse ECG")

        if analyse:
            prog = st.progress(0)
            for pct, msg in [
                (15, "P1 — Classifying condition..."),
                (35, "P2 — Extracting clinical features..."),
                (55, "P3 — Scoring risk level..."),
                (75, "P4 — ACDA making decision..."),
                (92, "P5 — Generating LLM report..."),
            ]:
                time.sleep(0.25)
                prog.progress(pct, text=msg)
            result, error = call_api(signal)
            prog.progress(100, text="Analysis complete")
            time.sleep(0.4)
            prog.empty()
            if error:
                st.error(f"Pipeline error: {error}")
            else:
                st.session_state["result"] = result
                st.session_state["feedback_done"] = False
                st.rerun()

        if st.session_state["result"]:
            result   = st.session_state["result"]
            features = result.get("clinical_features", {})
            risk     = result.get("risk_assessment", {})
            agent    = result.get("agent_decision", {})

            conf   = result['confidence']
            sev    = risk.get('severity', 'LOW')
            dec    = agent.get('decision', 'ACCEPT')
            rscore = risk.get('risk_score', 0)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # 5 metric cards
            col1, col2, col3, col4, col5 = st.columns(5)
            for col, label, value, sub, mc in [
                (col1, "Condition",  result['condition'],           "AI Classification",    "mc-blue"),
                (col2, "Confidence", f"{conf}%",                   "Model certainty",      "mc-green" if conf >= 80 else "mc-amber" if conf >= 65 else "mc-red"),
                (col3, "Risk Score", f"{rscore}/12",               f"Severity: {sev}",     {"LOW": "mc-green", "MODERATE": "mc-amber", "HIGH": "mc-red", "CRITICAL": "mc-red"}.get(sev, "mc-blue")),
                (col4, "Heart Rate", f"{features.get('heart_rate', '--')} bpm", "Clinical reading",
                    "mc-green" if features.get('heart_rate') and 50 <= features.get('heart_rate', 0) <= 110 else "mc-red"),
                (col5, "Decision",   dec,                          agent.get('triage_urgency', ''),
                    {"ACCEPT": "mc-green", "RECHECK": "mc-amber", "ALERT": "mc-red", "EMERGENCY": "mc-red", "REJECT": "mc-blue"}.get(dec, "mc-blue")),
            ]:
                with col:
                    st.markdown(f"""
                    <div class="metric-card {mc}">
                        <div class="mc-label">{label}</div>
                        <div class="mc-value">{value}</div>
                        <div class="mc-sub">{sub}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Decision banner
            dc    = decision_color(dec)
            dec_icons = {"ACCEPT": "●", "RECHECK": "↺", "ALERT": "▲", "EMERGENCY": "!", "REJECT": "✕"}
            st.markdown(f"""
            <div class="decision-banner db-{dec}">
                <div class="db-label" style="color:{dc}99;">ACDA — Autonomous Cardiac Decision Agent</div>
                <div class="db-title" style="color:{dc};">{dec_icons.get(dec, '●')} {dec}</div>
                <div class="db-reason" style="color:{T['text_secondary']};">{agent.get('decision_reason', '')}</div>
                <div class="db-badges">
                    <span class="db-badge" style="color:{dc}99;">Urgency: {agent.get('triage_urgency', '')}</span>
                    <span class="db-badge" style="color:{dc}99;">Risk {rscore}/12 · {sev}</span>
                    <span class="db-badge" style="color:{dc}99;">{agent.get('timestamp', '')[:16].replace('T', ' ')}</span>
                    <span class="db-badge" style="color:{dc}99;">{conf}% confidence</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            left, right = st.columns([1, 1], gap="large")

            with left:
                st.markdown('<div class="section-title"><span>+</span> CLINICAL MEASUREMENTS</div>', unsafe_allow_html=True)
                hr  = features.get('heart_rate')
                rr  = features.get('rr_interval_ms')
                qrs = features.get('qrs_duration_ms')
                pr  = features.get('pr_interval_ms')
                sq  = features.get('signal_quality', 0)
                irr = features.get('rr_irregular', False)

                st.markdown(measure_row("Heart Rate",   hr,  "bpm", (50, 110)),   unsafe_allow_html=True)
                st.markdown(measure_row("RR Interval",  rr,  "ms",  (600, 1000)), unsafe_allow_html=True)
                st.markdown(measure_row("QRS Duration", qrs, "ms",  (60, 120)),   unsafe_allow_html=True)
                st.markdown(measure_row("PR Interval",  pr,  "ms",  (120, 200)),  unsafe_allow_html=True)

                irr_c   = "#F87171" if irr else "#34D399"
                irr_txt = "Irregular — Possible AFib" if irr else "Regular Rhythm"
                st.markdown(f'<div class="measure-row"><span class="m-label">RR Rhythm</span><span class="m-value" style="color:{irr_c};">{ind(irr_c)} {irr_txt}</span></div>', unsafe_allow_html=True)

                sq_c = "#34D399" if sq > 0.6 else "#FBBF24" if sq > 0.3 else "#F87171"
                st.markdown(f'<div class="measure-row"><span class="m-label">Signal Quality</span><span class="m-value" style="color:{sq_c};">{ind(sq_c)} {sq}</span></div>', unsafe_allow_html=True)

                st.markdown('<br><div class="section-title"><span>%</span> CONFIDENCE BREAKDOWN</div>', unsafe_allow_html=True)
                all_scores = result.get("all_scores", {})
                if all_scores:
                    fig2 = go.Figure(go.Bar(
                        x=list(all_scores.values()), y=list(all_scores.keys()),
                        orientation='h',
                        marker=dict(color=['#38BDF8', '#34D399', '#FBBF24', '#F87171'][:len(all_scores)], opacity=0.85),
                        text=[f"{v:.1f}%" for v in all_scores.values()],
                        textposition='outside', textfont=dict(color=T['plot_text'], size=10)
                    ))
                    fig2.update_layout(
                        **plot_layout(190, l=0, r=60, t=5, b=30),
                        xaxis=dict(**axis_style('%'), range=[0, 115]),
                        yaxis=dict(color=T['plot_text'])
                    )
                    st.plotly_chart(fig2, use_container_width=True)

            with right:
                st.markdown('<div class="section-title"><span>!</span> RISK FACTORS</div>', unsafe_allow_html=True)
                reasons = risk.get('reasons', [])
                if reasons:
                    for r_item in reasons:
                        st.markdown(f'<div class="risk-item">-- {r_item}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="safe-item">No significant risk factors identified</div>', unsafe_allow_html=True)

                st.markdown(f"""
                <div style="margin-top:14px; padding:clamp(10px,2vw,16px) clamp(12px,2vw,18px);
                            background:{T['card_bg']};
                            border:1px solid {T['card_border']}; border-radius:12px;">
                    <div style="font-size:clamp(0.52rem,1.2vw,0.6rem); color:{T['text_muted']}; text-transform:uppercase;
                                letter-spacing:1.5px; font-weight:700; margin-bottom:8px;">Recommended Action</div>
                    <div style="color:{T['text_secondary']}; font-size:clamp(0.78rem,1.8vw,0.875rem); line-height:1.7;">
                        {risk.get('action', '')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if all_scores:
                    st.markdown('<br><div class="section-title"><span>O</span> CONDITION DISTRIBUTION</div>', unsafe_allow_html=True)
                    fig3 = go.Figure(go.Pie(
                        labels=list(all_scores.keys()), values=list(all_scores.values()),
                        hole=0.62,
                        marker=dict(colors=['#38BDF8', '#34D399', '#FBBF24', '#F87171'],
                                    line=dict(color=T['app_bg'], width=2)),
                        textinfo='label+percent',
                        textfont=dict(size=11, color=T['plot_text']),
                        hovertemplate='<b>%{label}</b>: %{value:.1f}%<extra></extra>'
                    ))
                    fig3.add_annotation(text=f"<b>{result['condition']}</b>", x=0.5, y=0.5,
                                        showarrow=False, font=dict(size=13, color=T['text_primary'], family='Inter'))
                    # FIX: showlegend is now inside plot_layout via show_legend param — no duplicate!
                    fig3.update_layout(**plot_layout(210, l=0, r=0, t=0, b=0, show_legend=False))
                    st.plotly_chart(fig3, use_container_width=True)
        else:
            st.markdown("""<div style="margin-top:24px;" class="empty-state">
                <div class="empty-icon">[ ? ]</div>
                <div class="empty-title">Ready to Analyse</div>
                <div class="empty-sub">Click "Analyse ECG" to run the full AI pipeline</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="empty-state">
            <div class="empty-icon">[ ECG ]</div>
            <div class="empty-title">No Signal Loaded</div>
            <div class="empty-sub">Load an ECG signal from the sidebar to begin</div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — REPORTS
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    if st.session_state["result"] and st.session_state["result"].get("report"):
        result   = st.session_state["result"]
        report   = result.get("report", {})
        risk     = result.get("risk_assessment", {})
        agent    = result.get("agent_decision", {})
        features = result.get("clinical_features", {})

        st.markdown('<div class="section-title"><span>+</span> CLINICAL REPORT</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="report-header">
            <div class="report-header-item" style="text-align:left;">
                <div class="report-header-label">Report ID</div>
                <div style="font-size:clamp(1rem,3vw,1.4rem); font-weight:900; color:#38BDF8; letter-spacing:3px; font-family:monospace;">#{report.get('report_id','')}</div>
            </div>
            <div class="report-header-item">
                <div class="report-header-label">Condition</div>
                <div class="report-header-value">{result['condition']}</div>
            </div>
            <div class="report-header-item">
                <div class="report-header-label">Severity</div>
                <div class="report-header-value" style="color:{severity_color(risk.get('severity','LOW'))};">{risk.get('severity','')}</div>
            </div>
            <div class="report-header-item">
                <div class="report-header-label">Decision</div>
                <div class="report-header-value" style="color:{decision_color(agent.get('decision',''))};">{agent.get('decision','')}</div>
            </div>
            <div class="report-header-item">
                <div class="report-header-label">Confidence</div>
                <div class="report-header-value">{result['confidence']}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        r1, r2 = st.columns([1, 1], gap="large")
        with r1:
            st.markdown(f"""
            <div class="summary-card">
                <div class="s-tag s-doctor">Clinical Summary — Doctor</div>
                <div class="s-text">{report.get('clinical_summary', '')}</div>
            </div>
            <div class="summary-card">
                <div class="s-tag s-action">Suggested Action</div>
                <div class="s-text">{report.get('suggested_action', '')}</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div class="summary-card">
                <div class="s-tag s-patient">Patient Summary — Plain English</div>
                <div class="s-text">{report.get('patient_summary', '')}</div>
            </div>
            <div class="summary-card">
                <div class="s-tag s-doctor">Key Measurements</div>
                <div class="measurements-mini-grid">
                    <div class="mini-metric">
                        <div class="mini-metric-val" style="color:#38BDF8;">{features.get('heart_rate','--')}</div>
                        <div class="mini-metric-lbl">BPM</div>
                    </div>
                    <div class="mini-metric">
                        <div class="mini-metric-val" style="color:#34D399;">{features.get('qrs_duration_ms','--')}</div>
                        <div class="mini-metric-lbl">QRS ms</div>
                    </div>
                    <div class="mini-metric">
                        <div class="mini-metric-val" style="color:#FBBF24;">{features.get('pr_interval_ms','--')}</div>
                        <div class="mini-metric-lbl">PR ms</div>
                    </div>
                    <div class="mini-metric">
                        <div class="mini-metric-val" style="color:#C084FC;">{risk.get('risk_score','--')}/12</div>
                        <div class="mini-metric-lbl">RISK</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        d1, d2, d3 = st.columns([1, 1, 1])
        with d2:
            pdf_path = resolve_pdf(report)
            if pdf_path:
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="Download Full PDF Report",
                        data=f,
                        file_name=f"ECG_Report_{report.get('report_id')}.pdf",
                        mime="application/pdf"
                    )
            else:
                st.warning("PDF not found. Re-run analysis.")
    else:
        st.markdown("""<div class="empty-state">
            <div class="empty-icon">[ PDF ]</div>
            <div class="empty-title">No Report Generated</div>
            <div class="empty-sub">Run an analysis in the Analysis tab first</div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — FEEDBACK
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    if st.session_state["result"]:
        result = st.session_state["result"]
        risk   = result.get("risk_assessment", {})
        agent  = result.get("agent_decision", {})
        dec    = agent.get('decision', '')
        dc     = decision_color(dec)

        st.markdown('<div class="section-title"><span>+</span> DOCTOR FEEDBACK LOOP</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:{T['card_bg']}; border:1px solid {T['card_border']};
                    border-radius:clamp(8px,1.5vw,14px); padding:clamp(12px,2vw,18px) clamp(14px,2vw,24px); margin-bottom:20px;
                    display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:14px;
                    box-shadow:0 2px 6px rgba(0,0,0,{'0.12' if dark else '0.04'});">
            <div>
                <div style="font-size:clamp(0.52rem,1.2vw,0.6rem); color:{T['text_muted']}; text-transform:uppercase; letter-spacing:2px; margin-bottom:5px;">Current AI Prediction</div>
                <div style="font-size:clamp(1rem,3vw,1.25rem); font-weight:800; color:{T['text_primary']};">{result['condition']}</div>
                <div style="font-size:clamp(0.65rem,1.5vw,0.78rem); color:{T['text_muted']}; margin-top:3px;">
                    {result['confidence']}% confidence · Risk {risk.get('risk_score')}/12 · {risk.get('severity')}
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:clamp(0.52rem,1.2vw,0.6rem); color:{T['text_muted']}; text-transform:uppercase; letter-spacing:2px; margin-bottom:5px;">Agent Decision</div>
                <div style="font-size:clamp(0.9rem,2.5vw,1.1rem); font-weight:800; color:{dc};">{dec}</div>
                <div style="font-size:clamp(0.62rem,1.4vw,0.75rem); color:{T['text_muted']};">{agent.get('triage_urgency', '')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state["feedback_done"]:
            st.markdown('<div class="feedback-card">', unsafe_allow_html=True)
            fc1, fc2 = st.columns(2, gap="large")
            with fc1:
                st.markdown(f'<div style="font-size:0.6rem; color:{T["text_muted"]}; font-weight:700; letter-spacing:2px; text-transform:uppercase; margin-bottom:10px;">Your Assessment</div>', unsafe_allow_html=True)
                override = st.radio("", ["Accept AI Prediction", "Override with Correct Label"], label_visibility="collapsed")
                doctor_label = result['condition']
                if "Override" in override:
                    doctor_label = st.selectbox("Correct Condition", ["Normal", "AFib", "PVC", "Tachycardia", "Bradycardia", "Other"])
            with fc2:
                st.markdown(f'<div style="font-size:0.6rem; color:{T["text_muted"]}; font-weight:700; letter-spacing:2px; text-transform:uppercase; margin-bottom:10px;">Reason</div>', unsafe_allow_html=True)
                reason = st.selectbox("", [
                    "N/A — Accepted prediction",
                    "Signal quality too low",
                    "Misidentified rhythm",
                    "Clinical context not captured",
                    "Model confidence too low",
                    "Other"
                ], label_visibility="collapsed")
                notes = st.text_area("Clinical Notes", placeholder="Additional observations...", height=90)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            sb1, sb2, sb3 = st.columns([1, 1, 1])
            with sb2:
                if st.button("Submit Feedback"):
                    save_feedback(result, doctor_label, "Override" in override, reason, notes)
                    st.session_state["feedback_done"] = True
                    st.success("Feedback saved successfully.")
                    st.balloons()
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="feedback-success">Feedback submitted for this analysis. It will improve the model in Phase 7.</div>', unsafe_allow_html=True)
    else:
        st.markdown("""<div class="empty-state">
            <div class="empty-icon">[ DR ]</div>
            <div class="empty-title">No Analysis to Review</div>
            <div class="empty-sub">Run an ECG analysis first, then come back here to submit feedback</div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — HISTORY
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title"><span>+</span> FEEDBACK HISTORY & MODEL PERFORMANCE</div>', unsafe_allow_html=True)

    if os.path.exists(FEEDBACK_FILE):
        df        = pd.read_csv(FEEDBACK_FILE)
        total     = len(df)
        overrides = int(df['override'].sum())
        agreement = round((1 - overrides / total) * 100, 1) if total > 0 else 0

        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-box">
                <div class="stat-num" style="color:#38BDF8;">{total}</div>
                <div class="stat-label">Total Cases</div>
            </div>
            <div class="stat-box">
                <div class="stat-num" style="color:#F87171;">{overrides}</div>
                <div class="stat-label">Doctor Overrides</div>
            </div>
            <div class="stat-box">
                <div class="stat-num" style="color:#34D399;">{agreement}%</div>
                <div class="stat-label">Agreement Rate</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        h1, h2 = st.columns([1, 1], gap="large")

        with h1:
            st.markdown('<div class="section-title"><span>~</span> AGREEMENT TREND</div>', unsafe_allow_html=True)
            if total > 1:
                df['cumulative_agreement'] = (~df['override'].astype(bool)).cumsum() / (df.index + 1) * 100
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(
                    y=df['cumulative_agreement'], mode='lines',
                    line=dict(color='#38BDF8', width=2, shape='spline'),
                    fill='tozeroy', fillcolor='rgba(56,189,248,0.06)',
                    hovertemplate='Case %{x}: %{y:.1f}%<extra></extra>'
                ))
                fig4.add_hline(y=80, line_dash="dot", line_color=T['divider'],
                               annotation_text="80% target", annotation_font_color=T['text_muted'])
                fig4.update_layout(
                    **plot_layout(220),
                    xaxis=dict(**axis_style('Case #')),
                    yaxis=dict(**axis_style('Agreement %'), range=[0, 105])
                )
                st.plotly_chart(fig4, use_container_width=True)

        with h2:
            st.markdown('<div class="section-title"><span>O</span> CONDITION DISTRIBUTION</div>', unsafe_allow_html=True)
            if 'ai_condition' in df.columns:
                cond_counts = df['ai_condition'].value_counts()
                fig5 = go.Figure(go.Pie(
                    labels=cond_counts.index.tolist(),
                    values=cond_counts.values.tolist(),
                    hole=0.6,
                    marker=dict(colors=['#38BDF8', '#34D399', '#FBBF24', '#F87171'],
                                line=dict(color=T['app_bg'], width=2)),
                    textinfo='label+percent',
                    textfont=dict(size=11, color=T['plot_text'])
                ))
                fig5.add_annotation(text=f"<b>{total}</b><br>cases", x=0.5, y=0.5,
                                    showarrow=False, font=dict(size=12, color=T['text_primary'], family='Inter'))
                # FIX: showlegend passed via show_legend param — no duplicate keyword error
                fig5.update_layout(**plot_layout(220, l=0, r=0, t=0, b=0, show_legend=False))
                st.plotly_chart(fig5, use_container_width=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><span>+</span> RECENT RECORDS</div>', unsafe_allow_html=True)
        display_cols = [c for c in ['timestamp', 'ai_condition', 'ai_confidence', 'severity', 'agent_decision', 'doctor_label', 'override', 'reason'] if c in df.columns]
        st.dataframe(
            df.tail(15)[display_cols].rename(columns={
                'timestamp': 'Time', 'ai_condition': 'AI Prediction',
                'ai_confidence': 'Confidence', 'severity': 'Severity',
                'agent_decision': 'Decision', 'doctor_label': 'Doctor Label',
                'override': 'Override', 'reason': 'Reason'
            }),
            use_container_width=True, hide_index=True
        )
    else:
        st.markdown("""<div class="empty-state">
            <div class="empty-icon">[ -- ]</div>
            <div class="empty-title">No History Yet</div>
            <div class="empty-sub">Submit doctor feedback to start tracking model performance</div>
        </div>""", unsafe_allow_html=True)