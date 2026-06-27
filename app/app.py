"""
Option Pricer & Risk Analyzer — Streamlit Dashboard.

Launch with: streamlit run app/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(
    page_title="Option Pricer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from components.theme import inject_theme, topbar_html
from components.sidebar import render_sidebar
from views import pricer, greeks_viz, vol_surface, exotics, hedging_sim, risk
from views import models, calibration_page, model_risk

inject_theme()

params = render_sidebar()
selected_page = params.pop("selected_page")

PAGES = {
    "01  Option Pricer": ("Option Pricer", pricer),
    "02  Greeks": ("Greeks", greeks_viz),
    "03  Vol Surface": ("Vol Surface", vol_surface),
    "04  Exotics": ("Exotics", exotics),
    "05  Delta Hedging": ("Delta Hedging", hedging_sim),
    "06  Risk / VaR": ("Risk / VaR", risk),
    "07  Adv. Models": ("Adv. Models", models),
    "08  Calibration": ("Calibration", calibration_page),
    "09  Model Risk": ("Model Risk", model_risk),
}

page_display, page_module = PAGES[selected_page]
page_num = selected_page[:2]

st.markdown(topbar_html(page_display, page_num, params), unsafe_allow_html=True)

page_module.render(params)
