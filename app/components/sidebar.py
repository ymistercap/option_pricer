"""Sidebar — OP logo, navigation, slider params, CALL/PUT toggle."""

import streamlit as st


_LOGO_HTML = """
<div style="padding:16px 0 14px 0; border-bottom:1px solid #2a3040; display:flex; align-items:center; gap:10px;">
    <div style="
        width:28px; height:28px; border-radius:4px;
        background:#f7b731; display:flex; align-items:center; justify-content:center;
        flex-shrink:0;
    ">
        <span style="font-family:'JetBrains Mono',monospace; font-size:13px;
            font-weight:600; color:#000;">OP</span>
    </div>
    <div style="font-family:'JetBrains Mono',monospace; font-size:11px; color:#8896a8; line-height:1.4;">
        <strong style="color:#f7b731; font-size:12px; display:block;">OPTION PRICER</strong>
        Risk Analyzer
    </div>
</div>
"""

_SECTION = (
    '<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
    'color:#556070;text-transform:uppercase;letter-spacing:0.12em;'
    'padding:14px 0 8px 0;">{}</div>'
)

PAGE_LABELS = [
    "01  Option Pricer",
    "02  Greeks",
    "03  Vol Surface",
    "04  Exotics",
    "05  Delta Hedging",
    "06  Risk / VaR",
    "07  Adv. Models",
    "08  Calibration",
    "09  Model Risk",
]


def render_sidebar() -> dict:
    """Render the styled sidebar: logo, navigation, parameters, type toggle."""
    st.sidebar.markdown(_LOGO_HTML, unsafe_allow_html=True)

    # --- Navigation (right after logo) ---
    st.sidebar.markdown(_SECTION.format("Navigation"), unsafe_allow_html=True)
    selected_page = st.sidebar.radio(
        "Navigation", PAGE_LABELS,
        label_visibility="collapsed", key="nav_radio",
    )

    # --- Parameters ---
    st.sidebar.markdown(_SECTION.format("Parameters"), unsafe_allow_html=True)

    S = st.sidebar.slider("Spot (S)", 10.0, 500.0, 100.0, 1.0, format="%.0f")
    K = st.sidebar.slider("Strike (K)", 10.0, 500.0, 100.0, 1.0, format="%.0f")
    T = st.sidebar.slider("Maturity (T)", 0.05, 5.0, 1.0, 0.05, format="%.2fy")
    r_pct = st.sidebar.slider("Rate (r)", 0.0, 20.0, 5.0, 0.5, format="%.1f%%")
    sigma_pct = st.sidebar.slider("Vol (σ)", 1.0, 100.0, 20.0, 1.0, format="%.0f%%")

    r = r_pct / 100.0
    sigma = sigma_pct / 100.0

    # --- Type toggle ---
    st.sidebar.markdown(_SECTION.format("Type"), unsafe_allow_html=True)
    option_type = st.sidebar.selectbox(
        "Option Type", ["call", "put"],
        label_visibility="collapsed",
    )

    # --- MC Paths ---
    n_paths_k = st.sidebar.slider("MC Paths", 1, 500, 100, 1, format="%dK")
    n_paths = n_paths_k * 1000

    # --- Footer ---
    st.sidebar.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;color:#556070;'
        'padding-top:8px;border-top:1px solid #2a3040;margin-top:12px;">'
        '<div style="margin-bottom:4px;">OPTION PRICER</div>'
        '<a href="https://github.com/ymistercap/option_pricer" target="_blank" '
        'style="color:#f7b731;text-decoration:none;">GitHub →</a>'
        '</div>',
        unsafe_allow_html=True,
    )

    return {
        "S": S, "K": K, "T": T, "r": r, "sigma": sigma,
        "option_type": option_type, "n_paths": int(n_paths),
        "selected_page": selected_page,
    }
