"""Model Risk page: compare models, local vol surface, parameter sensitivity."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.theme import card_title, plotly_layout
from pricing.binomial_tree import crr_european
from pricing.black_scholes import bs_price
from pricing.heston import heston_price
from pricing.jump_diffusion import merton_price
from models.local_vol import dupire_local_vol


def render(params: dict) -> None:
    """Render the model risk analysis page."""
    card_title("Model Risk Analysis")

    S, K, T, r, sigma = params["S"], params["K"], params["T"], params["r"], params["sigma"]
    option_type = params["option_type"]

    tab_compare, tab_localvol, tab_sensitivity = st.tabs(
        ["Model Comparison", "Local Vol Surface", "Parameter Sensitivity"]
    )

    with tab_compare:
        _render_comparison(S, K, T, r, sigma, option_type)

    with tab_localvol:
        _render_local_vol(S, r, sigma)

    with tab_sensitivity:
        _render_sensitivity(S, K, T, r, sigma, option_type)


def _render_comparison(S, K, T, r, sigma, option_type):
    if st.button("Compare All Models", key="btn_compare"):
        results = {}

        results["Black-Scholes"] = bs_price(S, K, T, r, sigma, option_type)
        results["CRR (500 steps)"] = crr_european(S, K, T, r, sigma, option_type, 500)
        results["Heston"] = heston_price(
            S, K, T, r, sigma**2, 2.0, sigma**2, 0.3, -0.7, option_type
        )
        results["Merton"] = merton_price(
            S, K, T, r, sigma, 1.0, -0.05, 0.1, option_type
        )

        bs_ref = results["Black-Scholes"]

        # Build HTML table with .tbl class
        rows_html = ""
        for name, price in results.items():
            diff = price - bs_ref
            if name == "Black-Scholes":
                diff_cell = "<td>—</td>"
            elif diff > 0:
                diff_cell = f'<td class="tbl-pos">${diff:+.4f}</td>'
            elif diff < 0:
                diff_cell = f'<td class="tbl-neg">${diff:+.4f}</td>'
            else:
                diff_cell = f"<td>${diff:+.4f}</td>"
            rows_html += (
                f'<tr><td class="tbl-hi">{name}</td>'
                f"<td>${price:.4f}</td>"
                f"{diff_cell}</tr>\n"
            )

        table_html = (
            '<table class="tbl">'
            "<thead><tr><th>Model</th><th>Price</th><th>Diff vs BS</th></tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            "</table>"
        )
        st.markdown(table_html, unsafe_allow_html=True)

        strikes = np.linspace(S * 0.8, S * 1.2, 21)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=strikes,
            y=[bs_price(S, k, T, r, sigma, option_type) for k in strikes],
            name="BS", line=dict(color="#4f8ef7", width=2, dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=strikes,
            y=[heston_price(S, k, T, r, sigma**2, 2.0, sigma**2, 0.3, -0.7, option_type) for k in strikes],
            name="Heston", line=dict(color="#f7b731", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=strikes,
            y=[merton_price(S, k, T, r, sigma, 1.0, -0.05, 0.1, option_type) for k in strikes],
            name="Merton", line=dict(color="#e74c3c", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=strikes,
            y=[crr_european(S, k, T, r, sigma, option_type, 500) for k in strikes],
            name="CRR", line=dict(color="#2ecc71", width=2, dash="dot"),
        ))
        fig.update_layout(**plotly_layout(
            title=dict(text="Price vs Strike"),
            xaxis=dict(title="Strike", gridcolor="#2a3040", zerolinecolor="#2a3040", tickfont=dict(size=9, color="#556070"), linecolor="#2a3040"),
            yaxis=dict(title="Price", gridcolor="#2a3040", zerolinecolor="#2a3040", tickfont=dict(size=9, color="#556070"), linecolor="#2a3040"),
        ))
        st.plotly_chart(fig, use_container_width=True)


def _render_local_vol(S, r, sigma):
    strikes = np.linspace(S * 0.7, S * 1.3, 13)
    maturities = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0])

    iv_surface = np.zeros((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            moneyness = K / S
            iv_surface[i, j] = sigma + 0.1 * (1.0 - moneyness) + 0.02 * np.sqrt(T)

    lv = dupire_local_vol(strikes, maturities, iv_surface, S, r)

    fig = go.Figure(data=[go.Surface(
        x=strikes, y=maturities, z=lv * 100,
        colorscale=[[0, "#1a3a5c"], [0.3, "#2980b9"], [0.6, "#00d4aa"], [1, "#f7b731"]],
    )])
    fig.update_layout(**plotly_layout(
        title=dict(text="Local Volatility Surface"),
        scene=dict(xaxis_title="Strike", yaxis_title="Maturity", zaxis_title="Local Vol (%)"),
        height=460,
    ))
    st.plotly_chart(fig, use_container_width=True)


def _render_sensitivity(S, K, T, r, sigma, option_type):
    st.markdown(
        '<div style="background:rgba(79,142,247,0.08);border:1px solid rgba(79,142,247,0.25);'
        "border-radius:4px;padding:10px 14px;font-family:'JetBrains Mono',monospace;"
        'font-size:11px;color:#4f8ef7;margin-bottom:16px">'
        "Price sensitivity to Heston ρ and ξ parameters. "
        "BS reference shown as benchmark."
        "</div>",
        unsafe_allow_html=True,
    )

    param = st.selectbox("Vary parameter", ["v0", "kappa", "theta", "xi", "rho"], key="sens_param")

    base = {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.7}

    ranges = {
        "v0": np.linspace(0.01, 0.16, 20),
        "kappa": np.linspace(0.5, 8.0, 20),
        "theta": np.linspace(0.01, 0.16, 20),
        "xi": np.linspace(0.05, 1.5, 20),
        "rho": np.linspace(-0.95, 0.95, 20),
    }

    vals = ranges[param]
    prices = []
    for v in vals:
        p = base.copy()
        p[param] = v
        prices.append(heston_price(S, K, T, r, p["v0"], p["kappa"], p["theta"], p["xi"], p["rho"], option_type))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vals, y=prices, name=f"Price vs {param}",
                             line=dict(color="#f7b731", width=2)))
    fig.add_hline(y=bs_price(S, K, T, r, sigma, option_type), line_dash="dash",
                 line_color="#00d4aa", annotation_text="BS ref")
    fig.update_layout(**plotly_layout(
        title=dict(text=f"Sensitivity to {param}"),
        xaxis=dict(title=param, gridcolor="#2a3040", zerolinecolor="#2a3040", tickfont=dict(size=9, color="#556070"), linecolor="#2a3040"),
        yaxis=dict(title="Price", gridcolor="#2a3040", zerolinecolor="#2a3040", tickfont=dict(size=9, color="#556070"), linecolor="#2a3040"),
    ))
    st.plotly_chart(fig, use_container_width=True)
