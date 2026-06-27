"""Calibration page: fit Heston/Merton/SABR models to market data."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.theme import card_title, section_title, plotly_layout
from calibration.calibrate import calibrate_heston, calibrate_merton
from models.sabr import sabr_calibrate, sabr_smile


_INFO_BOX = (
    '<div style="background:rgba(79,142,247,0.08);border:1px solid rgba(79,142,247,0.25);'
    "border-radius:4px;padding:10px 14px;font-family:'JetBrains Mono',monospace;"
    'font-size:11px;color:#4f8ef7;margin-bottom:14px;">'
    "Synthetic market: 9-strike vol surface with realistic skew. "
    "Calibration via differential evolution + Nelder-Mead."
    "</div>"
)


def render(params: dict) -> None:
    """Render the calibration page."""
    card_title("Model Calibration")

    S, r = params["S"], params["r"]

    tab_heston, tab_merton, tab_sabr = st.tabs(["Heston", "Merton", "SABR"])

    with tab_heston:
        _render_heston_cal(S, r)

    with tab_merton:
        _render_merton_cal(S, r)

    with tab_sabr:
        _render_sabr_cal(S, r, params["T"])


def _generate_synthetic_market(S, r):
    """Generate synthetic market data with a skew."""
    strikes = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120], dtype=float)
    T = 0.5
    base_vol = 0.20
    market_vols = np.array([
        base_vol + 0.06,
        base_vol + 0.04,
        base_vol + 0.025,
        base_vol + 0.01,
        base_vol,
        base_vol + 0.005,
        base_vol + 0.015,
        base_vol + 0.03,
        base_vol + 0.05,
    ])
    return strikes, T, market_vols


def _market_data_table(strikes, market_vols):
    """Render market data as a horizontal styled HTML table."""
    strike_headers = "".join(f"<th>{K:.0f}</th>" for K in strikes)
    vol_cells = "".join(f"<td>{v:.1%}</td>" for v in market_vols)
    st.markdown(
        f'<table class="tbl" style="max-width:480px"><thead><tr><th>Strike</th>{strike_headers}</tr></thead>'
        f'<tbody><tr><td class="tbl-hi">Market IV</td>{vol_cells}</tr></tbody></table>',
        unsafe_allow_html=True,
    )


def _fit_plot(strikes, market_vols, model_vols, model_name, valid=None):
    """Return a Plotly figure with market dots and model fit line."""
    if valid is None:
        valid = ~np.isnan(model_vols)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=strikes, y=market_vols * 100, name="Market",
        mode="markers", marker=dict(color="#f7b731", size=8),
    ))
    if valid.any():
        fig.add_trace(go.Scatter(
            x=strikes[valid], y=model_vols[valid] * 100,
            name=f"{model_name} Fit", line=dict(color="#00d4aa", width=2),
        ))
    fig.update_layout(**plotly_layout(
        title=dict(text=f"{model_name} Calibration"),
        xaxis=dict(title="Strike"),
        yaxis=dict(title="IV (%)"),
    ))
    return fig


def _show_params_metrics(params_dict, rmse):
    """Display calibrated parameters as a row of metrics plus RMSE."""
    keys = list(params_dict.keys())
    cols = st.columns(len(keys) + 1)
    for col, k in zip(cols, keys):
        with col:
            st.metric(k, f"{params_dict[k]:.4f}")
    with cols[-1]:
        st.metric("RMSE", f"{rmse:.4f}")


def _render_heston_cal(S, r):
    st.markdown(_INFO_BOX, unsafe_allow_html=True)

    strikes, T, market_vols = _generate_synthetic_market(S, r)
    maturities = np.full_like(strikes, T)

    section_title("Market Data (Synthetic)")
    _market_data_table(strikes, market_vols)

    if st.button("Calibrate Heston", key="cal_heston"):
        with st.spinner("Running differential evolution..."):
            result = calibrate_heston(S, r, strikes, maturities, market_vols)

        fig = _fit_plot(strikes, market_vols, result.model_vols, "Heston")
        st.plotly_chart(fig, use_container_width=True)

        section_title("Calibrated Parameters")
        _show_params_metrics(result.params, result.rmse)


def _render_merton_cal(S, r):
    st.markdown(_INFO_BOX, unsafe_allow_html=True)

    strikes, T, market_vols = _generate_synthetic_market(S, r)
    maturities = np.full_like(strikes, T)

    section_title("Market Data (Synthetic)")
    _market_data_table(strikes, market_vols)

    if st.button("Calibrate Merton", key="cal_merton"):
        with st.spinner("Running calibration..."):
            result = calibrate_merton(S, r, strikes, maturities, market_vols)

        fig = _fit_plot(strikes, market_vols, result.model_vols, "Merton")
        st.plotly_chart(fig, use_container_width=True)

        section_title("Calibrated Parameters")
        _show_params_metrics(result.params, result.rmse)


def _render_sabr_cal(S, r, T):
    st.markdown(_INFO_BOX, unsafe_allow_html=True)

    F = S * np.exp(r * T)
    beta = st.slider("Fixed beta", 0.0, 1.0, 0.5, 0.1, key="sabr_cal_beta")

    strikes, T_cal, market_vols = _generate_synthetic_market(S, r)

    section_title("Market Data (Synthetic)")
    _market_data_table(strikes, market_vols)

    if st.button("Calibrate SABR", key="cal_sabr"):
        result = sabr_calibrate(F, strikes, market_vols, T_cal, beta)
        model_vols = sabr_smile(F, strikes, T_cal, result["alpha"], beta, result["rho"], result["nu"])

        fig = _fit_plot(strikes, market_vols, model_vols, "SABR")
        st.plotly_chart(fig, use_container_width=True)

        section_title("Calibrated Parameters")
        cal_params = {
            "alpha": result["alpha"],
            "rho": result["rho"],
            "nu": result["nu"],
        }
        _show_params_metrics(cal_params, result["rmse"])
