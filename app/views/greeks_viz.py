"""Greeks visualization page."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.theme import card_title, section_title, plotly_layout
from greeks.analytical import delta, gamma, vega, theta, rho

_GREEKS = [
    ("Delta", "Δ", "#f7b731", lambda S, K, T, r, sigma, opt: delta(S, K, T, r, sigma, opt)),
    ("Gamma", "Γ", "#00d4aa", lambda S, K, T, r, sigma, opt: gamma(S, K, T, r, sigma)),
    ("Vega",  "ν", "#4f8ef7", lambda S, K, T, r, sigma, opt: vega(S, K, T, r, sigma)),
    ("Theta", "Θ", "#e74c3c", lambda S, K, T, r, sigma, opt: theta(S, K, T, r, sigma, opt)),
    ("Rho",   "ρ", "#c084fc", lambda S, K, T, r, sigma, opt: rho(S, K, T, r, sigma, opt)),
]


def _greek_chart(x_vals, y_vals, *, name: str, color: str, strike: float | None = None,
                 x_label: str = "Spot Price") -> go.Figure:
    """Create an individual Plotly figure for a single Greek."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals, mode="lines", showlegend=False,
        line=dict(color=color, width=2),
    ))
    if strike is not None:
        fig.add_vline(x=strike, line_dash="dash", line_color="#2a3040")
    fig.update_layout(**plotly_layout(
        height=260,
        title=dict(text=name, font=dict(size=11)),
        xaxis=dict(title=dict(text=x_label, font=dict(size=9))),
        yaxis=dict(title=dict(text=name, font=dict(size=9))),
        margin=dict(t=32, r=12, b=44, l=52),
    ))
    return fig


def render(params: dict) -> None:
    """Render the Greeks visualization page."""
    card_title("Greeks Visualization")

    with st.expander("Mathematical Background"):
        st.markdown(r"""
        **Greeks** measure the sensitivity of option price to parameters:
        - $\Delta = \frac{\partial V}{\partial S}$ — hedge ratio
        - $\Gamma = \frac{\partial^2 V}{\partial S^2}$ — delta curvature
        - $\mathcal{V} = \frac{\partial V}{\partial \sigma}$ — vol sensitivity
        - $\Theta = \frac{\partial V}{\partial t}$ — time decay
        - $\rho = \frac{\partial V}{\partial r}$ — rate sensitivity

        The BS PDE links them: $\Theta + \frac{1}{2}\sigma^2 S^2 \Gamma + rS\Delta - rV = 0$
        """)

    S, K, T, r_val, sigma = (
        params["S"], params["K"], params["T"], params["r"], params["sigma"],
    )
    option_type = params["option_type"]

    # --- Metrics row ---
    greek_values = [fn(S, K, T, r_val, sigma, option_type) for *_, fn in _GREEKS]
    cols = st.columns(5)
    for col, (_, symbol, _, _), val in zip(cols, _GREEKS, greek_values):
        col.metric(symbol, f"{val:.4f}")

    # --- Greeks vs Spot Price ---
    section_title("Greeks vs Spot Price (dashed = strike)")

    S_range = np.linspace(max(K * 0.5, 1), K * 1.5, 200)

    spot_cols = st.columns(3)
    for i, (name, _, color, fn) in enumerate(_GREEKS):
        y_vals = [fn(s, K, T, r_val, sigma, option_type) for s in S_range]
        fig = _greek_chart(S_range, y_vals, name=name, color=color, strike=K,
                           x_label="Spot Price")
        spot_cols[i % 3].plotly_chart(fig, use_container_width=True)

    # --- Greeks vs Time to Maturity ---
    section_title("Greeks vs Time to Maturity")

    T_range = np.linspace(0.01, max(T * 2, 2.0), 200)

    time_cols = st.columns(3)
    for i, (name, _, color, fn) in enumerate(_GREEKS):
        y_vals = [fn(S, K, t, r_val, sigma, option_type) for t in T_range]
        fig = _greek_chart(T_range, y_vals, name=name, color=color,
                           x_label="Time to Maturity")
        time_cols[i % 3].plotly_chart(fig, use_container_width=True)
