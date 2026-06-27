"""Exotic options pricing page."""

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.theme import card_title, section_title, plotly_layout
from exotics.barrier import barrier_analytical, barrier_mc
from exotics.asian import asian_geometric_analytical, asian_mc, asian_arithmetic_cv
from exotics.lookback import lookback_floating_analytical, lookback_mc


def render(params: dict) -> None:
    """Render the exotic options pricing page."""
    card_title("Exotic Options Pricer")

    S, K, T, r, sigma = params["S"], params["K"], params["T"], params["r"], params["sigma"]
    option_type = params["option_type"]
    n_paths = params["n_paths"]

    tab1, tab2, tab3 = st.tabs(["Barrier", "Asian", "Lookback"])

    with tab1:
        _render_barrier(S, K, T, r, sigma, option_type, n_paths)

    with tab2:
        _render_asian(S, K, T, r, sigma, option_type, n_paths)

    with tab3:
        _render_lookback(S, T, r, sigma, option_type, n_paths)


def _render_barrier(S, K, T, r, sigma, option_type, n_paths):
    with st.expander("Mathematical Background"):
        st.markdown(r"""
        **Barrier options** are activated/deactivated when spot hits a barrier $H$.
        - **In-Out Parity**: $V_{in} + V_{out} = V_{vanilla}$
        - Analytical formulas by Reiner & Rubinstein (1991)
        """)

    col1, col2 = st.columns(2)
    with col1:
        barrier_type = st.selectbox("Barrier Type",
            ["down-and-out", "down-and-in", "up-and-out", "up-and-in"])
    with col2:
        H = st.number_input("Barrier Level (H)", value=80.0 if "down" in barrier_type else 120.0,
                           min_value=0.01, step=5.0)

    if st.button("Price Barrier Option"):
        t0 = time.time()
        price_an = barrier_analytical(S, K, T, r, sigma, H, barrier_type, option_type)
        t1 = time.time() - t0

        t2 = time.time()
        mc = barrier_mc(S, K, T, r, sigma, H, barrier_type, option_type, n_paths)
        t3 = time.time() - t2

        from pricing.black_scholes import bs_price
        vanilla = bs_price(S, K, T, r, sigma, option_type)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Analytical", f"${price_an:.4f}")
            st.caption("Reiner-Rubinstein")
        with m2:
            st.metric("MC Price", f"${mc.price:.4f}")
            st.caption(f"Vanilla ref: ${vanilla:.4f}")
        with m3:
            st.metric("In-Out Parity", f"${vanilla:.4f}")
            st.caption("V_in + V_out ≈ Vanilla")

        section_title("Price vs Barrier Level")
        if "down" in barrier_type:
            H_range = np.linspace(max(S * 0.5, 1), S * 0.99, 50)
        else:
            H_range = np.linspace(S * 1.01, S * 1.5, 50)

        prices = [barrier_analytical(S, K, T, r, sigma, h, barrier_type, option_type) for h in H_range]
        fig = go.Figure(go.Scatter(x=H_range, y=prices, mode="lines",
                                   line=dict(color="#f7b731", width=2)))
        fig.update_layout(**plotly_layout(
            xaxis=dict(title="Barrier Level", gridcolor="#2a3040", zerolinecolor="#2a3040", tickfont=dict(size=9, color="#556070"), linecolor="#2a3040"),
            yaxis=dict(title="Option Price", gridcolor="#2a3040", zerolinecolor="#2a3040", tickfont=dict(size=9, color="#556070"), linecolor="#2a3040"),
            height=350,
        ))
        st.plotly_chart(fig, use_container_width=True)


def _render_asian(S, K, T, r, sigma, option_type, n_paths):
    with st.expander("Mathematical Background"):
        st.markdown(r"""
        **Asian options** payoff depends on the average price:
        - **Arithmetic avg**: No closed-form, priced via MC
        - **Geometric avg**: Closed-form exists (geometric avg of lognormals is lognormal)
        - The geometric price serves as a control variate for the arithmetic price
        """)

    if st.button("Price Asian Options"):
        t0 = time.time()
        geom_an = asian_geometric_analytical(S, K, T, r, sigma, option_type=option_type)
        t1 = time.time() - t0

        t2 = time.time()
        arith_mc = asian_mc(S, K, T, r, sigma, n_paths, average_type="arithmetic",
                           option_type=option_type)
        t3 = time.time() - t2

        t4 = time.time()
        arith_cv = asian_arithmetic_cv(S, K, T, r, sigma, n_paths, option_type=option_type)
        t5 = time.time() - t4

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Geometric (Analytical)", f"${geom_an:.4f}")
            st.caption(f"Time: {t1*1000:.1f} ms")
        with m2:
            st.metric("Arithmetic (MC)", f"${arith_mc.price:.4f}")
            st.caption(f"SE: {arith_mc.std_error:.4f}")
        with m3:
            st.metric("Arithmetic (CV)", f"${arith_cv.price:.4f}")
            st.caption(f"SE: {arith_cv.std_error:.4f} | VR: {(arith_mc.std_error/arith_cv.std_error)**2:.1f}x")

        section_title("Price vs Monitoring Steps")
        steps_range = [12, 24, 52, 104, 156, 208, 252, 504]
        cv_prices = []
        for n_s in steps_range:
            res = asian_arithmetic_cv(S, K, T, r, sigma, n_paths, n_steps=n_s,
                                      option_type=option_type)
            cv_prices.append(res.price)

        fig_conv = go.Figure(go.Scatter(
            x=steps_range, y=cv_prices, mode="lines+markers",
            line=dict(color="#00d4aa", width=2),
            marker=dict(color="#00d4aa", size=5),
        ))
        fig_conv.update_layout(**plotly_layout(
            xaxis=dict(title="Monitoring Steps", gridcolor="#2a3040", zerolinecolor="#2a3040", tickfont=dict(size=9, color="#556070"), linecolor="#2a3040"),
            yaxis=dict(title="Option Price", gridcolor="#2a3040", zerolinecolor="#2a3040", tickfont=dict(size=9, color="#556070"), linecolor="#2a3040"),
            height=350,
        ))
        st.plotly_chart(fig_conv, use_container_width=True)


def _render_lookback(S, T, r, sigma, option_type, n_paths):
    with st.expander("Mathematical Background"):
        st.markdown(r"""
        **Floating strike lookback** options:
        - Call: payoff = $S_T - \min(S_t)$ (buy at the low)
        - Put: payoff = $\max(S_t) - S_T$ (sell at the high)

        Analytical formula by Goldman-Sosin-Gatto (1979) for continuous monitoring.
        """)

    if st.button("Price Lookback Option"):
        t0 = time.time()
        an = lookback_floating_analytical(S, T, r, sigma, option_type)
        t1 = time.time() - t0

        t2 = time.time()
        mc = lookback_mc(S, T, r, sigma, option_type, n_paths)
        t3 = time.time() - t2

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Analytical (Continuous)", f"${an:.4f}")
            st.caption("GSG 1979")
        with m2:
            st.metric("MC (Discrete)", f"${mc.price:.4f}")
            st.caption("500 steps")

        section_title("Price vs Volatility")
        vol_range = np.linspace(0.05, 0.80, 40)
        lb_analytical = [lookback_floating_analytical(S, T, r, v, option_type) for v in vol_range]
        lb_mc = [lookback_mc(S, T, r, v, option_type, min(n_paths, 5000)).price for v in vol_range]

        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=vol_range * 100, y=lb_analytical, mode="lines",
            name="Analytical", line=dict(color="#f7b731", width=2),
        ))
        fig_vol.add_trace(go.Scatter(
            x=vol_range * 100, y=lb_mc, mode="lines",
            name="MC", line=dict(color="#e74c3c", width=1.5, dash="dot"),
        ))
        fig_vol.update_layout(**plotly_layout(
            xaxis=dict(title="Volatility (%)", gridcolor="#2a3040", zerolinecolor="#2a3040", tickfont=dict(size=9, color="#556070"), linecolor="#2a3040"),
            yaxis=dict(title="Lookback Price ($)", gridcolor="#2a3040", zerolinecolor="#2a3040", tickfont=dict(size=9, color="#556070"), linecolor="#2a3040"),
            height=320,
        ))
        st.plotly_chart(fig_vol, use_container_width=True)
