"""Pricer page: compare BS, MC, and FD pricing methods."""

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.theme import card_title, section_title, plotly_layout
from pricing.black_scholes import bs_price
from pricing.monte_carlo import mc_european, mc_antithetic, mc_control_variate
from pricing.finite_difference import fd_price


def render(params: dict) -> None:
    """Render the pricer comparison page."""
    card_title("Option Pricer — Method Comparison")

    with st.expander("Mathematical Background"):
        st.markdown(r"""
        **Black-Scholes Formula** (closed-form):
        $$C = S \cdot N(d_1) - K e^{-rT} N(d_2)$$
        where $d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}$, $d_2 = d_1 - \sigma\sqrt{T}$

        **Monte Carlo**: Simulate $N$ paths of $S_T = S \exp\left((r-\frac{\sigma^2}{2})T + \sigma\sqrt{T}Z\right)$,
        then $V = e^{-rT} \cdot \frac{1}{N}\sum \text{payoff}(S_T^i)$

        **Finite Differences** (Crank-Nicolson): Solve the BS PDE on a grid.
        """)

    S, K, T, r, sigma = params["S"], params["K"], params["T"], params["r"], params["sigma"]
    option_type = params["option_type"]
    n_paths = params["n_paths"]

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        compute = st.button("Price Option", type="primary")
    with col_info:
        st.markdown(
            f'<div class="param-summary">'
            f'S={S:.0f} K={K:.0f} T={T}y r={r} σ={sigma} {option_type.upper()}'
            f'</div>',
            unsafe_allow_html=True,
        )

    if not compute:
        return

    # --- Compute prices ---
    t0 = time.time()
    bs = bs_price(S, K, T, r, sigma, option_type)
    t_bs = time.time() - t0

    t0 = time.time()
    mc = mc_control_variate(S, K, T, r, sigma, n_paths, option_type)
    t_mc = time.time() - t0

    t0 = time.time()
    fd = fd_price(S, K, T, r, sigma, option_type, n_S=300, n_t=300)
    t_fd = time.time() - t0

    diff = mc.price - bs

    # --- 4 metric cards ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Black-Scholes", f"${bs:.4f}")
        st.caption(f"<1 ms · analytical")
    with c2:
        st.metric("Monte Carlo", f"${mc.price:.4f}")
        st.caption(f"SE: {mc.std_error:.4f} · ~{t_mc*1000:.0f}ms")
    with c3:
        st.metric("Finite Diff (CN)", f"${fd:.4f}")
        st.caption(f"300×300 grid · ~{t_fd*1000:.0f}ms")
    with c4:
        diff_str = f"+${diff:.4f}" if diff >= 0 else f"-${abs(diff):.4f}"
        st.metric("BS vs MC Diff", diff_str)
        st.caption("convergence")

    # --- Two charts side by side ---
    col_left, col_right = st.columns(2)

    with col_left:
        strikes = np.linspace(S * 0.7, S * 1.3, 31)
        prices = [bs_price(S, k, T, r, sigma, option_type) for k in strikes]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=strikes, y=prices, name="BS Price",
            line=dict(color="#f7b731", width=2),
        ))
        fig.update_layout(**plotly_layout(
            title=dict(text="Price vs Strike"),
            xaxis=dict(title="Strike"),
            yaxis=dict(title="Price ($)"),
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        n_sim_paths = 12
        n_steps = 60
        dt = T / n_steps
        t_grid = np.linspace(0, T, n_steps + 1)
        rng = np.random.default_rng(42)
        paths = np.zeros((n_sim_paths, n_steps + 1))
        paths[:, 0] = S
        for i in range(n_steps):
            z = rng.standard_normal(n_sim_paths)
            paths[:, i + 1] = paths[:, i] * np.exp(
                (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
            )
        colors = [
            "#f7b731", "#00d4aa", "#2ecc71", "#4f8ef7", "#c084fc", "#e74c3c",
            "#ff9f43", "#a29bfe", "#fd79a8", "#55efc4", "#74b9ff", "#e17055",
        ]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=[0, T], y=[K, K], mode="lines",
            line=dict(color="#444", dash="dot", width=1), showlegend=False,
        ))
        for j in range(n_sim_paths):
            fig2.add_trace(go.Scatter(
                x=t_grid, y=paths[j], mode="lines",
                line=dict(color=colors[j % len(colors)], width=1.5),
                showlegend=False,
            ))
        fig2.update_layout(**plotly_layout(
            title=dict(text="MC Path Simulation"),
            xaxis=dict(title="Time (years)"),
            yaxis=dict(title="Spot Price S_T"),
        ))
        st.plotly_chart(fig2, use_container_width=True)

    # --- Variance Reduction table ---
    section_title("Monte Carlo — Variance Reduction")
    mc_std = mc_european(S, K, T, r, sigma, n_paths, option_type)
    mc_anti = mc_antithetic(S, K, T, r, sigma, n_paths, option_type)
    mc_cv = mc

    vr_anti = (mc_std.std_error / mc_anti.std_error) ** 2 if mc_anti.std_error > 0 else 1.0
    vr_cv = (mc_std.std_error / mc_cv.std_error) ** 2 if mc_cv.std_error > 0 else 1.0

    st.markdown(f'''
    <table class="tbl">
    <thead><tr><th>Method</th><th>Price</th><th>Std Error</th><th>Variance Reduction</th></tr></thead>
    <tbody>
    <tr><td class="tbl-hi">Standard MC</td><td>${mc_std.price:.4f}</td><td>{mc_std.std_error:.6f}</td><td class="tbl-pos">1.00×</td></tr>
    <tr><td class="tbl-hi">Antithetic Variates</td><td>${mc_anti.price:.4f}</td><td>{mc_anti.std_error:.6f}</td><td class="tbl-pos">{vr_anti:.2f}×</td></tr>
    <tr><td class="tbl-hi">Control Variate</td><td>${mc_cv.price:.4f}</td><td>{mc_cv.std_error:.6f}</td><td class="tbl-pos">{vr_cv:.2f}×</td></tr>
    </tbody>
    </table>
    ''', unsafe_allow_html=True)

    # --- Benchmark table ---
    section_title("Benchmark vs Reference")
    diff_mc = mc.price - bs
    diff_fd = fd - bs

    st.markdown(f'''
    <table class="tbl">
    <thead><tr><th>Method</th><th>Price</th><th>Diff vs BS</th><th>Time</th></tr></thead>
    <tbody>
    <tr><td class="tbl-hi">Black-Scholes</td><td>${bs:.6f}</td><td>—</td><td style="color:#556070">&lt;1 ms</td></tr>
    <tr><td class="tbl-hi">Monte Carlo ({n_paths // 1000}K paths)</td><td>${mc.price:.6f}</td>
        <td class="{"tbl-pos" if diff_mc >= 0 else "tbl-neg"}">{diff_mc:+.6f}</td>
        <td style="color:#556070">~{t_mc*1000:.0f} ms</td></tr>
    <tr><td class="tbl-hi">Finite Differences (CN)</td><td>${fd:.6f}</td>
        <td class="{"tbl-pos" if diff_fd >= 0 else "tbl-neg"}">{diff_fd:+.6f}</td>
        <td style="color:#556070">~{t_fd*1000:.0f} ms</td></tr>
    </tbody>
    </table>
    ''', unsafe_allow_html=True)
