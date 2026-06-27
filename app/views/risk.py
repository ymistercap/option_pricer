"""Risk analysis page: VaR, CVaR, and stress testing."""

import plotly.graph_objects as go
import streamlit as st

from components.theme import card_title, section_title, plotly_layout
from risk.var import (
    historical_var,
    parametric_var,
    monte_carlo_var,
    expected_shortfall,
    backtest_var,
)
from risk.scenarios import spot_stress, vol_stress, scenario_matrix


def render(params: dict) -> None:
    """Render the risk analysis page."""
    card_title("Risk Analysis — VaR & Stress Testing")

    with st.expander("Mathematical Background"):
        st.markdown(r"""
        **Value at Risk (VaR)**: Maximum loss at confidence level $\alpha$ over horizon $h$:
        $$P(\text{Loss} > \text{VaR}_\alpha) = 1 - \alpha$$

        **Expected Shortfall (CVaR)**: Average loss beyond VaR:
        $$ES_\alpha = E[\text{Loss} \mid \text{Loss} > \text{VaR}_\alpha]$$

        ES is a **coherent** risk measure (subadditive), unlike VaR.
        """)

    S, K, T, r, sigma = params["S"], params["K"], params["T"], params["r"], params["sigma"]
    option_type = params["option_type"]

    tab1, tab2 = st.tabs(["VaR Analysis", "Stress Testing"])

    with tab1:
        _render_var(params)

    with tab2:
        _render_stress(S, K, T, r, sigma, option_type)


def _render_var(params):
    ticker = st.text_input("Ticker for Historical Data", value="SPY", key="var_ticker")
    confidence = st.select_slider("Confidence Level", options=[0.90, 0.95, 0.99], value=0.95)
    horizon = st.selectbox("Horizon (days)", [1, 5, 10], index=0)

    if st.button("Compute VaR", type="primary"):
        with st.spinner("Fetching data and computing..."):
            try:
                from data.fetcher import get_returns
                returns = get_returns(ticker, "2y")
                returns_arr = returns.values

                h_var = historical_var(returns_arr, confidence, horizon)
                p_var = parametric_var(returns_arr, confidence, horizon)
                mc_var = monte_carlo_var(returns_arr, confidence, horizon)
                es = expected_shortfall(returns_arr, confidence, horizon)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Historical VaR", f"{h_var*100:.2f}%")
                col2.metric("Parametric VaR", f"{p_var*100:.2f}%")
                col3.metric("Monte Carlo VaR", f"{mc_var*100:.2f}%")
                col4.metric("Expected Shortfall", f"{es*100:.2f}%")

                section_title("Returns Distribution")
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=returns_arr * 100, nbinsx=100,
                                          name="Daily Returns",
                                          marker_color="#f7b731", marker_line_color="#2a3040", marker_line_width=0.5))
                fig.add_vline(x=-h_var * 100, line_dash="dash", line_color="#e74c3c",
                             annotation_text=f"VaR: {h_var*100:.2f}%")
                fig.add_vline(x=-es * 100, line_dash="dash", line_color="#f7b731",
                             annotation_text=f"ES: {es*100:.2f}%")
                fig.update_layout(**plotly_layout(
                    xaxis=dict(title="Daily Return (%)", gridcolor="#2a3040", zerolinecolor="#2a3040", tickfont=dict(size=9, color="#556070"), linecolor="#2a3040"),
                    yaxis=dict(title="Frequency", gridcolor="#2a3040", zerolinecolor="#2a3040", tickfont=dict(size=9, color="#556070"), linecolor="#2a3040"),
                    height=350,
                ))
                st.plotly_chart(fig, use_container_width=True)

                section_title("VaR Backtesting")
                bt = backtest_var(returns_arr, confidence)
                violation_rate = bt["violation"].mean()
                expected_rate = 1 - confidence

                st.write(f"Violation rate: {violation_rate*100:.1f}% (expected: {expected_rate*100:.1f}%)")

                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(
                    x=bt["day"], y=bt["return"] * 100,
                    mode="markers", marker=dict(
                        color=["#e74c3c" if v else "#4f8ef7" for v in bt["violation"]],
                        size=3,
                    ), name="Returns",
                ))
                fig_bt.add_trace(go.Scatter(
                    x=bt["day"], y=-bt["var"] * 100,
                    mode="lines", line=dict(color="#e74c3c", dash="dash"),
                    name=f"VaR ({confidence*100:.0f}%)",
                ))
                fig_bt.update_layout(**plotly_layout(
                    xaxis=dict(title="Day", gridcolor="#2a3040", zerolinecolor="#2a3040", tickfont=dict(size=9, color="#556070"), linecolor="#2a3040"),
                    yaxis=dict(title="Return (%)", gridcolor="#2a3040", zerolinecolor="#2a3040", tickfont=dict(size=9, color="#556070"), linecolor="#2a3040"),
                    height=350,
                ))
                st.plotly_chart(fig_bt, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")


def _render_stress(S, K, T, r, sigma, option_type):
    from pricing.black_scholes import bs_price

    st.markdown(
        '<div style="background:rgba(79,142,247,0.08);border:1px solid rgba(79,142,247,0.25);'
        "border-radius:4px;padding:10px 14px;font-family:'JetBrains Mono',monospace;"
        'font-size:11px;color:#4f8ef7;margin-bottom:14px;">'
        "Scenario matrix shows option price across combinations of spot shock and vol shock."
        "</div>",
        unsafe_allow_html=True,
    )

    col_tbl, col_heat = st.columns(2)

    with col_tbl:
        section_title("Spot Shocks")
        df_spot = spot_stress(S, K, T, r, sigma, option_type)
        rows = ""
        for _, row in df_spot.iterrows():
            shock = row["shock_pct"]
            shock_cls = "tbl-pos" if shock > 0 else ("tbl-neg" if shock < 0 else "tbl-hi")
            rows += (
                f'<tr><td class="{shock_cls}">{shock:+.0f}%</td>'
                f'<td>${row["spot"]:.2f}</td>'
                f'<td>${row["price"]:.4f}</td>'
                f'<td>{row["delta"]:.4f}</td>'
                f'<td>{row["gamma"]:.6f}</td>'
                f'<td>{row["vega"]:.4f}</td>'
                f'<td>{row["theta"]:.4f}</td></tr>'
            )
        st.markdown(
            '<table class="tbl"><thead><tr><th>Shock</th><th>Spot</th><th>Price</th>'
            f"<th>Δ</th><th>Γ</th><th>Vega</th><th>Θ</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>",
            unsafe_allow_html=True,
        )

    with col_heat:
        spots, vols, matrix = scenario_matrix(S, K, T, r, sigma, option_type)
        fig = go.Figure(go.Heatmap(
            x=[f"{s:.0f}" for s in spots],
            y=[f"{v*100:.0f}%" for v in vols],
            z=matrix,
            colorscale=[[0, "#e74c3c"], [0.5, "#f7b731"], [1, "#2ecc71"]],
            colorbar=dict(title="Price ($)", len=0.8, tickfont=dict(color="#8896a8", size=8)),
        ))
        fig.update_layout(**plotly_layout(
            xaxis=dict(title="Spot Shock (%)", gridcolor="#2a3040", zerolinecolor="#2a3040", tickfont=dict(size=9, color="#556070"), linecolor="#2a3040"),
            yaxis=dict(title="Vol Shock (%)", gridcolor="#2a3040", zerolinecolor="#2a3040", tickfont=dict(size=9, color="#556070"), linecolor="#2a3040"),
            height=320,
        ))
        st.plotly_chart(fig, use_container_width=True)

    section_title("Volatility Shocks")
    df_vol = vol_stress(S, K, T, r, sigma, option_type)
    vol_rows = ""
    for _, row in df_vol.iterrows():
        vol_rows += (
            f'<tr><td class="tbl-hi">{row["vol_shock_pts"]:.0f} pts</td>'
            f'<td>{row["sigma"]:.3f}</td>'
            f'<td>${row["price"]:.4f}</td>'
            f'<td>{row["delta"]:.4f}</td>'
            f'<td>{row["vega"]:.4f}</td></tr>'
        )
    st.markdown(
        '<table class="tbl"><thead><tr><th>Vol Shock</th><th>σ</th><th>Price</th><th>Δ</th><th>Vega</th></tr></thead>'
        f"<tbody>{vol_rows}</tbody></table>",
        unsafe_allow_html=True,
    )
