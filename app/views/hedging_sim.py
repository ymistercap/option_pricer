"""Delta-hedging simulation page."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.theme import card_title, section_title, plotly_layout
from hedging.delta_hedge import simulate_delta_hedge, hedge_frequency_analysis


def render(params: dict) -> None:
    """Render the hedging simulation page."""
    card_title("Delta-Hedging Simulation")

    with st.expander("Mathematical Background"):
        st.markdown(r"""
        **Delta-hedging** replicates an option by holding $\Delta$ shares of the underlying.

        The residual P&L of a delta-hedged portfolio is driven by **gamma**:
        $$P\&L \approx \frac{1}{2} \Gamma S^2 (\sigma_{realized}^2 - \sigma_{implied}^2) \, dt$$

        - If $\sigma_{real} = \sigma_{impl}$: P&L $\to 0$ as rebalancing frequency $\to \infty$
        - If $\sigma_{real} > \sigma_{impl}$: short gamma **loses** on average
        - If $\sigma_{real} < \sigma_{impl}$: short gamma **profits** on average
        """)

    S, K, T, r, sigma = params["S"], params["K"], params["T"], params["r"], params["sigma"]

    col1, col2 = st.columns(2)
    with col1:
        sigma_real_pct = st.slider(
            "Realized Volatility",
            min_value=5,
            max_value=80,
            value=int(sigma * 100),
            step=1,
            format="%d%%",
        )
        sigma_real = sigma_real_pct / 100.0
        n_paths = st.number_input("Number of Paths", value=2000, min_value=100, max_value=50000, step=500)
    with col2:
        n_steps = st.slider(
            "Rebalancing Steps",
            min_value=10,
            max_value=2520,
            value=252,
            step=10,
        )
        tx_cost = st.number_input("Transaction Cost (bps)", value=0, min_value=0, max_value=100, step=5)

    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;color:#556070;">'
        f'σ<sub>implied</sub> = {sigma:.2f} &nbsp;→&nbsp; σ<sub>realized</sub> = {sigma_real:.2f}'
        f'</div>',
        unsafe_allow_html=True,
    )

    diff = abs(sigma_real - sigma)
    if diff < 0.02:
        chip_cls, chip_text = "chip-green", "BALANCED"
    elif sigma_real > sigma:
        chip_cls, chip_text = "chip-red", "SHORT GAMMA RISK"
    else:
        chip_cls, chip_text = "chip-amber", "POSITIVE CARRY"

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        compute = st.button("Run Simulation", type="primary")
    with col_info:
        st.markdown(
            f'<div class="param-summary">'
            f'σ_impl={sigma} &nbsp; σ_real={sigma_real} &nbsp; steps={int(n_steps)}'
            f'&nbsp;&nbsp;<span class="chip {chip_cls}">{chip_text}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if compute:
        with st.spinner("Simulating..."):
            result = simulate_delta_hedge(
                S, K, T, r, sigma, sigma_real,
                n_paths=int(n_paths), n_steps=int(n_steps),
                transaction_cost=tx_cost / 10000.0,
            )

        sharpe = f"{result.mean_pnl / result.std_pnl:.3f}" if result.std_pnl > 0 else "N/A"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean P&L", f"${result.mean_pnl:.4f}")
        m2.metric("P&L Std Dev", f"${result.std_pnl:.4f}")
        m3.metric("Sharpe (P&L)", sharpe)
        m4.metric("Implied σ", f"{sigma:.2f}", delta=f"realized {sigma_real:.2f}")

        t_grid = np.linspace(0, T, int(n_steps) + 1)
        n_show = min(50, result.paths.shape[0])

        section_title("Simulation Detail")
        r1c1, r1c2 = st.columns(2)

        with r1c1:
            fig_paths = go.Figure()
            for i in range(n_show):
                fig_paths.add_trace(go.Scatter(
                    x=t_grid, y=result.paths[i],
                    mode="lines",
                    line=dict(color="#f7b731", width=0.5),
                    opacity=0.25,
                    showlegend=False,
                    hoverinfo="skip",
                ))
            fig_paths.add_hline(y=K, line_dash="dash", line_color="#e74c3c",
                                annotation_text=f"K={K}",
                                annotation_font_color="#e74c3c",
                                annotation_font_size=9)
            fig_paths.update_layout(**plotly_layout(
                title=dict(text="Simulated Paths", font=dict(size=11, color="#8896a8")),
                xaxis=dict(title="Time (years)"),
                yaxis=dict(title="Spot Price"),
                height=340,
            ))
            st.plotly_chart(fig_paths, use_container_width=True)

        with r1c2:
            fig_delta = go.Figure()
            fig_delta.add_trace(go.Scatter(
                x=t_grid, y=result.deltas,
                mode="lines",
                line=dict(color="#00d4aa", width=1.5),
                name="Delta",
            ))
            fig_delta.update_layout(**plotly_layout(
                title=dict(text="Delta Hedge Ratio", font=dict(size=11, color="#8896a8")),
                xaxis=dict(title="Time (years)"),
                yaxis=dict(title="Delta (Δ)"),
                height=340,
                showlegend=False,
            ))
            st.plotly_chart(fig_delta, use_container_width=True)

        r2c1, r2c2 = st.columns(2)

        with r2c1:
            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Scatter(
                x=t_grid, y=result.hedge_pnl_path,
                mode="lines",
                line=dict(color="#f7b731", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(247,183,49,0.08)",
                name="Cumulative P&L",
            ))
            fig_pnl.add_hline(y=0, line_dash="dash", line_color="#2a3040")
            fig_pnl.update_layout(**plotly_layout(
                title=dict(text="Hedging P&L", font=dict(size=11, color="#8896a8")),
                xaxis=dict(title="Time (years)"),
                yaxis=dict(title="P&L ($)"),
                height=340,
                showlegend=False,
            ))
            st.plotly_chart(fig_pnl, use_container_width=True)

        with r2c2:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=result.pnl,
                nbinsx=80,
                marker_color="#f7b731",
                marker_line_color="#2a3040",
                marker_line_width=0.5,
            ))
            fig_hist.add_vline(x=0, line_dash="dash", line_color="#e74c3c")
            fig_hist.add_vline(
                x=result.mean_pnl,
                line_dash="dash",
                line_color="#2ecc71",
                annotation_text=f"Mean: {result.mean_pnl:.2f}",
                annotation_font_color="#2ecc71",
                annotation_font_size=9,
            )
            fig_hist.update_layout(**plotly_layout(
                title=dict(text="P&L Distribution", font=dict(size=11, color="#8896a8")),
                xaxis=dict(title="P&L ($)"),
                yaxis=dict(title="Frequency"),
                height=340,
            ))
            st.plotly_chart(fig_hist, use_container_width=True)

    section_title("Hedging Frequency Analysis")
    if st.button("Run Frequency Analysis"):
        with st.spinner("Running frequency sweep..."):
            df = hedge_frequency_analysis(S, K, T, r, sigma, n_paths=2000)

        fig_freq = go.Figure()
        fig_freq.add_trace(go.Scatter(
            x=df["n_steps"], y=df["std_pnl"],
            mode="lines+markers", name="P&L Std Dev",
            line=dict(color="#f7b731", width=2),
            marker=dict(color="#f7b731", size=5),
        ))
        fig_freq.update_layout(**plotly_layout(
            xaxis=dict(title="Rebalancing Steps"),
            yaxis=dict(title="P&L Standard Deviation ($)"),
            height=350,
        ))
        st.plotly_chart(fig_freq, use_container_width=True)

        rows = "".join(
            f'<tr><td class="tbl-hi">{int(row["n_steps"])}</td>'
            f'<td>${row["mean_pnl"]:.4f}</td>'
            f'<td>${row["std_pnl"]:.4f}</td></tr>'
            for _, row in df.iterrows()
        )
        st.markdown(
            '<table class="tbl"><thead><tr><th>Steps</th><th>Mean P&L</th><th>Std P&L</th></tr></thead>'
            f"<tbody>{rows}</tbody></table>",
            unsafe_allow_html=True,
        )
