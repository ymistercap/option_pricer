"""Volatility surface page with real market data."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.theme import card_title, section_title, plotly_layout
from volatility.implied_vol import implied_vol
from volatility.surface import build_surface


def render(params: dict) -> None:
    """Render the volatility surface page."""
    card_title("Implied Volatility Surface")

    with st.expander("Mathematical Background"):
        st.markdown(r"""
        The **implied volatility** $\sigma_{imp}$ is the unique value solving:
        $$BS(S, K, T, r, \sigma_{imp}) = V_{market}$$

        In a pure BS world, $\sigma_{imp}$ would be constant. In reality:
        - **Smile**: OTM puts and calls have higher IV than ATM (fat tails)
        - **Skew**: OTM puts have higher IV than OTM calls (crash protection demand)
        - **Term structure**: ATM vol varies across maturities
        """)

    ticker = st.text_input("Ticker Symbol", value="AAPL")
    fetch_data = st.button("Fetch Option Data", type="primary")

    if fetch_data:
        with st.spinner(f"Fetching option chain for {ticker}..."):
            try:
                import yfinance as yf
                from datetime import datetime

                stock = yf.Ticker(ticker)
                S = float(stock.history(period="1d")["Close"].iloc[-1])
                r = params["r"]

                st.info(f"Current spot price: ${S:.2f}")

                expirations = list(stock.options)
                if not expirations:
                    st.error("No option data available.")
                    return

                all_strikes, all_mats, all_ivs = [], [], []
                smiles = {}

                progress = st.progress(0)
                for idx, exp in enumerate(expirations[:10]):
                    try:
                        chain = stock.option_chain(exp)
                        exp_date = datetime.strptime(exp, "%Y-%m-%d")
                        T = max((exp_date - datetime.now()).days / 365.0, 0.01)

                        calls = chain.calls
                        calls = calls[(calls["volume"] > 10) & (calls["bid"] > 0) & (calls["ask"] > 0)]
                        calls["mid"] = (calls["bid"] + calls["ask"]) / 2.0
                        calls["moneyness"] = calls["strike"] / S

                        calls = calls[(calls["moneyness"] > 0.8) & (calls["moneyness"] < 1.2)]

                        smile_data = []
                        for _, row in calls.iterrows():
                            iv = implied_vol(row["mid"], S, row["strike"], T, r, "call")
                            if iv is not None and 0.01 < iv < 2.0:
                                all_strikes.append(row["strike"] / S)
                                all_mats.append(T)
                                all_ivs.append(iv)
                                smile_data.append({"moneyness": row["strike"] / S, "implied_vol": iv})

                        if smile_data:
                            smiles[f"{exp} (T={T:.2f}y)"] = pd.DataFrame(smile_data)
                    except Exception:
                        continue

                    progress.progress((idx + 1) / min(len(expirations), 10))

                if not all_ivs:
                    st.error("Could not compute implied volatilities. Try a different ticker.")
                    return

                section_title("Volatility Smiles")
                fig_smile = go.Figure()
                for label, df in smiles.items():
                    fig_smile.add_trace(go.Scatter(
                        x=df["moneyness"], y=df["implied_vol"] * 100,
                        mode="lines+markers", name=label,
                    ))
                fig_smile.update_layout(**plotly_layout(
                    xaxis=dict(title="Moneyness (K/S)", gridcolor="#2a3040", zerolinecolor="#2a3040", tickfont=dict(size=9, color="#556070"), linecolor="#2a3040"),
                    yaxis=dict(title="Implied Volatility (%)", gridcolor="#2a3040", zerolinecolor="#2a3040", tickfont=dict(size=9, color="#556070"), linecolor="#2a3040"),
                    height=400,
                ))
                st.plotly_chart(fig_smile, use_container_width=True)

                if len(all_ivs) > 20:
                    section_title("3D Volatility Surface")
                    K_grid, T_grid, IV_grid = build_surface(
                        np.array(all_strikes), np.array(all_mats), np.array(all_ivs),
                        n_strike_grid=40, n_maturity_grid=40,
                    )

                    fig_3d = go.Figure(data=[go.Surface(
                        x=K_grid, y=T_grid, z=IV_grid * 100,
                        colorscale=[[0, "#0d2137"], [0.2, "#1a4f7a"], [0.4, "#2980b9"], [0.6, "#00d4aa"], [0.8, "#f7b731"], [1, "#e74c3c"]],
                        colorbar=dict(title="IV (%)", len=0.5, thickness=12, tickfont=dict(color="#8896a8", size=8)),
                    )])
                    fig_3d.update_layout(**plotly_layout(
                        scene=dict(
                            xaxis_title="Moneyness (K/S)",
                            yaxis_title="Maturity (years)",
                            zaxis_title="Implied Vol (%)",
                        ),
                        height=600,
                    ))
                    st.plotly_chart(fig_3d, use_container_width=True)

                section_title("ATM Term Structure")
                atm_data = []
                for label, df in smiles.items():
                    atm_idx = (df["moneyness"] - 1.0).abs().idxmin()
                    T_val = float(label.split("T=")[1].rstrip("y)"))
                    atm_data.append({"maturity": T_val, "atm_iv": df.loc[atm_idx, "implied_vol"] * 100})

                if atm_data:
                    atm_df = pd.DataFrame(atm_data).sort_values("maturity")
                    fig_term = go.Figure(go.Scatter(
                        x=atm_df["maturity"], y=atm_df["atm_iv"],
                        mode="lines+markers",
                        line=dict(color="#f7b731", width=2),
                        marker=dict(color="#f7b731", size=6),
                    ))
                    fig_term.update_layout(**plotly_layout(
                        xaxis=dict(title="Maturity (years)", gridcolor="#2a3040", zerolinecolor="#2a3040", tickfont=dict(size=9, color="#556070"), linecolor="#2a3040"),
                        yaxis=dict(title="ATM Implied Vol (%)", gridcolor="#2a3040", zerolinecolor="#2a3040", tickfont=dict(size=9, color="#556070"), linecolor="#2a3040"),
                        height=350,
                    ))
                    st.plotly_chart(fig_term, use_container_width=True)

            except Exception as e:
                st.error(f"Error fetching data: {e}")
