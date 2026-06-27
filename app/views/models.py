"""Advanced Models page: Heston, Merton, SABR, Binomial, FFT pricing."""

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.theme import card_title, section_title, plotly_layout
from pricing.binomial_tree import crr_american, crr_european
from pricing.black_scholes import bs_price
from pricing.fourier import carr_madan_bs_price
from pricing.heston import heston_mc, heston_price
from pricing.jump_diffusion import merton_price
from models.sabr import sabr_smile


def render(params: dict) -> None:
    """Render the advanced models page."""
    card_title("Advanced Pricing Models")

    S, K, T, r, sigma = params["S"], params["K"], params["T"], params["r"], params["sigma"]
    option_type = params["option_type"]

    tab_binom, tab_heston, tab_merton, tab_sabr, tab_fft = st.tabs(
        ["Binomial Tree", "Heston", "Merton", "SABR", "FFT"]
    )

    with tab_binom:
        _render_binomial(S, K, T, r, sigma, option_type)

    with tab_heston:
        _render_heston(S, K, T, r, option_type)

    with tab_merton:
        _render_merton(S, K, T, r, sigma, option_type)

    with tab_sabr:
        _render_sabr(S, T, r, sigma)

    with tab_fft:
        _render_fft(S, K, T, r, sigma, option_type)


def _render_binomial(S, K, T, r, sigma, option_type):
    with st.expander("Mathematical Background"):
        st.markdown(r"""
        **Cox-Ross-Rubinstein (CRR) Binomial Tree**

        At each time step $\Delta t = T/N$:
        $$u = e^{\sigma\sqrt{\Delta t}}, \quad d = \frac{1}{u}, \quad p = \frac{e^{r\Delta t} - d}{u - d}$$

        The option price is computed by backward induction through the recombining tree.
        European options use discounted expectation; American options allow early exercise at each node.
        """)

    section_title("Parameters")
    n_steps = st.slider("Tree Steps", 50, 1000, 200, 50, key="binom_steps")

    if st.button("Price (Binomial)", key="btn_binom"):
        section_title("Pricing Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            eu = crr_european(S, K, T, r, sigma, option_type, n_steps)
            st.metric("European (CRR)", f"${eu:.4f}")
        with col2:
            am = crr_american(S, K, T, r, sigma, option_type, n_steps)
            st.metric("American (CRR)", f"${am:.4f}")
        with col3:
            bs = bs_price(S, K, T, r, sigma, option_type)
            st.metric("BS Reference", f"${bs:.4f}")

        if option_type == "put":
            st.info(f"Early exercise premium: ${am - eu:.4f}")

        section_title("Convergence Plot")
        steps_range = list(range(20, 501, 20))
        eu_prices = [crr_european(S, K, T, r, sigma, option_type, n) for n in steps_range]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=steps_range, y=eu_prices, name="CRR European",
                                 line=dict(color="#f7b731", width=2)))
        fig.add_hline(y=bs_price(S, K, T, r, sigma, option_type), line_dash="dash",
                     line_color="#00d4aa", annotation_text="BS")
        fig.update_layout(**plotly_layout(
            title=dict(text="Convergence"),
            xaxis=dict(title="Steps", gridcolor="#2a3040", zerolinecolor="#2a3040", tickfont=dict(size=9, color="#556070"), linecolor="#2a3040"),
            yaxis=dict(title="Price", gridcolor="#2a3040", zerolinecolor="#2a3040", tickfont=dict(size=9, color="#556070"), linecolor="#2a3040"),
        ))
        st.plotly_chart(fig, use_container_width=True)


def _render_heston(S, K, T, r, option_type):
    with st.expander("Mathematical Background"):
        st.markdown(r"""
        **Heston Stochastic Volatility Model**

        The asset and variance follow the coupled SDEs:
        $$dS_t = r S_t \, dt + \sqrt{v_t} \, S_t \, dW_t^S$$
        $$dv_t = \kappa(\theta - v_t) \, dt + \xi \sqrt{v_t} \, dW_t^v$$

        where $\text{Corr}(dW^S, dW^v) = \rho$.

        Semi-analytical pricing uses the characteristic function and numerical integration.
        """)

    section_title("Parameters")
    col1, col2 = st.columns(2)
    with col1:
        v0 = st.number_input("v0 (initial var)", 0.01, 1.0, 0.04, 0.01, key="h_v0")
        kappa = st.number_input("kappa (mean-rev)", 0.1, 10.0, 2.0, 0.1, key="h_kappa")
        theta = st.number_input("theta (long-run var)", 0.01, 1.0, 0.04, 0.01, key="h_theta")
    with col2:
        xi = st.number_input("xi (vol-of-vol)", 0.01, 2.0, 0.3, 0.05, key="h_xi")
        rho = st.number_input("rho (correlation)", -0.99, 0.99, -0.7, 0.05, key="h_rho")

    if st.button("Price (Heston)", key="btn_heston"):
        section_title("Pricing Results")
        t0 = time.time()
        price = heston_price(S, K, T, r, v0, kappa, theta, xi, rho, option_type)
        t_anal = time.time() - t0

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Semi-Analytical", f"${price:.4f}")
            st.caption(f"Computed in {t_anal*1000:.1f} ms")
        with col2:
            t0 = time.time()
            mc = heston_mc(S, K, T, r, v0, kappa, theta, xi, rho, option_type, n_paths=50_000, n_steps=100)
            t_mc = time.time() - t0
            st.metric("Monte Carlo", f"${mc.price:.4f}")
            st.caption(f"SE: {mc.std_error:.4f} | {t_mc*1000:.0f} ms")

        section_title("Price vs Strike")
        strikes = np.linspace(S * 0.8, S * 1.2, 21)
        heston_calls = [heston_price(S, k, T, r, v0, kappa, theta, xi, rho, "call") for k in strikes]
        bs_calls = [bs_price(S, k, T, r, np.sqrt(v0), "call") for k in strikes]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=strikes, y=heston_calls, name="Heston",
                                 line=dict(color="#f7b731", width=2)))
        fig.add_trace(go.Scatter(x=strikes, y=bs_calls, name="BS (flat vol)",
                                 line=dict(color="#00d4aa", dash="dash", width=2)))
        fig.update_layout(**plotly_layout(
            title=dict(text="Price vs Strike"),
            xaxis=dict(title="Strike", gridcolor="#2a3040", zerolinecolor="#2a3040", tickfont=dict(size=9, color="#556070"), linecolor="#2a3040"),
            yaxis=dict(title="Call Price", gridcolor="#2a3040", zerolinecolor="#2a3040", tickfont=dict(size=9, color="#556070"), linecolor="#2a3040"),
        ))
        st.plotly_chart(fig, use_container_width=True)


def _render_merton(S, K, T, r, sigma, option_type):
    with st.expander("Mathematical Background"):
        st.markdown(r"""
        **Merton Jump-Diffusion Model**

        The asset dynamics include a Poisson jump process:
        $$\frac{dS_t}{S_{t^-}} = (r - \lambda \bar{k}) \, dt + \sigma \, dW_t + (e^J - 1) \, dN_t$$

        where $N_t$ is a Poisson process with intensity $\lambda$,
        and $J \sim \mathcal{N}(m, \delta^2)$ is the log-jump size.

        Analytical pricing uses a series expansion over the number of jumps.
        """)

    section_title("Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        lam = st.number_input("lambda (jump freq)", 0.0, 10.0, 1.0, 0.1, key="m_lam")
    with col2:
        m = st.number_input("m (mean log-jump)", -1.0, 0.5, -0.05, 0.01, key="m_m")
    with col3:
        delta = st.number_input("delta (std log-jump)", 0.01, 1.0, 0.1, 0.01, key="m_delta")

    if st.button("Price (Merton)", key="btn_merton"):
        section_title("Pricing Results")
        t0 = time.time()
        price = merton_price(S, K, T, r, sigma, lam, m, delta, option_type)
        t_anal = time.time() - t0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Analytical", f"${price:.4f}")
            st.caption(f"{t_anal*1000:.1f} ms")
        with col2:
            bs = bs_price(S, K, T, r, sigma, option_type)
            st.metric("BS Reference", f"${bs:.4f}")
        with col3:
            st.metric("Jump Premium", f"${price - bs:.4f}")


def _render_sabr(S, T, r, sigma):
    with st.expander("Mathematical Background"):
        st.markdown(r"""
        **SABR Stochastic Alpha Beta Rho Model**

        The forward rate and volatility follow:
        $$dF_t = \alpha_t F_t^\beta \, dW_t^F$$
        $$d\alpha_t = \nu \alpha_t \, dW_t^\alpha$$

        where $\text{Corr}(dW^F, dW^\alpha) = \rho$.

        Hagan's asymptotic expansion gives an analytical implied volatility smile.
        """)

    section_title("Parameters")
    F = S * np.exp(r * T)
    col1, col2 = st.columns(2)
    with col1:
        beta = st.slider("beta", 0.0, 1.0, 0.5, 0.1, key="sabr_beta")
        alpha_default = min(sigma * F ** (1 - beta), 5.0)
        alpha = st.number_input("alpha", 0.01, 5.0, alpha_default, 0.01, key="sabr_alpha")
    with col2:
        rho = st.number_input("rho", -0.99, 0.99, -0.3, 0.05, key="sabr_rho")
        nu = st.number_input("nu (vol-of-vol)", 0.0, 2.0, 0.4, 0.05, key="sabr_nu")

    section_title("SABR Smile")
    strikes = np.linspace(S * 0.7, S * 1.3, 31)
    vols = sabr_smile(F, strikes, T, alpha, beta, rho, nu)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strikes, y=vols * 100, name="SABR Smile",
                             line=dict(color="#f7b731", width=2)))
    fig.add_vline(x=F, line_dash="dash", line_color="#2a3040", annotation_text="Forward")
    fig.update_layout(**plotly_layout(
        title=dict(text="SABR Implied Vol Smile"),
        xaxis=dict(title="Strike", gridcolor="#2a3040", zerolinecolor="#2a3040", tickfont=dict(size=9, color="#556070"), linecolor="#2a3040"),
        yaxis=dict(title="Implied Vol (%)", gridcolor="#2a3040", zerolinecolor="#2a3040", tickfont=dict(size=9, color="#556070"), linecolor="#2a3040"),
    ))
    st.plotly_chart(fig, use_container_width=True)


def _render_fft(S, K, T, r, sigma, option_type):
    with st.expander("Mathematical Background"):
        st.markdown(r"""
        **Carr-Madan FFT Pricing**

        The call price is recovered via the Fourier transform of a modified payoff:
        $$C(K) = \frac{e^{-\alpha \ln K}}{\pi} \int_0^\infty e^{-iv \ln K} \frac{e^{-rT} \phi_T(v - (\alpha+1)i)}{\alpha^2 + \alpha - v^2 + i(2\alpha+1)v} \, dv$$

        where $\phi_T$ is the characteristic function of $\ln S_T$.

        Discretisation and the FFT yield prices for an entire strike grid in $O(N \log N)$.
        """)

    st.markdown(
        '<div style="background:rgba(247,183,49,0.08); border:1px solid rgba(247,183,49,0.25); '
        'border-radius:4px; padding:12px 16px; margin-bottom:16px; '
        'font-family:\'JetBrains Mono\',monospace; font-size:11px; color:#f7b731;">'
        'FFT pricing prices a full strike grid (1024 strikes) in a single O(N log N) pass.'
        '</div>',
        unsafe_allow_html=True,
    )

    if st.button("Price (FFT)", key="btn_fft"):
        section_title("Pricing Results")
        t0 = time.time()
        fft_price = carr_madan_bs_price(S, K, T, r, sigma, option_type)
        t_fft = time.time() - t0
        bs = bs_price(S, K, T, r, sigma, option_type)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("FFT Price", f"${fft_price:.4f}")
            st.caption(f"{t_fft*1000:.1f} ms")
        with col2:
            st.metric("BS Reference", f"${bs:.4f}")
        with col3:
            st.metric("Difference", f"${fft_price - bs:.6f}")
