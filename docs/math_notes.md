# Mathematical Notes

## 1. Black-Scholes Model

### 1.1 Assumptions

The Black-Scholes-Merton model assumes:
1. The underlying follows a Geometric Brownian Motion (GBM): dS = μS dt + σS dW
2. Volatility σ and risk-free rate r are constant
3. No dividends, transaction costs, or taxes
4. Continuous trading is possible
5. Markets are complete (every contingent claim can be replicated)

### 1.2 Risk-Neutral Pricing

By Girsanov's theorem, under the risk-neutral measure Q:

    dS = rS dt + σS dW^Q

The price of any European contingent claim is:

    V(S, t) = e^{-r(T-t)} · E^Q[payoff(S_T) | S_t = S]

### 1.3 The PDE Approach

Construct a self-financing portfolio Π = V - ΔS. By Itô's lemma:

    dΠ = (∂V/∂t + ½σ²S²∂²V/∂S²) dt + (∂V/∂S - Δ) dS

Setting Δ = ∂V/∂S eliminates the stochastic term. No-arbitrage requires dΠ = rΠ dt, yielding the **Black-Scholes PDE**:

    ∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0

### 1.4 Closed-Form Solution

For a European call with terminal condition V(S, T) = max(S - K, 0):

    C = S·N(d₁) - K·e^{-rT}·N(d₂)

where:
    d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
    d₂ = d₁ - σ√T
    N(·) = standard normal CDF

Put price via put-call parity: P = C - S + K·e^{-rT}

---

## 2. Greeks

### 2.1 Delta: ∂V/∂S

    Δ_call = N(d₁)
    Δ_put = N(d₁) - 1

**Derivation**: Differentiating C = S·N(d₁) - K·e^{-rT}·N(d₂) w.r.t. S:

    ∂C/∂S = N(d₁) + S·n(d₁)·∂d₁/∂S - K·e^{-rT}·n(d₂)·∂d₂/∂S

Using ∂d₁/∂S = ∂d₂/∂S = 1/(Sσ√T) and the identity S·n(d₁) = K·e^{-rT}·n(d₂):

    ∂C/∂S = N(d₁)

### 2.2 Gamma: ∂²V/∂S²

    Γ = n(d₁) / (S·σ·√T)

Gamma is the same for calls and puts (since their deltas differ by a constant).

### 2.3 Vega: ∂V/∂σ

    ν = S·√T·n(d₁)

Always positive: higher vol → more optionality → higher price.

### 2.4 Theta: ∂V/∂t

    Θ_call = -S·n(d₁)·σ/(2√T) - r·K·e^{-rT}·N(d₂)
    Θ_put  = -S·n(d₁)·σ/(2√T) + r·K·e^{-rT}·N(-d₂)

Typically negative for long options (time decay).

### 2.5 Rho: ∂V/∂r

    ρ_call = K·T·e^{-rT}·N(d₂)
    ρ_put  = -K·T·e^{-rT}·N(-d₂)

### 2.6 BS PDE Verification

The Greeks satisfy: Θ + ½σ²S²Γ + rSΔ - rV = 0

This is verified numerically in our test suite.

---

## 3. Monte Carlo Methods

### 3.1 GBM Simulation

The exact solution of dS = rS dt + σS dW is:

    S(t+dt) = S(t) · exp((r - σ²/2)dt + σ√dt · Z),  Z ~ N(0,1)

This is used (not Euler-Maruyama) because it's exact for GBM.

### 3.2 Variance Reduction

**Antithetic Variates**: For each Z, also use -Z. The payoffs form a negatively correlated pair:

    V̂ = ½[f(Z) + f(-Z)]
    Var(V̂) = ½Var(f(Z))(1 + ρ),  where ρ < 0

**Control Variates**: Use a correlated variable C with known expectation:

    V̂_CV = V̂ - β(Ĉ - E[C])
    β* = Cov(V̂, Ĉ) / Var(Ĉ)

We use S_T as the control (E[S_T] = S·e^{rT} is known).

### 3.3 Convergence

By the CLT, the MC error is O(1/√N). Doubling paths reduces error by √2 ≈ 1.41.

---

## 4. Finite Difference Methods

### 4.1 Discretization

Transform to a uniform grid via x = ln(S). The BS PDE becomes:

    ∂V/∂t + ½σ²∂²V/∂x² + (r - ½σ²)∂V/∂x - rV = 0

### 4.2 Crank-Nicolson Scheme

Average of implicit and explicit:

    (V^{n+1} - V^n)/dt = ½[L·V^{n+1} + L·V^n]

This gives O(dt², dx²) accuracy and is unconditionally stable.

At each time step, solve the tridiagonal system:

    A · V^{n+1} = B · V^n + boundary terms

using the Thomas algorithm (O(N) complexity).

### 4.3 Explicit Scheme

    V^n = (I + dt·L) · V^{n+1}

O(dt, dx²) accuracy. Conditionally stable: requires dt ≤ dx²/σ² (CFL condition).

---

## 5. Implied Volatility

### 5.1 Existence and Uniqueness

Since Vega = ∂V/∂σ = S√T·n(d₁) > 0 for T > 0, the BS price is strictly monotonic in σ.
Therefore, for any market price V_mkt in the valid range, there exists a unique σ_imp such that BS(σ_imp) = V_mkt.

### 5.2 Newton-Raphson

    σ_{n+1} = σ_n - [BS(σ_n) - V_mkt] / Vega(σ_n)

Convergence is quadratic near the root. We use the Brenner-Subrahmanyam initial guess:

    σ₀ ≈ √(2π/T) · C/S

### 5.3 Brent's Method (Fallback)

Guaranteed convergence on a bracketed interval [σ_low, σ_high]. Combines bisection, secant, and inverse quadratic interpolation.

---

## 6. Exotic Options

### 6.1 Barrier Options (Reiner-Rubinstein)

For a down-and-out call with H < S:

    C_do = C_BS - (H/S)^{2λ} · C_BS(S → H²/S)

where λ = (r + σ²/2)/σ².

**In-Out Parity**: V_in + V_out = V_vanilla (one of them always activates).

### 6.2 Asian Options

The arithmetic average has no closed form. The geometric average of lognormal variables is lognormal:

    G̃ ~ LogN(μ_G·T, σ_G²·T)

where σ_G = σ·√((2n+1)/(6(n+1))).

The geometric price serves as a control variate for the arithmetic price.

### 6.3 Lookback Options (Goldman-Sosin-Gatto)

For a floating-strike lookback call (m = S at inception):

    C = S·N(a₁) - S·e^{-rT}·N(a₂) - S·η·[N(-a₁) - e^{-rT}·N(a₂)]

where η = σ²/(2r), a₁ = (r + σ²/2)√T/σ, a₂ = (r - σ²/2)√T/σ.

---

## 7. Delta-Hedging and the Gamma P&L

### 7.1 Hedging Mechanics

A trader short one call holds Δ shares. At each rebalancing:
1. Compute Δ_new = N(d₁) at current spot
2. Trade (Δ_new - Δ_old) shares
3. Finance via cash account at rate r

### 7.2 Gamma P&L Formula

The instantaneous P&L of a delta-hedged portfolio is:

    dP&L = ½ · Γ · S² · (dS/S)² - ½ · Γ · S² · σ²_impl · dt
         = ½ · Γ · S² · (σ²_real - σ²_impl) · dt

This means:
- If σ_real > σ_impl: long gamma profits, short gamma loses
- If σ_real = σ_impl: P&L → 0 as rebalancing frequency → ∞
- Discrete hedging introduces residual variance ∝ 1/n_steps

---

## 8. Value at Risk

### 8.1 Historical VaR

    VaR_α = -Quantile(returns, 1-α)

Non-parametric, captures fat tails naturally.

### 8.2 Parametric VaR

Assuming returns ~ N(μ, σ²):

    VaR_α = -(μ + z_α · σ)

where z_α = Φ^{-1}(1-α).

### 8.3 Expected Shortfall

    ES_α = E[Loss | Loss > VaR_α]

ES is coherent (subadditive), unlike VaR. ES ≥ VaR always.

### 8.4 Scaling

Under i.i.d. returns: VaR_h = VaR_1 · √h (square-root-of-time rule).

---

## 9. Binomial Trees (Cox-Ross-Rubinstein)

### 9.1 CRR Framework

Discretise time into N steps of dt = T/N. At each step the spot moves:

    u = exp(σ√dt)       — up factor
    d = 1/u = exp(-σ√dt) — down factor

The risk-neutral up-probability is:

    p = (e^{r·dt} - d) / (u - d)

The tree is recombining: after j up-moves and (n-j) down-moves,

    S_{n,j} = S · u^j · d^{n-j}

### 9.2 European Pricing

Terminal payoffs at step N: V_{N,j} = payoff(S_{N,j}). Backward induction:

    V_{n,j} = e^{-r·dt} [p·V_{n+1,j+1} + (1-p)·V_{n+1,j}]

Convergence is O(1/N) with oscillatory behaviour.

### 9.3 American Pricing

At each node, compare continuation vs early exercise:

    V_{n,j} = max(intrinsic(S_{n,j}), e^{-r·dt}[p·V_{n+1,j+1} + (1-p)·V_{n+1,j}])

For puts: V_american ≥ V_european (early exercise premium).
For calls (no dividends): V_american = V_european.

---

## 10. Merton Jump-Diffusion

### 10.1 Dynamics

    dS/S = (r - λk) dt + σ dW + J dN

where N(t) ~ Poisson(λt), J = exp(m + δZ) - 1, k = E[J] = exp(m + δ²/2) - 1.

### 10.2 Merton's Series Formula

The price is an infinite series of BS prices with adjusted parameters:

    V = Σ_{n=0}^∞  [e^{-λ'T}(λ'T)^n / n!] · BS(S, K, T, r_n, σ_n)

where:
    λ' = λ(1+k)
    r_n = r - λk + n·ln(1+k)/T
    σ_n = √(σ² + nδ²/T)

Each term weights the BS price under the scenario of exactly n jumps.

### 10.3 Put-Call Parity

Put-call parity holds under jump-diffusion since the forward relationship
S = F·e^{-rT} is preserved (jumps are compensated in the drift).

---

## 11. Heston Stochastic Volatility

### 11.1 Dynamics

    dS  = rS dt + √v S dW_S
    dv  = κ(θ - v) dt + ξ√v dW_v
    dW_S · dW_v = ρ dt

The variance follows a CIR (Cox-Ingersoll-Ross) process with:
- κ: mean-reversion speed
- θ: long-run variance
- ξ: vol-of-vol
- ρ: spot-vol correlation (typically negative, producing a skew)

Feller condition: 2κθ > ξ² ensures v > 0 a.s.

### 11.2 Semi-Analytical Pricing

    C = S·P₁ - K·e^{-rT}·P₂

where P_j = ½ + (1/π) ∫₀^∞ Re[e^{-iu·ln K} f_j(u)] / (iu) du.

f_j are the Heston characteristic functions for measures j=1,2:

    f_j(u) = exp(C_j + D_j·v₀ + iu·ln S)

With parameters (using Albrecher "little Heston trap" formulation):

    d_j = √((ρξiu - b_j)² - ξ²(2u_j·iu - u²))
    g_j = (b_j - ρξiu - d_j) / (b_j - ρξiu + d_j)

    C_j = r·iu·T + (κθ/ξ²)[(b_j - ρξiu - d_j)T - 2ln((1-g_j·e^{-d_j T})/(1-g_j))]
    D_j = ((b_j - ρξiu - d_j)/ξ²)(1 - e^{-d_j T})/(1 - g_j·e^{-d_j T})

where b₁ = κ - ρξ, u₁ = ½ and b₂ = κ, u₂ = -½.

### 11.3 Implied Vol Smile

With ρ < 0: OTM puts are more expensive (left skew), matching equity markets.
Higher ξ increases curvature (smile wings). Higher κ reduces term structure effects.

---

## 12. Carr-Madan FFT Pricing

### 12.1 The Carr-Madan Formula

For a damped call price c_T(k) = e^{αk} C(e^k) where k = ln K:

    c_T(k) = (e^{-αk}/π) ∫₀^∞ e^{-ivk} ψ(v) dv

where:
    ψ(v) = e^{-rT} φ(v - (α+1)i) / (α² + α - v² + i(2α+1)v)

φ(u) is the characteristic function of ln(S_T).

### 12.2 FFT Implementation

Discretise v_j = j·η (j=0,...,N-1) and k_m = -b + m·λ where λη = 2π/N:

    c_T(k_m) ≈ (e^{-αk_m}/π) · Σ_j e^{-iv_j k_m} ψ(v_j) w_j

The sum is a DFT, computed in O(N log N) via FFT.

Simpson weights: w_j = (η/3)(3 + (-1)^{j+1}), w₀ = η/3.

### 12.3 Advantages

Prices O(N) strikes simultaneously, making it ideal for calibration
where many strikes are evaluated per objective function call.

---

## 13. Dupire Local Volatility

### 13.1 Dupire's Formula

The unique deterministic volatility σ_loc(K,T) consistent with all
European option prices satisfies:

    σ²_loc(K,T) = 2(∂C/∂T + rK·∂C/∂K) / (K²·∂²C/∂K²)

In terms of implied volatility σ_imp(K,T):

    σ²_loc = (∂w/∂T) / [1 - (y/w)∂w/∂y + ¼(-¼ - 1/w + y²/w²)(∂w/∂y)² + ½·∂²w/∂y²]

where w = σ²_imp·T and y = ln(K/F).

### 13.2 Practical Computation

Derivatives are estimated by finite differences on the discrete IV grid.
Boundary effects and noise in market data require smoothing and
regularisation.

### 13.3 Properties

- Flat IV surface → σ_loc = σ_imp (constant, reducing to BS).
- σ_loc(K,T) > 0 whenever the IV surface is arbitrage-free.
- Local vol always overestimates the curvature of the smile.

---

## 14. SABR Model

### 14.1 Dynamics

    dF = α F^β dW₁
    dα = ν α dW₂
    dW₁·dW₂ = ρ dt

where F is the forward price.

### 14.2 Hagan's Approximation

The implied BS vol for strike K is (to leading order in ν):

For F ≠ K:
    σ_B(K) = [α / ((FK)^{(1-β)/2} · (1 + (1-β)²/24 · (ln F/K)² + ...))]
             · [z/x(z)]
             · [1 + ((1-β)²/24 · α²/(FK)^{1-β} + ρβνα/(4(FK)^{(1-β)/2})
                   + (2-3ρ²)/24 · ν²) · T]

where:
    z = (ν/α)(FK)^{(1-β)/2} · ln(F/K)
    x(z) = ln[(√(1-2ρz+z²) + z - ρ) / (1-ρ)]

For F = K (ATM):
    σ_B = [α/F^{1-β}] · [1 + ((1-β)²/24 · α²/F^{2-2β} + ρβνα/(4F^{1-β})
                              + (2-3ρ²)/24 · ν²) · T]

### 14.3 Calibration

For a fixed β, calibrate (α, ρ, ν) by minimising:

    Σ_i (σ_SABR(K_i; α, ρ, ν) - σ_market(K_i))²

β is typically fixed at 0 (normal), 0.5, or 1 (log-normal).

---

## 15. Model Calibration

### 15.1 Objective

Given N market implied vols σ^mkt_i at strikes K_i and maturities T_i:

    min_θ  RMSE = √((1/N) Σ_i (σ_model(K_i,T_i;θ) - σ^mkt_i)²)

### 15.2 Algorithm

1. **Global search**: Differential evolution explores the parameter space
   without gradient information, avoiding local minima.
2. **Local refinement**: Nelder-Mead polishes the global optimum.

### 15.3 Heston Calibration

Parameters: θ = (v₀, κ, θ, ξ, ρ). For each candidate:
1. Compute Heston price via semi-analytical formula
2. Invert to implied vol via Newton-Raphson
3. Compare to market vol

### 15.4 Merton Calibration

Parameters: θ = (σ, λ, m, δ). Similar workflow using the analytical
series formula.
