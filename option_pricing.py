import numpy as np
from scipy.stats import norm


# ============================================================
# OPTION PRICING: BLACK-SCHOLES VS JUMP-DIFFUSION
# ============================================================
#
# Goal of this file:
#
# This file prices the same European call option using two different
# assumptions about how oil prices behave.
#
# Model 1:
# Black-Scholes
#
# This is the classical benchmark. It assumes that prices move continuously,
# volatility is constant, and there are no sudden jumps.
#
# Model 2:
# Jump-diffusion Monte Carlo
#
# This model allows the oil price to experience sudden shocks. For this
# project, those shocks represent event-driven energy market risks such as
# geopolitical instability, supply disruptions, sanctions, or shipping-route
# concerns.
#
# Important finance note:
#
# For option pricing, I use risk-neutral valuation.
#
# In the jump-diffusion model, I also apply a jump compensation term:
#
# drift = r - lambda * kappa
#
# This matters because if jumps are positive on average, simply using r as the
# drift would artificially push the expected price path upward.
#
# ============================================================


# ------------------------------------------------------------
# BLACK-SCHOLES CALL OPTION FORMULA
# ------------------------------------------------------------
#
# This function calculates the closed-form Black-Scholes price of a
# European call option.
#
# Inputs:
#
# - S is the current price of the underlying asset
# - K is the strike price
# - T is time to maturity in years
# - r is the risk-free interest rate
# - sigma is annual volatility
#
# Black-Scholes assumes:
#
# - no jumps
# - continuous price movement
# - constant volatility
#
# In this project, Black-Scholes is the clean benchmark.

def black_scholes_call(S, K, T, r, sigma):

    d1 = (
        np.log(S / K)
        + (r + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    call_price = (
        S * norm.cdf(d1)
        - K * np.exp(-r * T) * norm.cdf(d2)
    )

    return call_price


# ------------------------------------------------------------
# JUMP-DIFFUSION SIMULATION FUNCTION
# ------------------------------------------------------------
#
# This function simulates oil price paths under a simplified Merton-style
# jump-diffusion process.
#
# The model has two components:
#
# 1. A GBM component
#    This captures normal day-to-day market movement.
#
# 2. A jump component
#    This captures sudden event-driven shocks.
#
# For this file, the simulation is used for option pricing, so the drift
# should be risk-neutral and jump-compensated.

def simulate_jump_diffusion(
    S0,
    drift,
    sigma,
    lam,
    jump_mean,
    jump_std,
    T,
    steps,
    n_sim
):

    dt = T / steps

    paths = np.zeros((n_sim, steps))
    paths[:, 0] = S0

    for t in range(1, steps):

        # Standard market randomness.
        Z = np.random.normal(0, 1, n_sim)

        # Poisson jump process.
        #
        # Most time steps have no jump.
        # Some paths experience one or more shock events.
        jump_occurs = np.random.poisson(lam * dt, n_sim)

        # Random jump magnitudes.
        #
        # The exponential form makes the jump multiplicative and keeps
        # simulated prices positive.
        jump_sizes = np.exp(
            jump_mean
            + jump_std * np.random.normal(0, 1, n_sim)
        ) - 1

        # Normal GBM movement under the pricing drift.
        gbm_component = np.exp(
            (drift - 0.5 * sigma**2) * dt
            + sigma * np.sqrt(dt) * Z
        )

        # Jump multiplier.
        #
        # If no jump occurs, this equals 1.
        # If a jump occurs, this scales the price by the jump size.
        jump_component = 1 + jump_occurs * jump_sizes

        paths[:, t] = (
            paths[:, t - 1]
            * gbm_component
            * jump_component
        )

    return paths


# ------------------------------------------------------------
# MONTE CARLO CALL OPTION PRICING
# ------------------------------------------------------------
#
# This function prices a European call option using simulated price paths.
#
# Payoff:
#
# max(S_T - K, 0)
#
# Then I discount the average payoff back to today.

def monte_carlo_call_price(paths, K, r, T):

    final_prices = paths[:, -1]

    payoffs = np.maximum(final_prices - K, 0)

    discounted_price = np.exp(-r * T) * np.mean(payoffs)

    return discounted_price


# ============================================================
# MODEL PARAMETERS
# ============================================================
#
# I normalize the starting oil price to 100 so the results are easier to
# interpret. The strike price is 110, meaning this is an out-of-the-money
# European call option at the start.

S0 = 100
K = 110
T = 1
r = 0.05

sigma = 0.20

steps = 252
n_sim = 10000


# ============================================================
# GEOPOLITICAL JUMP PARAMETERS
# ============================================================
#
# These parameters control the shock behavior in the jump-diffusion model.
#
# lam = 1.0 means about one major shock event per year on average.
# jump_mean = 0.04 means the average jump is around 4%.
# jump_std = 0.06 allows shock sizes to vary around that average.

lam = 1.0
jump_mean = 0.04
jump_std = 0.06


# ============================================================
# JUMP COMPENSATION TERM
# ============================================================
#
# kappa is the expected proportional jump size:
#
# kappa = E[J - 1]
#       = exp(jump_mean + 0.5 * jump_std^2) - 1
#
# The compensated risk-neutral drift is:
#
# r - lambda * kappa
#
# This removes the artificial expected growth created by positive average
# jumps, while still allowing jumps to increase tail risk.

kappa = np.exp(jump_mean + 0.5 * jump_std**2) - 1

risk_neutral_jump_drift = r - lam * kappa


# ============================================================
# BLACK-SCHOLES BENCHMARK PRICE
# ============================================================

bs_price = black_scholes_call(
    S0,
    K,
    T,
    r,
    sigma
)


# ============================================================
# JUMP-DIFFUSION MONTE CARLO PRICE
# ============================================================
#
# This prices the same European call option under jump-diffusion.
#
# The key correction:
#
# I pass risk_neutral_jump_drift into the simulation instead of plain r.

paths = simulate_jump_diffusion(
    S0,
    risk_neutral_jump_drift,
    sigma,
    lam,
    jump_mean,
    jump_std,
    T,
    steps,
    n_sim
)

jd_price = monte_carlo_call_price(
    paths,
    K,
    r,
    T
)


# ============================================================
# RESULTS
# ============================================================

print()
print("============================================================")
print("OPTION PRICING COMPARISON")
print("============================================================")
print(f"Black-Scholes Call Price: {bs_price:.2f}")
print(f"Jump-Diffusion Call Price: {jd_price:.2f}")
print(f"Difference: {jd_price - bs_price:.2f}")
print("============================================================")
print()

print("Jump Compensation:")
print(f"kappa = {kappa:.4f}")
print(f"Risk-Neutral Jump Drift = {risk_neutral_jump_drift:.4f}")
print()

print("Project Takeaway:")
print(
    "The jump-diffusion model prices the same European call option under "
    "shock-aware oil price dynamics, while Black-Scholes acts as the smooth "
    "market benchmark."
)
print()
print(
    "Because the jump-diffusion model uses a compensated risk-neutral drift, "
    "any price increase reflects jump-related tail risk rather than artificial "
    "upward expected growth."
)