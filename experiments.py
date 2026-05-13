import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os


# ============================================================
# GEOPOLITICAL ENERGY MARKET JUMP-DIFFUSION MODEL
# ============================================================
#
# Goal of this file:
#
# I want to compare a "normal" market model against a market model
# that actually allows for sudden shocks.
#
# The baseline model is GBM, which is basically the Black-Scholes world:
# prices move continuously, volatility is constant, and nothing randomly
# jumps because of external events.
#
# The second model is jump-diffusion, where prices still move normally most
# of the time, but can also experience sudden jumps. This is useful for
# energy markets because oil prices can react sharply to geopolitical shocks,
# supply disruptions, sanctions, or shipping-route concerns.
#
# Important finance distinction:
#
# For visual simulations, I use a real-world expected return mu.
#
# For option pricing, I use risk-neutral valuation.
#
# For jump-diffusion option pricing, I also apply a jump compensation term:
#
# drift = r - lambda * kappa
#
# This prevents positive average jumps from artificially increasing expected
# growth.
#
# Main question:
#
# Does ignoring sudden geopolitical shocks cause traditional models to
# underprice options?
#
# ============================================================


# ------------------------------------------------------------
# CREATE FOLDER FOR SAVED GRAPHS
# ------------------------------------------------------------

os.makedirs("plots", exist_ok=True)


# Optional: makes the output repeatable every time I run the file.
# This helps keep screenshots, graphs, and terminal outputs consistent.
np.random.seed(42)


# ------------------------------------------------------------
# GBM SIMULATION FUNCTION
# ------------------------------------------------------------

def simulate_gbm(S0, drift, sigma, T, steps, n_sim):

    dt = T / steps

    paths = np.zeros((n_sim, steps))
    paths[:, 0] = S0

    for t in range(1, steps):

        Z = np.random.normal(0, 1, n_sim)

        paths[:, t] = paths[:, t - 1] * np.exp(
            (drift - 0.5 * sigma**2) * dt
            + sigma * np.sqrt(dt) * Z
        )

    return paths


# ------------------------------------------------------------
# JUMP-DIFFUSION SIMULATION FUNCTION
# ------------------------------------------------------------

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
        jump_occurs = np.random.poisson(lam * dt, n_sim)

        # Random jump magnitudes.
        jump_sizes = np.exp(
            jump_mean
            + jump_std * np.random.normal(0, 1, n_sim)
        ) - 1

        # Normal GBM movement.
        gbm_component = np.exp(
            (drift - 0.5 * sigma**2) * dt
            + sigma * np.sqrt(dt) * Z
        )

        # Jump multiplier.
        jump_component = 1 + jump_occurs * jump_sizes

        paths[:, t] = (
            paths[:, t - 1]
            * gbm_component
            * jump_component
        )

    return paths


# ------------------------------------------------------------
# BLACK-SCHOLES CALL OPTION FORMULA
# ------------------------------------------------------------

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
# MONTE CARLO CALL OPTION PRICING
# ------------------------------------------------------------

def monte_carlo_call_price(paths, K, r, T):

    final_prices = paths[:, -1]

    payoffs = np.maximum(final_prices - K, 0)

    discounted_price = np.exp(-r * T) * np.mean(payoffs)

    return discounted_price


# ============================================================
# MODEL PARAMETERS
# ============================================================

S0 = 100

# Real-world expected return used for visualization only.
mu = 0.08

# Volatility assumption.
sigma = 0.20

# Time setup.
T = 1
steps = 252

# Number of simulations.
n_sim = 10000

# Option contract assumptions.
K = 110
r = 0.05


# ------------------------------------------------------------
# GEOPOLITICAL SHOCK PARAMETERS
# ------------------------------------------------------------

lam = 1.0
jump_mean = 0.04
jump_std = 0.06


# ------------------------------------------------------------
# JUMP COMPENSATION TERM
# ------------------------------------------------------------
#
# kappa is the expected proportional jump size:
#
# kappa = exp(jump_mean + 0.5 * jump_std^2) - 1
#
# In risk-neutral jump-diffusion pricing, I adjust the drift by:
#
# r - lambda * kappa

kappa = np.exp(jump_mean + 0.5 * jump_std**2) - 1


# ============================================================
# RUN REAL-WORLD SIMULATIONS FOR VISUALIZATION
# ============================================================
#
# These use mu = 0.08 because they are meant to show real-world style paths,
# not risk-neutral option valuation.

gbm_paths = simulate_gbm(
    S0,
    mu,
    sigma,
    T,
    steps,
    n_sim
)

jump_paths = simulate_jump_diffusion(
    S0,
    mu,
    sigma,
    lam,
    jump_mean,
    jump_std,
    T,
    steps,
    n_sim
)


# ============================================================
# GRAPH 1: FINAL PRICE DISTRIBUTION
# ============================================================

gbm_final = gbm_paths[:, -1]
jump_final = jump_paths[:, -1]

plt.figure(figsize=(12, 6))

plt.hist(
    gbm_final,
    bins=50,
    alpha=0.6,
    label="GBM"
)

plt.hist(
    jump_final,
    bins=50,
    alpha=0.6,
    label="Jump-Diffusion"
)

plt.title("Final Oil Price Distribution")
plt.xlabel("Final Oil Price")
plt.ylabel("Frequency")
plt.legend()

plt.savefig("plots/final_price_distribution.png", dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# GRAPH 2: SIDE-BY-SIDE PATH COMPARISON
# ============================================================

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)

for i in range(20):
    plt.plot(gbm_paths[i], alpha=0.7)

plt.title("GBM Oil Price Paths")
plt.xlabel("Trading Days")
plt.ylabel("Oil Price")


plt.subplot(1, 2, 2)

for i in range(20):
    plt.plot(jump_paths[i], alpha=0.7)

plt.title("Jump-Diffusion Oil Price Paths")
plt.xlabel("Trading Days")
plt.ylabel("Oil Price")

plt.tight_layout()
plt.savefig("plots/path_comparison.png", dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# OPTION PRICING COMPARISON
# ============================================================
#
# Black-Scholes gives the no-jump benchmark.
#
# Jump-diffusion uses a jump-compensated risk-neutral drift:
#
# r - lambda * kappa

bs_price = black_scholes_call(
    S0,
    K,
    T,
    r,
    sigma
)

risk_neutral_jump_drift = r - lam * kappa

pricing_jump_paths = simulate_jump_diffusion(
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
    pricing_jump_paths,
    K,
    r,
    T
)

print()
print("============================================================")
print("OPTION PRICING COMPARISON")
print("============================================================")
print(f"Black-Scholes Call Price: {bs_price:.2f}")
print(f"Jump-Diffusion Call Price: {jd_price:.2f}")
print(f"Difference: {jd_price - bs_price:.2f}")
print("============================================================")
print()


# ============================================================
# GRAPH 3: SENSITIVITY ANALYSIS
# ============================================================
#
# This is the clean research-style graph.
#
# lambda = 0 is forced to equal Black-Scholes.
#
# That is intentional because lambda = 0 means no jumps.
#
# lambda > 0 uses jump-diffusion with compensated risk-neutral drift.

lambda_values = [0, 1, 2, 3]

option_prices = []

for lam_test in lambda_values:

    if lam_test == 0:

        # No jumps = Black-Scholes benchmark.
        price = bs_price

    else:

        # Jump-compensated risk-neutral drift.
        risk_neutral_drift = r - lam_test * kappa

        simulated_paths = simulate_jump_diffusion(
            S0,
            risk_neutral_drift,
            sigma,
            lam_test,
            jump_mean,
            jump_std,
            T,
            steps,
            n_sim
        )

        price = monte_carlo_call_price(
            simulated_paths,
            K,
            r,
            T
        )

    option_prices.append(price)


plt.figure(figsize=(10, 6))

plt.plot(
    lambda_values,
    option_prices,
    marker="o"
)

plt.title("Option Price vs Geopolitical Shock Frequency")
plt.xlabel("Shock Frequency (λ)")
plt.ylabel("European Call Option Price")
plt.grid(True)

plt.savefig("plots/option_price_sensitivity.png", dpi=300, bbox_inches="tight")
plt.show()


print()
print("============================================================")
print("SENSITIVITY ANALYSIS")
print("============================================================")

for lam_val, price in zip(lambda_values, option_prices):
    print(f"Lambda = {lam_val}: Option Price = {price:.2f}")

print("============================================================")
print()


# ============================================================
# FINAL INTERPRETATION
# ============================================================

print("Project Takeaway:")
print(
    "Under geopolitical shock risk, the jump-diffusion model assigns "
    "greater value to European call options because it captures tail events "
    "that Black-Scholes ignores."
)
print()
print("Technical Note:")
print(
    "Real-world simulations use mu as the drift, while option-pricing "
    "simulations use jump-compensated risk-neutral drift."
)
print()
print("Jump Compensation:")
print(f"kappa = {kappa:.4f}")
print(
    "For jump-diffusion pricing, the drift is adjusted using "
    "r - lambda * kappa."
)
print()
print("Saved plots:")
print("1. plots/final_price_distribution.png")
print("2. plots/path_comparison.png")
print("3. plots/option_price_sensitivity.png")