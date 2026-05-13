import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# GEOPOLITICAL JUMP-DIFFUSION OIL PRICE SIMULATION
# ============================================================
#
# Goal of this file:
#
# This file extends the basic GBM simulation by allowing oil prices to
# experience sudden jumps.
#
# In the real world, energy markets do not always move smoothly. Oil prices
# can react sharply to geopolitical instability, supply disruptions, sanctions,
# shipping-route concerns, or unexpected changes in market expectations.
#
# That is the main reason I am using a jump-diffusion model here.
#
# GBM assumes:
# - continuous price movement
# - no sudden jumps
# - volatility explains all randomness
#
# Jump-diffusion assumes:
# - prices usually move normally
# - but sometimes a discrete shock occurs
# - those shocks can create sudden price moves
#
# In simple terms:
#
# GBM = normal market noise
# Jump-diffusion = normal market noise + event-driven shocks
#
# This file focuses only on the jump-diffusion simulation.
#
# Important:
#
# This standalone file is mainly for visualization. I use a slightly larger
# jump uncertainty here so the jump behavior is easier to see in the graph.
# The pricing files use more conservative assumptions because option pricing
# needs to be more technically defensible.
#
# ============================================================


# ------------------------------------------------------------
# JUMP-DIFFUSION SIMULATION FUNCTION
# ------------------------------------------------------------
#
# This function simulates oil price paths using a Merton-style
# jump-diffusion process.
#
# The model has two parts:
#
# 1. A normal GBM component
#    This captures regular day-to-day market movement.
#
# 2. A jump component
#    This captures sudden event-driven shocks.
#
# Financial interpretation of the inputs:
#
# - S0 is the starting oil price
# - mu is the real-world expected annual return
# - sigma is annual volatility
# - lam controls how often jumps happen
# - jump_mean controls the average jump size
# - jump_std controls how uncertain the jump size is
# - T is the time horizon in years
# - steps is the number of trading days
# - n_sim is the number of simulated price paths
#
# For this project, the jump component represents geopolitical energy-market
# shocks. I am not claiming to predict a specific conflict or exact oil price.
# The goal is to model the kind of discontinuous behavior that classical
# Black-Scholes assumptions ignore.

def simulate_jump_diffusion(
    S0,
    mu,
    sigma,
    lam,
    jump_mean,
    jump_std,
    T,
    steps,
    n_sim
):

    # dt is the size of each time step.
    #
    # With T = 1 and steps = 252, each step represents one trading day.
    dt = T / steps

    # Create a matrix to store every simulated price path.
    #
    # Rows = different simulations
    # Columns = trading days
    paths = np.zeros((n_sim, steps))

    # Every simulated path starts at the same initial oil price.
    paths[:, 0] = S0

    # Generate price paths one trading day at a time.
    for t in range(1, steps):

        # Standard GBM randomness.
        #
        # This represents normal day-to-day market noise.
        Z = np.random.normal(0, 1, n_sim)

        # Poisson jump process.
        #
        # This determines how many jump events occur in each path at this
        # specific time step.
        #
        # Most days, jump_occurs will be 0.
        # Occasionally, it may be 1 or more, meaning a shock occurred.
        #
        # lam * dt is the expected number of jumps during one trading day.
        jump_occurs = np.random.poisson(lam * dt, n_sim)

        # Random jump magnitude.
        #
        # The jump size is modeled with an exponential transformation.
        # This makes jumps multiplicative instead of additive, which is more
        # appropriate for asset prices because prices should stay positive.
        #
        # Example:
        # If jump_sizes = 0.05, the price jumps upward by about 5%.
        # If jump_sizes = -0.03, the price drops by about 3%.
        jump_sizes = np.exp(
            jump_mean
            + jump_std * np.random.normal(0, 1, n_sim)
        ) - 1

        # GBM growth component.
        #
        # This is the same smooth market movement used in the GBM baseline.
        # It captures normal price evolution when no major shock occurs.
        gbm_component = np.exp(
            (mu - 0.5 * sigma**2) * dt
            + sigma * np.sqrt(dt) * Z
        )

        # Jump component.
        #
        # If no jump occurs:
        # jump_occurs = 0, so jump_component = 1.
        #
        # If a jump occurs:
        # jump_component = 1 + jump_size.
        #
        # This means the jump directly scales the price up or down.
        jump_component = 1 + jump_occurs * jump_sizes

        # Combined jump-diffusion process.
        #
        # The new price equals:
        #
        # previous price
        # x normal GBM movement
        # x jump shock component
        #
        # This is what creates discontinuous price behavior.
        paths[:, t] = paths[:, t - 1] * gbm_component * jump_component

    return paths


# ============================================================
# MODEL PARAMETERS
# ============================================================
#
# These parameters define the basic oil-market environment.
#
# I normalize the starting oil price to 100 instead of using a specific
# real-world oil price. This makes the graph easier to interpret because
# all paths are relative to the same starting value.
#
# These parameters are also kept mostly consistent with the larger experiments
# file so that the project does not accidentally describe completely different
# market regimes in different scripts.
#
# The key difference is jump_std:
#
# - This standalone visualization file uses jump_std = 0.10
#   so the shock behavior is easier to see.
#
# - The option-pricing and experiments files use jump_std = 0.06
#   because those sections are meant to be more conservative and technically
#   defensible.

S0 = 100       # Normalized starting oil price

mu = 0.08      # Real-world expected annual return
sigma = 0.20   # Annual volatility assumption

T = 1          # Time horizon: 1 year
steps = 252    # Approximate number of trading days in one year
n_sim = 50     # Number of paths shown in the graph


# ============================================================
# GEOPOLITICAL SHOCK PARAMETERS
# ============================================================
#
# These are the parameters that make this model different from plain GBM.
#
# lambda controls how frequently shocks occur.
# jump_mean controls the average size of the shock.
# jump_std controls how unpredictable those shocks are.
#
# I am keeping these values reasonable, but I use slightly higher jump
# uncertainty here than in the pricing files so the visualization clearly shows
# discontinuous behavior.
#
# Interpretation:
#
# lam = 1.0 means the model expects about one major shock per year.
# jump_mean = 0.04 means the average jump is around 4%.
# jump_std = 0.10 allows the shock size to vary more visibly around that
# average for graphing purposes.

lam = 1.0
jump_mean = 0.04
jump_std = 0.10


# ============================================================
# RUN THE JUMP-DIFFUSION SIMULATION
# ============================================================
#
# This generates possible oil price paths over one year under a market
# environment where sudden geopolitical shocks can occur.
#
# Compared to the GBM simulation, I expect these paths to show more abrupt
# movement and slightly wider dispersion.

paths = simulate_jump_diffusion(
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
# PLOT THE RESULTS
# ============================================================
#
# This graph is meant to visually show the difference between a smooth market
# model and a shock-aware market model.
#
# What I expect to see:
# - all paths start at 100
# - most movement still looks like normal market randomness
# - some paths show sharper shifts because of jump events
# - the overall range of possible outcomes is wider than plain GBM
#
# This is the visual foundation for the larger project:
# energy markets under geopolitical stress can behave discontinuously.

plt.figure(figsize=(12, 6))

for i in range(n_sim):
    plt.plot(paths[i], alpha=0.7)

plt.title("Geopolitical Jump-Diffusion Oil Price Simulation")
plt.xlabel("Trading Days")
plt.ylabel("Oil Price")

plt.show()


# ============================================================
# QUICK SUMMARY
# ============================================================
#
# This prints the average final price across all simulated paths.
#
# I use this as a quick check to make sure the simulation is behaving
# reasonably and not producing unrealistic explosions.
#
# Some paths may move sharply because of jumps, but the average final price
# should still remain within a reasonable range for these visualization
# parameters.

print("Average Final Price:", paths[:, -1].mean())


# ============================================================
# INTERPRETATION
# ============================================================
#
# This file shows how adding jump risk changes the behavior of oil price paths.
#
# The key idea is that geopolitical shocks do not behave like normal daily
# volatility. They can create sudden price movements that a smooth GBM model
# does not capture.
#
# This is why the larger project compares Black-Scholes against jump-diffusion:
# if the underlying asset can jump, then traditional option pricing may
# underestimate tail risk.