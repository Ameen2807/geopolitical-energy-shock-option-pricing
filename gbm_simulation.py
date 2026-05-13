import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# GBM OIL PRICE SIMULATION
# ============================================================
#
# Goal of this file:
#
# This is the baseline model for the project.
#
# I am simulating oil price paths using Geometric Brownian Motion,
# which is the continuous price process behind the Black-Scholes model.
#
# The main purpose of this file is to show what a "normal" market looks like
# when prices move smoothly over time without sudden geopolitical shocks.
#
# Later, I compare this against a jump-diffusion model, where prices can
# suddenly move because of event-driven shocks like supply disruptions,
# sanctions, or energy-market instability.
#
# In simple terms:
#
# GBM = smooth market movement
# Jump-diffusion = smooth movement + sudden shocks
#
# This file only handles the smooth-market benchmark.
#
# ============================================================


# ------------------------------------------------------------
# GBM SIMULATION FUNCTION
# ------------------------------------------------------------
#
# GBM stands for Geometric Brownian Motion.
#
# This model assumes the asset price evolves continuously over time.
# That means there are no sudden jumps or breaks in the price path.
#
# Financial interpretation:
# - S0 is the starting oil price
# - mu is the expected annual return
# - sigma is annual volatility
# - T is the time horizon in years
# - steps is the number of time intervals
# - n_sim is the number of simulated paths
#
# For this project, I am using GBM as the clean benchmark because it represents
# the type of smooth price behavior assumed in classical Black-Scholes pricing.
#
# This is useful, but also limited, because real energy markets do not always
# move smoothly during geopolitical stress.

def simulate_gbm(S0, mu, sigma, T, steps, n_sim):

    # dt is the size of each time step.
    #
    # Since T = 1 year and steps = 252 trading days,
    # each step represents one trading day.
    dt = T / steps

    # Create a matrix to store every simulated price path.
    #
    # Rows = different simulation paths
    # Columns = trading days
    #
    # Example:
    # paths[0] is the first simulated oil price path.
    # paths[1] is the second simulated oil price path.
    paths = np.zeros((n_sim, steps))

    # Every simulated path starts at the same initial oil price.
    paths[:, 0] = S0

    # Generate price paths one trading day at a time.
    for t in range(1, steps):

        # Generate random standard normal shocks.
        #
        # Each simulated path gets its own random shock at each time step.
        # This is what creates different possible future price paths.
        Z = np.random.normal(0, 1, n_sim)

        # GBM update equation.
        #
        # The first part:
        # (mu - 0.5 * sigma^2) * dt
        # controls the drift-adjusted growth rate.
        #
        # The second part:
        # sigma * sqrt(dt) * Z
        # controls random volatility-driven movement.
        #
        # The exponential form keeps prices positive, which is important
        # because asset prices should not become negative in this model.
        paths[:, t] = paths[:, t - 1] * np.exp(
            (mu - 0.5 * sigma**2) * dt
            + sigma * np.sqrt(dt) * Z
        )

    return paths


# ============================================================
# MODEL PARAMETERS
# ============================================================
#
# These parameters define the baseline oil-price environment.
#
# I normalize the starting oil price to 100 instead of using a specific
# real-world oil price. This makes the model easier to interpret because
# all results are relative to the starting value.
#
# The key assumption here is that there are no jumps.
# So any movement in this file comes only from normal continuous volatility.

S0 = 100       # Normalized starting oil price
mu = 0.08      # Real-world expected annual return
sigma = 0.20   # Annual volatility assumption
T = 1          # Time horizon: 1 year
steps = 252    # Approximate number of trading days in one year
n_sim = 50     # Number of paths shown in the graph


# ============================================================
# RUN THE GBM SIMULATION
# ============================================================
#
# This generates 50 possible oil price paths over one year.
#
# Since this is only the GBM baseline, the paths should look relatively smooth.
# There should not be sudden vertical jumps or discontinuities.

paths = simulate_gbm(S0, mu, sigma, T, steps, n_sim)


# ============================================================
# PLOT THE RESULTS
# ============================================================
#
# This graph is meant to visually establish the benchmark case.
#
# What I expect to see:
# - all paths start at 100
# - prices move randomly over time
# - movement is continuous and smooth
# - some paths finish higher and some finish lower
#
# This becomes the "normal market" visual that I later compare against
# the jump-diffusion model.

plt.figure(figsize=(12, 6))

for i in range(n_sim):
    plt.plot(paths[i], alpha=0.6)

plt.title("GBM Oil Price Simulation")
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
# reasonably and not exploding to unrealistic values.
#
# With these parameters, the average final price should usually stay in a
# reasonable range around the starting value, depending on random variation.

print("Average Final Price:", paths[:, -1].mean())


# ============================================================
# INTERPRETATION
# ============================================================
#
# This file shows the smooth-market baseline.
#
# GBM is useful because it gives a clean model of continuous price movement.
# However, it does not capture sudden energy-market shocks.
#
# That limitation is exactly why the full project adds jump-diffusion later.