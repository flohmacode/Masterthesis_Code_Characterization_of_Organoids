import numpy as np
import matplotlib.pyplot as plt

# Load parameters
params = np.load('./modeling/parameters/jojo_april_linear_best_parameter_combined.npy', allow_pickle=True).item()
#Parameters
dt = 1  # min
total_time = 700  # min
steps = int(total_time / dt)
Volume_L = 7.0 * 1e-6  # Volume of the Experimental Setup

M_0 = params['M_0']
OXY_0 = params['OXY_0']
ORGANOID_VOLUME = params['ORGANOID_VOLUME']
CELLDENSITY = params['CELLDENSITY']
myu_glucose = params['myu_glucose']
myu_oxygen = params['myu_oxygen']
myu_waste_yield = params['myu_waste_yield']
k_m_m = params['k_m_m']
k_m_o = params['k_m_o']
k_m_i = params['k_m_i']
k_m_a = params['k_m_a']
yield_aerobic = params['yield_aerobic']
myu_fixed_costs = params['myu_fixed_costs']

# Initialize arrays for simulation
M = np.zeros(steps); M[0] = M_0  # mM
W = np.zeros(steps); W[0] = 0.0
OXY = np.zeros(steps); OXY[0] = OXY_0  # mM
ATP_p = np.zeros(steps);ATP_p[0] = 0.9

X = np.zeros(steps); X[0] = ORGANOID_VOLUME  # Biomass (Cell Viability)

# Scale rates by cell density and volume
myu_m = myu_glucose * CELLDENSITY * (X[0] / Volume_L) * 60
myu_o = myu_oxygen * CELLDENSITY * (X[0] / Volume_L) * 60
myu_w = myu_waste_yield * myu_m

# Michaelis-Menten constants
Km_m_bio = k_m_m * M_0  # mM
Km_o_bio = k_m_o * OXY_0  # mM

# Fixed constants
yield_aerobic = yield_aerobic
myu_fixed_cost = myu_fixed_costs

# Inhibition function
def inhibition(w, Ki=0.2):
    return (w) / (Ki + w)

# Q10 temperature dependency
Q10 = 2.5  # Metabolic rate increases 2.5x per 10°C rise
T_ref = 35  # Reference temperature (°C)
current_temp = 23  # Current temperature (°C)
q10_multiplier = Q10 ** ((current_temp - T_ref) / 10)

# Simulation loop
for t in range(1, steps):
    M_ratio = M[t-1] / (Km_m_bio + M[t-1])  # Michaelis-Menten curve
    OXY_ratio = OXY[t-1] / (Km_o_bio + OXY[t-1])

    aerobic = M_ratio * OXY_ratio * (1 - inhibition(W[t-1], k_m_i)) * (1 - inhibition(ATP_p[t-1], k_m_a)) * q10_multiplier

    prod = aerobic * yield_aerobic
    cost = myu_fixed_cost * q10_multiplier

    dATP_p = prod - cost
    dW = myu_w * aerobic
    dM = -myu_m * aerobic
    dOxy = -myu_o * aerobic

    ATP_p[t] = max(0, ATP_p[t-1] + dATP_p * dt)
    W[t] = max(0, W[t-1] + dW * dt)
    M[t] = max(0, M[t-1] + dM * dt)
    OXY[t] = max(0, OXY[t-1] + dOxy * dt)

# Custom color palette
my_colors = {
    'blueblue': '#003366',  # Deep Navy
    'redred': "#F8720C",    # Burnt Orange
    'greengreen': "#1B9E5A",  # Deep Teal
    'orange': '#4B0082'     # Deep Indigo
}

time_axis = np.linspace(0, total_time, steps)

# Create a figure with 2 rows and 2 columns
fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
fig.suptitle("Organoid Biological Simulation with Q10 Temperature Dependency", fontsize=16)

# 1. ATP Plot
axs[0, 0].plot(time_axis, ATP_p, color=my_colors["greengreen"], label="ATP")
axs[0, 0].set_ylabel("Peak Height (Unitless)")
axs[0, 0].set_title(f"ATP Levels (Q10 Multiplier: {q10_multiplier:.2f})")
axs[0, 0].legend()
axs[0, 0].grid(True)

# 2. Oxygen Plot
axs[0, 1].plot(time_axis, OXY, color=my_colors["blueblue"], linestyle='--', label="Oxygen")
axs[0, 1].set_ylabel("Relative Level")
axs[0, 1].set_title("Oxygen Saturation (mM)")
axs[0, 1].grid(True)
axs[0, 1].legend()
axs[0, 1].set_ylim(0, 0.25)

# 3. Glucose (Medium) Plot
axs[1, 0].plot(time_axis, M, color=my_colors["orange"], linestyle='--', linewidth=2, label="Medium")
axs[1, 0].set_ylabel("Relative Level (mM)")
axs[1, 0].set_xlabel("Time (min)")
axs[1, 0].set_title("Nutrient Availability")
axs[1, 0].grid(True)
axs[1, 0].legend()
axs[1, 0].set_ylim(0, 25)

# 4. Waste Plot
axs[1, 1].plot(time_axis, W, color=my_colors["redred"], linestyle='--', label="Waste")
axs[1, 1].set_ylabel("Accumulation")
axs[1, 1].set_xlabel("Time (min)")
axs[1, 1].set_title("Metabolic Waste")
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()