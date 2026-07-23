from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import src.connection_to_bio as ctb
import matplotlib.pyplot as plt

plt.close()

# Define your parameter ranges
problem = {
    'num_vars': 12,
    'names': ['M_0', 'OXY_0', 'ORGANOID_VOLUME',
              'CELLDENSITY','myu_glucose','myu_oxygen','myu_waste_yield',
              'k_m_m','k_m_o','k_m_i','k_m_a',
              'myu_fixed_costs'],
    'bounds': [[5.0, 20.0],       # medium (mM)
                [0.18, 0.24],      # oxygen (mM)
                [2e-8, 4.5e-8],    # organoid_volume (m^3)
                [3e11, 6e11],      # celldensity (cells/m^3)
                [1e-16, 3e-16],    # myu_glucose
                [5e-16, 1e-15],    # myu_oxygen
                [1, 100],            # myu_waste
                [0.2, 0.4],        # k_m_m
                [0.1, 0.3],        # k_m_o
                [0.1, 0.3],        # k_m_i
                [0.1, 0.3],        # k_m_a
                [0.04, 0.1],       # myu_fixed_costs
                ] 
}

# problem = {
#     'num_vars': 13,
#     'names': ['M_0', 'OXY_0', 'ORGANOID_VOLUME',
#               'CELLDENSITY','myu_glucose','myu_oxygen','myu_waste_yield',
#               'k_m_m','k_m_o','k_m_i','k_m_a',
#               'aerobic_yield','myu_fixed_costs'],
#     'bounds': [[5.0, 20.0],       # medium (mM)
#                 [0.18, 0.24],      # oxygen (mM)
#                 [2e-8, 4.5e-8],    # organoid_volume (m^3)
#                 [3e11, 6e11],      # celldensity (cells/m^3)
#                 [1e-16, 3e-16],    # myu_glucose
#                 [5e-16, 1e-15],    # myu_oxygen
#                 [0.5, 15.0],      # myu_waste
#                 [0.2, 0.4],        # k_m_m
#                 [0.1, 0.3],        # k_m_o
#                 [0.1, 0.3],        # k_m_i
#                 [0.1, 0.3],        # k_m_a
#                 [0.1, 1.0],        # yield
#                 [0.04, 0.1],       # myu_fixed_costs
#                 ] 
# }


dt = 1 #min
total_time =700#min
steps = int(total_time/dt)
Volume = 7.0  * 1e-6

# Generate samples (~1000-10000 model runs needed)
param_values = saltelli.sample(problem, N=512)

# Run your model for each sample
#outputs = np.array([ctb.simulation_sensitivity_organoids(steps,dt,Volume
#                                                         ,p)[-1] for p in param_values])
results = [ctb.simulation_sensitivity_organoids(steps, dt, Volume, p) for p in param_values]

outputs_ATP = np.array([r[0].mean() for r in results])  # ATP_p last value, all param sets
outputs_M   = np.array([r[1].mean() for r in results])  # M last value, all param sets
outputs_O  = np.array([r[2].mean() for r in results])  # Oxy last value, all param sets
outputs_W  = np.array([r[3].mean() for r in results])  # Oxy last value, all param sets

# Analyze
Si = sobol.analyze(problem, outputs_W)
print(Si['S1'])   # first-order indices
print(Si['ST'])   # total-order indices

# Data preparation
names = problem['names']
s1 = Si['S1']
st = Si['ST']
indices = np.arange(len(names))

# Sort by Total Effect for better visual flow
idx_sorted = np.argsort(st)
names_sorted = [names[i] for i in idx_sorted]
s1_sorted = s1[idx_sorted]
st_sorted = st[idx_sorted]

# Plotting
plt.figure(figsize=(10, 8))
plt.barh(indices, st_sorted, color='lightskyblue', label='Total Effect ($S_T$)')
plt.barh(indices, s1_sorted, color='steelblue', label='First Order ($S_1$)')

# Formatting
plt.xlabel('Sensitivity Index', fontsize=12)
plt.title('Global Sensitivity Analysis (Sobol Indices)', fontsize=14)
plt.yticks(indices, names_sorted)
plt.legend(loc='lower right')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save for thesis
plt.savefig('./figs/sobol_analysis_atp_W.png', dpi=300)
plt.show()