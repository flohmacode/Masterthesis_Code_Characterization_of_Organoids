"""
Optimal scan-time analysis for the organoid ODE model.

Idea (in one sentence): a scan at time t is "informative" if a small nudge
to a parameter causes a big change in the predicted ATP signal at time t.
We never need real data at new timepoints -- we only need the MODEL,
evaluated at your best-fit parameters (theta_star), queried at candidate
times you never actually scanned.

Steps:
  1. Run the model once at theta_star  -> baseline trajectory.
  2. For each parameter, nudge it up/down a tiny bit, rerun the model,
     and take the finite-difference derivative of ATP(t) w.r.t. that
     parameter, at EVERY timestep t (not just your 8-min scan grid --
     the ODE solver already gives you a value at every dt).
  3. Square and sum these sensitivities (weighted by 1/noise_variance)
     at each t --> this is a (diagonal / local) Fisher-information curve
     over time. Its peaks = your most informative scan times.
  4. Optionally: greedily pick N scan times that add the most NEW
     information, to compare against your actual 8.3-min NSPECT cadence.
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------
# 1. Your ODE model (unchanged, copy-pasted from your code)
# -----------------------------------------------------------------------
def organoid_sim_spec_params_linear(steps, dt, Volume_L, params):
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

    M = np.zeros(steps);   M[0] = M_0
    W = np.zeros(steps);   W[0] = 0.0
    OXY = np.zeros(steps); OXY[0] = OXY_0
    ATP_p = np.zeros(steps); ATP_p[0] = 0.9
    X = np.zeros(steps);   X[0] = ORGANOID_VOLUME

    myu_m = myu_glucose * CELLDENSITY * (X[0] / Volume_L) * 60
    myu_o = myu_oxygen * CELLDENSITY * (X[0] / Volume_L) * 60
    myu_w = myu_waste_yield * myu_m

    Km_m_bio = k_m_m * M_0
    Km_o_bio = k_m_o * OXY_0
    yield_aer = yield_aerobic
    myu_fixed_cost = myu_fixed_costs

    def inhibition(w, Ki):
        return w / (Ki + w)

    for t in range(1, steps):
        M_ratio = M[t-1] / (Km_m_bio + M[t-1])
        OXY_ratio = OXY[t-1] / (Km_o_bio + OXY[t-1])
        aerobic = (M_ratio * OXY_ratio
                   * (1 - inhibition(W[t-1], k_m_i))
                   * (1 - inhibition(ATP_p[t-1], k_m_a)))
        prod = aerobic * yield_aer
        cost = myu_fixed_cost
        dATP_p = prod - cost
        dW = myu_w * aerobic
        dM = -myu_m * aerobic
        dOxy = -myu_o * aerobic

        ATP_p[t] = max(0, ATP_p[t-1] + dATP_p * dt)
        W[t] = max(0, W[t-1] + dW * dt)
        M[t] = max(0, M[t-1] + dM * dt)
        OXY[t] = max(0, OXY[t-1] + dOxy * dt)

    return ATP_p, M, OXY, W


# -----------------------------------------------------------------------
# 2. Reference parameters (theta_star) -- REPLACE with your ABC posterior
#    means from Table 5.2 / your ABC results (Section 6.12).
# -----------------------------------------------------------------------
# theta_star = {
#     'M_0': 12.5,
#     'OXY_0': 0.21,
#     'ORGANOID_VOLUME': 5.09e-7,
#     'CELLDENSITY': 4.82e11,
#     'myu_glucose': 2.0e-16,
#     'myu_oxygen': 7.7e-16,
#     'myu_waste_yield': 1.0,        # placeholder -- set to your fitted value
#     'k_m_m': 0.3,
#     'k_m_o': 0.2,
#     'k_m_i': 0.28,                 # e.g. your ABC posterior mean for k_m_i
#     'k_m_a': 0.9,                  # placeholder -- inhibition constant for ATP
#     'yield_aerobic': 0.65,         # your ABC posterior mean
#     'myu_fixed_costs': 0.09,       # your ABC posterior mean
# }


theta_star = np.load('./parameters/jojo_april_linear_best_parameter.npy',allow_pickle=True).item()


print(theta_star)

Volume_L = 7.0e-6      # from Table 5.2 (VSetup)
dt = 1.0                # 1 minute per step -- fine resolution, NOT your scan grid
total_minutes = 250
steps = int(total_minutes / dt)

# Only vary parameters that are actually free / of interest.
# (No point testing sensitivity to fixed/measured quantities like Volume_L.)
free_params = ['myu_waste_yield', 'k_m_m', 'k_m_o', 'k_m_i',
               'k_m_a', 'yield_aerobic', 'myu_fixed_costs']

# Measurement noise variance -- from your GP fit (Section 5.15), ~0.03
noise_variance = 0.03 ** 2


# -----------------------------------------------------------------------
# 3. Local sensitivity of ATP(t) to each parameter, at every timestep
# -----------------------------------------------------------------------
def run_atp(params):
    ATP_p, _, _, _ = organoid_sim_spec_params_linear(steps, dt, Volume_L, params)
    return ATP_p

baseline_ATP = run_atp(theta_star)

eps = 1e-3  # relative perturbation size
sensitivity = np.zeros((steps, len(free_params)))

for j, pname in enumerate(free_params):
    params_plus = dict(theta_star)
    params_minus = dict(theta_star)
    h = theta_star[pname] * eps

    params_plus[pname] = theta_star[pname] + h
    params_minus[pname] = theta_star[pname] - h

    atp_plus = run_atp(params_plus)
    atp_minus = run_atp(params_minus)

    # d(ATP)/d(theta_j) at every timestep, central difference
    sensitivity[:, j] = (atp_plus - atp_minus) / (2 * h)


# -----------------------------------------------------------------------
# 4. Information curve over time
#    (diagonal / local Fisher information: sum over params of
#     (dATP/dtheta_j)^2 / noise_variance)
# -----------------------------------------------------------------------
info_curve = np.sum(sensitivity ** 2, axis=1) / noise_variance
time_axis = np.arange(steps) * dt

# -----------------------------------------------------------------------
# 5. Greedy selection: given a fixed budget of N scans, which N timepoints
#    give the most information? (ignores parameter correlations -- a
#    simple, honest approximation, not full D-optimal design)
# -----------------------------------------------------------------------
def greedy_scan_times(info_curve, time_axis, n_scans, min_spacing_min=8):
    """Pick n_scans timepoints maximizing cumulative info curve,
    respecting a minimum spacing (can't scan faster than your
    real acquisition time, e.g. 8.3 min)."""
    candidates = list(range(len(info_curve)))
    chosen = []
    info_copy = info_curve.copy()
    for _ in range(n_scans):
        idx = np.argmax(info_copy)
        chosen.append(idx)
        # suppress nearby timepoints so we don't cluster all scans together
        lo = max(0, idx - int(min_spacing_min / dt))
        hi = min(len(info_copy), idx + int(min_spacing_min / dt))
        info_copy[lo:hi] = -np.inf
    chosen.sort()
    return time_axis[chosen]

n_actual_scans = int(total_minutes / 8.3)  # your real NSPECT cadence
optimal_times = greedy_scan_times(info_curve, time_axis, n_actual_scans)
uniform_times = np.linspace(0, total_minutes, n_actual_scans)

info_at_optimal = np.interp(optimal_times, time_axis, info_curve).sum()
info_at_uniform = np.interp(uniform_times, time_axis, info_curve).sum()

print(f"Total info, your actual (~uniform, 8.3 min) schedule: {info_at_uniform:.2f}")
print(f"Total info, greedily-optimized schedule:              {info_at_optimal:.2f}")
print(f"Suggested (near-)optimal scan times (min): {np.round(optimal_times, 1)}")


# -----------------------------------------------------------------------
# 6. Plot
# -----------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

axes[0].plot(time_axis, baseline_ATP, color='black')
axes[0].set_ylabel('ATP (model)')
axes[0].set_title('Baseline ATP trajectory at theta*')

axes[1].plot(time_axis, info_curve, color='tab:blue', label='Information curve')
axes[1].scatter(uniform_times, np.interp(uniform_times, time_axis, info_curve),
                 color='gray', marker='x', label='Your actual scan times (uniform)')
axes[1].scatter(optimal_times, np.interp(optimal_times, time_axis, info_curve),
                 color='tab:red', marker='o', facecolors='none',
                 label='Greedily-optimal scan times')
axes[1].set_xlabel('Time (min)')
axes[1].set_ylabel('Local Fisher information')
axes[1].set_title('When a scan is most informative')
axes[1].legend()

plt.tight_layout()
#plt.savefig('/mnt/user-data/outputs/optimal_scan_time.png', dpi=150)
plt.show()