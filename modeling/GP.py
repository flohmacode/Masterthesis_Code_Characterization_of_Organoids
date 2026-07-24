import numpy as np
import matplotlib.pyplot as plt
import src.connection_to_bio as ctb
import src.helper as helper
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

import modeling.src.organoid_sim_linear as mosl

# Simulation Params
dt = 1 # min
total_time = 700 # min
steps = int(total_time / dt)
Volume = 7.0 * 1e-6
time_sim_1d = np.arange(steps)
time_sim_2d = time_sim_1d[:, None]

# Load parameters (Assumes your loading/conversion block from earlier ran here)
name = 'jojo_april'

# 1. Load the numpy arrays
accepted_parameters_jojo_arr = np.load('./parameters/jojo_april_linear_accepted_parameter_combined.npy', allow_pickle=True)
best_parameters_jojo_arr = np.load('./parameters/jojo_april_linear_best_parameter_combined.npy', allow_pickle=True)


# 2. Convert them back to Dict and List of Dicts
best_parameters_jojo = best_parameters_jojo_arr.item()          # Becomes a dict
accepted_parameters_jojo = accepted_parameters_jojo_arr.tolist()  # Becomes a list of dicts

name = 'leupold_feb'

accepted_parameters_leupold_arr = np.load('./parameters/leupold_feb_linear_accepted_parameter_combined.npy', allow_pickle=True)
best_parameters_leupold_arr = np.load('./parameters/leupold_feb_linear_best_parameter_combined.npy', allow_pickle=True)

best_parameters_leupold = best_parameters_leupold_arr.item()     # Becomes a dict
accepted_parameters_leupold = accepted_parameters_leupold_arr.tolist() # Becomes a list of dicts

# Run both baseline mechanistic models (700 steps)
atp_model_jojo, _, _, ip_model_jojo = mosl.organoid_sim_spec_params_linear(steps, dt, Volume, best_parameters_jojo)
atp_model_leupold, _, _, ip_model_leupold = mosl.organoid_sim_spec_params_linear(steps, dt, Volume, best_parameters_leupold)

# Load both real experimental datasets
time_exp_jojo, normed_datapoints_jojo = helper.get_real_data('jojo_april')
time_exp_leupold, normed_datapoints_leupold = helper.get_real_data('leupold_feb')

time_ip_jojo,ip_normed_jojo = helper.get_inorganic_phosphate('jojo_april')
time_ip_leupold,ip_normed_leupold = helper.get_inorganic_phosphate('leupold_feb')

# Flatten experimental data vectors just 
normed_datapoints_jojo = normed_datapoints_jojo.squeeze()
normed_datapoints_leupold = normed_datapoints_leupold.squeeze()

ip_normed_jojo = ip_normed_jojo.squeeze()
ip_normed_leupold = ip_normed_leupold.squeeze()

# Experiment 2: Leupold
#ATP Residual
leupold_at_obs = np.interp(time_exp_leupold, time_sim_1d, atp_model_leupold)
res_leupold = normed_datapoints_leupold - leupold_at_obs

#IP Residual
ip_leupold_at_obs = np.interp(time_ip_leupold,time_sim_1d,ip_model_leupold)
res_ip_leupold_at_obs = ip_normed_leupold - ip_leupold_at_obs

# Experiment 3: JOJO
#ATP residual
jojo_at_obs = np.interp(time_exp_jojo, time_sim_1d, atp_model_jojo)
res_jojo = normed_datapoints_jojo - jojo_at_obs

#IP Residual
ip_jojo_at_obs = np.interp(time_ip_jojo,time_sim_1d,ip_model_jojo)
res_ip_jojo_at_obs = ip_normed_jojo - ip_jojo_at_obs

print('var',(0.5*(np.var(res_jojo)+np.var(res_leupold))))

# Fit GP to each residual separately
# Same kernel for both so hyperparameters are comparable
kernel = RBF(length_scale=135, length_scale_bounds="fixed") \
     + WhiteKernel(noise_level=0.5*(np.var(res_jojo)+np.var(res_leupold)), noise_level_bounds="fixed")

gp_leupold = GaussianProcessRegressor(kernel=kernel, normalize_y=False, n_restarts_optimizer=20)
gp_leupold.fit(time_exp_leupold.reshape(-1,1), res_leupold)

gp_jojo = GaussianProcessRegressor(kernel=kernel, normalize_y=False, n_restarts_optimizer=20)
gp_jojo.fit(time_exp_jojo.reshape(-1,1), res_jojo)

kernel_ip = RBF(length_scale=135, length_scale_bounds="fixed") \
     + WhiteKernel(noise_level=0.5*(np.var(res_ip_leupold_at_obs)+np.var(res_ip_jojo_at_obs)), noise_level_bounds="fixed")

gp_leupold_ip = GaussianProcessRegressor(kernel=kernel_ip, normalize_y=False, n_restarts_optimizer=20)
gp_leupold_ip.fit(time_exp_leupold.reshape(-1,1), res_ip_leupold_at_obs)

gp_jojo_ip = GaussianProcessRegressor(kernel=kernel_ip, normalize_y=False, n_restarts_optimizer=20)
gp_jojo_ip.fit(time_exp_jojo.reshape(-1,1), res_ip_jojo_at_obs)

print("Leupold discrepancy kernel: ", gp_leupold.kernel_)
print("Jojo discrepancy kernel: ", gp_jojo.kernel_)

fig, axes = plt.subplots(2, 2, figsize=(10, 6),squeeze=False)

# =====================================================================
# Predict full timeline for both GPs
# =====================================================================

mean_leupold, std_leupold = gp_leupold.predict(time_sim_2d, return_std=True)
mean_jojo, std_jojo = gp_jojo.predict(time_sim_2d, return_std=True)

mean_ip_leupold, std_ip_leupold = gp_leupold_ip.predict(time_sim_2d, return_std=True)
mean_ip_jojo, std_ip_jojo = gp_jojo_ip.predict(time_sim_2d, return_std=True)

fitted_leupold = atp_model_leupold + mean_leupold
fitted_jojo = atp_model_jojo + mean_jojo

fitted_ip_leupold = ip_model_leupold + mean_ip_leupold
fitted_ip_jojo = ip_model_jojo + mean_ip_jojo

my_colors = {
    "3. Experiment": "#0C68C4",   # Deep Navy
    "2. Experiment": "#F8720C"   # Burnt Orange
}

# =====================================================================
# Top Left: Experiment 2 model + GP discrepancy vs Leupold data
# =====================================================================
ax = axes[0, 0]
ax.plot(time_sim_1d, atp_model_leupold, 'r--', alpha=0.6, label='Mechanistic Model 2. Experiment')
ax.scatter(time_exp_leupold, normed_datapoints_leupold, color='black', zorder=5, s=30, label='Data (Exp 2)')
ax.plot(time_sim_1d, fitted_leupold, color=my_colors['2. Experiment'], lw=2, label='Model + GP Discrepancy')
ax.fill_between(time_sim_1d,
                fitted_leupold + 1.96 * std_leupold,
                fitted_leupold - 1.96 * std_leupold,
                color=my_colors['2. Experiment'], alpha=0.3, label='95% Uncertainty Bound')
ax.set_title('2.Experiment: Model Discrepancy', fontsize=13)
#ax.set_xlabel('Time (min)')
ax.set_ylabel('ATP')
ax.set_xlim(0, 400)
ax.set_ylim(bottom=-0.5,top = 1.5)
ax.legend(fontsize=8)
ax.grid(True, linestyle=':', alpha=0.5)

# =====================================================================
# Top Right: Experiment 3 model + GP discrepancy vs Jojo data
# =====================================================================
ax = axes[0, 1]
ax.plot(time_sim_1d, atp_model_jojo, 'b--', alpha=0.6, label='Mechanistic Model 3.Experiment')
ax.scatter(time_exp_jojo, normed_datapoints_jojo, color='black', zorder=5, s=30, label='Data 3.Exp')
ax.plot(time_sim_1d, fitted_jojo, color=my_colors['3. Experiment'], lw=2, label='Model + GP Discrepancy')
ax.fill_between(time_sim_1d,
                fitted_jojo + 1.96*std_jojo,
                fitted_jojo - 1.96*std_jojo,
                color=my_colors['3. Experiment'], alpha=0.3, label='95% Uncertainty Bound')
ax.set_title('3.Experiment: Model Discrepancy', fontsize=13)
#ax.set_xlabel('Time (min)')
ax.set_ylabel('ATP')
ax.set_xlim(0, 400)
ax.set_ylim(bottom=-0.5,top = 1.5)
ax.legend(fontsize=8)
ax.grid(True, linestyle=':', alpha=0.5)


ax = axes[1, 0]
ax.plot(time_sim_1d, ip_model_leupold, 'r--', alpha=0.6, label='Mechanistic Model 2. Experiment')
ax.scatter(time_exp_leupold, ip_normed_leupold, color='black', zorder=5, s=30, label='Data (Exp 2)')
ax.plot(time_sim_1d, fitted_ip_leupold, color=my_colors['2. Experiment'], lw=2, label='Model + GP Discrepancy')
ax.fill_between(time_sim_1d,
                fitted_ip_leupold + 1.96 * std_ip_leupold,
                fitted_ip_leupold - 1.96 * std_ip_leupold,
                color=my_colors['2. Experiment'], alpha=0.3, label='95% Uncertainty Bound')
#ax.set_title('2.Experiment: Model Discrepancy', fontsize=13)
ax.set_xlabel('Time (min)')
ax.set_ylabel('Inorganic Phosphate')
ax.set_xlim(0, 400)
ax.set_ylim(bottom=-1,top = 1.5)
ax.legend(fontsize=8)
ax.grid(True, linestyle=':', alpha=0.5)

ax = axes[1, 1]
ax.plot(time_sim_1d, ip_model_jojo, 'b--', alpha=0.6, label='Mechanistic Model 3. Experiment')
ax.scatter(time_exp_jojo, ip_normed_jojo, color='black', zorder=5, s=30, label='Data (Exp 3)')
ax.plot(time_sim_1d, fitted_ip_jojo, color=my_colors['3. Experiment'], lw=2, label='Model + GP Discrepancy')
ax.fill_between(time_sim_1d,
                fitted_ip_jojo + 1.96 * std_ip_jojo,
                fitted_ip_jojo - 1.96 * std_ip_jojo,
                color=my_colors['3. Experiment'], alpha=0.3, label='95% Uncertainty Bound')
#ax.set_title('3.Experiment: Model Discrepancy', fontsize=13)
ax.set_xlabel('Time (min)')
ax.set_ylabel('Inorganic Phosphate')
ax.set_xlim(0, 400)
ax.set_ylim(bottom=-0.5,top = 1.5)
ax.legend(fontsize=8)
ax.grid(True, linestyle=':', alpha=0.5)

plt.show()

# ======================================================================
# Bottom Left: Raw residuals + GP mean overlaid — shows what GP learned
# ======================================================================

fig, ax = plt.subplots(figsize=(10, 5))

ax.scatter(time_exp_leupold, res_leupold, color=my_colors['2. Experiment'], s=30, zorder=5, label='Residuals 2.Experiment')
ax.scatter(time_exp_jojo, res_jojo, color=my_colors['3. Experiment'], s=30, zorder=5, label='Residuals 3.Experiment')
ax.plot(time_sim_1d, mean_leupold, color=my_colors['2. Experiment'], lw=2, label='GP Mean 2.Experiment')
ax.plot(time_sim_1d, mean_jojo, color=my_colors['3. Experiment'], lw=2, label='GP Mean 3.Experiment')
ax.plot(time_sim_1d,
                mean_leupold+1.96*std_leupold,
                color=my_colors['2. Experiment'],linestyle = '--', alpha=0.6, label='95% Uncertainty Bound')
ax.plot(time_sim_1d,
                mean_leupold-1.96*std_leupold,
                color=my_colors['2. Experiment'],linestyle = '--', alpha=0.6)
ax.plot(time_sim_1d,
                mean_jojo +1.96*std_jojo,
                color=my_colors['3. Experiment'],linestyle = '--', alpha=0.6, label='95% Uncertainty Bound')
ax.plot(time_sim_1d,
                mean_jojo -1.96*std_jojo,
                color=my_colors['3. Experiment'],linestyle = '--', alpha=0.6)

ax.axhline(0, color='black', linestyle='--', lw=1.5, alpha=0.8, zorder=10)

ax.set_title('Residuals and GP Discrepancy Functions', fontsize=13)
ax.set_xlabel('Time (min)')
ax.set_ylabel('Residual ATP')
ax.set_xlim(0, 300)
ax.legend(fontsize=8)
ax.grid(True, linestyle=':', alpha=0.5)
plt.show()



fig, ax = plt.subplots(figsize=(10, 5))

ax.scatter(time_exp_leupold, res_ip_leupold_at_obs, color=my_colors['2. Experiment'], s=30, zorder=5, label='Residuals 2.Experiment')
ax.scatter(time_exp_jojo, res_ip_jojo_at_obs, color=my_colors['3. Experiment'], s=30, zorder=5, label='Residuals 3.Experiment')
ax.plot(time_sim_1d, mean_ip_leupold, color=my_colors['2. Experiment'], lw=2, label='GP Mean 2.Experiment')
ax.plot(time_sim_1d, mean_ip_jojo, color=my_colors['3. Experiment'], lw=2, label='GP Mean 3.Experiment')
ax.plot(time_sim_1d,
                mean_ip_leupold+1.96*std_ip_leupold,
                color=my_colors['2. Experiment'],linestyle = '--', alpha=0.6, label='95% Uncertainty Bound')
ax.plot(time_sim_1d,
                mean_ip_leupold-1.96*std_ip_leupold,
                color=my_colors['2. Experiment'],linestyle = '--', alpha=0.6)
ax.plot(time_sim_1d,
                mean_ip_jojo +1.96*std_ip_jojo,
                color=my_colors['3. Experiment'],linestyle = '--', alpha=0.6, label='95% Uncertainty Bound')
ax.plot(time_sim_1d,
                mean_ip_jojo -1.96*std_ip_jojo,
                color=my_colors['3. Experiment'],linestyle = '--', alpha=0.6)

ax.axhline(0, color='black', linestyle='--', lw=1.5, alpha=0.8, zorder=10)

ax.set_title('Residuals and GP Discrepancy Functions', fontsize=13)
ax.set_xlabel('Time (min)')
ax.set_ylabel('Residual Inorganic Phosphate')
ax.set_xlim(0, 300)
ax.legend(fontsize=8)
ax.grid(True, linestyle=':', alpha=0.5)
plt.show()



print("\n=== GP HYPERPARAMETERS FOR THESIS METHOD SECTION ===")

for name, gp in [("Leupold", gp_leupold), ("Jojo", gp_jojo)]:
    # 1. Get the fitted kernel
    fitted_kernel = gp.kernel_
    
    # 2. Extract individual parameter values from the Sum kernel object
    # scikit-learn combines kernels as: Sum(RBF, WhiteKernel) -> k1 is RBF, k2 is WhiteKernel
    rbf_length_scale = fitted_kernel.k1.length_scale
    white_noise_level = fitted_kernel.k2.noise_level
    
    print(f"\n[{name} Dataset]")
    print(f"  - Kernel Structure: {fitted_kernel}")
    print(f"  - RBF Length Scale (\u2113): {rbf_length_scale:.4f}")
    print(f"  - Noise Level Variance (\u03c3\u00b2): {white_noise_level:.6f}")
    # Log-marginal likelihood is great to report for model selection/fit quality
    print(f"  - Log-Marginal Likelihood: {gp.log_marginal_likelihood():.4f}")
