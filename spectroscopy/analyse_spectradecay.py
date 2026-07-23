import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import src.processing as processing
import pandas as pd
import seaborn as sns

plt.close('all')
name_bimsb = "leupold_dec"
spectras = np.load(f"./processed_data/{name_bimsb}/spectra.npy")
scantime = np.load(f"./processed_data/{name_bimsb}/scantime.npy",allow_pickle=True)
ppm_axis = np.load(f"./processed_data/{name_bimsb}/ppm_axis.npy")
spectras = spectras.squeeze()
spectras = np.flip(spectras, axis=-1)
spectras = np.roll(spectras,235)
print(spectras.shape)


if ppm_axis[0] < ppm_axis[-1]:
    ppm_axis = np.flip(ppm_axis)

#np.save("./fig/tracking_check/spectra.npy",spectras[0])

# A professional, colorblind-friendly palette
# Source: seaborn 'colorblind' palette or custom professional set
my_colors = {
    'Inorganic Phosphate': '#003f5c', # Deep Navy (Professional, authoritative)
    'gamma-ATP':           "#ff1900", # Burnt Orange (High contrast, very legible)
    'alpha-ATP':           "#44B5B5", # Deep Teal (Calm, distinct from navy)
    'beta-ATP':            "#df8d12"  # Deep Violet (Darker, sophisticated, not pink)
}
# 1. DEFINE TARGETS (Update your dictionary)
metabolites = {
    'Inorganic Phosphate': {'ppm':  5.02,  'color': my_colors['Inorganic Phosphate']},
    'gamma-ATP':           {'ppm': -2.48,  'color': my_colors['gamma-ATP']},
    'alpha-ATP':           {'ppm': -7.52,  'color': my_colors['alpha-ATP']},
    'beta-ATP':            {'ppm': -16.26, 'color': my_colors['beta-ATP']}
}

# "ticks" or "whitegrid" are usually best for academic papers
sns.set_theme(style="ticks", font="sans-serif", context="paper")
# Force the global color palette
sns.set_palette(list(my_colors.values()))

# 3. RUN TRACKING
all_peak_data = [] # We'll store results here

print("Starting tracking and generating debug plots...")
results_list = []

print("Processing scans... check the './tracking_check' folder when done.")
for i, spectra in enumerate(spectras):
    current_abs = np.abs(spectra)
    
    # Run tracking + plotting
    intensities = processing.peak_integral(current_abs, ppm_axis, metabolites,tolerance=1.5, scan_idx=i)
    results_list.append(intensities)

# 4. FINAL DATAFRAME
df_final = pd.DataFrame(results_list)

time_start = scantime[0]
seconds_elapsed = [(t - time_start).total_seconds()/60 for t in scantime]

# 4. FINAL DATAFRAME
df_final = pd.DataFrame(results_list)

# --- TIME CALCULATION ---
time_start = scantime[0]
df_final['time'] = [(t - time_start).total_seconds()/60 for t in scantime]

# --- BINNING ---
bin_size = 5
df_final['bin'] = df_final.index // bin_size

# Calculate the middle time point for each bin
binned_times_pd = df_final.groupby('bin')['time'].apply(lambda x: x.iloc[len(x)//2])

# --- PLOTTING ---
fig, ax = plt.subplots(figsize=(8, 6))

# Loop through the dictionary to plot each metabolite automatically
for name, info in metabolites.items():
    # if name == 'Inorganic Phosphate':
    #     continue
    color = info['color']
    time_limit = 300
    df_final_filtered = df_final[df_final['time']<=300].copy()
    
    # 1. Get raw data and binned stats from df_final
    raw_intensities = df_final_filtered[name]
    binned_stats = df_final_filtered.groupby('bin')[name].agg(['mean', 'std'])
    #binned_stats = df_final_filtered.groupby('bin')[name].agg(['mean', 'std'])
    
    mask = binned_times_pd<= time_limit
    # 4. Filter the data for plotting
    plot_times = binned_times_pd[mask]
    plot_means = binned_stats.loc[mask, 'mean'] # Use .loc to ensure index matching
    plot_stds = binned_stats.loc[mask, 'std']

    # 2. Scatter raw points
    ax.scatter(df_final_filtered['time'], raw_intensities, alpha=0.3, s=30, label=name, color=color)
    
    # 3. Plot error bars (Binned data)

    # ax.errorbar(binned_times_pd, binned_stats['mean'], yerr=binned_stats['std'], 
    #             fmt='o-', capsize=5, capthick=2, markersize=10, color=color, linewidth=2)
    ax.errorbar(plot_times, plot_means, yerr=plot_stds, 
            fmt='o-', capsize=5, capthick=2, markersize=10, color=color, linewidth=2)
    

    

# --- AESTHETICS ---
ax.grid()
ax.set_xlabel('Scan Time in Minutes', fontsize=16)
ax.set_ylabel('31P Peak Integral', fontsize=16)
ax.set_title('Trends of 31P Resonances', fontsize=16)
#to make it comparable and not misleading
ax.set_xlim(-25,330)
ax.set_ylim(bottom=0)
#ax.set_yscale('symlog', linthresh=1e5) 


# #print(df_final_filtered['beta-ATP'])
# print('beta 0',df_final_filtered['beta-ATP'].iloc[0])
# print('beta-1',df_final_filtered['beta-ATP'].iloc[-1])
# print('quantification beta atp',df_final_filtered['beta-ATP'].iloc[-1]/df_final_filtered['beta-ATP'].iloc[0])

# #print(df_final_filtered['alpha-ATP'])
# print('alpha 0',df_final_filtered['alpha-ATP'].iloc[0])
# print('alpha-1',df_final_filtered['alpha-ATP'].iloc[-1])
# print('quantification alpha atp',df_final_filtered['alpha-ATP'].iloc[-1]/df_final_filtered['alpha-ATP'].iloc[0])

# #print(df_final_filtered['gamma-ATP'])
# print('gamma 0',df_final_filtered['gamma-ATP'].iloc[0])
# print('gamma -1',df_final_filtered['gamma-ATP'].iloc[-1])
# print('quantification gamma atp',df_final_filtered['gamma-ATP'].iloc[-1]/df_final_filtered['gamma-ATP'].iloc[0])

# #print(df_final_filtered['Inorganic Phosphate'])
# print('Inorganic Phosphate 0',df_final_filtered['Inorganic Phosphate'].iloc[0])
# print('Inorganic Phosphate -1',df_final_filtered['Inorganic Phosphate'].iloc[-1])
# print('quantification Inorganic Phosphate',df_final_filtered['Inorganic Phosphate'].iloc[-1]/df_final_filtered['Inorganic Phosphate'].iloc[0])



# # Define the columns/metabolites you want to check
# metabolites = ['beta-ATP', 'alpha-ATP', 'gamma-ATP', 'Inorganic Phosphate']

# # Print the table header
# print(f"{'Metabolite':<20} | {'Initial (0)':<12} | {'Final (-1)':<12} | {'Ratio (Final/Initial)':<20}")
# print("-" * 78)

# # Loop through and print each row with aligned formatting
# for met in metabolites:
#     initial = df_final_filtered[met].iloc[0]
#     final = df_final_filtered[met].iloc[-1]
#     ratio = (initial-final) / initial if initial != 0 else float('nan') # Avoid division by zero
    
#     # :<25 aligns left with 25 spaces, :.4f formats floats to 4 decimal places
#     print(f"{met:<25} | {initial:<12.4f} | {final:<12.4f} | {ratio:<20.4f}")



#ax.axhline(y=threshold, color='grey', linestyle='--', linewidth=1.5, label='Noise Floor')


# Fix the legend
# handles, labels = ax.get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# ax.legend(by_label.values(), by_label.keys(), fontsize=14)

# Fix the legend by placing it outside the axes
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))

# Move the legend outside to the right (x=1.05, y=1)
# loc='upper left' means the top-left corner of the legend box is at (1.05, 1)
ax.legend(by_label.values(), by_label.keys(), 
          fontsize=10,           # Slightly smaller font
          loc='upper right',     # Force specific location
          bbox_to_anchor=(0.98, 0.7),
          framealpha=0.6,        # Semi-transparent background
          edgecolor='none')      # Remove the hard border

plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'./fig/spectra_overtime/{name_bimsb}')

# Save individual metabolite intensities as .npy files

if name_bimsb == 'leupold_feb':
    df_final = df_final.iloc[:37]


for name in metabolites.keys():
    safe_name = name.replace(' ', '_').replace('-', '_')
    #np.save(f'./processed_data/{name_bimsb}/{safe_name}.npy', df_final[name].to_numpy())

print("Saved intensity .npy files.")


# --- QUANTIFICATION BASED ON PLOTTED BINNED MEANS ---

# 1. Get the binned data
metabolite_names = list(metabolites.keys())
binned_data = df_final_filtered.groupby('bin')[metabolite_names].mean()
# 2. Extract the first and last bins
first_bin_means = binned_data.iloc[0]
last_bin_means = binned_data.iloc[-1]

print("\n--- Quantification based on plotted binned means (First Bin vs Last Bin) ---")
print(f"{'Metabolite':<20} | {'First Bin Avg':<15} | {'Last Bin Avg':<15} | {'Reduction %':<12}")
print("-" * 75)

for met in metabolites:
    start_val = first_bin_means[met]
    end_val = last_bin_means[met]
    
    # Calculate percentage reduction (how much it decreased relative to the start)
    reduction = ((start_val - end_val) / start_val) * 100
    
    print(f"{met:<20} | {start_val:<15.2f} | {end_val:<15.2f} | {reduction:<12.2f}%")