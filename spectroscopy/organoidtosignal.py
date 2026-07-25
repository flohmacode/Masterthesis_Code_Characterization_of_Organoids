import numpy as np
import matplotlib.pyplot as plt
import src.spectroscopy as spectroscopy
import src.file_utils as file_utils
import src.processing as processing

study_directory_org =  "./spectroscopy/data/jojo_april"
file_utils.read_bruker_study(study_directory_org)
fids_org, _, ppm_axis, header = spectroscopy.read_NSPECT(study_directory_org,3) 
filtered_fid= processing.linebroadening(fids_org,20)
spect_org = np.fft.fftshift(np.fft.fft(filtered_fid, axis=0), axes=0)
spect_org = spect_org.squeeze()
spect_org = np.flip(spect_org, axis=-1)
spect_org = np.roll(spect_org,235)

study_directory_stressed =  "./spectroscopy/data/minimum_amount"
fids_stress, spect_stress, ppm_axis, header = spectroscopy.read_NSPECT(study_directory_stressed, 2)
filtered_fid= processing.linebroadening(fids_stress,20)
spect_stress = np.fft.fftshift(np.fft.fft(filtered_fid, axis=0), axes=0)
spect_stress = spect_stress.squeeze()
spect_stress = np.flip(spect_stress, axis=-1)
spect_stress = np.roll(spect_stress,235)

# --- 2. CALCULATE THE INTEGRALS (RAW SCANNER UNITS) ---
def findpeak(spectra, area, ppm_axis, tolerance=4):
    area_mask = (ppm_axis < area + tolerance) & (ppm_axis > area - tolerance)
    
    # Find the maximum value and its local index inside the masked area
    max_val = np.max(spectra[area_mask])
    local_idx = np.argmax(spectra[area_mask])
    
    # Map the local index back to the global ppm_axis
    global_idx = np.where(area_mask)[0][local_idx]
    ppm_coordinate = ppm_axis[global_idx]
    
    return max_val, ppm_coordinate

# Raw integral value for your 18 organoid sample (e.g., around 1e8)
measured_org18,idx1 = findpeak(np.abs(spect_org), 18, ppm_axis, tolerance=4)

# Raw integral value for your new, lower-biomass stressed sample
measured_stressed_signal,idx2= findpeak(np.abs(spect_stress), 18, ppm_axis, tolerance=4)

# --- 3. THE SIMPLIFIED CALIBRATION ---
num_org_healthy = 18
# Calculate the expected raw signal intensity contributed per healthy organoid
signal_per_healthy_org = measured_org18 / num_org_healthy

# # Generate the line based entirely on raw scanner units
organoid_counts = np.arange(0, 26) 
expected_signals = signal_per_healthy_org * organoid_counts

num_stressed_org = 12

signal_per_stressed_org = measured_stressed_signal / num_stressed_org
expected_signals_stressed = signal_per_stressed_org * organoid_counts

plt.plot(organoid_counts, expected_signals,label = '3mm Organoid')
plt.scatter([18],measured_org18)
plt.plot(organoid_counts, expected_signals_stressed,label = '1mm Organoid')
plt.scatter([12],measured_stressed_signal)
plt.legend()
plt.grid()
plt.xlabel('Number of Orgnaoids')
plt.ylabel('Expected Signal')
plt.title('Estimated Signal of Organoids of varying sizes')
plt.show()

def organoid_volume(diameter):
    volume_one_organoid = (4/3) * np.pi * ((diameter/2) ** 3) 
    return volume_one_organoid

print(organoid_volume(1)*12)
