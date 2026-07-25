import numpy as np
import src.file_utils as file_utils
import src.spectroscopy as spectroscopy
import src.processing as processing
import matplotlib.pyplot as plt
import seaborn as sns

'''This script can overlay two spectra of choice'''

study_directory_org =  "./spectroscopy/data/jojo_april"
file_utils.read_bruker_study(study_directory_org)
fids_org, spect_org, ppm_axis, header = spectroscopy.read_NSPECT(study_directory_org,3) 


study_directory_org =  "./spectroscopy/data/leupold_feb"
file_utils.read_bruker_study(study_directory_org)
fids_org, spect_org, ppm_axis, header = spectroscopy.read_NSPECT(study_directory_org,13) 

fid_org_filtered = processing.modLineBroadening(fids_org,80)
fid_org_filtered2 = processing.linebroadening(fids_org,20)

spects_org = np.fft.fftshift(np.fft.fft(fid_org_filtered, axis=0), axes=0)
spects_org_flip = np.flip(spects_org, axis=-1)
spects_org_shift = np.roll(spects_org_flip,-250)

spects_org = np.fft.fftshift(np.fft.fft(fid_org_filtered2, axis=0), axes=0)
spects_org_flip = np.flip(spects_org, axis=-1)
spects_org_shift2 = np.roll(spects_org_flip,-250)

study_directory_reference =  "./spectroscopy/data/reference/20260107_145049_MDC_0230_fm_organoids_260107_organoid_atp_spec2_1_15"
file_utils.read_bruker_study(study_directory_reference)
fids_ref, spect_ref, ppm_axis, header = spectroscopy.read_NSPECT(study_directory_reference,11) 

spect_ref = np.roll(spect_ref,110)

spec1_norm = np.abs(spects_org_shift[:,:]) / np.max(np.abs(spects_org_shift[:,:]))
spec1_norm2 = np.abs(spects_org_shift2[:,:]) / np.max(np.abs(spects_org_shift2[:,:]))
spec2_norm = np.abs(spect_ref[:, :]) / np.max(np.abs(spect_ref))

fig,ax = plt.subplots(figsize = (14,6))

sns.lineplot(x=ppm_axis[:], y=spec1_norm.flatten(),label= 'gaussian')
sns.lineplot(x=ppm_axis[:], y=spec1_norm2.flatten(),label= 'linebroad')
plt.title('Measured Signal and Reference ATP')
plt.xlabel('Chemical Shift (ppm)')
plt.ylabel('Normed Signal Intensity')
plt.legend()
plt.xlim(25,-25)
plt.grid()
#plt.savefig('./fig/filtering')
plt.show()