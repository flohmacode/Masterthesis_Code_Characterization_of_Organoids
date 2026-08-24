import os
import numpy as np
import src.file_utils as file_utils
import src.processing as processing
import matplotlib.pyplot as plt
import src.spectroscopy as spectroscopy
import src.processing as processing
import seaborn as sns

'''Plots the minimum amount experiment'''

name_bimsb = 'minimum_amount'
study_directory =  f"./spectroscopy/data/{name_bimsb}"

file_utils.read_bruker_study(study_directory)

#LOAD Only NSPECT SCANS
studylist = file_utils.read_bruker_study(study_directory)
nspectlist = []
for element in studylist:
    if '<Bruker:NSPECT>' in element:
        nspectlist.append(element)
nspecdict =  dict(nspectlist)

#PROCESS NSPECT SCANS to Spectra
specls = []
expidxls = []
missing_data = []
scantimeduration = []
scantime = []

for i in nspectlist:
    idx,scan = i
    if os.path.isfile(os.path.join(study_directory, str(idx), 'pdata', '1', 'fid_proc.64')):
        fids, spects, ppm_axis, header = spectroscopy.read_NSPECT(study_directory,idx) 
        filtered_fid= processing.linebroadening(fids,20)
        spects = np.fft.fftshift(np.fft.fft(filtered_fid, axis=0), axes=0)
        specls.append(spects)
        expidxls.append(idx)
        scantimeduration.append(processing.scanduration(header))
        scantime.append(processing.date_of_scan(header))
    else:
        missing_data.append(idx)

# SORT ALL DATA BY SCANTIME
idx = np.argsort(scantime)
expidx_arr = np.array(expidxls)[idx]
scantime_arr = np.array(scantime)[idx]
specls_arr = np.array(specls)[idx]

# roll of the first two spectra since they were recorded 
# with the wrong settings compared to the other scans
if name_bimsb == 'leupold_feb':
    specls_arr[0] = np.roll(specls_arr[0],-15)
    specls_arr[1] = np.roll(specls_arr[1],325)

if name_bimsb == 'leupold_dec':
    specls_arr[0] = np.roll(specls_arr[0],340)
    specls_arr[1] = np.roll(specls_arr[1],0)

# PLOT AND SAVE ALL SPECTRA
for key,value in enumerate(specls_arr[:]):
    spec = value
    spec = np.roll(spec,-235)
    lpidx = key
    passed_time = (key + 1) * 8
    fig, ax = plt.subplots()
    timestamp_str = scantime[lpidx].strftime("%H:%M:%S")
    colorpalette = sns.color_palette("viridis")
    sns.lineplot(x=ppm_axis[:], y=np.abs(spec[:, :]).flatten())
    plt.xlabel("Chemical shift (ppm)")
    plt.axvline(5.02,label = 'Inorganic Phosphate: +5.02ppm', color = 'black', linestyle='--')
    plt.axvline(x=-2.48, color='red', linestyle='--', label='γ-ATP: -2.48ppm')
    plt.axvline(x=-7.52, color='green', linestyle='--', label='α-ATP: -7.52ppm')
    plt.axvline(x=-16.26, color='purple', linestyle='--', label='β-ATP: -16.26ppm')

    plt.legend(loc = 'upper right')
    plt.xlim(25,-25)
    plt.ylabel(f"Signal Intensity")
    plt.title(f'31P Spectrum of 12 Organoids of Size 1mm: Scan No. {key}')
    plt.grid()
    plt.tight_layout()
    plt.show()
    