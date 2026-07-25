import sys
import os

# Ensure the project root (parent of this file's folder) is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import src.file_utils as file_utils
import src.processing as processing
import matplotlib.pyplot as plt


'''This Script shows represntative FIDs'''


study_directory = "/Users/neo/Documents/ComputerEngMs/MDCMaster/Masterthesis_Code/data/MDC_0230_fm_organoids_MDC_0230_fm_organoids_334710_1_Default_251202_organoid_water_368290_360.3.6/20251202_104612_MDC_0230_fm_organoids_251202_organoid_water_1_3"
scan_no = 16
file_utils.read_bruker_study(study_directory)

header = file_utils.read_bruker_all_headers(study_directory, scan_no)
fid_ls = []
fid_file = os.path.join(study_directory, str(scan_no), 'pdata', '1', 'fid_proc.64')

with open(fid_file, 'rb') as f:
    raw_fid = np.fromfile(f, dtype=np.float64)
    print('rawfid0',raw_fid.shape)
    fid_ls.append(raw_fid)


rawdata = fid_ls[0]
rawdata = processing.remove_bruker_filterartifacts(rawdata)
cdata = processing.raw_to_complex(rawdata)
spatial,ppm_axis = processing.spatial_axes(header)

plt.plot(rawdata[2500000:2510000])
plt.xlabel('Sample Index')
plt.ylabel('Sample Amplitude (unitless)')
plt.show()

plt.plot(rawdata[2502500:2503500:2])
plt.title('Representative FID')
plt.xlabel('Sample Index')
plt.ylabel('Sample Amplitude (unitless)')
plt.show()