import numpy as np
import os
import pyAMARES
import src.spectroscopy as spectroscopy
import src.file_utils as file_utils
import src.processing as processing


# import sys
# import os

# # Ensure the project root (parent of this file's folder) is on sys.path
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def save_fid_for_pyamares():

    #LOAD ALL DATA
    study_directory =  "./data/20260218_181048_MDC_0230_fm_organoids_260218_cerebral_organoids_paula_1_19"
    file_utils.read_bruker_study(study_directory)

    #LOAD Only NSPECT SCANS
    studylist = file_utils.read_bruker_study(study_directory)
    nspectlist = []
    for element in studylist:
        if '<Bruker:NSPECT>' in element:
            nspectlist.append(element)
    nspecdict =  dict(nspectlist)

    freeinductiondecays = []

    headerlist = []

    for i in nspectlist:
        idx,scan = i

        if os.path.isfile(os.path.join(study_directory, str(idx), 'pdata', '1', 'fid_proc.64')):

            fids, spects, ppm_axis, header = spectroscopy.read_NSPECT(study_directory,idx) 
            
            freeinductiondecays.append(fids)
            headerlist.append(header)
            
    np.save('./processed_data/fid_amares.npy',np.array(freeinductiondecays))


fid_data_all = np.load('./spectroscopy/processed_data/fid_amares.npy')

fid_data_all = fid_data_all.squeeze()

fid_data = fid_data_all[0]

filtered_fid= processing.linebroadening(fid_data,20)

# 2. Scanner Parameters (Extracted from the Header)
SWH = 7936.507936507936    # PVM_SpecSWH
MHz = 162.04150323     # PVM_FrqWork
# PVM DeadTime is 0.05ms -> Convert to seconds for pyAMARES
dead_time_s = 0.05 * 1e-3 

#SHIFTING
current_peak_ppm = -4  
shift_needed = -10 - current_peak_ppm

# 2. Apply a frequency shift to the FID
# A shift in frequency domain = linear phase ramp in time domain
t = np.linspace(0, len(filtered_fid)/SWH, len(filtered_fid))
shift_hz = shift_needed * MHz
filtered_fid = filtered_fid * np.exp(1j * 2 * np.pi * shift_hz * t)

FIDobj = pyAMARES.initialize_FID(
    fid=filtered_fid,
    MHz=MHz,
    sw=SWH,
    deadtime=dead_time_s,
    normalize_fid=False,
    priorknowledgefile="./spectroscopy/pknowledge_normal.csv",
    preview=False,
    xlim=(25,-25),
    #delta_phase=4
)

params_hsvd = pyAMARES.HSVDinitializer(
    fid_parameters=FIDobj,
    num_of_component=12,  # If error happens, decreasae this number
    fitting_parameters=FIDobj.initialParams,
    preview=False,
)

FIDresult1 = pyAMARES.fitAMARES(
    fid_parameters=FIDobj,
    fitting_parameters=params_hsvd,
    method="least_squares",
    ifplot=True,
    inplace=False,
)
#FIDresult1.result_sum.to_csv('fitted_tabular.csv')
