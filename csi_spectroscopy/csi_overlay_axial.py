
import sys
import os

# Ensure the project root (parent of this file's folder) is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import csi_spectroscopy.src.file_utils as file_utils
import csi_spectroscopy.src.processing as processing
import csi_spectroscopy.src.csiplots2stable as csiplots
import matplotlib.pyplot as plt
import csi_spectroscopy.src.mr_imaging as mr_imaging
study_directory = './csi_data/atpspec/20260107_145049_MDC_0230_fm_organoids_260107_organoid_atp_spec2_1_15'

file_utils.read_bruker_study(study_directory)

experimentindice = [16,17,18]
#experimentindice = [94,96]

experiments = []
spatial = None
ppm_axis = None
p = (12,10,10,2048)


#We load and filter the experiments described in experimentindice.
for i in experimentindice:

    scan_no = i
    header = file_utils.read_bruker_all_headers(study_directory, scan_no)
    p,transposition = processing.check_orientation(header['PVM_SPackArrGradOrient'])
    rawdata = processing.read_fid_proc64file(study_directory,scan_no)
    rawdata = processing.remove_bruker_filterartifacts(rawdata)
    cdata = processing.raw_to_complex(rawdata)
    spatial,ppm_axis = processing.spatial_axes(header)
    kspace = cdata.reshape(tuple(p))
    kspace = processing.linebroadening(kspace,30)
    spects = processing.fourierpipeline(kspace)
    spatial = spatial
    ppm_axis = ppm_axis
    experiments.append(spects)


study_directory_img = './csi_data/atpspec/20260107_145049_MDC_0230_fm_organoids_260107_organoid_atp_spec2_1_15'
scan_img_number = 10
slice_idx = 1

fig_temp, ax_temp, img_mat = mr_imaging.get_reconstructed_img(study_directory_img, scan_img_number, slice_idx, 'axial')
plt.close()

header_img = file_utils.read_bruker_all_headers(study_directory_img, scan_img_number)
fov_img = header_img['PVM_Fov']

header_spec = file_utils.read_bruker_all_headers(study_directory, scan_no = 18)
fov_spec = header_spec['PVM_Fov']
offset = header_spec['ACQ_slice_offset']

fig, ax = csiplots.csi_overlay_axial(
    spects=spects,
    img_array=img_mat,
    idxtoslice=slice_idx,
    spatial=spatial,
    fov_spec = fov_spec,
    fov_img = fov_img,
    offset=offset,
    figsize = (8,8),
    title = 'Chemical Shift Image Axial'
)
plt.savefig('./fig_csi/axial_nmr')
plt.show()