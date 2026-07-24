import sys
import os

# Ensure the project root (parent of this file's folder) is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.mr_imaging as mr_imaging
import matplotlib.pyplot as plt


"""This Script allows to read TURBORARE-SCANS"""


study_directory = './csi_data/20251222_112930_MDC_0230_fm_organoids_251222_Organoid_paula_leupold__1_13/25/pdata/1'
idx = 3
mr_imaging.plot_turborare_dataset(mr_imaging.turborareload(study_directory),idx)
plt.savefig('./fig_csi/axial_organoids')
plt.show()

study_directory2 = './csi_data/jojo_april/4/pdata/1'
idx2= 7
mr_imaging.plot_turborare_dataset(mr_imaging.turborareload(study_directory2),idx2)
plt.savefig('./fig_csi/coronal_organoids')
plt.show()