import os
import numpy as np
import src.file_utils as file_utils
import src.processing as processing
import src.spectroscopy as spectroscopy

import src.filtering as filtering
import matplotlib.pyplot as plt

import seaborn as sns

study_directory =  "./data/20260218_181048_MDC_0230_fm_organoids_260218_cerebral_organoids_paula_1_19"
file_utils.read_bruker_study(study_directory)



studylist = file_utils.read_bruker_study(study_directory)
nspectlist = []
for element in studylist:
    if '<Bruker:NSPECT>' in element:
        nspectlist.append(element)
nspecdict =  dict(nspectlist)

header_list = []
for i in nspectlist:
    idx,scan = i

    if os.path.isfile(os.path.join(study_directory, str(idx), 'pdata', '1', 'fid_proc.64')):

        fids, spects, ppm_axis, header = spectroscopy.read_NSPECT(study_directory,idx) 

        header_list.append(header)

print(header_list[4]['PVM_FrqWork'])

#print(header_list[0].keys())
# print(header_list[4]['PVM_FrqWorkOffset'])
# print(header_list[4]['PVM_FrqWorkOffsetPpm'])
# print(header_list[4]['PVM_FrqWorkPpm'])
# print(header_list[4]['PVM_NucleiPpmWork'])
