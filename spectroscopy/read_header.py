import os
import numpy as np
import src.file_utils as file_utils
import src.spectroscopy as spectroscopy


"""This Script reads headerfiles of scans for extracting specific scaninformation"""

study_directory =  "./spectroscopy/data/jojo_april"
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

print(header_list[0].keys())
