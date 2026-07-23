import numpy as np
import src.file_utils as file_utils
import seaborn as sns
import matplotlib.pyplot as plt

#code written by giorgi
def read_NSPECT(study_directory, scan_no):
    """
    Processes Bruker NSPECT data to compute FIDs, spectra, ppm axis, and header information.

    Parameters:
        study_directory (str): Path to the directory containing the Bruker study.
        scan_no (int): The scan number to process.

    Returns:
        tuple: Contains the following elements:
            - np.ndarray: FIDs.
            - np.ndarray: Spectra.
            - np.ndarray: PPM axis.
            - dict: Header information.
    """
    
    fids, header = file_utils.read_bruker_readout(study_directory, scan_no,'spectroscopy')
    n_points = int(header['PVM_SpecMatrix'])
    filter_points = 76

    # Process FIDs
    fids = np.roll(fids, -filter_points, axis=0)
    fids[-filter_points:] = 0

    # Compute spectra and ppm axis
    spects = np.fft.fftshift(np.fft.fft(fids, axis=0), axes=0)
    
   # print(float(header['PVM_SpecSW']))
    ppm_axis = file_utils.compute_ppm_axis(header, n_points)
    #print(ppm_axis)

    return fids, spects, ppm_axis, header


def find_peak(spectra,x_corrdinate,ppm_axis,tolerance = 3):

    area = (ppm_axis < x_corrdinate+tolerance)&(ppm_axis > x_corrdinate-tolerance)#((ppm_axis < x_corrdinate))
    
    peakidx_local = np.argmax(spectra[area])#idx where biggest peak is to be found

    peakidx_global = np.where(area)[0][peakidx_local]

    return peakidx_global, spectra[peakidx_global]


def plot_spectra(spectra,ppm_axis,key = 0):
    fig, ax = plt.subplots()
    #timestamp_str = scantime[lpidx].strftime("%H:%M:%S")
    #ax.text(0.5,0.9,s= f"Scan Time: {timestamp_str} Duration {passed_time}min", 
            # transform=ax.transAxes,
            # fontsize=9, 
            # bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    sns.lineplot(x=ppm_axis[:], y=np.abs(spectra[:, :]).flatten())

    plt.xlabel("PPM")
    # plt.axvline(x=9.5, color='yellow', linestyle='--', label='PCr (theoretical)')
    
    # plt.axvline(x=3, color='red', linestyle='--', label='a-ATP (theoretical)')
    # plt.axvline(x=-1.8, color='green', linestyle='--', label='b-ATP (theoretical)')
    # plt.axvline(x=-11, color='purple', linestyle='--', label='y-ATP (theoretical)')
    plt.xlim(25,-25)
    plt.ylabel(f"Spectrum Intensity exp: {key}")
    plt.title(f'Spectra of Organoids')
    plt.grid()
    plt.show()