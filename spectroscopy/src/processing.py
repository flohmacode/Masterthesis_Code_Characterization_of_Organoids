import numpy as np
import src.file_utils as file_utils
import src.processing as processing
import src.filtering as filtering
import os
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal.windows import hamming


def linebroadening(rawdata,lb= 30):
    bw =  7936.5 # written in header 
    lb = lb
   # Find the spectral (time-domain) axis - the one with size 2048
    spec_axis = None
    for axis, size in enumerate(rawdata.shape):
        if size == 2048:
            spec_axis = axis
            break
    
    if spec_axis is None:
        raise ValueError("Could not find spectral axis with size 2048 in data shape", rawdata.shape)

    # Create time vector
    t = np.arange(rawdata.shape[spec_axis]) / bw
    
    # Create 1D exponential window
    apod_window_1d = np.exp(-t * np.pi * 2.0 * lb)
    
    # Build target shape: place the 1D window along spec_axis, size 1 for others
    target_shape = [1] * len(rawdata.shape)
    target_shape[spec_axis] = rawdata.shape[spec_axis]
    
    # Reshape 1D window for broadcasting
    apod_window = apod_window_1d.reshape(target_shape)
    
    # Apply filter
    filtered_data = rawdata * apod_window
    
    return filtered_data


def modLineBroadening(rawdata, lb=30):
    bw = 7936.5  # bandwidth in Hz

    # Find the spectral (time-domain) axis - the one with size 2048
    spec_axis = None
    for axis, size in enumerate(rawdata.shape):
        if size == 2048:
            spec_axis = axis
            break
    if spec_axis is None:
        raise ValueError("Could not find spectral axis with size 2048 in data shape", rawdata.shape)

    m = rawdata.shape[spec_axis]
    print('m',m)

    # Time scale (equivalent to MATLAB: t = (0:(m-1))' / recBandwidth)
    t = np.arange(m) / bw

    # Scale by line broadening factor (MATLAB: t = abs(t * lb))
    t = np.abs(t * lb)

    # Modified LB window (MATLAB: modlb = exp(-t) .* (1 + 5*t).^0.2)
    modlb = np.exp(-t) * (1 + 5 * t) ** 0.2

    # Welsh window to avoid cutoff artifacts
    # (MATLAB: modlb = modlb .* (1 - ((0:(m-1))'/m).^2))
    welsh = 1 - (np.arange(m) / m) ** 2
    modlb = modlb * welsh

    #SHAPE FIXING FOR MULTIPLICATION
    # Reshape for broadcasting along the correct axis
    target_shape = [1] * len(rawdata.shape)
    target_shape[spec_axis] = m
    modlb = modlb.reshape(target_shape)
    print('modlb',modlb)

    return rawdata * modlb

def create_hamming_window_for_axis(size, ndim, axis):
    """
    Erstellt ein Hamming Window und formt es für Broadcasting auf eine bestimmte Achse.
    
    Parameters:
    - size: Länge der Dimension
    - ndim: Gesamtanzahl der Dimensionen
    - axis: Achse, auf die das Fenster angewendet werden soll
    
    Returns:
    - window: reshaped Fenster für Broadcasting
    """
    # Hamming window: 0.54 + 0.46 * cos(pi * [-0.5 : 1/(size) : 0.5 - 1/(size)] / 0.5)
    window_1d = hamming(size)
    
    # Shape für Broadcasting: 1 überall außer auf axis
    target_shape = [1] * ndim
    target_shape[axis] = size
    
    window = window_1d.reshape(target_shape)
    return window


def remove_bruker_filterartifacts(fids):

    '''in our bruker system when we scan the first 76 points are not correct and have to be discarded.'''

    filter_points= 76
    fids = np.roll(fids, -filter_points, axis=0)
    fids[-filter_points:] = 0
    return fids
def raw_to_complex(fid):
    '''
    parameters: 
        -fid: the free induction decays as a collection of floats from the rawdatafile (fid_proc.64)

    
    Notes: Takes the raw fid and converts them into real and imaginary values by indexing. 
    the raw data is interleaved of real and imaginary'''
    complex_data = fid[::2] + 1j* fid[1::2]
    return complex_data

def spatial_axes(header):
    # Spatial axes
    enc = header['PVM_EncMatrix']
    fov = header['PVM_Fov']
    
    x_axis = np.linspace(-float(fov[0])/2, float(fov[0])/2, int(enc[0]))
    y_axis = np.linspace(-float(fov[1])/2, float(fov[1])/2, int(enc[1]))
    z_axis = np.linspace(-float(fov[2])/2, float(fov[2])/2, int(enc[2]))
    ppm_axis = file_utils.compute_ppm_axis(header,int(header['PVM_SpecMatrix']))
    spatial_axes = {'x': x_axis, 'y': y_axis, 'z': z_axis, 'ppm': ppm_axis}
    return spatial_axes,ppm_axis

def find_spectral_axis(cdata):

    return cdata.shape.index(2048)

def find_spatial_axis(cdata):
    perm = cdata.shape
    result = []
    
    first_10 = perm.index(10)
    # Find the second 10 by searching from the position after the first 10
    second_10 = perm.index(10, first_10 + 1)
    first_12 = perm.index(12)
    #result.append((first_10, second_10, first_12))
    return (first_10,second_10,first_12)


def fourierpipeline(cdata):
    
    ax_spec = processing.find_spectral_axis(cdata)
    ax_spat = processing.find_spatial_axis(cdata)
    
    ax_spat = tuple(sorted(ax_spat))#??? does this make a difference?

    # Spectral FFT
    #spectra = np.fft.fftshift(cdata, axes=ax_spec) #commented out bc clemens didnt do it, 
    spectra = np.fft.fft(cdata, axis=ax_spec)
    spectra = np.fft.fftshift(spectra, axes=ax_spec)
    

    ##HAMMING WINDOW MIGHT BE WRONG, FOR NOW COMMMENTED OUT
    
    # for axis in ax_spat:
    #     size = cdata.shape[axis]
    #     window = filtering.create_hamming_window_for_axis(size, cdata.ndim, axis)
        
    #     print(f"Applying Hamming window to axis {axis} (size={size})")
        
    #     # fftshift -> multiply window -> fft -> fftshift
        
    #     spectra = spectra * window
    #     spectra = np.fft.fft(spectra, axis=axis)
    #     spectra = np.fft.fftshift(spectra, axes=axis)
    
    # return spectra
    
    # Spatial FFT (FFT on spatial axes works best)
    spectra = np.fft.fftshift(np.fft.fftn(spectra, axes=ax_spat), axes=ax_spat)
    #spectra = np.fft.fftshift(spectra, axes=ax_spec)
    #spectra = np.fft.fftn(spectra, axes=ax_spat)


    return spectra



def read_fid_proc64file(study_directory,scan_no):
    '''For csi scans from Clemens and NSPECTS u have to load the fid_proc.64 file'''

    fid_ls = []
    fid_file = os.path.join(study_directory, str(scan_no), 'pdata', '1', 'fid_proc.64')

    with open(fid_file, 'rb') as f:
        raw_fid = np.fromfile(f, dtype=np.float64)
        fid_ls.append(raw_fid)

    return fid_ls[0]


#giorgi
def signaltonoise(ppm, spects_X, signal_i, signal_f, boolean=1, noise_i=None, noise_f=None):
    """
    Computes the signal-to-noise ratio (SNR) for specified spectral regions.

    Parameters:
        ppm (np.ndarray): The PPM (parts per million) axis values of the spectrum.
        spects_X (np.ndarray): The spectral data corresponding to the PPM axis.
        signal_i (float, optional): The starting value (in ppm) of the signal region.
        signal_f (float, optional): The ending value (in ppm) of the signal region.
        boolean (int, optional): Determines the noise region selection method:
                                 - 0: Explicit noise region specified by `noise_i` and `noise_f`.
                                 - 1(default): Noise region automatically inferred as all data outside the signal region.
        noise_i (float, optional): The starting value (in ppm) of the noise region. Only used if `boolean` is 0. Default is None.
        noise_f (float, optional): The ending value (in ppm) of the noise region. Only used if `boolean` is 0. Default is None.

    Returns:
        tuple: A tuple containing:
            - float: The computed signal-to-noise ratio (SNR).
            - float: The maximum amplitude of the signal region.
            - float: The noise level, calculated as the standard deviation of the noise region.
    """
    # Mask the signal and noise regions
    signal_mask = (ppm >= signal_i) & (ppm <= signal_f)
    if boolean == 0:
        noise_mask = (ppm >= noise_i) & (ppm <= noise_f)
    else:
        noise_mask = ~signal_mask
    # Extract the signal and noise data
    signal_data = spects_X[signal_mask]
    noise_data = spects_X[noise_mask]

    # Calculate the signal amplitude 
    signal_amplitude = np.max(signal_data)

    # Calculate the noise level (use standard deviation)
    noise_level = np.std(noise_data)

    # Compute SNR
    snr = signal_amplitude / noise_level

    return snr, signal_amplitude, noise_level


def date_of_scan(header):
    from datetime import datetime

    part = header['ACQ_abs_time']

    timefrom1970,_,_= part.strip("()").split(",")

    dt_object = datetime.fromtimestamp(int(timefrom1970))
    #print(dt_object)
    return dt_object

def scanduration(header):
    #print(header['PVM_ScanTime'])
    scantime_ms = header['PVM_ScanTime']

    total_seconds = scantime_ms / 1000

    # 2. Calculate minutes and remaining seconds
    minutes, seconds = divmod(total_seconds, 60)
    return minutes,seconds


def check_spectras(spectras,ppm_axis):
    #spectras is a list

    # PLOT AND SAVE ALL SPECTRA
    for key,value in enumerate(spectras[:]):
        spec = value
        spec = np.roll(spec,-235)
        #plt.figure(figsize=(12, 6))
        lpidx = key
        passed_time = (key + 1) * 8

        fig, ax = plt.subplots()
        #timestamp_str = scantime[lpidx].strftime("%H:%M:%S")

        colorpalette = sns.color_palette("viridis")
        sns.lineplot(x=ppm_axis[:], y=np.abs(spec).flatten())
        plt.xlabel("Chmical shift (ppm)")
        plt.axvline(5.02,label = 'Inorganic Phosphate (theoretical)', color = 'black', linestyle='--')
        plt.axvline(x=-2.48, color='red', linestyle='--', label='γ-ATP (theoretical)')
        plt.axvline(x=-7.52, color='green', linestyle='--', label='α-ATP (theoretical)')
        plt.axvline(x=-16.26, color='purple', linestyle='--', label='β-ATP (theoretical)')

        plt.legend(loc = 'upper right')
        plt.ylim(0,600000)
        plt.xlim(25,-25)
        plt.ylabel(f"Signal Intensity")
        plt.title(f'31P Spectrum of 20 Organoids: Scan No. {key}')
        plt.grid()
        plt.tight_layout()
        plt.show()
        #plt.savefig(f'./fig/leupoldspec/scan_no{key}')
        plt.close()




def find_peak(spectra,x_coordinate,ppm_axis,tolerance = 2):
    """
    Finds the peak location and intensity within a specified range of the x-axis.
    
    Parameters
    ----------
    spectra : array-like
        The spectral intensity values.
    x_coordinate : float
        The center point on the x-axis around which to search for the peak.
    ppm_axis : array-like
        The x-axis values (typically in ppm for NMR spectra).
    tolerance : float, optional
        The range around x_coordinate to search (default is 3).
        The search range is [x_coordinate - tolerance, x_coordinate + tolerance].
    
    Returns
    -------
    tuple
        peakidx_global : int
            The index of the peak in the original spectra array.
        peak_intensity : float
            The intensity value at the peak location.
    """

    # 1. Define the search mask
    area = (ppm_axis < x_coordinate + tolerance) & (ppm_axis > x_coordinate - tolerance)
    
    # 2. Extract the indices where 'area' is True
    indices_in_window = np.where(area)[0]
    
    if len(indices_in_window) == 0:
        print(f"DEBUG: No data found for {x_coordinate} ppm within tolerance {tolerance}")
        return None, 0

    # 3. Find peak in the subset
    peakidx_local = np.argmax(spectra[area])
    peakidx_global = indices_in_window[peakidx_local]
    peak_intensity = spectra[peakidx_global]


    return peakidx_global, peak_intensity

def find_peak_debug(spectra, x_coordinate, ppm_axis, tolerance=2):
    # 1. Define the search mask
    area = (ppm_axis < x_coordinate + tolerance) & (ppm_axis > x_coordinate - tolerance)
    
    # 2. Extract the indices where 'area' is True
    indices_in_window = np.where(area)[0]
    
    if len(indices_in_window) == 0:
        print(f"DEBUG: No data found for {x_coordinate} ppm within tolerance {tolerance}")
        return None, 0

    # 3. Find peak in the subset
    peakidx_local = np.argmax(spectra[area])
    peakidx_global = indices_in_window[peakidx_local]
    peak_intensity = spectra[peakidx_global]

    # # --- DEBUG PLOTTING ---
    # plt.figure(figsize=(10, 4))
    # # Plot the full spectrum vs indices so we can't be confused by PPM units
    # plt.plot(spectra, label='Full Spectrum (Indices)', color='gray', alpha=0.5)
    
    # # Highlight the search window
    # plt.axvspan(indices_in_window[0], indices_in_window[-1], color='yellow', alpha=0.3, label='Search Window')
    
    # # Mark the found peak
    # plt.axvline(peakidx_global, color='red', linestyle='--', label=f'Found Peak at Index {peakidx_global}')
    # plt.scatter(peakidx_global, peak_intensity, color='red', zorder=5)
    
    # plt.title(f"Searching for {x_coordinate}ppm | Found at idx: {peakidx_global} | Val: {peak_intensity:.2f}")
    # plt.legend()
    # plt.show()
    # # -----------------------

    return peakidx_global, peak_intensity


def verify_scan_tracking(spectra, ppm_axis, targets, tolerance=2, scan_idx=0, save_dir="./tracking_check"):
    """
    Saves a plot with the exact colors requested:
    Red=Pi, Orange=gamma, Green=alpha, Blue=beta
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(14, 6))
    plt.plot(spectra, color='gray', alpha=0.3, label='Data')
    
    found_values = {}

    for name, info in targets.items():
        # Define range
        target_ppm = info['ppm']
        color = info['color']
        
        area = (ppm_axis < target_ppm + tolerance) & (ppm_axis > target_ppm - tolerance)
        indices = np.where(area)[0]
        
        if len(indices) == 0:
            found_values[name] = 0
            continue

        # Find peak
        local_idx = np.argmax(spectra[area])
        global_idx = indices[local_idx]
        peak_val = spectra[global_idx]
        found_values[name] = peak_val

        # Plot window and peak
        plt.axvspan(indices[0], indices[-1], color=color, alpha=0.15, label=f'{name} search')
        plt.scatter(global_idx, peak_val, color=color, s=60, edgecolors='black', zorder=5)
        plt.annotate(f"{name}", (global_idx, peak_val), textcoords="offset points", 
                     xytext=(0,10), ha='center', fontsize=9, fontweight='bold', color=color)

    plt.title(f"Metabolite Tracking: Scan {scan_idx} (Verification Plot)")
    plt.xlabel("Array Index")
    plt.ylabel("Intensity")
    plt.legend(loc='upper right', ncol=2)
    plt.grid(alpha=0.2)
    
    plt.savefig(f"{save_dir}/scan_{scan_idx:03d}.png")
    plt.close() 

    return found_values


def peak_integral(spectra, ppm_axis, targets, tolerance=2,scan_idx = 0,normalize_total = False):

    #plt.figure(figsize=(14, 6))
    #plt.plot(spectra, color='gray', alpha=0.3, label='Data')

    # 1. Define the search mask
    found_values = {}
    noise_floor = np.mean(spectra[1500:2000])
    
    for name, info in targets.items():
        # Define range
        target_ppm = info['ppm']
        color = info['color']

        area = (ppm_axis < target_ppm + tolerance) & (ppm_axis > target_ppm - tolerance)
    
        # 2. Extract the indices where 'area' is True
        indices_in_window = np.where(area)[0]

        #plt.axvspan(indices_in_window[0], indices_in_window[-1], color='blue', alpha=0.15, label=f'inorganic phosphate')

        raw_sum = np.sum(spectra[indices_in_window])
        baseline_area = noise_floor * len(indices_in_window)
        
        # Max(0, ...) ensures we don't get negative areas from random noise
        found_values[name] = max(0, raw_sum - baseline_area)

    # --- 3. OPTIONAL: CONSTANT SUM NORMALIZATION ---
    if normalize_total:
        total_sum = sum(found_values.values())
        if total_sum > 0:
            for name in found_values:
                found_values[name] = found_values[name] / total_sum


    return found_values
