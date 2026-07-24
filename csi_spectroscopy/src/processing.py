import numpy as np
import csi_spectroscopy.src.file_utils as file_utils
import csi_spectroscopy.src.processing as processing
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.signal.windows import hamming

def linebroadening(rawdata,lb= 30):
    """
    Applies exponential line broadening to spectral data along the spectral (time-domain) axis.

    Parameters:
    - rawdata (numpy.ndarray): Input spectral data. Must contain an axis with size 2048.
    - lb (float, optional): Line broadening factor. Defaults to 30.

    Returns:
    - numpy.ndarray: Filtered data with exponential line broadening applied.

    Notes:
    - The spectral axis is dynamically identified as the axis with size 2048.
    - The exponential window is created and reshaped for broadcasting.
    """
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
    """
    Removes artifacts from Bruker data by discarding the first 76 points and zero-padding the end.

    Parameters:
    - fids (numpy.ndarray): Input FID (Free Induction Decay) data.

    Returns:
    - numpy.ndarray: FID data with the first 76 points removed and zero-padded at the end.
    """
    filter_points= 76
    fids = np.roll(fids, -filter_points, axis=0)
    fids[-filter_points:] = 0
    return fids
def raw_to_complex(fid):
    """
    Converts raw FID data into complex values by interleaving real and imaginary parts.

    Parameters:
    - fid (numpy.ndarray): Raw FID data as a collection of floats (interleaved real and imaginary).

    Returns:
    - numpy.ndarray: Complex-valued FID data.

    Notes:
    - The raw data is assumed to be interleaved as [real0, imag0, real1, imag1, ...].
    """
    complex_data = fid[::2] + 1j* fid[1::2]
    return complex_data

def spatial_axes(header):
    """
    Computes spatial axes (x, y, z) and the ppm axis from Bruker header information.

    Parameters:
    - header (dict): Bruker header containing encoding matrix and field of view.

    Returns:
    - tuple: (spatial_axes, ppm_axis)
      - spatial_axes (dict): Dictionary with keys 'x', 'y', 'z' containing the spatial axis arrays.
      - ppm_axis (numpy.ndarray): PPM (parts per million) axis for spectral data.
    """
    enc = header['PVM_EncMatrix']
    fov = header['PVM_Fov']
    
    x_axis = np.linspace(-float(fov[0])/2, float(fov[0])/2, int(enc[0]))
    y_axis = np.linspace(-float(fov[1])/2, float(fov[1])/2, int(enc[1]))
    z_axis = np.linspace(-float(fov[2])/2, float(fov[2])/2, int(enc[2]))
    ppm_axis = file_utils.compute_ppm_axis(header,int(header['PVM_SpecMatrix']))
    spatial_axes = {'x': x_axis, 'y': y_axis, 'z': z_axis, 'ppm': ppm_axis}
    return spatial_axes,ppm_axis


def find_spectral_axis(cdata, spec_size=2048):
    """
    Dynamically finds the spectral axis in a multi-dimensional array.

    Parameters:
    - cdata (numpy.ndarray): Input data array.
    - spec_size (int, optional): Expected size of the spectral axis. Defaults to 2048.

    Returns:
    - int: Index of the spectral axis.

    Notes:
    - If `spec_size` is not found, the largest dimension is assumed to be the spectral axis.
    """
    if spec_size in cdata.shape:
        return cdata.shape.index(spec_size)
    else:
        # Fallback: assume the largest dimension is the spectral axis
        return cdata.shape.index(max(cdata.shape))

def find_spatial_axis(cdata):
    """
    Dynamically finds the spatial axes in a multi-dimensional array by excluding the spectral axis.

    Parameters:
    - cdata (numpy.ndarray): Input data array.

    Returns:
    - tuple: Indices of the spatial axes.

    Notes:
    - The spectral axis is identified using `find_spectral_axis`.
    - All other axes are considered spatial.
    """
    spec_axis = find_spectral_axis(cdata)
    
    # Return a tuple of all dimension indices that are NOT the spectral axis
    spatial_axes = tuple(i for i in range(cdata.ndim) if i != spec_axis)
    return spatial_axes



def fourierpipeline(cdata):
    """
    Applies a Fourier transform pipeline to multi-dimensional data.

    Parameters:
    - cdata (numpy.ndarray): Input data array (e.g., k-space or FID data).

    Returns:
    - numpy.ndarray: Fourier-transformed data with spectral and spatial FFTs applied.

    Notes:
    - Spectral FFT is applied along the spectral axis.
    - Spatial FFT is applied along all spatial axes.
    - Both transforms include a shift to center the zero frequency.
    """
    
    ax_spec = processing.find_spectral_axis(cdata)
    ax_spat = processing.find_spatial_axis(cdata)
    
    ax_spat = tuple(sorted(ax_spat))

    spectra = np.fft.fft(cdata, axis=ax_spec)
    spectra = np.fft.fftshift(spectra, axes=ax_spec)
    
    spectra = np.fft.fftshift(np.fft.fftn(spectra, axes=ax_spat), axes=ax_spat)
    return spectra



def read_fid_proc64file(study_directory,scan_no):
    """
    Reads a Bruker FID file (fid_proc.64) for CSI scans.

    Parameters:
    - study_directory (str): Path to the Bruker study directory.
    - scan_no (int): Scan number.

    Returns:
    - numpy.ndarray: Raw FID data as a 1D array of floats.
    """

    fid_ls = []
    fid_file = os.path.join(study_directory, str(scan_no), 'pdata', '1', 'fid_proc.64')

    with open(fid_file, 'rb') as f:
        raw_fid = np.fromfile(f, dtype=np.float64)
        fid_ls.append(raw_fid)

    return fid_ls[0]



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
    """
    Extracts the date and time of the scan from the Bruker header.

    Parameters:
    - header (dict): Bruker header containing acquisition time.

    Returns:
    - datetime.datetime: The date and time of the scan.
    """
    part = header['ACQ_abs_time']
    timefrom1970,_,_= part.strip("()").split(",")
    dt_object = datetime.fromtimestamp(int(timefrom1970))

    return dt_object

def scanduration(header):
    """
    Computes the duration of the scan in minutes and seconds.

    Parameters:
    - header (dict): Bruker header containing scan time.

    Returns:
    - tuple: (minutes, seconds) as integers.
    """

    scantime_ms = header['PVM_ScanTime']
    total_seconds = scantime_ms / 1000

    # 2. Calculate minutes and remaining seconds
    minutes, seconds = divmod(total_seconds, 60)
    return minutes,seconds


def check_spectras(spectras,ppm_axis):
    """
    Plots and displays all spectra in a list with annotated theoretical peaks.

    Parameters:
    - spectras (list): List of spectral data arrays.
    - ppm_axis (numpy.ndarray): PPM axis for the spectra.

    Notes:
    - Each spectrum is shifted by 235 points for visualization.
    - Theoretical peaks for Inorganic Phosphate, γ-ATP, α-ATP, and β-ATP are annotated.
    """

    # PLOT AND SAVE ALL SPECTRA
    for key,value in enumerate(spectras[:]):
        spec = value
        spec = np.roll(spec,-235)

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


def peak_integral(spectra, ppm_axis, targets, tolerance=2,normalize_total = True):

    # 1. Define the search mask
    found_values = {}
    noise_floor = np.mean(spectra[1500:2000])
    
    for name, info in targets.items():
        # Define range
        target_ppm = info['ppm']

        area = (ppm_axis < target_ppm + tolerance) & (ppm_axis > target_ppm - tolerance)
    
        # 2. Extract the indices where 'area' is True, Subtract baseline noise
        indices_in_window = np.where(area)[0]
        raw_sum = np.sum(spectra[indices_in_window])
        baseline_area = noise_floor * len(indices_in_window)
        found_values[name] = max(0, raw_sum - baseline_area)

    # --- 3. OPTIONAL: CONSTANT SUM NORMALIZATION ---
    if normalize_total:
        total_sum = sum(found_values.values())
        if total_sum > 0:
            for name in found_values:
                found_values[name] = found_values[name] / total_sum

    return found_values


def check_orientation(m, p_spectra=2048, p_dims=(12, 10, 10)):
    """
    Returns reshape tuple and transpose order to normalize to (Z, X, Y, spectra).
    
    Coronal:  dominant axes [Z, X, Y] → no transpose needed  → (0,1,2,3)
    Axial:    dominant axes [X, Y, Z] → need (Z,X,Y,spectra) → (2,0,1,3)
    Sagittal: dominant axes [Y, Z, X] → need (Z,X,Y,spectra) → (1,2,0,3)
    """
    m = np.array(m).squeeze()  # shape (3,3)
    
    # dominant physical axis per acquisition dimension (0=X, 1=Y, 2=Z)
    dominant = [int(np.argmax(np.abs(row))) for row in m]
    # e.g. coronal → [2, 0, 1]  (acqdim0=Z, acqdim1=X, acqdim2=Y)
    
    # Find which acquisition dimension each physical axis lives in
    axis_to_acqdim = {phys: acq for acq, phys in enumerate(dominant)}
    # We want output order (Z, X, Y, spectra) = physical axes (2, 0, 1)
    transpose_order = (
        axis_to_acqdim[2],  # Z first
        axis_to_acqdim[0],  # then X
        axis_to_acqdim[1],  # then Y
        3                   # spectra always last
    )
    
    # reshape tuple follows acquisition order
    shape = tuple(p_dims) + (p_spectra,)
    
    return shape, transpose_order