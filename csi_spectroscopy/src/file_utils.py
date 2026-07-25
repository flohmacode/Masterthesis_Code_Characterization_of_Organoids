import sys
print(sys.executable)

import os
import numpy as np
import warnings

def strtok(val, delims):
    """
    Tokenizes a string into substrings based on multiple delimiters.

    Parameters:
        val (list of str): A list of strings to tokenize.
        delims (list of str): A list of delimiter characters.

    Returns:
        list of str: A list of tokens after splitting by the delimiters.
    """
    for delim in delims:
        val = [subtoken for token in val for subtoken in token.split(delim) if subtoken.strip()]
    return val

def read_bruker_header(filepath):
    """
    Parses a Bruker header file into a dictionary format.

    Parameters:
        filepath (str): Path to the header file.

    Returns:
        dict: A dictionary containing parsed header data.

    Raises:
        FileNotFoundError: If the file at the specified path does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r') as file:
        rawdata = file.read()

    rawitems = strtok([rawdata], ['##', '$$'])
    header_dict = {}

    for item in rawitems:
        if item.startswith((' @vis', 'END=', 'PVM_StartupShimList=')):
            continue

        # Parse name and value
        try:
            name, value = item.split('=', 1)
        except ValueError:
            continue

        name = name.replace('-', '_').replace(' ', '_').replace('$', '')

        if value.startswith('( '):  # Handle arrays
            try:
                size, values = value[2:].split(')', 1)
                size = list(map(int, size.split(',')))
                size += [1] * (2 - len(size))  # Ensure 2D size
                header_dict[name] = np.array(values.split(), dtype=float).reshape(size)
            except ValueError:
                header_dict[name] = value.strip('<>').replace('\n', '')
        else:  # Single value
            try:
                header_dict[name] = float(value)
            except ValueError:
                header_dict[name] = value.strip()

    # Flatten single-element arrays
    for key, value in header_dict.items():
        if isinstance(value, np.ndarray) and value.shape == (1, 1):
            header_dict[key] = value[0, 0]

    return header_dict

def read_bruker_all_headers(study_directory, scan_no):
    """
    Reads and merges Bruker headers from multiple files for a given study directory and scan number.

    Parameters:
        study_directory (str): Path to the directory containing the Bruker study.
        scan_no (int): The scan number to process.

    Returns:
        dict: A dictionary containing merged header information from the subject, method, and acquisition parameter (acqp) files.
    """
    headers = {}

    # File paths
    method_path = os.path.join(study_directory, str(scan_no), 'method')
    acqp_path = os.path.join(study_directory, str(scan_no), 'acqp')
    subject_path = os.path.join(study_directory, 'subject')

    # Read headers
    method = read_bruker_header(method_path)
    acqp = read_bruker_header(acqp_path)
    subject = read_bruker_header(subject_path)

    # Merge headers
    headers.update(subject)
    duplicate_names = {'TITLE', 'JCAMPDX', 'DATATYPE', 'ORIGIN', 'OWNER'}

    # Remove duplicate keys from `method` and `acqp`
    method = {k: v for k, v in method.items() if k not in duplicate_names}
    acqp = {k: v for k, v in acqp.items() if k not in duplicate_names}

    # Update `headers` with `method` and handle conflicts
    for key, value in method.items():
        if key in headers:
            warnings.warn(
                f"*WARNING* {key} exists in both method and subject. Using value from method."
            )
        headers[key] = value

    # Update `headers` with `acqp` and handle conflicts
    for key, value in acqp.items():
        if key in headers:
            warnings.warn(
                f"*WARNING* {key} exists in both method/subject and acqp. Using value from acqp."
            )
        headers[key] = value

    return headers

def read_bruker_readout(study_directory, scan_no,type_of_scan):
    """
    Reads and processes the Bruker readout file to extract free induction decays (FIDs).

    Parameters:
        study_directory (str): Path to the directory containing the Bruker study.
        scan_no (int): The scan number to process.

    Returns:
        tuple: Contains the following elements:
            - np.ndarray: Complex FID data.
            - dict: Header information.
    """
    
    if type_of_scan == 'image':
        header = read_bruker_all_headers(study_directory, scan_no)
        ls = []
        fid_file = os.path.join(study_directory,str(scan_no),"rawdata.job0")

        with open(fid_file, 'rb') as f:
            raw_fid = np.fromfile(f, dtype=np.int32)
            ls.append(raw_fid)

        return ls
    
    if type_of_scan == 'csi':
        header = read_bruker_all_headers(study_directory, scan_no)
        ls = []
        fid_file = os.path.join(study_directory, str(scan_no), 'pdata', '1', 'fid_proc.64')

        with open(fid_file, 'rb') as f:
            raw_fid = np.fromfile(f, dtype=np.int32)
            ls.append(raw_fid)

        return ls

    if type_of_scan == 'spectroscopy':

        header = read_bruker_all_headers(study_directory, scan_no)
        try:
            n_points = int(header.get("PVM_SpecMatrix"))
        except (KeyError,ValueError,TypeError):
            raise ValueError("No PVM_SpecMatrix in headerfile, perhaps u selected the wrong scan?")
        
        min_points = 128
        # Ensure `tmp_n_points` is a power of 2 and greater than or equal to `min_points`
        tmp_n_points = max(min_points, 2 ** int(np.ceil(np.log2(max(n_points, min_points)))))

        # Load the FID data
        fid_file = os.path.join(study_directory, str(scan_no), 'pdata', '1', 'fid_proc.64')
        with open(fid_file, 'rb') as f:
            raw_fid = np.fromfile(f, dtype=np.float64)

        # Convert raw data to complex format and reshape
        complex_fid = raw_fid[::2] + 1j * raw_fid[1::2]
        remainder = len(complex_fid) // tmp_n_points
        reshaped_fid = complex_fid[:remainder * tmp_n_points].reshape((remainder, tmp_n_points)).T
        fids = reshaped_fid[:n_points, :]

        return fids, header

def compute_ppm_axis(header, n_points):
    """
    Computes the parts per million (ppm) axis based on the header information and number of points.

    Parameters:
        header (dict): Header data containing spectral information.
        n_points (int): Number of points in the spectrum.

    Returns:
        np.ndarray: PPM axis values.    
    """
    ppm_axis = np.linspace(0.0, float(header['PVM_SpecSW']), n_points)
    ppm_axis -= np.mean(ppm_axis)
    return ppm_axis

def read_NSPECT(study_directory, scan_no,type_of_scan):
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
    fids, header = read_bruker_readout(study_directory, scan_no,type_of_scan)

    try:
        n_points = int(header.get("PVM_SpecMatrix"))
    except (KeyError,ValueError,TypeError):
        raise ValueError("This Function is for Spectroscopy - No PVM_SpecMatrix in headerfile, perhaps u selected the wrong scan?")
    
    filter_points = 76

    # Process FIDs
    fids = np.roll(fids, -filter_points, axis=0)
    fids[-filter_points:] = 0

    # Compute spectra and ppm axis
    spects = np.fft.fftshift(np.fft.fft(fids, axis=0), axes=0)
    ppm_axis = compute_ppm_axis(header, n_points)

    return fids, spects, ppm_axis, header

def read_bruker_study(study_dir):
    """
    Reads all scans in a Bruker study directory and prints information about each scan.

    Parameters:
        study_dir (str): Path to the Bruker study directory.

    Returns:
        None: Prints information about the scans in the study directory.
    """
    scans = sorted(int(folder.name) for folder in os.scandir(study_dir) if folder.is_dir() and folder.name.isdigit())
    tmpheaderls = []
    for scan in scans:
        method_file = os.path.join(study_dir, str(scan), 'method')
        if os.path.isfile(method_file):
            tmp_header = read_bruker_header(method_file)
            print(f"Scan No: {scan} is a {tmp_header['Method']} Scan.")
            tmpheaderls.append((scan,tmp_header['Method']))
        else:
            print(f"Scan No: {scan} has no header.")
    return tmpheaderls

def get_scan_time(header):
    """
    Extracts the scan duration from a Bruker header and converts it into minutes and seconds.

    Parameters:
    - header (dict): Bruker header containing the scan time information.
                     The key 'PVM_ScanTime' must be present, representing the scan duration in milliseconds.

    Returns:
    - tuple: (minutes, seconds)
      - minutes (int): The total number of full minutes in the scan duration.
      - seconds (int): The remaining seconds after extracting full minutes.

    Notes:
    - The scan time is converted from milliseconds to seconds, then split into minutes and seconds.
    - Uses `divmod` to compute both the quotient (minutes) and remainder (seconds) in one step.
    """
    scantime_ms = header['PVM_ScanTime']

    total_seconds = scantime_ms / 1000
    # 2. Calculate minutes and remaining seconds
    minutes, seconds = divmod(total_seconds, 60)
    return minutes,seconds
