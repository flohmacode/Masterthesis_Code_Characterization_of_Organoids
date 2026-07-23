import numpy as np
import math 
import matplotlib.pyplot as plt
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
    
    print(f"Data shape: {rawdata.shape}")
    print(f"Spectral axis: {spec_axis}")
    
    # Create time vector
    t = np.arange(rawdata.shape[spec_axis]) / bw
    print(f"Time vector shape: {t.shape}")
    
    # Create 1D exponential window
    apod_window_1d = np.exp(-t * np.pi * 2.0 * lb)
    
    # Build target shape: place the 1D window along spec_axis, size 1 for others
    target_shape = [1] * len(rawdata.shape)
    target_shape[spec_axis] = rawdata.shape[spec_axis]
    
    # Reshape 1D window for broadcasting
    apod_window = apod_window_1d.reshape(target_shape)
    
    print(f"Apodization window shape: {apod_window.shape}")
    
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


#u can filter sinus with this
##FILTERING
def smoothing_filter(raw_signal):
    alpha = 0.9
    filtered_signal = np.empty_like(raw_signal, dtype=float)

    previous_y = 0.0

    for i in range(raw_signal.shape[0]):
        x = raw_signal[i, 0, 0, 0]
        y = alpha * x + (1 - alpha) * previous_y
        filtered_signal[i, 0, 0, 0] = y
        previous_y = y

    return filtered_signal

def create_example():
    ones = np.ones((2048,1,1,1),dtype = np.int8)
    x_axis = np.arange(2048)

    signal = np.sin(x_axis/256)[:,None,None,None]
    noise = np.random.normal(0,0.5,size=signal.shape)

    noisy_signal = signal+noise
    fig = plt.figure()

    plt.plot(noisy_signal[:,0,0,0])

    filtered_signal = smoothing_filter(noisy_signal)
    plt.plot(filtered_signal[:,0,0,0])

    return noisy_signal,noisy_signal,fig
