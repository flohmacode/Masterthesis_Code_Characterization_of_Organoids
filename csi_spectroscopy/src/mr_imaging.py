import csi_spectroscopy.src.file_utils as file_utils
import sigpy as sp
import numpy as np
import matplotlib.pyplot as plt

from brukerapi.dataset import Dataset

def combine_coils(kspace):
    """
    Combine multi-coil k-space data by summing across the coil axis.

    Notes:
    - This function *assumes* one axis of `kspace` has length 2 (the coil/channel axis).
    - It finds that axis using `shape.index(2)` and then sums across it.
    - The result is divided by 2 (i.e. averaged across the two coils).
    """
    shape = kspace.shape
    shapeidx = shape.index(2)

    sumofchannels = np.sum(kspace,axis= shapeidx) /2

    return sumofchannels 

def inversefft(kspace):
    """
    Perform inverse FFT on k-space and return the magnitude image.

    Notes:
    - Uses scipy.fft.ifft applied along axes (0, 2) — these must match your k-space layout.
    - Returns the absolute value (magnitude) of the complex image.
    """
    
    shape = kspace.shape
    
    img = sp.ifft(kspace, axes=(0,2))
    
    recon_img = np.abs(img)
    
    return recon_img

def get_header_info(studydirectory,scan_no):
    '''function that reads the header info of number of channels, number of slices, expected number of pixels'''

    header = file_utils.read_bruker_all_headers(studydirectory, scan_no)


    len_encoding= header['PVM_EncSteps1']
    num_slice = header['NSLICES']
    num_coils = header['PVM_EncNReceivers']

    return int(len(len_encoding)), int(num_slice), int(num_coils)


def get_reconstructed_img(study_directory_img,scan_img_number,slice_mri,view):
    """
    Loads raw k-space data, reconstructs the full MRI image volume, 
    extracts a specified slice, and plots the magnitude of the slice.
    
    The function handles orientation correction for the 'coronal' view 
    by transposing the entire 3D reconstruction volume.

    Args:
        study_directory_img (str): Path to the Bruker study directory.
        scan_img_number (int): The scan number containing the image data.
        slice_mri (int): The index of the slice to extract (along the 
                         second axis of the 3D volume).
        view (str): The anatomical view of the acquisition. Expected values 
                    are 'axial', 'sagital', or 'coronal'.

    Returns:
        tuple: (matplotlib.figure.Figure, matplotlib.axes.Axes, numpy.ndarray)
               The figure, axes, and the 2D image array of the extracted slice.
    """

    rawfid = file_utils.read_bruker_readout(study_directory_img,scan_img_number,'image')
    rawfidarr = rawfid[0]

    # If data is interleaved real/imaginary:
    complex_data = rawfidarr[::2] + 1j * rawfidarr[1::2]  # 98,304 complex values

    pixelmatrix,slices,coils= get_header_info(study_directory_img,scan_img_number)

    #[numberofpixels, slices, coils, numberofpixels]

    kspace = complex_data.reshape(pixelmatrix,slices,coils,pixelmatrix)

    k_space_sumcoil_j = combine_coils(kspace)

    recon = inversefft(k_space_sumcoil_j)

    
    if view == 'coronal':
        #the cornal case somehow needed to be transposed so that it made sense visually. 
        #this might be a bug.
        fig,ax = plt.subplots()

        recon= recon.transpose()
        
        ax.imshow(recon[:,slice_mri,:], cmap = 'gray',origin= 'upper')
        ax.set_title(slice_mri)
        ax.set_xticks([])
        ax.set_yticks([])


        return fig,ax,recon[:,slice_mri,:]
    
    if view ==  'axial' or view == 'sagital':
    
        fig,ax = plt.subplots()

        #recon= recon.transpose()
        
        ax.imshow(recon[:,slice_mri,:], cmap = 'gray',origin= 'upper')
        ax.set_title(slice_mri)
        ax.set_xticks([])
        ax.set_yticks([])

        return fig,ax,recon[:,slice_mri,:]

    else:
        raise ValueError('View not known, has to be coronal, sagital, or axial')
    

# def get_full_imageplot(study_directory_img,scan_img_number,slice_mri):

#     rawfid = file_utils.read_bruker_readout(study_directory_img,scan_img_number,'image')

#     rawfidarr = rawfid[0]

#     # If data is interleaved real/imaginary:
#     complex_data = rawfidarr[::2] + 1j * rawfidarr[1::2]  # 98,304 complex values

#     pixelmatrix,slices,coils= get_header_info(study_directory_img,scan_img_number)

#     #[numberofpixels, slices, coils, numberofpixels]

#     kspace = complex_data.reshape(pixelmatrix,slices,coils,pixelmatrix)

#     k_space_sumcoil_j = combine_coils(kspace)

#     recon = inversefft(k_space_sumcoil_j)

#     plt.figure()
#     fig,ax =plt.subplots(ncols=2)

#     ax[0].imshow(np.abs(k_space_sumcoil_j[:,slice_mri,:]) , cmap = 'gray',origin= 'upper')
#     ax[0].set_title(slice_mri)

#     ax[1].imshow(recon[:,slice_mri,:], cmap = 'gray',origin= 'upper')
#     ax[1].set_title(slice_mri)

#     return fig,ax

def turborareload(study_directory):
    return Dataset(study_directory)


def plot_turborare_dataset(dataset,idx = 3):
    plt.imshow(dataset.data[:,:,idx],cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('Organoids in NMR Tubes')