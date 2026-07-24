

### NOT BEING USED!!!!!!!!e
def read_bruker_csi(study_directory, scan_no,type_of_scan):
    """
    Processes Bruker Chemical Shift Imaging (CSI) data and computes spatial and spectral information.

    also makes a distinction between 2dimensional and 3dimensional csi

    Parameters:
        study_directory (str): Path to the directory containing the Bruker study.
        scan_no (int): The scan number to process.

    Returns:
        tuple: Contains the following elements:
            - np.ndarray: FIDs.
            - np.ndarray: Spectra.
            - np.ndarray: X spatial axis.
            - np.ndarray: Y spatial axis.
            - np.ndarray: PPM axis.
            - dict: Header information.

    Raises:
        ValueError: If the header method does not match Bruker CSI.
    """
    fids, header = read_bruker_readout(study_directory, scan_no,type_of_scan)
    print(fids.shape)





    

    n_points = int(header['PVM_SpecMatrix'])
    filter_points = 76# apparently it stays 76
    n_phase_encodes = header['PVM_EncMatrix']
    fov = header['PVM_Fov']

    spatial_dims = len(n_phase_encodes)

    fids = np.roll(fids, -filter_points, axis=0)
    fids[-filter_points:] = 0

    if spatial_dims == 2:
        print('producing 2D csi')
        # Process FIDs
        # 2D CSI
        new_shape = (n_points, int(n_phase_encodes[0]), int(n_phase_encodes[1]))
        fids = fids.reshape(new_shape)
        print(f"Reshaped to 2D: {fids.shape}")
        
        # Compute spectra (FFT spectral + spatial)
        spects = np.fft.fftshift(np.fft.fft(fids, axis=0), axes=0)
        
        # Spatial axes
        x_axis = np.linspace(-fov[0]/2, fov[0]/2, int(n_phase_encodes[0]))
        y_axis = np.linspace(-fov[1]/2, fov[1]/2, int(n_phase_encodes[1]))
        z_axis = None

    if spatial_dims ==3 :
        print("producing 3d CSI")
        new_shape = (n_points, int(n_phase_encodes[0]), 
                     int(n_phase_encodes[1]), int(n_phase_encodes[2]))
        fids = fids.reshape(new_shape)
        print(f"Reshaped to 3D: {fids.shape}")
        
        # Compute spectra (FFT spectral + spatial)
        spects = np.fft.fftshift(np.fft.fft(fids, axis=0), axes=0)

        # Apply spatial FFTs for all three spatial dimensions
        spects = np.fft.fftshift(np.fft.fft(spects, axis=1), axes=1)
        spects = np.fft.fftshift(np.fft.fft(spects, axis=2), axes=2)
        spects = np.fft.fftshift(np.fft.fft(spects, axis=3), axes=3)
        
        
        # Spatial axes
        x_axis = np.linspace(-fov[0]/2, fov[0]/2, int(n_phase_encodes[0]))
        y_axis = np.linspace(-fov[1]/2, fov[1]/2, int(n_phase_encodes[1]))
        z_axis = np.linspace(-fov[2]/2, fov[2]/2, int(n_phase_encodes[2]))

    ppm_axis = compute_ppm_axis(header, n_points)

    return fids, spects, x_axis, y_axis,z_axis, ppm_axis, header




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
