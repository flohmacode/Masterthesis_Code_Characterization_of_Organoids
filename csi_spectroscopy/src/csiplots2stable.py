
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_csi_spec_axial(spects, idxtoslice, spatial):
    """
    Plot axial-view CSI spectra for a given slice as a grid of subplots.

    This function visualizes chemical shift imaging (CSI) spectra in an
    axial orientation. For a selected slice index, spectra are extracted
    for each (x, y) spatial location and plotted in a 2D grid where each
    subplot corresponds to one voxel. All spectra share the same y-axis
    scaling based on the global maximum intensity to allow visual
    comparison across voxels.

    Parameters
    ----------
    spects : np.ndarray
        Multidimensional array containing CSI spectral data. The expected
        dimension order is:
        (slice, x, y, spectral_points).
    view : str
        View orientation. Currently, only 'axial' is supported.
    idxtoslice : int
        Index of the slice to be visualized.
    spatial : dict
        Dictionary containing spatial and spectral axes with the following
        required keys:
        - 'x' : array-like
            X-axis spatial coordinates.
        - 'y' : array-like
            Y-axis spatial coordinates.
        - 'z' : array-like
            Z-axis spatial coordinates (not directly used here).
        - 'ppm' : array-like
            Chemical shift axis in parts per million (ppm).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib figure object containing the subplot grid.
    ax : np.ndarray of matplotlib.axes.Axes
        Array of Axes objects corresponding to each voxel subplot.

    Notes
    -----
    - The spectra are plotted as magnitude spectra (absolute value).
    - A fixed spectral shift is applied using `np.roll` to align peaks.
    - X-axis tick labels and Y-axis tick labels are hidden for clarity.
    - The ppm axis is displayed in reverse order (25 to -25 ppm),
      following common MRS conventions.
    - All subplots are scaled to the same y-axis limit, determined by
      the maximum intensity across all plotted spectra.
    """

    x_axis, y_axis, z_axis = spatial['x'], spatial['y'], spatial['z'] 

    # 1. Extract axes from spatial dictionary
    ppm_axis = spatial['ppm']



    fig,ax = plt.subplots(len(x_axis),len(y_axis))
    
    fig.set_figheight(5)
    fig.set_figwidth(5)
    
    newslice = [None,None,None,None]

    y_max = 0
    for y in range(len(y_axis)):
        for x in range(len(x_axis)):
        

            newslice[0]= idxtoslice
            newslice[1]= x
            newslice[2]= y
            newslice[3]= slice(None)

            spe = np.abs(spects [tuple(newslice)])
            spe = np.roll(spe,0)

            y_maxnew = np.max(np.abs(spects [tuple(newslice)]))

            if y_max < y_maxnew:
                y_max = y_maxnew

            ax[y,x].plot(ppm_axis,spe,linewidth  = 0.7)
            #ax[x,y].axvline(x= 3,linestyle = ':',color = 'red')#waterpeak
            # ax[x,y].axvline(x=-2.48, color='red', linestyle='--', label='γ-ATP (theoretical)')
            # ax[x,y].axvline(x=-7.52, color='green', linestyle='--', label='α-ATP (theoretical)')
            # ax[x,y].axvline(x=-16.26, color='purple', linestyle='--', label='β-ATP (theoretical)')
            ax[y,x].set_xticklabels([])
            ax[y,x].set_yticklabels([])
            ax[y,x].set_xlim(25,-25)
            fig.suptitle(f"Spectra for shape {spects.shape},axialview") 

    #scale the y axis by the highest point found in spectra
    for ax_obj in ax.flat:
        ax_obj.set_ylim(0, y_max)
    
    return fig,ax#, np.abs(spects [tuple(newslice)])



def plot_csi_spec_coronal(spects, idxtoslice, spatial):
    """
    Plot coronal-view CSI spectra for a given slice as a grid of subplots.

    This function visualizes chemical shift imaging (CSI) spectra in a
    coronal orientation. For a selected coronal slice index, spectra are
    extracted for each (z, x) spatial location and displayed in a 2D grid
    of subplots, where each subplot represents the spectrum from a single
    voxel. All spectra are normalized to a common y-axis scale based on
    the global maximum intensity across the slice.

    Parameters
    ----------
    spects : np.ndarray
        Multidimensional array containing CSI spectral data. The expected
        dimension order is:
        (z, x, y, spectral_points).
    view : str
        View orientation. This function executes only when set to 'coronal'.
    idxtoslice : int
        Index of the coronal slice (y-dimension) to be visualized.
    spatial : dict
        Dictionary containing spatial and spectral axes with the following
        required keys:
        - 'x' : array-like
            X-axis spatial coordinates.
        - 'y' : array-like
            Y-axis spatial coordinates.
        - 'z' : array-like
            Z-axis spatial coordinates.
        - 'ppm' : array-like
            Chemical shift axis in parts per million (ppm).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib figure object containing the subplot grid.
    ax : np.ndarray of matplotlib.axes.Axes
        Array of Axes objects corresponding to each voxel subplot.

    Notes
    -----
    - Spectra are plotted as magnitude spectra using the absolute value.
    - A fixed circular shift is applied to each spectrum using `np.roll`
      to align spectral features.
    - Axis tick labels are removed for cleaner visualization of the grid.
    - The ppm axis is displayed in reverse order (25 to -25 ppm), following
      standard MRS plotting conventions.
    - All subplots share the same y-axis limits, determined by the maximum
      spectral intensity across the entire slice.
    """
    
    x_axis, y_axis, z_axis = spatial['x'], spatial['y'], spatial['z'] 
    # 1. Extract axes from spatial dictionary
    
    ppm_axis = spatial['ppm']



    fig,ax = plt.subplots(len(z_axis),len(x_axis))
    
    fig.set_figheight(5)
    fig.set_figwidth(5)
    
    newslice = [None,None,None,None]

    y_max = 0
    for x in range(len(x_axis)):
        for z in range(len(z_axis)):
        

            newslice[0]= z
            newslice[1]= x
            newslice[2]= idxtoslice
            newslice[3]= slice(None)

            spe = np.abs(spects [tuple(newslice)])
            spe = np.roll(spe,-235)

            y_maxnew = np.max(np.abs(spects [tuple(newslice)]))

            if y_max < y_maxnew:
                y_max = y_maxnew

            #ax[x,y].plot(ppm_axis,spec2_norm_shifted,linewidth  = 0.7)
            ax[z,x].plot(ppm_axis,spe,linewidth  = 0.7)
            # ax[x,y].axvline(x=-2.48, color='red', linestyle='--', label='γ-ATP (theoretical)')
            # ax[x,y].axvline(x=-7.52, color='green', linestyle='--', label='α-ATP (theoretical)')
            # ax[x,y].axvline(x=-16.26, color='purple', linestyle='--', label='β-ATP (theoretical)')

            #ax[z,x].plot(ppm_axis,spe,linewidth  = 0.7)
            ax[z,x].set_xticklabels([])
            ax[z,x].set_yticklabels([])
            ax[z,x].set_xlim(25,-25)
            fig.suptitle(f"Spectra for shape {spects.shape},coronalview") 

    #scale the y axis by the highest point found in spectra
    for ax_obj in ax.flat:
        ax_obj.set_ylim(0, y_max)
    
    return fig,ax#, np.abs(spects [tuple(newslice)])



def plot_csi_spec_sagital(spects, idxtoslice, spatial):
    """
    Plot sagittal-view CSI spectra for a given slice as a grid of subplots.

    This function visualizes chemical shift imaging (CSI) spectra in a
    sagittal orientation. For a selected sagittal slice index, spectra are
    extracted for each (y, z) spatial location and displayed in a 2D grid
    of subplots, where each subplot corresponds to the spectrum from a
    single voxel. All spectra are displayed using a shared y-axis scale
    determined by the maximum intensity across the slice.

    Parameters
    ----------
    spects : np.ndarray
        Multidimensional array containing CSI spectral data. The expected
        dimension order is:
        (z, x, y, spectral_points).
    view : str
        View orientation. This function executes only when set to 'sagital'.
    idxtoslice : int
        Index of the sagittal slice (x-dimension) to be visualized.
    spatial : dict
        Dictionary containing spatial and spectral axes with the following
        required keys:
        - 'x' : array-like
            X-axis spatial coordinates.
        - 'y' : array-like
            Y-axis spatial coordinates.
        - 'z' : array-like
            Z-axis spatial coordinates.
        - 'ppm' : array-like
            Chemical shift axis in parts per million (ppm).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib figure object containing the subplot grid.
    ax : np.ndarray of matplotlib.axes.Axes
        Array of Axes objects corresponding to each voxel subplot.

    Notes
    -----
    - Spectra are plotted as magnitude spectra using the absolute value.
    - A fixed circular shift is applied using `np.roll` to align spectral
      features.
    - X- and Y-axis tick labels are removed for clarity when visualizing
      large voxel grids.
    - The ppm axis is displayed in reverse order (25 to -25 ppm), following
      standard MRS conventions.
    - All subplots share the same y-axis limits, set by the maximum spectral
      intensity found across the entire slice.
    """

    x_axis, y_axis, z_axis = spatial['x'], spatial['y'], spatial['z'] 

    # 1. Extract axes from spatial dictionary
    
    ppm_axis = spatial['ppm']


    fig,ax = plt.subplots(len(y_axis),len(z_axis))
    
    fig.set_figheight(5)
    fig.set_figwidth(5)
    
    newslice = [None,None,None,None]

    y_max = 0
    for y in range(len(y_axis)):
        for z in range(len(z_axis)):
        

            newslice[0]= z
            newslice[1]= idxtoslice
            newslice[2]= y
            newslice[3]= slice(None)

            spe = np.abs(spects [tuple(newslice)])
            spe = np.roll(spe,-235)

            y_maxnew = np.max(np.abs(spects [tuple(newslice)]))

            if y_max < y_maxnew:
                y_max = y_maxnew

            #ax[x,y].plot(ppm_axis,spec2_norm_shifted,linewidth  = 0.7)
            ax[y,z].plot(ppm_axis,spe,linewidth  = 0.7)
            # ax[x,y].axvline(x=-2.48, color='red', linestyle='--', label='γ-ATP (theoretical)')
            # ax[x,y].axvline(x=-7.52, color='green', linestyle='--', label='α-ATP (theoretical)')
            # ax[x,y].axvline(x=-16.26, color='purple', linestyle='--', label='β-ATP (theoretical)')

            #ax[y,x].plot(ppm_axis,spe,linewidth  = 0.7)
            ax[y,z].set_xticklabels([])
            ax[y,z].set_yticklabels([])
            ax[y,z].set_xlim(25,-25)
            fig.suptitle(f"Spectra for shape {spects.shape},sagitalview") 

    #scale the y axis by the highest point found in spectra
    for ax_obj in ax.flat:
        ax_obj.set_ylim(0, y_max)
    
    return fig,ax#, np.abs(spects [tuple(newslice)])




#HEATMAPS START HERE

def plot_csi_heatmap_axial(spects, idxtoslice, spatial):
    """
    Plot an axial heatmap of maximum CSI spectral intensity for a given slice.

    This function generates a 2D heatmap representing the maximum magnitude
    of the CSI spectrum at each (x, y) voxel location for a selected axial
    slice. The heatmap provides a coarse visualization of signal intensity
    distribution across the slice, independent of spectral shape.

    Parameters
    ----------
    spects : np.ndarray
        Multidimensional array containing CSI spectral data. Expected
        dimension order:
        (slice, x, y, spectral_points).
    view : str
        View orientation. The heatmap is generated only when set to 'axial'.
    idxtoslice : int
        Index of the axial slice to be visualized.
    spatial : dict
        Dictionary containing spatial and spectral axes. Required keys:
        'x', 'y', 'z', and 'ppm'.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib figure object containing the heatmap.
    ax : matplotlib.axes.Axes
        The Axes object associated with the heatmap.

    Notes
    -----
    - Each voxel value is computed as the maximum absolute spectral
      intensity across the spectral dimension.
    - The heatmap uses a fixed color scale (`vmin=25000`, `vmax=600000`)
      and the 'hot' colormap.
    - The image is transposed prior to display to match spatial orientation.
    """
    x_axis, y_axis, z_axis = spatial['x'], spatial['y'], spatial['z'] 

    # 1. Extract axes from spatial dictionary
    
    ppm_axis = spatial['ppm']

    fig,ax = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(5)
    
    newslice = [None,None,None,None]

    y_max = 0

    img = np.zeros((len(x_axis),len(y_axis)))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i,j] = np.max(np.abs(spects[idxtoslice,i,j,:]))

    plt.imshow(img.T,vmin=25000, vmax=1200000,cmap = 'hot')
    plt.xticks([])
    plt.yticks([])
    
    return fig,ax
    


def plot_csi_heatmap_coronal(spects,  idxtoslice, spatial):
    """
    Plot a coronal heatmap of maximum CSI spectral intensity for a given slice.

    This function produces a 2D heatmap showing the maximum magnitude of
    the CSI spectrum at each (x, z) voxel location for a selected coronal
    slice. The heatmap summarizes overall signal strength per voxel without
    preserving spectral information.

    Parameters
    ----------
    spects : np.ndarray
        Multidimensional array containing CSI spectral data. Expected
        dimension order:
        (z, x, y, spectral_points).
    view : str
        View orientation. The heatmap is generated only when set to 'coronal'.
    idxtoslice : int
        Index of the coronal slice (y-dimension) to be visualized.
    spatial : dict
        Dictionary containing spatial and spectral axes. Required keys:
        'x', 'y', 'z', and 'ppm'.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib figure object containing the heatmap.
    ax : matplotlib.axes.Axes
        The Axes object associated with the heatmap.

    Notes
    -----
    - Voxel intensity is defined as the maximum absolute value across the
      spectral dimension.
    - A fixed intensity range (`vmin=25000`, `vmax=600000`) and the 'hot'
      colormap are used for visualization.
    - The image is transposed before display to align with coronal
      anatomical orientation.
    """
    x_axis, y_axis, z_axis = spatial['x'], spatial['y'], spatial['z'] 

    # 1. Extract axes from spatial dictionary
    
    ppm_axis = spatial['ppm']
    
    fig,ax = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(5)
    
    newslice = [None,None,None,None]

    y_max = 0

    img = np.zeros((len(x_axis),len(z_axis)))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i,j] = np.max(np.abs(spects[j,i,idxtoslice,:]))

    plt.imshow(img.T,vmin=25000, vmax=1200000,cmap='hot')
    plt.xticks([])
    plt.yticks([])
    
    return fig,ax



def plot_csi_heatmap_sagital(spects, idxtoslice, spatial):
    """
    Plot a sagittal heatmap of maximum CSI spectral intensity for a given slice.

    This function generates a 2D heatmap representing the maximum magnitude
    of the CSI spectrum at each (y, z) voxel location for a selected sagittal
    slice. The heatmap provides a spatial overview of signal intensity
    distribution across the sagittal plane.

    Parameters
    ----------
    spects : np.ndarray
        Multidimensional array containing CSI spectral data. Expected
        dimension order:
        (z, x, y, spectral_points).
    view : str
        View orientation. The heatmap is generated only when set to 'sagital'.
    idxtoslice : int
        Index of the sagittal slice (x-dimension) to be visualized.
    spatial : dict
        Dictionary containing spatial and spectral axes. Required keys:
        'x', 'y', 'z', and 'ppm'.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib figure object containing the heatmap.
    ax : matplotlib.axes.Axes
        The Axes object associated with the heatmap.

    Notes
    -----
    - Each voxel value corresponds to the maximum absolute spectral
      intensity across the spectral dimension.
    - The heatmap uses a fixed color scale (`vmin=25000`, `vmax=600000`)
      with the 'hot' colormap.
    - No axis labels or anatomical overlays are applied.
    """
    x_axis, y_axis, z_axis = spatial['x'], spatial['y'], spatial['z'] 

    # 1. Extract axes from spatial dictionary
    
    ppm_axis = spatial['ppm']

    fig,ax = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(5)
    
    newslice = [None,None,None,None]

    y_max = 0

    img = np.zeros((len(y_axis),len(z_axis)))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i,j] = np.max(np.abs(spects[j,idxtoslice,i,:]))

    plt.imshow(img,vmin=25000, vmax=600000,cmap='hot')
    
    return fig,ax


# Overlay 

def compute_alignment(fov_spec, fov_img, offset_y, offset_x=0):
    """
    Calculates GridSpec boundaries (0 to 1) for a CSI grid.
    
    Args:
        fov_spec: Spectral FOV (e.g., [[40.], [40.]])
        fov_img: Image FOV (e.g., [[25.], [25.], [72.]])
        offset_y: The physical shift in mm (e.g., 2.89)
        offset_x: Physical shift in x (default 0)
    """
    # 1. Flatten the inputs to get simple floats
    f_spec_x = float(np.array(fov_spec).flatten()[0])
    f_spec_y = float(np.array(fov_spec).flatten()[1])
    
    f_img_x = float(np.array(fov_img).flatten()[0])
    f_img_y = float(np.array(fov_img).flatten()[1])

    # 2. Calculate the size of the CSI grid relative to the image
    # (e.g., if CSI is 40mm and Image is 25mm, this will be 1.6)
    width_ratio = f_spec_x / f_img_x
    height_ratio = f_spec_y / f_img_y

    # 3. Calculate the center position in normalized coordinates
    # 0.5 is the center of the image. 
    # We add (offset / total_fov) to shift that center.
    center_x = 0.5 + (offset_x / f_img_x)
    center_y = 0.5 + (offset_y / f_img_y)

    # 4. Calculate final GridSpec boundaries
    left = center_x - (width_ratio / 2)
    right = center_x + (width_ratio / 2)
    bottom = center_y - (height_ratio / 2)
    top = center_y + (height_ratio / 2)

    return left, right, bottom, top


def compute_alignment_3d(fov_spec_3d, fov_img_2d, offsets_3d, view='axial'):
    """
    fov_spec_3d: [[25.], [25.], [72.]] -> Physical X, Y, Z
    fov_img_2d:  [40, 40]              -> The 2D dimensions of the active slice
    offsets_3d:  [off_x, off_y, off_z] -> Physical shifts in mm
    """
    # 1. Flatten inputs to simple 1D arrays
    fs = np.array(fov_spec_3d).flatten() # Should be [25, 25, 72]
    fi = np.array(fov_img_2d).flatten()   # Should be [40, 40]
    off = np.array(offsets_3d).flatten() # Should be [x, y, z]

    # 2. Map physical CSI axes (0=X, 1=Y, 2=Z) to 2D Plot axes
    # The image FOV [40, 40] is always [Horizontal, Vertical] on screen
    if view.lower() == 'axial':
        spec_h_idx, spec_v_idx = 0, 1  # CSI Width=X, CSI Height=Y
    elif view.lower() == 'coronal':
        spec_h_idx, spec_v_idx = 0, 2  # CSI Width=X, CSI Height=Z
    elif view.lower() == 'sagittal':
        spec_h_idx, spec_v_idx = 1, 2  # CSI Width=Y, CSI Height=Z
    else:
        raise ValueError("View must be 'axial', 'coronal', or 'sagittal'")

    # 3. Extract the physical sizes for the current view
    f_spec_h = fs[spec_h_idx]
    f_spec_v = fs[spec_v_idx]
    
    f_img_h = fi[0] # Image Width (usually 40)
    f_img_v = fi[1] # Image Height (usually 40)
    
    off_h = off[spec_h_idx]
    off_v = off[spec_v_idx]

    # 4. Calculate Ratios and Normalized Centers
    width_ratio = f_spec_h / f_img_h
    height_ratio = f_spec_v / f_img_v

    # 0.5 is the center of the 40x40 image
    center_h = 0.5 + (off_h / f_img_h)
    center_v = 0.5 + (off_v / f_img_v)

    # 5. Calculate GridSpec boundaries (0 to 1)
    left = center_h - (width_ratio / 2)
    right = center_h + (width_ratio / 2)
    bottom = center_v - (height_ratio / 2)
    top = center_v + (height_ratio / 2)

    return left, right, bottom, top

def compute_alignment2(fov_spec, fov_img, offset_y, offset_x, nx, ny):
    """
    Calculates GridSpec boundaries with a half-voxel correction.
    
    Args:
        nx, ny: Number of voxels in the spectral grid (len(x_axis), len(y_axis))
    """
    f_spec_x = float(np.array(fov_spec).flatten()[0])
    f_spec_y = float(np.array(fov_spec).flatten()[1])
    
    f_img_x = float(np.array(fov_img).flatten()[0])
    f_img_y = float(np.array(fov_img).flatten()[1])

    # Calculate physical size of one spectral voxel
    voxel_size_x = f_spec_x / nx
    voxel_size_y = f_spec_y / ny

    # width_ratio is the total span of the grid in image-space units
    width_ratio = f_spec_x / f_img_x
    height_ratio = f_spec_y / f_img_y

    # Calculate the center position (0.5 is image center)
    # If the shift is off, you may need to SUBTRACT the half-voxel instead of adding
    # depending on the scanner's coordinate convention.
    center_x = 0.5 - (offset_x / f_img_x)
    #center_y = 0.5 + (offset_y / f_img_y)
    
    # APPLY HALF-VOXEL CORRECTION
    # We shift the boundaries by half the width of one voxel (in normalized units)
    half_vox_x = (voxel_size_x / f_img_x) / 2
    half_vox_y = (voxel_size_y / f_img_y) / 2
    print(half_vox_y)
    # 1. Calculate the normalized height of one FULL voxel
    # (Voxel height / Total Image Height)
    full_vox_y = (f_spec_y / ny) / f_img_y

    # 2. Update center_y by adding that full voxel
    # We use '+' because in your current Matplotlib setup, 
    # increasing the value moves the grid UP the image.
    center_y = 0.5 + (offset_y / f_img_y) + full_vox_y

    # Adjusting boundaries. Try adding or subtracting half_vox if alignment 
    # gets worse; usually, 'center' needs to be shifted to 'edge'.
    left   = center_x - (width_ratio / 2) + half_vox_x
    right  = center_x + (width_ratio / 2) + half_vox_x
    bottom = center_y - (height_ratio / 2) + half_vox_y
    top    = center_y + (height_ratio / 2) + half_vox_y

    return left, right, bottom, top

def csi_overlay_axial(spects, img_array,idxtoslice, spatial,  fov_img ,fov_spec ,offset,figsize, title):
    """
    Overlay axial CSI spectra on a 2D anatomical image using GridSpec alignment.

    This function overlays voxel-wise CSI spectra onto an axial anatomical
    image (e.g. MRI) by embedding a grid of transparent spectral subplots
    within the spatial extent of the background image. Each spectrum is
    plotted at its corresponding (x, y) voxel location using a manually
    computed field-of-view alignment.

    Parameters
    ----------
    spects : np.ndarray
        Multidimensional CSI spectral array with expected dimension order:
        (slice, x, y, spectral_points).
    img_array : np.ndarray
        2D anatomical image used as the background for the overlay.
    idxtoslice : int
        Index of the axial slice to visualize.
    spatial : dict
        Dictionary containing spatial and spectral axes with keys:
        'x', 'y', 'z', and 'ppm'.
    fov_img : tuple or array-like
        Field-of-view of the anatomical image (in physical units).
    fov_spec : tuple or array-like
        Field-of-view of the CSI grid (in physical units).
    offset : tuple or array-like
        Spatial offset between the CSI grid and the anatomical image.
    figsize : tuple
        Size of the Matplotlib figure.
    title : str
        Title for the figure (currently not displayed by default).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib figure object containing the overlay.
    ax_objects : np.ndarray of matplotlib.axes.Axes
        Array of Axes objects corresponding to the CSI voxel spectra.

    Notes
    -----
    - Alignment between CSI and anatomical space is handled manually using
      `compute_alignment` and Matplotlib `GridSpec`.
    - Spectra are plotted as magnitude spectra and circularly shifted using
      `np.roll` for peak alignment.
    - All voxel spectra share a global y-axis scale based on the maximum
      spectral intensity across the slice.
    - Subplot backgrounds are made transparent to reveal the anatomical
      image beneath.
    """
    x_axis,y_axis,z_axis,ppm_axis = spatial['x'],spatial['y'],spatial['z'],spatial['ppm']
    # Keep your original figure setup
    fig = plt.figure(figsize=figsize)
    ax_main = fig.add_subplot(111)

    # Display the background image exactly as you had it
    ax_main.imshow(img_array, cmap='gray', extent=[0, img_array.shape[1], 0, img_array.shape[0]], origin='upper', zorder=0)
    ax_main.set_axis_off()

    pos = ax_main.get_position()
    img_h, img_w = img_array.shape[0], img_array.shape[1]

    #l_local, r_local, b_local, t_local = compute_alignment(fov_spec, fov_img, offset)


    l_local, r_local, b_local, t_local = compute_alignment2(
        fov_spec, 
        fov_img, 
        offset_y=offset, 
        offset_x=offset, # Add if you have x-offset
        nx=len(x_axis), 
        ny=len(y_axis)
    )
    final_left   = pos.x0 + (l_local * pos.width)
    final_right  = pos.x0 + (r_local * pos.width)
    final_bottom = pos.y0 + (b_local * pos.height)
    final_top    = pos.y0 + (t_local * pos.height)


    # GridSpec setup to contain the subplots
    gs = gridspec.GridSpec(
        len(y_axis), 
        len(x_axis), 
        figure=fig,
        left=final_left,
        right=final_right,
        bottom=final_bottom,
        top = final_top,
        wspace=0, 
        hspace=0
    )


    # Extract axes from spatial dictionary exactly as your code did
    
    
    # Your original slice initialization
    newslice = [None, None, None, None]

    # First pass: Get y_max using your specific slicing logic
    y_max = 0
    for y in range(len(y_axis)):
        for x in range(len(x_axis)):
            newslice[0] = idxtoslice
            newslice[1] = x
            newslice[2] = y
            newslice[3] = slice(None)
            
            y_maxnew = np.max(np.abs(spects[tuple(newslice)]))
            if y_max < y_maxnew:
                y_max = y_maxnew

    # Second pass: Plotting using your logic but targeting the GridSpec
    ax_objects = np.empty((len(y_axis), len(x_axis)), dtype=object)

    for y in range(len(y_axis)):
        for x in range(len(x_axis)):
            # CREATE SUBPLOT AT GRID POSITION
            # This is the "fusion" point: creating the axis inside the figure
            ax = fig.add_subplot(gs[y, x])
            
            newslice[0] = idxtoslice
            newslice[1] = x
            newslice[2] = y
            newslice[3] = slice(None)

            # Your specific data processing
            spe = np.abs(spects[tuple(newslice)])
            spe = np.roll(spe, -235)

            # Your plotting parameters
            ax.plot(ppm_axis, spe,color = 'orange', linewidth=1.5)
            
            # Formatting as per your code
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlim(25, -25)
            ax.set_ylim(0, y_max)
            
            # Optional: make background of small plots transparent to see MRI
            ax.patch.set_alpha(0) 
            
            ax_objects[y, x] = ax

    #fig.suptitle(title) 
    
    return fig, ax_objects


def csi_overlay_coronal(spects, img_array, idxtoslice, spatial, fov_img, fov_spec, offset,figsize, title):
    """
    Overlay coronal CSI spectra on a 2D anatomical image using GridSpec alignment.

    This function overlays voxel-wise CSI spectra onto a coronal anatomical
    image by placing a grid of transparent spectral plots within the
    anatomical image bounds. Each subplot corresponds to one (z, x) CSI
    voxel and is aligned using a manually computed field-of-view mapping.

    Parameters
    ----------
    spects : np.ndarray
        Multidimensional CSI spectral array with expected dimension order:
        (z, x, y, spectral_points).
    img_array : np.ndarray
        2D anatomical image used as the background for the overlay.
    idxtoslice : int
        Index of the coronal slice (y-dimension) to visualize.
    spatial : dict
        Dictionary containing spatial and spectral axes with keys:
        'x', 'y', 'z', and 'ppm'.
    fov_img : tuple or array-like
        Field-of-view of the anatomical image.
    fov_spec : tuple or array-like
        Field-of-view of the CSI grid.
    offset : tuple or array-like
        Spatial offset between CSI and anatomical image.
    figsize : tuple
        Figure size passed to Matplotlib.
    title : str
        Title for the figure (currently not displayed by default).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib figure object.
    ax_objects : np.ndarray of matplotlib.axes.Axes
        Array of Axes objects corresponding to CSI voxel spectra.

    Notes
    -----
    - Alignment is computed using `compute_alignment_3d` and applied via
      Matplotlib `GridSpec`.
    - Spectra are plotted as magnitude spectra with a fixed spectral shift.
    - All spectra are scaled to a shared y-axis limit determined from the
      maximum intensity across the slice.
    - Subplot axes are hidden and rendered with transparent backgrounds
      to preserve visibility of the anatomical image.
    """
    x_axis, y_axis, z_axis, ppm_axis = spatial['x'], spatial['y'], spatial['z'], spatial['ppm']
    
    fig = plt.figure(figsize=figsize)
    ax_main = fig.add_subplot(111)

    # Display the background image
    ax_main.imshow(img_array, cmap='gray', extent=[0, img_array.shape[1], 0, img_array.shape[0]], origin='lower', zorder=0)
    ax_main.set_axis_off()

    pos = ax_main.get_position()
    
    # Calculate alignment
    #l_local, r_local, b_local, t_local = compute_alignment2(fov_spec, fov_img, offset)
    l_local, r_local, b_local, t_local = compute_alignment2(
    fov_spec, 
    fov_img, 
    offset_y=offset, 
    offset_x=offset, # Add if you have x-offset
    nx=len(x_axis), 
    ny=len(y_axis)
    )

    final_left   = pos.x0 + (l_local * pos.width)
    final_right  = pos.x0 + (r_local * pos.width)
    final_bottom = pos.y0 + (b_local * pos.height)
    final_top    = pos.y0 + (t_local * pos.height)

    # GridSpec setup: Rows = Z, Cols = X (Standard Coronal orientation)
    gs = gridspec.GridSpec(
        len(z_axis), 
        len(y_axis), 
        figure=fig,
        left=final_left,
        right=final_right,
        bottom=final_bottom,
        top=final_top,
        wspace=0, 
        hspace=0
    )

    newslice = [None, None, None, None]

    # First pass: Get global y_max for scaling
    y_max = 0

    ax_objects = np.empty((len(z_axis), len(y_axis)), dtype=object)
    
    newslice = [None,None,None,None]
    for x in range(len(x_axis)):
        for z in range(len(z_axis)):
        
            ax = fig.add_subplot(gs[z, x])

            newslice[0]= z
            newslice[1]= x
            newslice[2]= idxtoslice
            newslice[3]= slice(None)

            spe = np.abs(spects [tuple(newslice)])
            spe = np.roll(spe,-235)

            y_max = max(y_max, np.max(np.abs(spects[tuple(newslice)])))

            #ax[x,y].plot(ppm_axis,spec2_norm_shifted,linewidth  = 0.7)
            ax.plot(ppm_axis,spe,color = 'orange',linewidth  = 2)
            #ax[x,y].axvline(x= 3,linestyle = ':',color = 'red')#waterpeak
            # ax[x,y].axvline(x=-2.48, color='red', linestyle='--', label='γ-ATP (theoretical)')
            # ax[x,y].axvline(x=-7.52, color='green', linestyle='--', label='α-ATP (theoretical)')
            # ax[x,y].axvline(x=-16.26, color='purple', linestyle='--', label='β-ATP (theoretical)')
            
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlim(25,-25)

            ax.patch.set_alpha(0) 
            
            ax_objects[z, x] = ax

    #fig.suptitle(title)
    for ax in ax_objects.flat:
        ax.set_ylim(0, y_max)

    return fig, ax_objects



def csi_overlay_sagital(spects, img_array, idxtoslice, spatial, fov_img, fov_spec, offset,figsize, title):
    """
    Overlay sagittal CSI spectra on a 2D anatomical image using GridSpec alignment.

    This function overlays voxel-wise CSI spectra onto a sagittal anatomical
    image by embedding a grid of transparent spectral plots aligned to the
    CSI voxel layout. Each subplot represents one (y, z) voxel and is
    positioned using a manually defined field-of-view alignment.

    Parameters
    ----------
    spects : np.ndarray
        Multidimensional CSI spectral array with expected dimension order:
        (z, x, y, spectral_points).
    img_array : np.ndarray
        2D anatomical image serving as the background.
    idxtoslice : int
        Index of the sagittal slice (x-dimension) to visualize.
    spatial : dict
        Dictionary containing spatial and spectral axes with keys:
        'x', 'y', 'z', and 'ppm'.
    fov_img : tuple or array-like
        Field-of-view of the anatomical image.
    fov_spec : tuple or array-like
        Field-of-view of the CSI grid.
    offset : tuple or array-like
        Spatial offset between CSI and anatomical image.
    figsize : tuple
        Size of the Matplotlib figure.
    title : str
        Title for the figure (currently not displayed).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib figure object.
    ax_objects : np.ndarray of matplotlib.axes.Axes
        Array of Axes objects for the overlaid CSI spectra.

    Notes
    -----
    - Alignment is handled manually using `compute_alignment` and GridSpec.
    - Spectra are plotted as magnitude spectra with a fixed circular shift.
    - All voxel spectra share a common y-axis scale determined globally.
    - Subplot backgrounds are transparent to allow the anatomical image
      to remain visible beneath the CSI overlay.
    """
    x_axis, y_axis, z_axis, ppm_axis = spatial['x'], spatial['y'], spatial['z'], spatial['ppm']
    
    fig = plt.figure(figsize=figsize)
    ax_main = fig.add_subplot(111)

    # Display the background image
    ax_main.imshow(img_array, cmap='gray', extent=[0, img_array.shape[1], 0, img_array.shape[0]], origin='lower', zorder=0)
    ax_main.set_axis_off()

    pos = ax_main.get_position()
    
    # Calculate alignment
    l_local, r_local, b_local, t_local = compute_alignment2(
        fov_spec, 
        fov_img, 
        offset_y=offset, 
        offset_x=offset, # Add if you have x-offset
        nx=len(x_axis), 
        ny=len(y_axis)
        )

    final_left   = pos.x0 + (l_local * pos.width)
    final_right  = pos.x0 + (r_local * pos.width)
    final_bottom = pos.y0 + (b_local * pos.height)
    final_top    = pos.y0 + (t_local * pos.height)

    # GridSpec setup: Rows = Z, Cols = X (Standard Coronal orientation)
    gs = gridspec.GridSpec(
        len(y_axis), 
        len(z_axis), 
        figure=fig,
        left=final_left,
        right=final_right,
        bottom=final_bottom,
        top=final_top,
        wspace=0, 
        hspace=0
    )

    newslice = [None, None, None, None]

    # First pass: Get global y_max for scaling
    y_max = 0

    ax_objects = np.empty((len(y_axis), len(z_axis)), dtype=object)
    
    newslice = [None,None,None,None]
    for y in range(len(y_axis)):
        for z in range(len(z_axis)):
        
            ax = fig.add_subplot(gs[y, z])

            newslice[0]= z
            newslice[1]= idxtoslice
            newslice[2]= y
            newslice[3]= slice(None)

            spe = np.abs(spects [tuple(newslice)])
            spe = np.roll(spe,-235)

            y_max = max(y_max, np.max(np.abs(spects[tuple(newslice)])))

            #ax[x,y].plot(ppm_axis,spec2_norm_shifted,linewidth  = 0.7)
            ax.plot(ppm_axis,spe,color = 'orange',linewidth  = 2)
            #ax[x,y].axvline(x= 3,linestyle = ':',color = 'red')#waterpeak
            # ax[x,y].axvline(x=-2.48, color='red', linestyle='--', label='γ-ATP (theoretical)')
            # ax[x,y].axvline(x=-7.52, color='green', linestyle='--', label='α-ATP (theoretical)')
            # ax[x,y].axvline(x=-16.26, color='purple', linestyle='--', label='β-ATP (theoretical)')
            #plt.axvline(x=3, color='red', linestyle='--'
            #plt.axvline(x=3, color='red', linestyle='--'
            
            
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlim(25,-25)

            ax.patch.set_alpha(0) 
            
            ax_objects[y, z] = ax

    #fig.suptitle(title)
    
    for ax in ax_objects.flat:
        ax.set_ylim(0, y_max)

    return fig, ax_objects

