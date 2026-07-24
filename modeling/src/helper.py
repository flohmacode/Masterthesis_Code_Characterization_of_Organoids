import numpy as np

def get_real_data(name):
    """
    Loads and normalizes real experimental data for ATP (beta-ATP) from a specified dataset.

    Parameters:
    - name (str): Name of the dataset (used to construct the file path).

    Returns:
    - tuple: (time_exp, normed_datapoints)
      - time_exp (numpy.ndarray): Time points for the experimental data (in minutes).
      - normed_datapoints (numpy.ndarray): Normalized ATP data points (scaled to [0, 1]).

    Notes:
    - The data is loaded from a .npy file located in `./modeling/processed_data/{name}/beta_atp.npy`.
    - Time points start at 60 minutes and are spaced 8 minutes apart.
    - The data is normalized by dividing by its maximum value.
    """
    datapoints =np.load(f'./modeling/processed_data/{name}/beta_atp.npy')
    normed_datapoints = datapoints/np.max(datapoints)
    time_exp = 60 + np.arange(len(normed_datapoints)) * 8
    return time_exp,normed_datapoints

def get_inorganic_phosphate(name):
    """
    Loads and normalizes real experimental data for inorganic phosphate from a specified dataset.

    Parameters:
    - name (str): Name of the dataset (used to construct the file path).

    Returns:
    - tuple: (time_exp, normed_datapoints)
      - time_exp (numpy.ndarray): Time points for the experimental data (in minutes).
      - normed_datapoints (numpy.ndarray): Normalized inorganic phosphate data points (scaled to [0, 1] and baseline-corrected).

    Notes:
    - The data is loaded from a .npy file located in `./modeling/processed_data/{name}/Inorganic_Phosphate.npy`.
    - Time points start at 60 minutes and are spaced 8 minutes apart.
    - The data is normalized by dividing by its maximum value and then baseline-corrected by subtracting the first value.
    """
    datapoints =np.load(f'./modeling/processed_data/{name}/Inorganic_Phosphate.npy')
    normed_datapoints = datapoints/np.max(datapoints)
    normed_datapoints = normed_datapoints - normed_datapoints[0]    
    time_exp = 60 + np.arange(len(normed_datapoints)) * 8
    return time_exp,normed_datapoints


def sim_rmse(simcurve,datapoints,time_exp):
    """
    Computes the Root Mean Square Error (RMSE) between a simulated curve and experimental data points.

    Parameters:
    - simcurve (numpy.ndarray): Simulated curve (e.g., ATP or waste levels over time).
    - datapoints (numpy.ndarray): Experimental data points to compare against.
    - time_exp (numpy.ndarray): Time points corresponding to the experimental data.

    Returns:
    - float: RMSE value between the simulated curve and experimental data at the specified time points.

    Notes:
    - The simulated curve is sampled at the experimental time points for comparison.
    """
    simvalues = np.array([simcurve[t] for t in time_exp])
    return np.sqrt(np.mean((simvalues-datapoints)**2))

def mse(ATP_sim,ATP_real):
    """
    Computes the Mean Squared Error (MSE) between two arrays.

    Parameters:
    - ATP_sim (numpy.ndarray): Simulated ATP data.
    - ATP_real (numpy.ndarray): Real ATP data.

    Returns:
    - float: MSE value between the simulated and real data.

    Notes:
    - This is a simple wrapper for the RMSE calculation without the square root.
    """
    distance = np.sqrt(np.mean((ATP_sim-ATP_real)**2))
    return distance

