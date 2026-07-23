import numpy as np

def get_real_data(name):
    #norm data
    datapoints =np.load(f'./processed_data/{name}/beta_atp.npy')
    normed_datapoints = datapoints/np.max(datapoints)


    #timeaxis_exp = np.load('./processed_data/time_axis.npy')
    
    time_exp = 60 + np.arange(len(normed_datapoints)) * 8
    return time_exp,normed_datapoints

def get_inorganic_phosphate(name):
    datapoints =np.load(f'./processed_data/{name}/Inorganic_Phosphate.npy')
    normed_datapoints = datapoints/np.max(datapoints)

    normed_datapoints = normed_datapoints - normed_datapoints[0]


    #timeaxis_exp = np.load('./processed_data/time_axis.npy')
    
    time_exp = 60 + np.arange(len(normed_datapoints)) * 8
    return time_exp,normed_datapoints


def sim_rmse(simcurve,datapoints,time_exp):
    simvalues = np.array([simcurve[t] for t in time_exp])
    return np.sqrt(np.mean((simvalues-datapoints)**2))

def mse(ATP_sim,ATP_real):
    distance = np.sqrt(np.mean((ATP_sim-ATP_real)**2))
    return distance

import numpy as np
from scipy import stats



def rbf_kernel(t1, t2, lengthscale, sigma_sq):
    """Calculates the Squared Exponential covariance matrix."""
    # Pairwise squared distances between time points
    dist_sq = (t1[:, None] - t2[None, :])**2
    return sigma_sq * np.exp(-dist_sq / (2 * lengthscale**2))

#def marginal_likelyhood():
def calculate_gp_log_likelihood(time_obs, residuals, lengthscale, gp_sigma_sq, sigma_noise):

    
    # 1. Build the GP Covariance Matrix for the observation time points
    K = rbf_kernel(time_obs, time_obs, lengthscale, gp_sigma_sq)
    
    # 2. Add the observation noise (White Noise) to the diagonal
    # This is K + sigma^2 * I
    Ky = K + (sigma_noise**2) * np.eye(len(time_obs))
    
    # 3. Use the Multivariate Normal to get the Log-Likelihood
    # Mean is 0 because the GP models the *deviation* from the mechanistic model
    try:
        # We use logpdf because likelihoods can get extremely small; 
        # log-space prevents numerical underflow.
        log_prob = stats.multivariate_normal.logpdf(
            residuals, 
            mean=np.zeros(len(residuals)), 
            cov=Ky, 
            allow_singular=True
        )
        return log_prob
    
    except np.linalg.LinAlgError:
        # Return a very small number if the matrix is non-invertible
        return -1e10
